#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M",
}

CUTOFFS = [3.9, 5.0, 8.0]


@dataclass(frozen=True)
class Residue:
    chain: str
    residue_id: str
    resname: str
    one_letter: str
    label: str
    heavy_atoms: tuple[tuple[float, float, float], ...]


def norm_chain(value: str | None) -> str:
    value = (value or "").strip()
    if value in {"", "_", "blank", "BLANK"}:
        return "_"
    return value


def residue_id_from_pdb_fields(resseq: str, insertion_code: str) -> str:
    insertion_code = insertion_code.strip()
    return f"{resseq}{insertion_code}" if insertion_code else resseq


def parse_pdb_residues(pdb_path: Path, chain_filter: str | None = None) -> list[Residue]:
    wanted_chain = None if chain_filter is None else norm_chain(chain_filter)

    residues: list[Residue] = []
    current_key: tuple[str, str, str, str] | None = None
    current_atoms: list[tuple[float, float, float]] = []

    def flush() -> None:
        nonlocal current_key, current_atoms
        if current_key is None:
            return
        chain, residue_id, _icode, resname = current_key
        one_letter = THREE_TO_ONE.get(resname)
        if one_letter and current_atoms:
            residues.append(
                Residue(
                    chain=chain,
                    residue_id=residue_id,
                    resname=resname,
                    one_letter=one_letter,
                    label=f"{chain}.{resname}.{residue_id}",
                    heavy_atoms=tuple(current_atoms),
                )
            )
        current_key = None
        current_atoms = []

    with pdb_path.open() as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            altloc = line[16].strip()
            if altloc not in ("", "A"):
                continue

            chain = line[21].strip() or "_"
            if wanted_chain is not None and chain != wanted_chain:
                continue

            resname = line[17:20].strip()
            if resname not in THREE_TO_ONE:
                continue

            resseq = line[22:26].strip()
            if not resseq:
                continue

            insertion_code = line[26]
            residue_id = residue_id_from_pdb_fields(resseq, insertion_code)
            key = (chain, residue_id, insertion_code.strip(), resname)

            if key != current_key:
                flush()
                current_key = key

            atom_name = line[12:16].strip()
            element = line[76:78].strip().upper() or atom_name[:1].upper()
            if element == "H":
                continue

            current_atoms.append(
                (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            )

    flush()
    return residues


def needleman_wunsch_map(query_seq: str, bound_seq: str) -> tuple[list[int | None], dict[str, int | float]]:
    match_score = 2
    mismatch_score = -1
    gap_score = -2
    n = len(query_seq)
    m = len(bound_seq)

    scores = [[0] * (m + 1) for _ in range(n + 1)]
    trace = [[""] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        scores[i][0] = scores[i - 1][0] + gap_score
        trace[i][0] = "up"
    for j in range(1, m + 1):
        scores[0][j] = scores[0][j - 1] + gap_score
        trace[0][j] = "left"

    for i in range(1, n + 1):
        qi = query_seq[i - 1]
        for j in range(1, m + 1):
            bj = bound_seq[j - 1]
            diag = scores[i - 1][j - 1] + (match_score if qi == bj else mismatch_score)
            up = scores[i - 1][j] + gap_score
            left = scores[i][j - 1] + gap_score
            if diag >= up and diag >= left:
                scores[i][j] = diag
                trace[i][j] = "diag"
            elif up >= left:
                scores[i][j] = up
                trace[i][j] = "up"
            else:
                scores[i][j] = left
                trace[i][j] = "left"

    query_to_bound: list[int | None] = [None] * n
    mapped_positions = 0
    exact_matches = 0
    query_gap_positions = 0
    bound_gap_positions = 0

    i, j = n, m
    while i > 0 or j > 0:
        move = trace[i][j]
        if move == "diag":
            query_to_bound[i - 1] = j - 1
            mapped_positions += 1
            if query_seq[i - 1] == bound_seq[j - 1]:
                exact_matches += 1
            i -= 1
            j -= 1
        elif move == "up":
            query_to_bound[i - 1] = None
            query_gap_positions += 1
            i -= 1
        elif move == "left":
            bound_gap_positions += 1
            j -= 1
        else:
            break

    identity = exact_matches / mapped_positions if mapped_positions else 0.0
    return query_to_bound, {
        "query_length": n,
        "bound_length": m,
        "mapped_positions": mapped_positions,
        "exact_matches": exact_matches,
        "alignment_identity_mapped": identity,
        "query_gap_positions": query_gap_positions,
        "bound_gap_positions": bound_gap_positions,
    }


def residue_pair_min_distance(r1: Residue, r2: Residue) -> float:
    best_sq = math.inf
    for x1, y1, z1 in r1.heavy_atoms:
        for x2, y2, z2 in r2.heavy_atoms:
            dx = x1 - x2
            dy = y1 - y2
            dz = z1 - z2
            d2 = dx * dx + dy * dy + dz * dz
            if d2 < best_sq:
                best_sq = d2
    return math.sqrt(best_sq)


def nearest_partner_distances(
    bound_a: list[Residue],
    bound_b: list[Residue],
) -> tuple[list[float], list[float], list[str], list[str]]:
    dist_a = [math.inf] * len(bound_a)
    dist_b = [math.inf] * len(bound_b)
    partner_a = [""] * len(bound_a)
    partner_b = [""] * len(bound_b)

    for i, ra in enumerate(bound_a):
        for j, rb in enumerate(bound_b):
            d = residue_pair_min_distance(ra, rb)
            if d < dist_a[i]:
                dist_a[i] = d
                partner_a[i] = rb.label
            if d < dist_b[j]:
                dist_b[j] = d
                partner_b[j] = ra.label

    return dist_a, dist_b, partner_a, partner_b


def build_query_lookup(residues: list[Residue]) -> tuple[dict[str, int], dict[tuple[str, str], list[int]]]:
    by_label = {r.label: i for i, r in enumerate(residues)}
    by_id_resname: dict[tuple[str, str], list[int]] = {}
    for i, r in enumerate(residues):
        by_id_resname.setdefault((r.residue_id, r.resname), []).append(i)
    return by_label, by_id_resname


def locate_query_index(
    row: dict[str, str],
    by_label: dict[str, int],
    by_id_resname: dict[tuple[str, str], list[int]],
) -> tuple[int | None, str]:
    label = row.get("pdb_residue_label", "").strip()
    residue_id = row.get("pdb_residue_id", "").strip()
    resname = row.get("pdb_resname", "").strip()

    if label and label in by_label:
        return by_label[label], "label"

    candidates = by_id_resname.get((residue_id, resname), [])
    if len(candidates) == 1:
        return candidates[0], "residue_id_resname"
    if len(candidates) > 1:
        return None, "ambiguous_residue_id_resname"
    return None, "not_found"


def resolve_query_pdb(out_dir: Path, row: dict[str, str], query_side: str) -> tuple[Path, str | None, str]:
    """Return query/unbound PDB path, optional chain filter, and source label.

    Prefer the single-chain PDB emitted by combine_ifrag_radi.py when present.
    If it is absent, fall back to the BM5 manifest unbound PDB plus query-side chain.
    """
    candidates = sorted(out_dir.glob(f"{query_side}.surface_input_chain_*.pdb"))
    if len(candidates) == 1:
        return candidates[0], None, "combine_surface_input"
    if len(candidates) > 1:
        raise RuntimeError(
            f"{out_dir}: expected at most one {query_side}.surface_input_chain_*.pdb, found {len(candidates)}"
        )

    role = row[f"{query_side}_role"].strip()
    if role == "receptor":
        return Path(row["receptor_unbound_pdb"]), row[f"{query_side}_chain"], "manifest_unbound_chain"
    if role == "ligand":
        return Path(row["ligand_unbound_pdb"]), row[f"{query_side}_chain"], "manifest_unbound_chain"

    raise RuntimeError(f"{row['chainpair_id']}: unsupported {query_side}_role={role!r}")


def side_bound_spec(row: dict[str, str], query_side: str) -> tuple[Path, str, str]:
    role = row[f"{query_side}_role"].strip()
    if role == "receptor":
        return Path(row["receptor_bound_pdb"]), row["receptor_chain"], role
    if role == "ligand":
        return Path(row["ligand_bound_pdb"]), row["ligand_chain"], role
    raise RuntimeError(f"{row['chainpair_id']}: unsupported {query_side}_role={role!r}")


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader.fieldnames or []), list(reader)


def process_case(row: dict[str, str], score_fields_seen: list[str]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    cid = row["chainpair_id"]
    out_dir = Path(row["planned_output_dir"])

    query_pdb: dict[str, Path] = {}
    query_chain_filter: dict[str, str | None] = {}
    query_pdb_source: dict[str, str] = {}
    for side in ("query1", "query2"):
        qpath, qchain, qsource = resolve_query_pdb(out_dir, row, side)
        query_pdb[side] = qpath
        query_chain_filter[side] = qchain
        query_pdb_source[side] = qsource

    bound_pdb_1, bound_chain_1, role1 = side_bound_spec(row, "query1")
    bound_pdb_2, bound_chain_2, role2 = side_bound_spec(row, "query2")

    query_residues = {
        "query1": parse_pdb_residues(query_pdb["query1"], query_chain_filter["query1"]),
        "query2": parse_pdb_residues(query_pdb["query2"], query_chain_filter["query2"]),
    }

    for side in ("query1", "query2"):
        if not query_residues[side]:
            raise RuntimeError(
                f"{cid}: no query residues found for {side} chain "
                f"{query_chain_filter[side]!r} in {query_pdb[side]}"
            )
    bound_residues = {
        "query1": parse_pdb_residues(bound_pdb_1, bound_chain_1),
        "query2": parse_pdb_residues(bound_pdb_2, bound_chain_2),
    }

    if not bound_residues["query1"]:
        raise RuntimeError(f"{cid}: no bound query1 residues found for chain {bound_chain_1} in {bound_pdb_1}")
    if not bound_residues["query2"]:
        raise RuntimeError(f"{cid}: no bound query2 residues found for chain {bound_chain_2} in {bound_pdb_2}")

    q_to_b = {}
    align_stats = {}
    for side in ("query1", "query2"):
        qseq = "".join(r.one_letter for r in query_residues[side])
        bseq = "".join(r.one_letter for r in bound_residues[side])
        q_to_b[side], align_stats[side] = needleman_wunsch_map(qseq, bseq)

    nearest_1, nearest_2, partner_1, partner_2 = nearest_partner_distances(
        bound_residues["query1"], bound_residues["query2"]
    )
    nearest = {"query1": nearest_1, "query2": nearest_2}
    partners = {"query1": partner_1, "query2": partner_2}

    records: list[dict[str, str]] = []
    summaries: list[dict[str, str]] = []

    for side in ("query1", "query2"):
        score_path = out_dir / f"{side}_branch_scores.tsv"
        score_fields, score_rows = read_tsv(score_path)
        for f in score_fields:
            if f not in score_fields_seen:
                score_fields_seen.append(f)

        by_label, by_id_resname = build_query_lookup(query_residues[side])

        mapped = 0
        unmapped = 0
        interface_counts = {3.9: 0, 5.0: 0, 8.0: 0}

        for srow in score_rows:
            qidx, lookup_method = locate_query_index(srow, by_label, by_id_resname)
            bidx = None
            bound_residue = None
            mapping_status = lookup_method
            min_dist = math.inf
            nearest_partner = ""

            if qidx is None:
                unmapped += 1
            else:
                bidx = q_to_b[side][qidx]
                if bidx is None:
                    unmapped += 1
                    mapping_status = "alignment_gap"
                else:
                    mapped += 1
                    mapping_status = f"mapped_via_{lookup_method}"
                    bound_residue = bound_residues[side][bidx]
                    min_dist = nearest[side][bidx]
                    nearest_partner = partners[side][bidx]
                    for c in CUTOFFS:
                        if min_dist <= c:
                            interface_counts[c] += 1

            rec = {
                "chainpair_id": cid,
                "entity_id": row.get("entity_id", ""),
                "table_complex_id": row.get("table_complex_id", ""),
                "difficulty": row.get("difficulty", ""),
                "category_code": row.get("category_code", ""),
                "query_side": side,
                "query_role": row.get(f"{side}_role", ""),
                "query_pdb": str(query_pdb[side]),
                "query_chain": norm_chain(row.get(f"{side}_chain", "")),
                "query_pdb_source": query_pdb_source[side],
                "bound_pdb": str(bound_pdb_1 if side == "query1" else bound_pdb_2),
                "bound_chain": norm_chain(bound_chain_1 if side == "query1" else bound_chain_2),
                "query_sequence_index": "" if qidx is None else str(qidx + 1),
                "bound_sequence_index": "" if bidx is None else str(bidx + 1),
                "bound_residue_label": "" if bound_residue is None else bound_residue.label,
                "bound_residue_id": "" if bound_residue is None else bound_residue.residue_id,
                "bound_resname": "" if bound_residue is None else bound_residue.resname,
                "mapping_status": mapping_status,
                "min_partner_atom_distance_A": "" if not math.isfinite(min_dist) else f"{min_dist:.3f}",
                "nearest_partner_residue_label": nearest_partner,
                "interface_3p9A": "1" if math.isfinite(min_dist) and min_dist <= 3.9 else "0",
                "interface_5A": "1" if math.isfinite(min_dist) and min_dist <= 5.0 else "0",
                "interface_8A": "1" if math.isfinite(min_dist) and min_dist <= 8.0 else "0",
            }
            for f in score_fields:
                rec[f"score_{f}"] = srow.get(f, "")
            records.append(rec)

        summaries.append(
            {
                "chainpair_id": cid,
                "query_side": side,
                "query_role": row.get(f"{side}_role", ""),
                "query_pdb_residue_count": str(len(query_residues[side])),
                "bound_pdb_residue_count": str(len(bound_residues[side])),
                "score_rows": str(len(score_rows)),
                "mapped_score_rows": str(mapped),
                "unmapped_score_rows": str(unmapped),
                "interface_residue_count_3p9A": str(interface_counts[3.9]),
                "interface_residue_count_5A": str(interface_counts[5.0]),
                "interface_residue_count_8A": str(interface_counts[8.0]),
                "alignment_query_length": str(align_stats[side]["query_length"]),
                "alignment_bound_length": str(align_stats[side]["bound_length"]),
                "alignment_mapped_positions": str(align_stats[side]["mapped_positions"]),
                "alignment_exact_matches": str(align_stats[side]["exact_matches"]),
                "alignment_identity_mapped": f'{align_stats[side]["alignment_identity_mapped"]:.6f}',
            }
        )

    return records, summaries


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=Path("benchmark/manifests/bm5_phase1_runnable_chainpairs.tsv"))
    p.add_argument("--output-dir", type=Path, default=Path("benchmark/labels"))
    p.add_argument("--label-prefix", default="bm5_phase1_native_interface")
    p.add_argument("--only-chainpair-ids", default="")
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(args.manifest.open(), delimiter="\t"))
    only = {x.strip() for x in args.only_chainpair_ids.split(",") if x.strip()}
    if only:
        rows = [r for r in rows if r["chainpair_id"] in only]
    if args.limit is not None:
        rows = rows[: args.limit]

    score_fields_seen: list[str] = []
    all_records: list[dict[str, str]] = []
    all_summaries: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []

    for i, row in enumerate(rows, start=1):
        cid = row["chainpair_id"]
        print(f"[{i}/{len(rows)}] {cid}", flush=True)
        try:
            records, summaries = process_case(row, score_fields_seen)
            all_records.extend(records)
            all_summaries.extend(summaries)
        except Exception as e:
            errors.append({"chainpair_id": cid, "error": repr(e)})
            print(f"[ERROR] {cid}: {e}", flush=True)

    base_fields = [
        "chainpair_id", "entity_id", "table_complex_id", "difficulty", "category_code",
        "query_side", "query_role", "query_pdb", "query_chain", "query_pdb_source",
        "bound_pdb", "bound_chain",
        "query_sequence_index", "bound_sequence_index", "bound_residue_label",
        "bound_residue_id", "bound_resname", "mapping_status",
        "min_partner_atom_distance_A", "nearest_partner_residue_label",
        "interface_3p9A", "interface_5A", "interface_8A",
    ]
    label_path = args.output_dir / f"{args.label_prefix}.labels.tsv"
    with label_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            delimiter="\t",
            fieldnames=base_fields + [f"score_{x}" for x in score_fields_seen],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(all_records)

    summary_fields = [
        "chainpair_id", "query_side", "query_role", "query_pdb_residue_count",
        "bound_pdb_residue_count", "score_rows", "mapped_score_rows",
        "unmapped_score_rows", "interface_residue_count_3p9A",
        "interface_residue_count_5A", "interface_residue_count_8A",
        "alignment_query_length", "alignment_bound_length",
        "alignment_mapped_positions", "alignment_exact_matches",
        "alignment_identity_mapped",
    ]
    summary_path = args.output_dir / f"{args.label_prefix}.summary.tsv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=summary_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_summaries)

    error_path = args.output_dir / f"{args.label_prefix}.errors.tsv"
    with error_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=["chainpair_id", "error"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(errors)

    meta = {
        "manifest": str(args.manifest),
        "n_cases_requested": len(rows),
        "n_cases_with_errors": len(errors),
        "n_label_rows": len(all_records),
        "n_summary_rows": len(all_summaries),
        "cutoffs_A": CUTOFFS,
        "label_path": str(label_path),
        "summary_path": str(summary_path),
        "error_path": str(error_path),
    }
    meta_path = args.output_dir / f"{args.label_prefix}.meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    print("wrote:", label_path)
    print("wrote:", summary_path)
    print("wrote:", error_path)
    print("wrote:", meta_path)
    print("cases requested:", len(rows))
    print("cases with errors:", len(errors))
    print("label rows:", len(all_records))
    print("summary rows:", len(all_summaries))


if __name__ == "__main__":
    main()
