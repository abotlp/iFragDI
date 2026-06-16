#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path


THREE_TO_ONE = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "MSE": "M",
}


@dataclass(frozen=True)
class Residue:
    chain: str
    residue_id: str
    resname: str
    one_letter: str
    label: str
    heavy_atoms: tuple[tuple[float, float, float], ...]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compare BM5 docking-guidance residues from combine_ifrag_radi.py against the native "
            "interface seen in the bound receptor/ligand structures. "
            "The native interface is defined here as any heavy-atom inter-chain contact within "
            "--contact-cutoff angstroms."
        )
    )
    p.add_argument("--combine-out-dir", required=True, type=Path)
    p.add_argument("--bound-query1-pdb", required=True, type=Path)
    p.add_argument("--bound-query2-pdb", required=True, type=Path)
    p.add_argument(
        "--query1-pdb",
        type=Path,
        default=None,
        help=(
            "Optional query-side structure used by the combine run for query1. "
            "If omitted, the script auto-detects query1.surface_input_chain_*.pdb in --combine-out-dir."
        ),
    )
    p.add_argument(
        "--query2-pdb",
        type=Path,
        default=None,
        help=(
            "Optional query-side structure used by the combine run for query2. "
            "If omitted, the script auto-detects query2.surface_input_chain_*.pdb in --combine-out-dir."
        ),
    )
    p.add_argument(
        "--contact-cutoff",
        type=float,
        default=5.0,
        help="Heavy-atom distance cutoff in angstroms used to define native interface residues. Default: 5.0",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=None,
        help=(
            "Optional additional check on the top N rows from query1_residue_scores.tsv and "
            "query2_residue_scores.tsv."
        ),
    )
    return p.parse_args()


def residue_id_from_pdb_fields(resseq: str, insertion_code: str) -> str:
    insertion_code = insertion_code.strip()
    return f"{resseq}{insertion_code}" if insertion_code else resseq


def parse_pdb_residues(pdb_path: Path) -> list[Residue]:
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
            label = f"{chain}.{resname}.{residue_id}"
            residues.append(
                Residue(
                    chain=chain,
                    residue_id=residue_id,
                    resname=resname,
                    one_letter=one_letter,
                    label=label,
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
            resname = line[17:20].strip()
            resseq = line[22:26].strip()
            if not resseq:
                continue
            chain = line[21].strip() or "_"
            insertion_code = line[26]
            residue_id = residue_id_from_pdb_fields(resseq, insertion_code)
            key = (chain, residue_id, insertion_code.strip(), resname)
            if key != current_key:
                flush()
                current_key = key
            atom_name = line[12:16].strip()
            element = line[76:78].strip().upper()
            if not element:
                element = atom_name[:1].upper()
            if element == "H":
                continue
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            current_atoms.append((x, y, z))
    flush()
    return residues


def auto_detect_query_pdb(combine_out_dir: Path, query_name: str) -> Path:
    candidates = sorted(combine_out_dir.glob(f"{query_name}.surface_input_chain_*.pdb"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise SystemExit(
            f"Could not auto-detect the query-side PDB for {query_name} in {combine_out_dir}. "
            f"Please pass --{query_name}-pdb explicitly."
        )
    raise SystemExit(
        f"Found multiple possible query-side PDBs for {query_name} in {combine_out_dir}: "
        + ", ".join(str(path) for path in candidates)
        + f". Please pass --{query_name}-pdb explicitly."
    )


def needleman_wunsch_map(query_seq: str, bound_seq: str) -> tuple[list[int | None], dict[str, int]]:
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
    i = n
    j = m
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
    stats = {
        "query_length": n,
        "bound_length": m,
        "mapped_positions": mapped_positions,
        "exact_matches": exact_matches,
        "query_gap_positions": query_gap_positions,
        "bound_gap_positions": bound_gap_positions,
    }
    return query_to_bound, stats


def residue_pair_min_distance(res1: Residue, res2: Residue) -> float:
    best = math.inf
    for x1, y1, z1 in res1.heavy_atoms:
        for x2, y2, z2 in res2.heavy_atoms:
            dx = x1 - x2
            dy = y1 - y2
            dz = z1 - z2
            dist_sq = dx * dx + dy * dy + dz * dz
            if dist_sq < best:
                best = dist_sq
    return math.sqrt(best)


def compute_native_interface(
    bound_query1: list[Residue], bound_query2: list[Residue], cutoff: float
) -> tuple[set[int], set[int], list[float], list[float], list[str | None], list[str | None]]:
    interface_query1: set[int] = set()
    interface_query2: set[int] = set()
    nearest_partner_dist_query1 = [math.inf] * len(bound_query1)
    nearest_partner_dist_query2 = [math.inf] * len(bound_query2)
    nearest_partner_label_query1: list[str | None] = [None] * len(bound_query1)
    nearest_partner_label_query2: list[str | None] = [None] * len(bound_query2)

    for idx1, residue1 in enumerate(bound_query1):
        for idx2, residue2 in enumerate(bound_query2):
            distance = residue_pair_min_distance(residue1, residue2)
            if distance < nearest_partner_dist_query1[idx1]:
                nearest_partner_dist_query1[idx1] = distance
                nearest_partner_label_query1[idx1] = residue2.label
            if distance < nearest_partner_dist_query2[idx2]:
                nearest_partner_dist_query2[idx2] = distance
                nearest_partner_label_query2[idx2] = residue1.label
            if distance <= cutoff:
                interface_query1.add(idx1)
                interface_query2.add(idx2)

    return (
        interface_query1,
        interface_query2,
        nearest_partner_dist_query1,
        nearest_partner_dist_query2,
        nearest_partner_label_query1,
        nearest_partner_label_query2,
    )


def build_query_lookup(residues: list[Residue]) -> tuple[dict[str, int], dict[tuple[str, str], list[int]]]:
    by_label = {residue.label: idx for idx, residue in enumerate(residues)}
    by_id_resname: dict[tuple[str, str], list[int]] = {}
    for idx, residue in enumerate(residues):
        by_id_resname.setdefault((residue.residue_id, residue.resname), []).append(idx)
    return by_label, by_id_resname


def locate_query_index(
    row: dict[str, str], by_label: dict[str, int], by_id_resname: dict[tuple[str, str], list[int]]
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


def read_table_rows(tsv_path: Path, top_n: int | None = None) -> list[dict[str, str]]:
    with tsv_path.open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
    if top_n is None:
        return rows
    filtered: list[dict[str, str]] = []
    for row in rows:
        rank_text = row.get("rank", "").strip()
        if not rank_text:
            continue
        try:
            rank = int(rank_text)
        except ValueError:
            continue
        if rank <= top_n:
            filtered.append(row)
    return filtered


def evaluate_rows(
    *,
    rows: list[dict[str, str]],
    source_name: str,
    source_kind: str,
    query_name: str,
    query_residues: list[Residue],
    bound_residues: list[Residue],
    query_to_bound: list[int | None],
    native_interface_indices: set[int],
    nearest_partner_distances: list[float],
    nearest_partner_labels: list[str | None],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    by_label, by_id_resname = build_query_lookup(query_residues)
    result_rows: list[dict[str, object]] = []
    native_hits = 0
    mapped_rows = 0
    role_counts: dict[str, int] = {}
    role_hits: dict[str, int] = {}

    for row in rows:
        query_index, lookup_method = locate_query_index(row, by_label, by_id_resname)
        mapping_status = lookup_method
        bound_index: int | None = None
        bound_residue: Residue | None = None
        native_interface_hit = False
        min_partner_distance = None
        nearest_partner_label = None

        if query_index is not None:
            bound_index = query_to_bound[query_index]
            if bound_index is None:
                mapping_status = "alignment_gap"
            else:
                mapping_status = f"mapped_via_{lookup_method}"
                mapped_rows += 1
                bound_residue = bound_residues[bound_index]
                native_interface_hit = bound_index in native_interface_indices
                min_partner_distance = nearest_partner_distances[bound_index]
                nearest_partner_label = nearest_partner_labels[bound_index]
                if native_interface_hit:
                    native_hits += 1

        role = row.get("role", "").strip() or "all"
        role_counts[role] = role_counts.get(role, 0) + 1
        if native_interface_hit:
            role_hits[role] = role_hits.get(role, 0) + 1

        result_rows.append(
            {
                "query": query_name,
                "source_name": source_name,
                "source_kind": source_kind,
                "role": role,
                "role_rank": row.get("role_rank", ""),
                "global_rank": row.get("global_rank", row.get("rank", "")),
                "query_residue_label": row.get("pdb_residue_label", ""),
                "query_residue_id": row.get("pdb_residue_id", ""),
                "query_resname": row.get("pdb_resname", ""),
                "query_residue_score": row.get("residue_score", ""),
                "query_sequence_index": "" if query_index is None else query_index + 1,
                "bound_residue_label": "" if bound_residue is None else bound_residue.label,
                "bound_residue_id": "" if bound_residue is None else bound_residue.residue_id,
                "bound_resname": "" if bound_residue is None else bound_residue.resname,
                "bound_sequence_index": "" if bound_index is None else bound_index + 1,
                "native_interface_hit": int(native_interface_hit),
                "min_partner_atom_distance_A": (
                    "" if min_partner_distance is None else f"{min_partner_distance:.3f}"
                ),
                "nearest_partner_residue_label": nearest_partner_label or "",
                "mapping_status": mapping_status,
            }
        )

    role_summary = {}
    for role, count in sorted(role_counts.items()):
        hits = role_hits.get(role, 0)
        role_summary[role] = {
            "predicted_count": count,
            "native_interface_hits": hits,
            "native_interface_precision": (hits / count) if count else None,
        }

    summary = {
        "source_name": source_name,
        "source_kind": source_kind,
        "predicted_count": len(rows),
        "mapped_count": mapped_rows,
        "native_interface_hits": native_hits,
        "native_interface_precision": (native_hits / len(rows)) if rows else None,
        "role_summary": role_summary,
    }
    return result_rows, summary


def write_query_tsv(out_path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "query",
        "source_name",
        "source_kind",
        "role",
        "role_rank",
        "global_rank",
        "query_residue_label",
        "query_residue_id",
        "query_resname",
        "query_residue_score",
        "query_sequence_index",
        "bound_residue_label",
        "bound_residue_id",
        "bound_resname",
        "bound_sequence_index",
        "native_interface_hit",
        "min_partner_atom_distance_A",
        "nearest_partner_residue_label",
        "mapping_status",
    ]
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def maybe_relative(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def main() -> None:
    args = parse_args()
    combine_out_dir = args.combine_out_dir.resolve()
    bound_query1_pdb = args.bound_query1_pdb.resolve()
    bound_query2_pdb = args.bound_query2_pdb.resolve()
    query1_pdb = (args.query1_pdb.resolve() if args.query1_pdb else auto_detect_query_pdb(combine_out_dir, "query1"))
    query2_pdb = (args.query2_pdb.resolve() if args.query2_pdb else auto_detect_query_pdb(combine_out_dir, "query2"))

    query1_residues = parse_pdb_residues(query1_pdb)
    query2_residues = parse_pdb_residues(query2_pdb)
    bound_query1_residues = parse_pdb_residues(bound_query1_pdb)
    bound_query2_residues = parse_pdb_residues(bound_query2_pdb)

    query1_to_bound, query1_alignment = needleman_wunsch_map(
        "".join(residue.one_letter for residue in query1_residues),
        "".join(residue.one_letter for residue in bound_query1_residues),
    )
    query2_to_bound, query2_alignment = needleman_wunsch_map(
        "".join(residue.one_letter for residue in query2_residues),
        "".join(residue.one_letter for residue in bound_query2_residues),
    )

    (
        native_interface_query1,
        native_interface_query2,
        nearest_partner_dist_query1,
        nearest_partner_dist_query2,
        nearest_partner_label_query1,
        nearest_partner_label_query2,
    ) = compute_native_interface(bound_query1_residues, bound_query2_residues, args.contact_cutoff)

    query_specs = [
        (
            "query1",
            query1_residues,
            bound_query1_residues,
            query1_to_bound,
            query1_alignment,
            native_interface_query1,
            nearest_partner_dist_query1,
            nearest_partner_label_query1,
        ),
        (
            "query2",
            query2_residues,
            bound_query2_residues,
            query2_to_bound,
            query2_alignment,
            native_interface_query2,
            nearest_partner_dist_query2,
            nearest_partner_label_query2,
        ),
    ]

    summary: dict[str, object] = {
        "parameters": {
            "combine_out_dir": maybe_relative(combine_out_dir),
            "query1_pdb": maybe_relative(query1_pdb),
            "query2_pdb": maybe_relative(query2_pdb),
            "bound_query1_pdb": maybe_relative(bound_query1_pdb),
            "bound_query2_pdb": maybe_relative(bound_query2_pdb),
            "contact_cutoff": args.contact_cutoff,
            "top_n": args.top_n,
            "native_interface_definition": (
                "A residue is native-interface if any heavy atom is within "
                f"{args.contact_cutoff:.2f} A of the partner chain in the bound structures."
            ),
        }
    }

    for (
        query_name,
        query_residues,
        bound_residues,
        query_to_bound,
        alignment_stats,
        native_interface_indices,
        nearest_partner_distances,
        nearest_partner_labels,
    ) in query_specs:
        query_rows: list[dict[str, object]] = []
        evaluation_summaries: dict[str, object] = {}

        source_paths = [
            (
                f"{query_name}_docking_residues.tsv",
                "docking_residues_primary",
                combine_out_dir / f"{query_name}_docking_residues.tsv",
                None,
            ),
            (
                f"{query_name}_docking_residues.strict.tsv",
                "docking_residues_strict",
                combine_out_dir / f"{query_name}_docking_residues.strict.tsv",
                None,
            ),
            (
                f"{query_name}_docking_residues.loose.tsv",
                "docking_residues_loose",
                combine_out_dir / f"{query_name}_docking_residues.loose.tsv",
                None,
            ),
        ]
        if args.top_n is not None:
            source_paths.append(
                (
                    f"{query_name}_top_{args.top_n}_residue_scores",
                    "top_n_residue_scores",
                    combine_out_dir / f"{query_name}_residue_scores.tsv",
                    args.top_n,
                )
            )

        for source_name, source_kind, tsv_path, top_n in source_paths:
            if not tsv_path.exists():
                continue
            rows = read_table_rows(tsv_path, top_n=top_n)
            if not rows:
                continue
            evaluated_rows, evaluated_summary = evaluate_rows(
                rows=rows,
                source_name=source_name,
                source_kind=source_kind,
                query_name=query_name,
                query_residues=query_residues,
                bound_residues=bound_residues,
                query_to_bound=query_to_bound,
                native_interface_indices=native_interface_indices,
                nearest_partner_distances=nearest_partner_distances,
                nearest_partner_labels=nearest_partner_labels,
            )
            query_rows.extend(evaluated_rows)
            evaluation_summaries[source_name] = evaluated_summary

        query_rows.sort(
            key=lambda row: (
                str(row["source_name"]),
                str(row["role"]),
                int(row["global_rank"]) if str(row["global_rank"]).isdigit() else 10**9,
            )
        )
        write_query_tsv(combine_out_dir / f"native_interface_proximity.{query_name}.tsv", query_rows)

        summary[query_name] = {
            "query_residue_count": len(query_residues),
            "bound_residue_count": len(bound_residues),
            "alignment": alignment_stats,
            "bound_native_interface_residue_count": len(native_interface_indices),
            "bound_native_interface_residue_labels": [
                bound_residues[idx].label for idx in sorted(native_interface_indices)
            ],
            "evaluations": evaluation_summaries,
        }

    summary_path = combine_out_dir / "native_interface_proximity.summary.json"
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Wrote {maybe_relative(summary_path)}")
    for query_name in ("query1", "query2"):
        query_summary = summary.get(query_name, {})
        if not isinstance(query_summary, dict):
            continue
        native_count = query_summary.get("bound_native_interface_residue_count", 0)
        bound_count = query_summary.get("bound_residue_count", 0)
        print(f"{query_name}: native interface residues {native_count}/{bound_count}")
        evaluations = query_summary.get("evaluations", {})
        if not isinstance(evaluations, dict):
            continue
        for source_name, source_summary in evaluations.items():
            if not isinstance(source_summary, dict):
                continue
            predicted_count = source_summary.get("predicted_count", 0)
            hits = source_summary.get("native_interface_hits", 0)
            precision = source_summary.get("native_interface_precision")
            precision_text = "n/a" if precision is None else f"{100.0 * float(precision):.1f}%"
            print(f"  {source_name}: {hits}/{predicted_count} native-interface hits ({precision_text})")


if __name__ == "__main__":
    main()
