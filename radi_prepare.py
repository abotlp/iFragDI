#!/usr/bin/env python3
"""
Build the paired alignment used by raDI.

Stable biology:
- reuse the shared template-backed MMseqs homolog search
- keep one best resolved hit per accession on each query side
- build paired interolog rows only from interaction-supported homolog pairs
- do not hard-filter to same-taxid; keep taxid as diagnostics
- fetch full homolog sequences for the selected accessions
- build one real per-chain FAMSA MSA for each side
- trim columns where the query has a gap
- concatenate the trimmed left/right rows into the paired raDI alignment
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from conservation import (
    PairSupport,
    canonicalize_accession,
    load_pair_graph,
    load_sequences_from_sources,
    make_pair_key,
    parse_alignment_fasta,
    pair_dataset_defaults,
    parse_int_field,
    read_single_fasta,
    run_famsa,
    trim_query_gap_columns,
    write_alignment_fasta,
    write_paired_msa,
    write_sequence_fasta,
    write_ssa,
)
from template_mmseqs import (
    HOMOLOG_SEARCH_MODE_CHOICES,
    ResolvedMmseqsHit,
    default_template_fasta,
    load_resolved_hits_tsv,
)


QUERY1_ROW_ID = "QUERY1"
QUERY2_ROW_ID = "QUERY2"
PAIR_DATASET_CHOICES = ("intact_biogrid", "intact_biogrid_string")


@dataclass(frozen=True)
class CandidatePair:
    query1_accession: str
    query2_accession: str
    query1_sequence_id: str
    query2_sequence_id: str
    query1_taxid: str | None
    query2_taxid: str | None
    combined_bitscore: float
    combined_evalue: float
    combined_aligned_query_positions: int
    mean_pident: float
    interaction_supported: bool
    pair_sources_label: str
    pair_detection_methods: str
    pair_support_score: float
    pair_source_count: int
    pair_pubmed_count: int
    pair_string_score: int

    @property
    def same_taxid(self) -> bool:
        return (
            self.query1_taxid is not None
            and self.query2_taxid is not None
            and self.query1_taxid == self.query2_taxid
        )


@dataclass(frozen=True)
class PairedRow:
    query1_accession: str
    query2_accession: str
    combined_row: str
    query1_taxid: str | None
    query2_taxid: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the interaction-supported paired MSA used by the inter-chain raDI branch."
    )
    parser.add_argument("--query1", required=True, type=Path, help="Single-sequence FASTA for protein 1.")
    parser.add_argument("--query2", required=True, type=Path, help="Single-sequence FASTA for protein 2.")
    parser.add_argument(
        "--query1-search-tsv",
        type=Path,
        required=True,
        help="Precomputed resolved homolog TSV for query1 from homolog_search.py.",
    )
    parser.add_argument(
        "--query2-search-tsv",
        type=Path,
        required=True,
        help="Precomputed resolved homolog TSV for query2 from homolog_search.py.",
    )
    parser.add_argument(
        "--prepared-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory with precomputed interaction-supported per-chain MSA outputs. "
            "When provided, reuse its trimmed per-chain FASTAs and sequence-backed pair table "
            "instead of rebuilding the per-chain FAMSAs."
        ),
    )
    parser.add_argument(
        "--pair-dataset",
        choices=PAIR_DATASET_CHOICES,
        default="intact_biogrid_string",
        help=(
            "Homolog-side pair universe. "
            "'intact_biogrid' is the curated core; "
            "'intact_biogrid_string' is the STRING-expanded universe."
        ),
    )
    parser.add_argument(
        "--pairs",
        type=Path,
        default=None,
        help="Optional override for the homolog-side interaction table.",
    )
    parser.add_argument(
        "--pairs-meta",
        type=Path,
        default=None,
        help="Optional override for the homolog-side pair metadata table.",
    )
    parser.add_argument(
        "--shared-search-mode",
        choices=HOMOLOG_SEARCH_MODE_CHOICES,
        default="template_iterative",
        help="Shared template-backed homolog-search mode used upstream by homolog_search.py.",
    )
    parser.add_argument(
        "--sequence-fasta",
        type=Path,
        default=None,
        help="Sequence FASTA used to recover the selected homolog sequences before FAMSA. Defaults to the template FASTA for --pair-dataset.",
    )
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory.")
    parser.add_argument("--famsa-bin", default="famsa", help="famsa executable.")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--verbose", action="store_true", help="Print progress information.")
    args = parser.parse_args()

    if args.threads <= 0:
        raise SystemExit("--threads must be > 0")
    if not args.query1_search_tsv.exists():
        raise SystemExit(f"Precomputed query1 search TSV not found: {args.query1_search_tsv}")
    if not args.query2_search_tsv.exists():
        raise SystemExit(f"Precomputed query2 search TSV not found: {args.query2_search_tsv}")
    if args.sequence_fasta is None:
        args.sequence_fasta = default_template_fasta(args.pair_dataset)
    if not args.sequence_fasta.exists():
        raise SystemExit(f"Sequence FASTA not found: {args.sequence_fasta}")
    if args.prepared_dir is not None and not args.prepared_dir.exists():
        raise SystemExit(f"Prepared directory not found: {args.prepared_dir}")
    return args


def hit_sort_key(hit: ResolvedMmseqsHit) -> tuple[float, float, int, float, str]:
    return (hit.evalue, -hit.bitscore, -hit.aligned_query_positions, -hit.pident, hit.accession)


def build_candidate_pair(
    *,
    query1_accession: str,
    query2_accession: str,
    query1_hit: ResolvedMmseqsHit,
    query2_hit: ResolvedMmseqsHit,
    pair_support: PairSupport | None,
) -> CandidatePair:
    return CandidatePair(
        query1_accession=query1_accession,
        query2_accession=query2_accession,
        query1_sequence_id=query1_hit.sequence_id,
        query2_sequence_id=query2_hit.sequence_id,
        query1_taxid=query1_hit.taxid,
        query2_taxid=query2_hit.taxid,
        combined_bitscore=query1_hit.bitscore + query2_hit.bitscore,
        combined_evalue=query1_hit.evalue * query2_hit.evalue,
        combined_aligned_query_positions=query1_hit.aligned_query_positions + query2_hit.aligned_query_positions,
        mean_pident=0.5 * (query1_hit.pident + query2_hit.pident),
        interaction_supported=True,
        pair_sources_label=pair_support.sources_label if pair_support else "unknown",
        pair_detection_methods=pair_support.detection_methods if pair_support else "",
        pair_support_score=pair_support.support_score if pair_support else 0.0,
        pair_source_count=pair_support.source_count if pair_support else 0,
        pair_pubmed_count=pair_support.pubmed_count if pair_support else 0,
        pair_string_score=pair_support.string_score_max if pair_support else 0,
    )


def candidate_pair_sort_key(pair: CandidatePair) -> tuple[float, float, float, int, float, str, str]:
    return (
        -pair.pair_support_score,
        -pair.combined_bitscore,
        pair.combined_evalue,
        -pair.combined_aligned_query_positions,
        -pair.mean_pident,
        pair.query1_accession,
        pair.query2_accession,
    )


def build_interaction_only_pairs(
    q1_hits: Dict[str, ResolvedMmseqsHit],
    q2_hits: Dict[str, ResolvedMmseqsHit],
    adjacency: Dict[str, set[str]],
    pair_support_by_key: Dict[tuple[str, str], PairSupport],
    allow_self_pairs: bool,
) -> tuple[List[CandidatePair], dict[str, int]]:
    q2_accessions = set(q2_hits)
    selected_pairs: List[CandidatePair] = []
    total_candidates = 0
    self_pairs_skipped = 0
    same_taxid_candidates = 0
    cross_taxid_candidates = 0
    missing_taxid_candidates = 0

    for acc1, hit1 in sorted(q1_hits.items(), key=lambda item: hit_sort_key(item[1])):
        compatible = [acc2 for acc2 in adjacency.get(acc1, ()) if acc2 in q2_accessions]
        compatible.sort(key=lambda accession: hit_sort_key(q2_hits[accession]))
        for acc2 in compatible:
            total_candidates += 1
            if not allow_self_pairs and acc1 == acc2:
                self_pairs_skipped += 1
                continue
            hit2 = q2_hits[acc2]
            if hit1.taxid is None or hit2.taxid is None:
                missing_taxid_candidates += 1
            elif hit1.taxid == hit2.taxid:
                same_taxid_candidates += 1
            else:
                cross_taxid_candidates += 1
            selected_pairs.append(
                build_candidate_pair(
                    query1_accession=acc1,
                    query2_accession=acc2,
                    query1_hit=hit1,
                    query2_hit=hit2,
                    pair_support=pair_support_by_key.get(make_pair_key(acc1, acc2)),
                )
            )

    stats = {
        "interaction_supported_candidates_total": total_candidates,
        "interaction_supported_self_pairs_skipped": self_pairs_skipped,
        "interaction_supported_same_taxid_candidates": same_taxid_candidates,
        "interaction_supported_cross_taxid_candidates": cross_taxid_candidates,
        "interaction_supported_missing_taxid_candidates": missing_taxid_candidates,
        "interaction_only_rows": len(selected_pairs),
    }
    return sorted(selected_pairs, key=candidate_pair_sort_key), stats


def write_detected_accessions(path: Path, hits: Dict[str, ResolvedMmseqsHit]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "accession",
                "sequence_id",
                "search_tier",
                "taxid",
                "evalue",
                "bitscore",
                "aligned_query_positions",
                "pident",
            ]
        )
        for accession in sorted(hits):
            hit = hits[accession]
            writer.writerow(
                [
                    accession,
                    hit.sequence_id,
                    hit.search_tier,
                    hit.taxid or "",
                    f"{hit.evalue:.6g}",
                    f"{hit.bitscore:.6g}",
                    hit.aligned_query_positions,
                    f"{hit.pident:.6g}",
                ]
            )


def write_pairable_pairs(path: Path, pairs: Iterable[CandidatePair]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "query1_accession",
                "query2_accession",
                "query1_sequence_id",
                "query2_sequence_id",
                "query1_taxid",
                "query2_taxid",
                "directional_pair_key",
                "undirected_pair_key",
                "same_taxid",
                "interaction_supported",
                "pair_sources_label",
                "pair_detection_methods",
                "pair_support_score",
                "pair_source_count",
                "pair_pubmed_count",
                "pair_string_score",
                "combined_bitscore",
                "combined_evalue",
            ]
        )
        for pair in pairs:
            writer.writerow(
                [
                    pair.query1_accession,
                    pair.query2_accession,
                    pair.query1_sequence_id,
                    pair.query2_sequence_id,
                    pair.query1_taxid or "",
                    pair.query2_taxid or "",
                    f"{pair.query1_accession}->{pair.query2_accession}",
                    "|".join(sorted((pair.query1_accession, pair.query2_accession))),
                    "1" if pair.same_taxid else "0",
                    "1",
                    pair.pair_sources_label,
                    pair.pair_detection_methods,
                    f"{pair.pair_support_score:.3f}",
                    pair.pair_source_count,
                    pair.pair_pubmed_count,
                    pair.pair_string_score,
                    f"{pair.combined_bitscore:.6g}",
                    f"{pair.combined_evalue:.6g}",
                ]
            )


def load_prepared_sequence_backed_pairs(
    path: Path,
    q1_hits: Dict[str, ResolvedMmseqsHit],
    q2_hits: Dict[str, ResolvedMmseqsHit],
) -> List[CandidatePair]:
    if not path.exists():
        raise FileNotFoundError(f"Prepared pair table not found: {path}")
    pairs: List[CandidatePair] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"query1_accession", "query2_accession"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise RuntimeError(f"{path}: expected a pairable-pairs TSV with query1_accession/query2_accession columns")
        for row in reader:
            acc1 = canonicalize_accession(row.get("query1_accession") or "")
            acc2 = canonicalize_accession(row.get("query2_accession") or "")
            if acc1 is None or acc2 is None:
                continue
            hit1 = q1_hits.get(acc1)
            hit2 = q2_hits.get(acc2)
            if hit1 is None or hit2 is None:
                continue
            pairs.append(
                CandidatePair(
                    query1_accession=acc1,
                    query2_accession=acc2,
                    query1_sequence_id=hit1.sequence_id,
                    query2_sequence_id=hit2.sequence_id,
                    query1_taxid=hit1.taxid,
                    query2_taxid=hit2.taxid,
                    combined_bitscore=hit1.bitscore + hit2.bitscore,
                    combined_evalue=hit1.evalue * hit2.evalue,
                    combined_aligned_query_positions=hit1.aligned_query_positions + hit2.aligned_query_positions,
                    mean_pident=0.5 * (hit1.pident + hit2.pident),
                    interaction_supported=True,
                    pair_sources_label=(row.get("sources_label") or "unknown").strip() or "unknown",
                    pair_detection_methods=(row.get("detection_methods") or "").strip(),
                    pair_support_score=float(row.get("pair_support_score") or 0.0),
                    pair_source_count=parse_int_field(row.get("source_count") or "0"),
                    pair_pubmed_count=parse_int_field(row.get("pubmed_count") or "0"),
                    pair_string_score=parse_int_field(row.get("pair_string_score") or "0"),
                )
            )
    return pairs


def ordered_unique_accessions(pairs: Iterable[CandidatePair], side: str) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for pair in pairs:
        accession = pair.query1_accession if side == "left" else pair.query2_accession
        if accession not in seen:
            seen.add(accession)
            ordered.append(accession)
    return ordered


def build_paired_rows_from_msas(
    pairs: Iterable[CandidatePair],
    left_rows: Dict[str, str],
    right_rows: Dict[str, str],
) -> tuple[List[PairedRow], int]:
    paired_rows: List[PairedRow] = []
    missing_pairs = 0
    for pair in pairs:
        left = left_rows.get(pair.query1_accession)
        right = right_rows.get(pair.query2_accession)
        if left is None or right is None:
            missing_pairs += 1
            continue
        paired_rows.append(
            PairedRow(
                query1_accession=pair.query1_accession,
                query2_accession=pair.query2_accession,
                combined_row=left + right,
                query1_taxid=pair.query1_taxid,
                query2_taxid=pair.query2_taxid,
            )
        )
    return paired_rows, missing_pairs


def rows_to_msa_records(rows: Iterable[PairedRow]) -> List[Tuple[str, str, str]]:
    return [(row.query1_accession, row.query2_accession, row.combined_row) for row in rows]


def main() -> int:
    args = parse_args()
    dataset_defaults = pair_dataset_defaults(args.pair_dataset)
    if args.pairs is None:
        args.pairs = dataset_defaults["pairs"]
    if args.pairs_meta is None:
        args.pairs_meta = dataset_defaults["pairs_meta"]
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    q1_header, q1_seq = read_single_fasta(args.query1)
    q2_header, q2_seq = read_single_fasta(args.query2)
    allow_self_pairs = q1_seq == q2_seq
    if args.verbose:
        print(f"[INFO] Query1: {q1_header} (length {len(q1_seq)})")
        print(f"[INFO] Query2: {q2_header} (length {len(q2_seq)})")

    q1_search_out = args.query1_search_tsv
    q2_search_out = args.query2_search_tsv
    q1_hits = load_resolved_hits_tsv(q1_search_out)
    q2_hits = load_resolved_hits_tsv(q2_search_out)
    write_detected_accessions(out_dir / "q1_detected_accessions.tsv", q1_hits)
    write_detected_accessions(out_dir / "q2_detected_accessions.tsv", q2_hits)

    reused_prepared_dir = args.prepared_dir is not None
    if reused_prepared_dir:
        assert args.prepared_dir is not None
        pair_graph_stats = {
            "metadata_used": True,
            "input_pair_rows": 0,
            "metadata_pair_rows": 0,
            "retained_pair_rows": 0,
            "filtered_pair_rows": 0,
            "pairs_missing_metadata": 0,
        }
        interaction_pairs = load_prepared_sequence_backed_pairs(
            args.prepared_dir / "pairable_pairs_sequence_backed.tsv",
            q1_hits,
            q2_hits,
        )
        interaction_pair_stats = {
            "interaction_supported_candidates_total": len(interaction_pairs),
            "interaction_supported_self_pairs_skipped": 0,
            "interaction_supported_same_taxid_candidates": sum(1 for pair in interaction_pairs if pair.same_taxid),
            "interaction_supported_cross_taxid_candidates": sum(
                1
                for pair in interaction_pairs
                if not pair.same_taxid and pair.query1_taxid is not None and pair.query2_taxid is not None
            ),
            "interaction_supported_missing_taxid_candidates": sum(
                1 for pair in interaction_pairs if pair.query1_taxid is None or pair.query2_taxid is None
            ),
            "interaction_only_rows": len(interaction_pairs),
        }
        write_pairable_pairs(out_dir / "pairable_pairs.tsv", interaction_pairs)
        write_pairable_pairs(out_dir / "pairable_pairs_interaction_only.tsv", interaction_pairs)
        write_pairable_pairs(out_dir / "pairable_pairs_sequence_backed.tsv", interaction_pairs)

        sequence_backed_pairs = interaction_pairs
        selected_left_accessions = ordered_unique_accessions(sequence_backed_pairs, "left")
        selected_right_accessions = ordered_unique_accessions(sequence_backed_pairs, "right")
        q1_sequence_backed_hits = {
            accession: q1_hits[accession] for accession in selected_left_accessions if accession in q1_hits
        }
        q2_sequence_backed_hits = {
            accession: q2_hits[accession] for accession in selected_right_accessions if accession in q2_hits
        }
        missing_sequence_ids: List[str] = []
        chain1_raw_fasta = args.prepared_dir / "chain1_raw.fasta"
        chain2_raw_fasta = args.prepared_dir / "chain2_raw.fasta"
        chain1_aligned_fasta = args.prepared_dir / "chain1_famsa.fasta"
        chain2_aligned_fasta = args.prepared_dir / "chain2_famsa.fasta"
        chain1_trimmed_fasta = args.prepared_dir / "chain1_trimmed.fasta"
        chain2_trimmed_fasta = args.prepared_dir / "chain2_trimmed.fasta"
        for required_path in (
            chain1_trimmed_fasta,
            chain2_trimmed_fasta,
            chain1_aligned_fasta,
            chain2_aligned_fasta,
        ):
            if not required_path.exists():
                raise FileNotFoundError(f"Prepared MSA file not found: {required_path}")
        chain1_trimmed = parse_alignment_fasta(chain1_trimmed_fasta)
        chain2_trimmed = parse_alignment_fasta(chain2_trimmed_fasta)
        chain1_trimmed_length = len(chain1_trimmed.get(q1_header, q1_seq))
        chain2_trimmed_length = len(chain2_trimmed.get(q2_header, q2_seq))
    else:
        adjacency, pair_support_by_key, pair_graph_stats = load_pair_graph(
            pairs_path=args.pairs,
            pairs_meta_path=args.pairs_meta,
            skip_self_pairs=False,
        )
        interaction_pairs, interaction_pair_stats = build_interaction_only_pairs(
            q1_hits,
            q2_hits,
            adjacency,
            pair_support_by_key=pair_support_by_key,
            allow_self_pairs=allow_self_pairs,
        )
        write_pairable_pairs(out_dir / "pairable_pairs.tsv", interaction_pairs)
        write_pairable_pairs(out_dir / "pairable_pairs_interaction_only.tsv", interaction_pairs)

        wanted_sequence_ids = {
            q1_hits[pair.query1_accession].sequence_id
            for pair in interaction_pairs
            if pair.query1_accession in q1_hits
        } | {
            q2_hits[pair.query2_accession].sequence_id
            for pair in interaction_pairs
            if pair.query2_accession in q2_hits
        }
        selected_sequences = load_sequences_from_sources([args.sequence_fasta], wanted_sequence_ids)
        missing_sequence_ids = sorted(wanted_sequence_ids - set(selected_sequences))

        sequence_backed_pairs = [
            pair
            for pair in interaction_pairs
            if pair.query1_sequence_id in selected_sequences and pair.query2_sequence_id in selected_sequences
        ]
        write_pairable_pairs(out_dir / "pairable_pairs_sequence_backed.tsv", sequence_backed_pairs)

        selected_left_accessions = ordered_unique_accessions(sequence_backed_pairs, "left")
        selected_right_accessions = ordered_unique_accessions(sequence_backed_pairs, "right")
        q1_sequence_backed_hits = {accession: q1_hits[accession] for accession in selected_left_accessions if accession in q1_hits}
        q2_sequence_backed_hits = {accession: q2_hits[accession] for accession in selected_right_accessions if accession in q2_hits}

    same_taxid_rows = sum(1 for pair in sequence_backed_pairs if pair.same_taxid)
    cross_taxid_rows = sum(1 for pair in sequence_backed_pairs if not pair.same_taxid and pair.query1_taxid and pair.query2_taxid)
    missing_taxid_rows = sum(1 for pair in sequence_backed_pairs if pair.query1_taxid is None or pair.query2_taxid is None)

    summary = {
        "query1_header": q1_header,
        "query2_header": q2_header,
        "query1_length": len(q1_seq),
        "query2_length": len(q2_seq),
        "homomer_query_mode": allow_self_pairs,
        "pair_dataset": args.pair_dataset,
        "pairs_file": str(args.pairs),
        "pairs_meta_file": str(args.pairs_meta),
        "pair_universe_used": args.pair_dataset,
        "pair_support_meta_used": pair_graph_stats["metadata_used"],
        "pair_universe_rows_loaded": pair_graph_stats["input_pair_rows"],
        "pair_universe_rows_with_metadata": pair_graph_stats["metadata_pair_rows"],
        "pair_universe_rows_retained": pair_graph_stats["retained_pair_rows"],
        "pair_universe_rows_filtered": pair_graph_stats["filtered_pair_rows"],
        "pair_universe_rows_missing_metadata": pair_graph_stats["pairs_missing_metadata"],
        "shared_search_logic": args.shared_search_mode,
        "reused_prepared_dir": str(args.prepared_dir) if args.prepared_dir is not None else None,
        "sequence_fasta": str(args.sequence_fasta),
        "pairing_mode_requested": "interaction_only",
        "pairing_mode_used": "interaction_only",
        "q1_homolog_accessions_found": len(q1_hits),
        "q2_homolog_accessions_found": len(q2_hits),
        "interaction_supported_homolog_pair_count": len(interaction_pairs),
        "selected_candidate_pair_count": len(interaction_pairs),
        "sequence_backed_candidate_pair_count": len(sequence_backed_pairs),
        "unique_selected_q1_side_homolog_count": len(selected_left_accessions),
        "unique_selected_q2_side_homolog_count": len(selected_right_accessions),
        "selected_homolog_accessions_missing_sequences": len(missing_sequence_ids),
        "missing_sequence_ids_sample": missing_sequence_ids[:50],
        "interaction_pair_stats": interaction_pair_stats,
        "same_taxid_rows_sequence_backed": same_taxid_rows,
        "cross_taxid_rows_sequence_backed": cross_taxid_rows,
        "missing_taxid_rows_sequence_backed": missing_taxid_rows,
        "query1_search_tsv": str(q1_search_out),
        "query2_search_tsv": str(q2_search_out),
        "warnings": [],
    }

    if not sequence_backed_pairs:
        summary["error"] = "No sequence-backed interaction-supported homolog pairs were found for raDI."
        (out_dir / "radi_prepare_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        raise SystemExit("No sequence-backed interaction-supported homolog pairs were found for raDI.")

    if not reused_prepared_dir:
        chain1_raw_fasta = out_dir / "chain1_raw.fasta"
        chain2_raw_fasta = out_dir / "chain2_raw.fasta"
        write_sequence_fasta(
            chain1_raw_fasta,
            QUERY1_ROW_ID,
            q1_seq,
            selected_left_accessions,
            q1_sequence_backed_hits,
            selected_sequences,
        )
        write_sequence_fasta(
            chain2_raw_fasta,
            QUERY2_ROW_ID,
            q2_seq,
            selected_right_accessions,
            q2_sequence_backed_hits,
            selected_sequences,
        )

        chain1_aligned_fasta = out_dir / "chain1_famsa.fasta"
        chain2_aligned_fasta = out_dir / "chain2_famsa.fasta"
        run_famsa(args.famsa_bin, chain1_raw_fasta, chain1_aligned_fasta, args.threads)
        run_famsa(args.famsa_bin, chain2_raw_fasta, chain2_aligned_fasta, args.threads)

        chain1_alignment = parse_alignment_fasta(chain1_aligned_fasta)
        chain2_alignment = parse_alignment_fasta(chain2_aligned_fasta)
        chain1_trimmed, chain1_trimmed_length = trim_query_gap_columns(chain1_alignment, QUERY1_ROW_ID, q1_seq)
        chain2_trimmed, chain2_trimmed_length = trim_query_gap_columns(chain2_alignment, QUERY2_ROW_ID, q2_seq)

        chain1_trimmed_fasta = out_dir / "chain1_trimmed.fasta"
        chain2_trimmed_fasta = out_dir / "chain2_trimmed.fasta"
        write_alignment_fasta(chain1_trimmed_fasta, chain1_trimmed, [QUERY1_ROW_ID, *selected_left_accessions])
        write_alignment_fasta(chain2_trimmed_fasta, chain2_trimmed, [QUERY2_ROW_ID, *selected_right_accessions])

    paired_rows, pairs_missing_after_alignment = build_paired_rows_from_msas(
        sequence_backed_pairs,
        chain1_trimmed,
        chain2_trimmed,
    )
    if not paired_rows:
        summary["error"] = "No paired rows survived sequence retrieval and MSA construction."
        summary["pairs_missing_after_alignment"] = pairs_missing_after_alignment
        (out_dir / "radi_prepare_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        raise SystemExit("No paired rows survived sequence retrieval and MSA construction.")

    alignment_array, msa_path = write_paired_msa(out_dir, q1_seq, q2_seq, rows_to_msa_records(paired_rows))
    ssa_path = write_ssa(out_dir, q1_seq, q2_seq)

    warnings: List[str] = []
    if cross_taxid_rows > 0:
        warnings.append(
            "Cross-taxid interaction-supported pairs were retained for the paired MSA; taxid is diagnostic only in the strict interaction-supported mode."
        )
    if len(paired_rows) < 20:
        warnings.append("Paired MSA depth is low; raDI anchors should be treated cautiously.")

    summary.update(
        {
            "chain1_raw_msa_depth": 1 + len(selected_left_accessions),
            "chain2_raw_msa_depth": 1 + len(selected_right_accessions),
            "chain1_trimmed_length": chain1_trimmed_length,
            "chain2_trimmed_length": chain2_trimmed_length,
            "final_paired_row_count": len(paired_rows),
            "paired_rows_used": len(paired_rows),
            "alignment_rows_total_including_query": 1 + len(paired_rows),
            "weak_msa_warning": len(paired_rows) < 20,
            "pairs_missing_after_alignment": pairs_missing_after_alignment,
            "paired_msa_path": str(msa_path),
            "paired_ssa_path": str(ssa_path),
            "paired_msa_rows_shape": list(alignment_array.shape),
            "warnings": warnings,
            "outputs": {
                "paired_msa": str(msa_path),
                "paired_ssa": str(ssa_path),
                "query1_search_tsv": str(q1_search_out),
                "query2_search_tsv": str(q2_search_out),
                "q1_detected_accessions_tsv": str(out_dir / "q1_detected_accessions.tsv"),
                "q2_detected_accessions_tsv": str(out_dir / "q2_detected_accessions.tsv"),
                "pairable_pairs_tsv": str(out_dir / "pairable_pairs.tsv"),
                "pairable_pairs_interaction_only_tsv": str(out_dir / "pairable_pairs_interaction_only.tsv"),
                "pairable_pairs_sequence_backed_tsv": str(out_dir / "pairable_pairs_sequence_backed.tsv"),
                "chain1_raw_fasta": str(chain1_raw_fasta),
                "chain2_raw_fasta": str(chain2_raw_fasta),
                "chain1_famsa_fasta": str(chain1_aligned_fasta),
                "chain2_famsa_fasta": str(chain2_aligned_fasta),
                "chain1_trimmed_fasta": str(chain1_trimmed_fasta),
                "chain2_trimmed_fasta": str(chain2_trimmed_fasta),
                "radi_prepare_summary_json": str(out_dir / "radi_prepare_summary.json"),
            },
        }
    )
    (out_dir / "radi_prepare_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.verbose:
        print(f"[INFO] Reusing shared homolog-search TSVs: {q1_search_out} and {q2_search_out}")
        print(f"[INFO] Homolog accessions: q1={len(q1_hits)} q2={len(q2_hits)}")
        print(f"[INFO] Interaction-supported homolog pairs: {len(interaction_pairs)}")
        print(f"[INFO] Sequence-backed homolog pairs: {len(sequence_backed_pairs)}")
        print(f"[INFO] Final paired rows: {len(paired_rows)}")
        for warning in warnings:
            print(f"[WARN] {warning}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
