#!/usr/bin/env python3
"""
Run the shared template-backed homolog search once for both query chains.

The stable default now searches directly against the interaction-template
sequence universe used by iFrag. This keeps conservation and raDI aligned
with the same template/resource choice as the PPI-driven branch.

Two modes are supported:
- template_iterative: one high-sensitivity iterative MMseqs search on the template DB
- template_single_pass: one lighter single-pass MMseqs search for smoke tests

The resolved accession-level TSVs are then reused by both:
- conservation.py
- radi_prepare.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from conservation import read_single_fasta, write_query_fasta
from template_mmseqs import (
    HOMOLOG_SEARCH_MODE_CHOICES,
    default_template_mmseqs_db,
    default_template_proteins,
    mmseqs_db_exists,
    normalize_homolog_search_mode,
    run_template_mmseqs_search,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the shared template-backed MMseqs homolog search for query1/query2."
    )
    parser.add_argument("--query1", required=True, type=Path, help="Single-sequence FASTA for protein 1.")
    parser.add_argument("--query2", required=True, type=Path, help="Single-sequence FASTA for protein 2.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory.")
    parser.add_argument(
        "--template-dataset",
        choices=("intact_biogrid", "intact_biogrid_string"),
        default="intact_biogrid_string",
        help="Interaction-template dataset used to resolve the default template MMseqs DB and proteins table.",
    )
    parser.add_argument(
        "--template-mmseqs-db",
        type=Path,
        default=None,
        help="Template MMseqs DB prefix used for the shared homolog search. Defaults to the selected template dataset resource.",
    )
    parser.add_argument(
        "--template-proteins",
        type=Path,
        default=None,
        help="proteins.final.tsv for the selected template dataset, used to recover per-accession taxids.",
    )
    parser.add_argument(
        "--search-mode",
        choices=HOMOLOG_SEARCH_MODE_CHOICES,
        default="template_iterative",
        help=(
            "Shared homolog-search mode. "
            "'template_iterative' runs one 4-iteration MMseqs search on the template DB; "
            "'template_single_pass' is a faster smoke-test mode."
        ),
    )
    parser.add_argument("--mmseqs-bin", default="mmseqs", help="mmseqs executable.")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-hits", type=int, default=100000)
    parser.add_argument(
        "--evalue",
        type=float,
        default=None,
        help="Optional MMseqs E-value cutoff. If omitted, keep the MMseqs default like the original RADI buildmsa.py workflow.",
    )
    parser.add_argument("--mmseqs-sensitivity", type=float, default=7.5)
    parser.add_argument(
        "--stage1-iterations",
        type=int,
        default=4,
        help="Number of MMseqs iterations in template_iterative mode. Defaults to 4.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.template_mmseqs_db is None:
        args.template_mmseqs_db = default_template_mmseqs_db(args.template_dataset)
    if args.template_proteins is None:
        args.template_proteins = default_template_proteins(args.template_dataset)
    if args.threads <= 0:
        raise SystemExit("--threads must be > 0")
    if args.max_hits <= 0:
        raise SystemExit("--max-hits must be > 0")
    if args.stage1_iterations <= 0:
        raise SystemExit("--stage1-iterations must be > 0")
    if not mmseqs_db_exists(args.template_mmseqs_db):
        raise SystemExit(
            f"Template MMseqs database not found: {args.template_mmseqs_db}\n"
            "Build it with mmseqs createdb and pass the database prefix to --template-mmseqs-db."
        )
    if not args.template_proteins.exists():
        raise SystemExit(f"Template proteins table not found: {args.template_proteins}")
    return args


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    q1_header, q1_seq = read_single_fasta(args.query1)
    q2_header, q2_seq = read_single_fasta(args.query2)
    if args.verbose:
        print(f"[INFO] Query1: {q1_header} (length {len(q1_seq)})")
        print(f"[INFO] Query2: {q2_header} (length {len(q2_seq)})")

    q1_query_fasta = out_dir / "query1_query.fa"
    q2_query_fasta = out_dir / "query2_query.fa"
    write_query_fasta(q1_query_fasta, q1_header, q1_seq)
    write_query_fasta(q2_query_fasta, q2_header, q2_seq)

    q1_search_out = out_dir / "query1_mmseqs.tsv"
    q2_search_out = out_dir / "query2_mmseqs.tsv"

    normalized_mode = normalize_homolog_search_mode(args.search_mode)

    q1_summary = run_template_mmseqs_search(
        mmseqs_bin=args.mmseqs_bin,
        query_fasta=q1_query_fasta,
        query_length=len(q1_seq),
        out_tsv=q1_search_out,
        work_dir=out_dir / "query1_mmseqs",
        template_db=args.template_mmseqs_db,
        template_proteins=args.template_proteins,
        threads=args.threads,
        max_hits=args.max_hits,
        sensitivity=args.mmseqs_sensitivity,
        evalue=args.evalue,
        stage1_iterations=args.stage1_iterations,
        search_mode=normalized_mode,
    )
    q2_summary = run_template_mmseqs_search(
        mmseqs_bin=args.mmseqs_bin,
        query_fasta=q2_query_fasta,
        query_length=len(q2_seq),
        out_tsv=q2_search_out,
        work_dir=out_dir / "query2_mmseqs",
        template_db=args.template_mmseqs_db,
        template_proteins=args.template_proteins,
        threads=args.threads,
        max_hits=args.max_hits,
        sensitivity=args.mmseqs_sensitivity,
        evalue=args.evalue,
        stage1_iterations=args.stage1_iterations,
        search_mode=normalized_mode,
    )

    summary = {
        "query1_header": q1_header,
        "query2_header": q2_header,
        "query1_length": len(q1_seq),
        "query2_length": len(q2_seq),
        "shared_search_logic": normalized_mode,
        "template_dataset": args.template_dataset,
        "template_mmseqs_db": str(args.template_mmseqs_db),
        "template_proteins": str(args.template_proteins),
        "stage1_iterations": args.stage1_iterations if normalized_mode == "template_iterative" else None,
        "evalue": args.evalue,
        "max_hits": args.max_hits,
        "mmseqs_sensitivity": args.mmseqs_sensitivity,
        "query1_search_summary": q1_summary,
        "query2_search_summary": q2_summary,
        "outputs": {
            "query1_search_tsv": str(q1_search_out),
            "query2_search_tsv": str(q2_search_out),
            "homolog_search_summary_json": str(out_dir / "homolog_search_summary.json"),
        },
    }
    (out_dir / "homolog_search_summary.json").write_text(json.dumps(summary, indent=2))

    if args.verbose:
        print(
            "[INFO] Resolved homolog accessions: "
            f"q1={q1_summary['resolved_accessions']} q2={q2_summary['resolved_accessions']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
