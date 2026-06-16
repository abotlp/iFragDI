#!/usr/bin/env python3
"""Build template FASTA resources from a merged UniProt-based PPI dataset.

This script reads proteins.final.tsv from the merged dataset, extracts matching
UniProt sequences from the local Swiss-Prot and TrEMBL FASTA archives, writes a
templates.fasta file, and copies the pair tables into a template-resource
directory for downstream iFragDI use.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import shutil
import sys
from pathlib import Path
from typing import Iterable, Iterator


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Build template FASTA from a merged dataset proteins.final.tsv",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=root / "data" / "datasets" / "intact_biogrid",
        help="Directory containing proteins.final.tsv and template_pairs files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=root / "data" / "interaction_templates" / "intact_biogrid",
        help="Output directory for templates.fasta and copied pair tables.",
    )
    parser.add_argument(
        "--uniprot-sprot",
        type=Path,
        default=root / "data" / "raw" / "uniprot_sprot.fasta.gz",
        help="Swiss-Prot FASTA archive.",
    )
    parser.add_argument(
        "--uniprot-trembl",
        type=Path,
        default=root / "data" / "raw" / "uniprot_trembl.fasta.gz",
        help="TrEMBL FASTA archive.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000000,
        help="Print progress to stderr every N FASTA entries scanned. Use 0 to disable.",
    )
    return parser.parse_args()


def die(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def ensure_exists(path: Path) -> None:
    if not path.exists():
        die(f"Required file not found: {path}")


def read_requested_accessions(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if "protein_id" not in (reader.fieldnames or []):
            die(f"{path} does not contain a protein_id column")
        accessions = [row["protein_id"].strip() for row in reader if row["protein_id"].strip()]
    if not accessions:
        die(f"No protein_id values found in {path}")
    return accessions


def iter_fasta(path: Path) -> Iterator[tuple[str, str]]:
    header: str | None = None
    seq_lines: list[str] = []
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_lines)
                header = line[1:]
                seq_lines = []
            else:
                seq_lines.append(line.strip())
    if header is not None:
        yield header, "".join(seq_lines)


def extract_accession(header: str) -> str:
    token = header.split(None, 1)[0]
    parts = token.split("|")
    if len(parts) >= 2 and parts[0] in {"sp", "tr"}:
        return parts[1]
    return token


def maybe_report(source: str, count: int, progress_every: int) -> None:
    if progress_every > 0 and count > 0 and count % progress_every == 0:
        print(f"[{source}] scanned {count:,} FASTA entries", file=sys.stderr)


def scan_source(
    fasta_path: Path,
    source_name: str,
    remaining: set[str],
    found_source: dict[str, str],
    found_header: dict[str, str],
    found_sequence: dict[str, str],
    progress_every: int,
) -> None:
    scanned = 0
    for header, sequence in iter_fasta(fasta_path):
        scanned += 1
        maybe_report(source_name, scanned, progress_every)
        accession = extract_accession(header)
        if accession in remaining and accession not in found_sequence:
            found_source[accession] = source_name
            found_header[accession] = header
            found_sequence[accession] = sequence
    print(f"[{source_name}] finished after {scanned:,} FASTA entries", file=sys.stderr)


def copy_if_present(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copy2(src, dst)


def write_tsv(path: Path, header: Iterable[str], rows: Iterable[Iterable[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(list(header))
        writer.writerows(rows)


def main() -> int:
    args = parse_args()

    dataset_dir = args.dataset_dir
    out_dir = args.out_dir
    proteins_path = dataset_dir / "proteins.final.tsv"
    pairs_path = dataset_dir / "template_pairs.final.tsv"
    pairs_meta_path = dataset_dir / "template_pairs.meta.final.tsv"

    ensure_exists(proteins_path)
    ensure_exists(pairs_path)
    ensure_exists(pairs_meta_path)
    ensure_exists(args.uniprot_sprot)
    ensure_exists(args.uniprot_trembl)

    requested_order = read_requested_accessions(proteins_path)
    requested = set(requested_order)
    remaining = set(requested_order)

    out_dir.mkdir(parents=True, exist_ok=True)

    found_source: dict[str, str] = {}
    found_header: dict[str, str] = {}
    found_sequence: dict[str, str] = {}

    print(f"Requested proteins: {len(requested_order):,}", file=sys.stderr)
    scan_source(
        fasta_path=args.uniprot_sprot,
        source_name="Swiss-Prot",
        remaining=remaining,
        found_source=found_source,
        found_header=found_header,
        found_sequence=found_sequence,
        progress_every=args.progress_every,
    )
    remaining -= set(found_sequence)
    scan_source(
        fasta_path=args.uniprot_trembl,
        source_name="TrEMBL",
        remaining=remaining,
        found_source=found_source,
        found_header=found_header,
        found_sequence=found_sequence,
        progress_every=args.progress_every,
    )

    templates_fasta = out_dir / "templates.fasta"
    with templates_fasta.open("w", encoding="utf-8") as handle:
        for accession in requested_order:
            sequence = found_sequence.get(accession)
            if not sequence:
                continue
            source = found_source[accession]
            original_header = found_header[accession]
            handle.write(f">{accession} source={source} original_header={original_header}\n")
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start:start + 80] + "\n")

    copy_if_present(pairs_path, out_dir / "template_pairs.final.tsv")
    copy_if_present(pairs_meta_path, out_dir / "template_pairs.meta.final.tsv")
    copy_if_present(proteins_path, out_dir / "proteins.final.tsv")

    write_tsv(
        out_dir / "template_sequence_sources.tsv",
        ("protein_id", "sequence_source"),
        (
            (accession, found_source[accession])
            for accession in requested_order
            if accession in found_source
        ),
    )
    write_tsv(
        out_dir / "missing_accessions.tsv",
        ("protein_id",),
        ((accession,) for accession in requested_order if accession not in found_sequence),
    )
    sprot_found = sum(1 for src in found_source.values() if src == "Swiss-Prot")
    trembl_found = sum(1 for src in found_source.values() if src == "TrEMBL")
    found_total = len(found_sequence)
    requested_total = len(requested_order)
    missing_total = requested_total - found_total
    write_tsv(
        out_dir / "fasta_build_summary.tsv",
        ("metric", "value"),
        (
            ("requested_proteins", requested_total),
            ("found_sequences_total", found_total),
            ("found_in_swissprot", sprot_found),
            ("found_in_trembl", trembl_found),
            ("missing_sequences_total", missing_total),
            (
                "sequence_coverage_fraction",
                f"{(found_total / requested_total):.6f}" if requested_total else "0.000000",
            ),
        ),
    )

    print(f"Wrote template FASTA to: {templates_fasta}")
    print(f"Found sequences: {found_total:,}")
    print(f"Missing sequences: {missing_total:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
