#!/usr/bin/env python3
"""
Build the interaction-supported conservation prior.

Stable biology:
- use the shared template-backed MMseqs homolog search
- keep homologs that participate in at least one interaction-supported pair
- build one per-chain full-sequence FAMSA MSA for each query
- trim columns where the query has a gap
- compute per-residue conservation and alignment coverage on each chain
- combine the two per-chain profiles into a broad 2D patch prior

Important:
- conservation stays per-chain biologically
- the 2D matrix is a combined prior for downstream scoring, not a contact map
- pairable homolog pairs are still written as diagnostics
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, TextIO, Tuple

import numpy as np

from template_mmseqs import (
    HOMOLOG_SEARCH_MODE_CHOICES,
    ResolvedMmseqsHit,
    default_template_fasta,
    load_resolved_hits_tsv,
)


UNIPROT_BASE_RE = (
    r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9][A-Z0-9]{3}[0-9]|[A-Z0-9]{10})$"
)
UNIPROT_BASE_RX = re.compile(UNIPROT_BASE_RE)
TRAILING_DASH_NUMBER_RX = re.compile(r"-\d+$")
PAIR_DATASET_CHOICES = ("intact_biogrid", "intact_biogrid_string")
NO_EVIDENCE_REASON = "No sequence-backed interaction-supported homologs were found for conservation."


@dataclass(frozen=True)
class PairSupport:
    acc_a: str
    acc_b: str
    src_intact: int
    src_biogrid: int
    src_string: int
    source_count: int
    pubmed_count: int
    string_score_max: int
    string_experiments_max: int
    string_database_max: int
    evidence_count: int
    sources_label: str
    interaction_types: str
    detection_methods: str
    support_score: float


def pair_dataset_defaults(dataset_name: str) -> Dict[str, Path]:
    return {
        "pairs": Path("data") / "datasets" / dataset_name / "template_pairs.final.tsv",
        "pairs_meta": Path("data") / "datasets" / dataset_name / "template_pairs.meta.final.tsv",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the interaction-supported conservation prior from shared template-backed homolog hits."
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
        "--interaction-mode",
        choices=("heteromer", "homomer", "auto"),
        default="heteromer",
        help="Interpret query pairing as heteromer, homomer, or auto-resolve from identical sequences.",
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
    parser.add_argument(
        "--allow-no-evidence",
        "--allow-empty",
        dest="allow_no_evidence",
        action="store_true",
        help=(
            "When no sequence-backed interaction-supported homologs exist, "
            "write zero-filled conservation outputs and exit successfully instead of failing."
        ),
    )
    parser.add_argument("--no-heatmap", action="store_true", help="Skip heatmap PNG output.")
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
    return args


def resolve_interaction_mode(requested_mode: str, q1_seq: str, q2_seq: str) -> str:
    if requested_mode == "auto":
        return "homomer" if q1_seq == q2_seq else "heteromer"
    if requested_mode == "homomer" and q1_seq != q2_seq:
        raise ValueError("homomer mode currently requires identical query sequences")
    return requested_mode


def open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def canonicalize_accession(token: str) -> Optional[str]:
    value = token.strip()
    if not value:
        return None
    if ":" in value:
        prefix, rest = value.split(":", 1)
        if prefix.isdigit():
            value = rest
    value = value.split()[0]
    lowered = value.lower()
    if lowered.startswith("uniprotkb:"):
        value = value.split(":", 1)[1]
    if lowered.startswith(("sp|", "tr|")) and "|" in value:
        parts = value.split("|")
        if len(parts) >= 2:
            value = parts[1]
    elif "|" in value:
        value = value.split("|", 1)[0]
    value = value.upper()
    match = TRAILING_DASH_NUMBER_RX.search(value)
    base = value[: match.start()] if match else value
    if not UNIPROT_BASE_RX.match(base):
        return None
    return base


def make_pair_key(acc_a: str, acc_b: str) -> tuple[str, str]:
    return (acc_a, acc_b) if acc_a <= acc_b else (acc_b, acc_a)


def parse_int_field(value: str) -> int:
    text = value.strip()
    if not text:
        return 0
    try:
        return int(text)
    except ValueError:
        return int(float(text))


def compute_pair_support_score(
    *,
    src_intact: int,
    src_biogrid: int,
    src_string: int,
    source_count: int,
    pubmed_count: int,
    string_score_max: int,
    evidence_count: int,
) -> float:
    score = 0.0
    score += 4.0 * int(bool(src_intact))
    score += 3.5 * int(bool(src_biogrid))
    score += 1.0 * int(bool(src_string))
    score += 0.5 * min(source_count, 4)
    score += 0.2 * min(pubmed_count, 5)
    score += 0.05 * min(evidence_count, 10)
    score += min(string_score_max, 1000) / 1000.0
    return score


def sources_label_from_flags(src_intact: int, src_biogrid: int, src_string: int) -> str:
    labels: list[str] = []
    if src_intact:
        labels.append("intact")
    if src_biogrid:
        labels.append("biogrid")
    if src_string:
        labels.append("string")
    if not labels:
        return "unknown"
    if len(labels) == 1:
        return f"{labels[0]}_only"
    return "_plus_".join(labels)


def read_single_fasta(path: Path) -> Tuple[str, str]:
    opener = gzip.open if path.suffix == ".gz" else open
    header: Optional[str] = None
    chunks: List[str] = []
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    raise ValueError(f"{path}: expected exactly one FASTA record")
                header = line[1:].strip() or "query"
            else:
                chunks.append("".join(line.split()))
    sequence = "".join(chunks).upper()
    if not header or not sequence:
        raise ValueError(f"{path}: expected a non-empty single-sequence FASTA")
    if "-" in sequence or "." in sequence:
        raise ValueError(f"{path}: sequence must be ungapped")
    return header.split()[0], sequence


def write_query_fasta(path: Path, header: str, sequence: str) -> None:
    path.write_text(f">{header}\n{sequence}\n", encoding="utf-8")


def iter_fasta_records(path: Path) -> Iterator[Tuple[str, str]]:
    header: Optional[str] = None
    seq_chunks: List[str] = []
    with open_text(path) as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_chunks).upper()
                header = line[1:].strip().split()[0]
                seq_chunks = []
            else:
                if header is None:
                    raise ValueError(f"{path}: sequence data found before FASTA header")
                seq_chunks.append("".join(line.split()))
        if header is not None:
            yield header, "".join(seq_chunks).upper()


def load_sequences_from_sources(paths: Iterable[Path], wanted: set[str]) -> Dict[str, str]:
    sequences: Dict[str, str] = {}
    if not wanted:
        return sequences
    for path in paths:
        if not path.exists():
            continue
        for header, sequence in iter_fasta_records(path):
            header_id = header.strip()
            accession = canonicalize_accession(header_id)
            for record_id in (header_id, accession):
                if record_id and record_id in wanted and record_id not in sequences:
                    sequences[record_id] = sequence
            if len(sequences) == len(wanted):
                return sequences
    return sequences


def write_sequence_fasta(
    path: Path,
    query_id: str,
    query_sequence: str,
    accessions: Iterable[str],
    hits: Dict[str, ResolvedMmseqsHit],
    sequences: Dict[str, str],
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f">{query_id}\n{query_sequence}\n")
        for accession in accessions:
            hit = hits.get(accession)
            if hit is None:
                continue
            sequence = sequences.get(hit.sequence_id)
            if sequence is None:
                continue
            handle.write(f">{accession}\n{sequence}\n")


def write_alignment_fasta(path: Path, records: Dict[str, str], ordered_ids: Iterable[str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record_id in ordered_ids:
            sequence = records.get(record_id)
            if sequence is None:
                continue
            handle.write(f">{record_id}\n{sequence}\n")


def run_famsa(famsa_bin: str, in_fasta: Path, out_fasta: Path, threads: int) -> None:
    input_records = list(iter_fasta_records(in_fasta))
    if len(input_records) <= 1:
        out_fasta.write_text(in_fasta.read_text(encoding="utf-8"), encoding="utf-8")
        return
    cmd = [famsa_bin, "-t", str(threads), str(in_fasta), str(out_fasta)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            "FAMSA failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )


def parse_alignment_fasta(path: Path) -> Dict[str, str]:
    records = dict(iter_fasta_records(path))
    if not records:
        raise RuntimeError(f"{path}: alignment FASTA is empty")
    lengths = {len(sequence) for sequence in records.values()}
    if len(lengths) != 1:
        raise RuntimeError(f"{path}: aligned FASTA contains rows with inconsistent lengths")
    return records


def trim_query_gap_columns(
    alignment: Dict[str, str],
    query_id: str,
    query_sequence: str,
) -> tuple[Dict[str, str], int]:
    query_row = alignment.get(query_id)
    if query_row is None:
        raise RuntimeError(f"Aligned FASTA is missing the query row {query_id}")
    keep_indices = [idx for idx, char in enumerate(query_row) if char != "-"]
    trimmed = {
        record_id: "".join(sequence[idx] for idx in keep_indices)
        for record_id, sequence in alignment.items()
    }
    trimmed_query = trimmed[query_id]
    if "-" in trimmed_query or trimmed_query != query_sequence:
        raise RuntimeError(
            f"Query-gap trimming failed for {query_id}: expected {len(query_sequence)} ungapped residues."
        )
    return trimmed, len(keep_indices)


def load_pair_keys(path: Path, skip_self_pairs: bool) -> set[tuple[str, str]]:
    pair_keys: set[tuple[str, str]] = set()
    with open_text(path) as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                parts = line.split()
            if len(parts) < 2:
                continue
            if parts[0].lower() == "acca" and parts[1].lower() == "accb":
                continue
            acc_a = canonicalize_accession(parts[0])
            acc_b = canonicalize_accession(parts[1])
            if not acc_a or not acc_b:
                continue
            if skip_self_pairs and acc_a == acc_b:
                continue
            pair_keys.add(make_pair_key(acc_a, acc_b))
    return pair_keys


def adjacency_from_pair_keys(pair_keys: Iterable[tuple[str, str]]) -> Dict[str, set[str]]:
    adjacency: Dict[str, set[str]] = defaultdict(set)
    for acc_a, acc_b in pair_keys:
        adjacency[acc_a].add(acc_b)
        adjacency[acc_b].add(acc_a)
    return adjacency


def load_pair_graph(
    *,
    pairs_path: Path,
    pairs_meta_path: Path | None,
    skip_self_pairs: bool,
) -> tuple[Dict[str, set[str]], Dict[tuple[str, str], PairSupport], dict[str, int | bool | str | None]]:
    allowed_pair_keys = load_pair_keys(pairs_path, skip_self_pairs)
    if not pairs_meta_path:
        return (
            adjacency_from_pair_keys(allowed_pair_keys),
            {},
            {
                "metadata_used": False,
                "pairs_meta_file": None,
                "input_pair_rows": len(allowed_pair_keys),
                "metadata_pair_rows": 0,
                "retained_pair_rows": len(allowed_pair_keys),
                "filtered_pair_rows": 0,
                "pairs_missing_metadata": len(allowed_pair_keys),
            },
        )
    if not pairs_meta_path.exists():
        raise FileNotFoundError(f"Pair metadata file not found: {pairs_meta_path}")

    aggregated: dict[tuple[str, str], dict[str, int | set[str]]] = {}
    with open_text(pairs_meta_path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            acc_a = canonicalize_accession(row.get("accA") or row.get("protein_1") or "")
            acc_b = canonicalize_accession(row.get("accB") or row.get("protein_2") or "")
            if not acc_a or not acc_b:
                continue
            if skip_self_pairs and acc_a == acc_b:
                continue
            key = make_pair_key(acc_a, acc_b)
            if key not in allowed_pair_keys:
                continue
            sources_field = (row.get("sources") or "").replace(",", ";")
            if sources_field.strip():
                source_tokens = {token.strip().lower() for token in sources_field.split(";") if token.strip()}
                src_intact = int("intact" in source_tokens)
                src_biogrid = int("biogrid" in source_tokens)
                src_string = int("string" in source_tokens)
            else:
                src_intact = parse_int_field(row.get("src_intact", "0"))
                src_biogrid = parse_int_field(row.get("src_biogrid", "0"))
                src_string = parse_int_field(row.get("src_string", "0"))
            current = aggregated.setdefault(
                key,
                {
                    "src_intact": 0,
                    "src_biogrid": 0,
                    "src_string": 0,
                    "pubmed_count_sum": 0,
                    "string_score_max": 0,
                    "string_experiments_max": 0,
                    "string_database_max": 0,
                    "evidence_count": 0,
                    "interaction_types": set(),
                    "detection_methods": set(),
                },
            )
            current["src_intact"] = max(int(current["src_intact"]), src_intact)
            current["src_biogrid"] = max(int(current["src_biogrid"]), src_biogrid)
            current["src_string"] = max(int(current["src_string"]), src_string)
            current["pubmed_count_sum"] = int(current["pubmed_count_sum"]) + parse_int_field(row.get("pubmed_count", "0"))
            current["string_score_max"] = max(int(current["string_score_max"]), parse_int_field(row.get("string_score_max", "0")))
            current["string_experiments_max"] = max(
                int(current["string_experiments_max"]),
                parse_int_field(row.get("string_experiments_max", "0")),
            )
            current["string_database_max"] = max(
                int(current["string_database_max"]),
                parse_int_field(row.get("string_database_max", "0")),
            )
            evidence_count = parse_int_field(row.get("evidence_count") or row.get("support_count") or "0")
            current["evidence_count"] = int(current["evidence_count"]) + evidence_count
            for value in (row.get("interaction_types") or "").replace(",", ";").split(";"):
                cleaned = value.strip()
                if cleaned:
                    cast = current["interaction_types"]
                    assert isinstance(cast, set)
                    cast.add(cleaned)
            for value in (row.get("detection_methods") or "").replace(",", ";").split(";"):
                cleaned = value.strip()
                if cleaned:
                    cast = current["detection_methods"]
                    assert isinstance(cast, set)
                    cast.add(cleaned)

    support_by_key: Dict[tuple[str, str], PairSupport] = {}
    retained_pair_keys: set[tuple[str, str]] = set()
    for key in allowed_pair_keys:
        aggregated_row = aggregated.get(key)
        if aggregated_row is None:
            continue
        source_count = int(bool(aggregated_row["src_intact"])) + int(bool(aggregated_row["src_biogrid"])) + int(
            bool(aggregated_row["src_string"])
        )
        support = PairSupport(
            acc_a=key[0],
            acc_b=key[1],
            src_intact=int(aggregated_row["src_intact"]),
            src_biogrid=int(aggregated_row["src_biogrid"]),
            src_string=int(aggregated_row["src_string"]),
            source_count=source_count,
            pubmed_count=int(aggregated_row["pubmed_count_sum"]),
            string_score_max=int(aggregated_row["string_score_max"]),
            string_experiments_max=int(aggregated_row["string_experiments_max"]),
            string_database_max=int(aggregated_row["string_database_max"]),
            evidence_count=int(aggregated_row["evidence_count"]),
            sources_label=sources_label_from_flags(
                int(aggregated_row["src_intact"]),
                int(aggregated_row["src_biogrid"]),
                int(aggregated_row["src_string"]),
            ),
            interaction_types="; ".join(sorted(aggregated_row["interaction_types"])),  # type: ignore[arg-type]
            detection_methods="; ".join(sorted(aggregated_row["detection_methods"])),  # type: ignore[arg-type]
            support_score=compute_pair_support_score(
                src_intact=int(aggregated_row["src_intact"]),
                src_biogrid=int(aggregated_row["src_biogrid"]),
                src_string=int(aggregated_row["src_string"]),
                source_count=source_count,
                pubmed_count=int(aggregated_row["pubmed_count_sum"]),
                string_score_max=int(aggregated_row["string_score_max"]),
                evidence_count=int(aggregated_row["evidence_count"]),
            ),
        )
        retained_pair_keys.add(key)
        support_by_key[key] = support

    missing_metadata_count = len(allowed_pair_keys) - len(aggregated)
    if missing_metadata_count:
        examples: list[str] = []
        for acc_a, acc_b in sorted(allowed_pair_keys):
            if (acc_a, acc_b) not in aggregated:
                examples.append(f"{acc_a}-{acc_b}")
                if len(examples) >= 5:
                    break
        raise RuntimeError(
            "Pair metadata is incomplete for the requested pair universe. "
            f"pairs={pairs_path} pairs_meta={pairs_meta_path} "
            f"missing_rows={missing_metadata_count} "
            f"example_missing_pairs=[{', '.join(examples)}]"
        )

    return (
        adjacency_from_pair_keys(retained_pair_keys),
        support_by_key,
        {
            "metadata_used": True,
            "pairs_meta_file": str(pairs_meta_path),
            "input_pair_rows": len(allowed_pair_keys),
            "metadata_pair_rows": len(aggregated),
            "retained_pair_rows": len(retained_pair_keys),
            "filtered_pair_rows": 0,
            "pairs_missing_metadata": missing_metadata_count,
        },
    )


def build_interacting_pairs(
    q1_hits: Dict[str, ResolvedMmseqsHit],
    q2_hits: Dict[str, ResolvedMmseqsHit],
    adjacency: Dict[str, set[str]],
    pair_support_by_key: Optional[Dict[tuple[str, str], PairSupport]] = None,
) -> List[Tuple[str, str]]:
    interacting_pairs: List[Tuple[str, str]] = []
    for acc1 in sorted(q1_hits):
        partners = sorted(
            adjacency.get(acc1, ()),
            key=lambda acc2: (
                -(
                    pair_support_by_key.get(make_pair_key(acc1, acc2)).support_score
                    if pair_support_by_key and make_pair_key(acc1, acc2) in pair_support_by_key
                    else 0.0
                ),
                acc2,
            ),
        )
        for acc2 in partners:
            if acc2 in q2_hits:
                interacting_pairs.append((acc1, acc2))
    return interacting_pairs


def select_interacting_homologs(
    q1_hits: Dict[str, ResolvedMmseqsHit],
    q2_hits: Dict[str, ResolvedMmseqsHit],
    adjacency: Dict[str, set[str]],
    pair_support_by_key: Optional[Dict[tuple[str, str], PairSupport]] = None,
) -> Tuple[Dict[str, ResolvedMmseqsHit], Dict[str, ResolvedMmseqsHit], List[Tuple[str, str]]]:
    interacting_pairs = build_interacting_pairs(q1_hits, q2_hits, adjacency, pair_support_by_key=pair_support_by_key)
    q1_selected_accessions = {acc1 for acc1, _ in interacting_pairs}
    q2_selected_accessions = {acc2 for _, acc2 in interacting_pairs}
    q1_selected = {accession: q1_hits[accession] for accession in q1_hits if accession in q1_selected_accessions}
    q2_selected = {accession: q2_hits[accession] for accession in q2_hits if accession in q2_selected_accessions}
    return q1_selected, q2_selected, interacting_pairs


def write_detected_accessions(path: Path, hits: Dict[str, ResolvedMmseqsHit]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["accession", "sequence_id", "search_tier", "taxid", "aligned_query_positions"])
        for accession in sorted(hits):
            hit = hits[accession]
            writer.writerow(
                [
                    accession,
                    hit.sequence_id,
                    hit.search_tier,
                    hit.taxid or "",
                    hit.aligned_query_positions,
                ]
            )


def write_pairable_pairs(
    path: Path,
    pairs: Iterable[Tuple[str, str]],
    pair_support_by_key: Optional[Dict[tuple[str, str], PairSupport]] = None,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "query1_accession",
                "query2_accession",
                "directional_pair_key",
                "undirected_pair_key",
                "sources_label",
                "detection_methods",
                "interaction_types",
                "pair_support_score",
                "source_count",
                "pubmed_count",
            ]
        )
        for acc1, acc2 in pairs:
            support = pair_support_by_key.get(make_pair_key(acc1, acc2)) if pair_support_by_key else None
            writer.writerow(
                [
                    acc1,
                    acc2,
                    f"{acc1}->{acc2}",
                    "|".join(sorted((acc1, acc2))),
                    support.sources_label if support else "unknown",
                    support.detection_methods if support else "",
                    support.interaction_types if support else "",
                    f"{support.support_score:.3f}" if support else "0.000",
                    support.source_count if support else 0,
                    support.pubmed_count if support else 0,
                ]
            )


def write_paired_msa(
    out_dir: Path,
    q1_seq: str,
    q2_seq: str,
    rows: List[Tuple[str, str, str]],
) -> Tuple[np.ndarray, Path]:
    total_len = len(q1_seq) + len(q2_seq)
    msa_path = out_dir / "paired_msa.txt"
    alignment_rows: List[List[str]] = [list(q1_seq + q2_seq)]
    with msa_path.open("w", encoding="utf-8") as handle:
        handle.write(">Query\n")
        handle.write(q1_seq + q2_seq + "\n")
        for acc1, acc2, combined in rows:
            combined_row = combined.ljust(total_len, "-")[:total_len]
            handle.write(f"> q1:{acc1} q2:{acc2}\n")
            handle.write(combined_row + "\n")
            alignment_rows.append(list(combined_row))
    return np.array(alignment_rows), msa_path


def write_ssa(out_dir: Path, q1_seq: str, q2_seq: str) -> Path:
    ssa_path = out_dir / "paired_ssa.txt"
    full_query = q1_seq + q2_seq
    full_ssa = ("H" * len(q1_seq)) + ("B" * len(q2_seq))
    with ssa_path.open("w", encoding="utf-8") as handle:
        handle.write(">Query\n")
        handle.write(full_query + "\n")
        handle.write(">Structure\n")
        handle.write(full_ssa + "\n")
    return ssa_path


def compute_chain_conservation_profile(
    query_seq: str,
    homolog_alignment_array: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    if homolog_alignment_array.ndim != 2 or homolog_alignment_array.shape[0] == 0:
        raise RuntimeError("Conservation scoring requires at least one homolog row for each query chain.")
    query_len = len(query_seq)
    if homolog_alignment_array.shape[1] != query_len:
        raise RuntimeError(
            f"Chain conservation alignment width {homolog_alignment_array.shape[1]} does not match query length {query_len}."
        )
    num_rows = homolog_alignment_array.shape[0]
    conservation_freq: List[float] = []
    alignment_freq: List[float] = []
    for col_idx in range(query_len):
        column = homolog_alignment_array[:, col_idx]
        non_gap = column != "-"
        non_gap_count = int(non_gap.sum())
        match_count = int((column == query_seq[col_idx]).sum())
        conservation_freq.append(match_count / float(non_gap_count) if non_gap_count > 0 else 0.0)
        alignment_freq.append(non_gap_count / float(num_rows))
    return np.array(conservation_freq, dtype=float), np.array(alignment_freq, dtype=float)


def build_conservation_matrix(
    cons1: np.ndarray,
    cons2: np.ndarray,
    align1: np.ndarray,
    align2: np.ndarray,
) -> np.ndarray:
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError as exc:
        raise RuntimeError(
            "scipy is required to smooth the conservation matrix. Install scipy in the active environment."
        ) from exc

    base_scores = np.outer(cons1, cons2) * np.outer(align1, align2)
    return gaussian_filter(base_scores, sigma=(3.0, 2.0), mode="constant")


def save_matrix_tsv(path: Path, matrix: np.ndarray) -> None:
    np.savetxt(path, matrix, fmt="%.6f", delimiter="\t")


def save_matrix_npy(path: Path, matrix: np.ndarray) -> None:
    np.save(path, matrix)


def write_profile_tsv(
    path: Path,
    sequence: str,
    conservation_freq: np.ndarray,
    alignment_freq: np.ndarray,
) -> None:
    profile = conservation_freq * alignment_freq
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["residue_index", "aa", "conservation_freq", "alignment_freq", "profile_score"])
        for idx, aa in enumerate(sequence, start=1):
            writer.writerow(
                [
                    idx,
                    aa,
                    f"{float(conservation_freq[idx - 1]):.6f}",
                    f"{float(alignment_freq[idx - 1]):.6f}",
                    f"{float(profile[idx - 1]):.6f}",
                ]
            )


def write_no_evidence_outputs(
    *,
    out_dir: Path,
    q1_header: str,
    q2_header: str,
    q1_seq: str,
    q2_seq: str,
    q1_search_out: Path,
    q2_search_out: Path,
    summary: dict[str, object],
    write_heatmap_png: bool,
) -> None:
    q1_len = len(q1_seq)
    q2_len = len(q2_seq)
    conservation_matrix = np.zeros((q1_len, q2_len), dtype=float)
    cons1 = np.zeros(q1_len, dtype=float)
    cons2 = np.zeros(q2_len, dtype=float)
    align1 = np.zeros(q1_len, dtype=float)
    align2 = np.zeros(q2_len, dtype=float)

    save_matrix_tsv(out_dir / "conservation_matrix.tsv", conservation_matrix)
    save_matrix_npy(out_dir / "conservation_matrix.npy", conservation_matrix)
    np.savetxt(out_dir / "conservation_freq_q1.tsv", cons1, fmt="%.6f", delimiter="	")
    np.savetxt(out_dir / "conservation_freq_q2.tsv", cons2, fmt="%.6f", delimiter="	")
    np.savetxt(out_dir / "alignment_freq_q1.tsv", align1, fmt="%.6f", delimiter="	")
    np.savetxt(out_dir / "alignment_freq_q2.tsv", align2, fmt="%.6f", delimiter="	")
    write_profile_tsv(out_dir / "query1_conservation_profile.tsv", q1_seq, cons1, align1)
    write_profile_tsv(out_dir / "query2_conservation_profile.tsv", q2_seq, cons2, align2)

    if write_heatmap_png:
        write_heatmap(
            out_dir / "conservation_heatmap.png",
            conservation_matrix,
            "Conservation Heatmap",
            q2_header,
            q1_header,
        )

    outputs: dict[str, str] = {
        "conservation_matrix_tsv": str(out_dir / "conservation_matrix.tsv"),
        "conservation_matrix_npy": str(out_dir / "conservation_matrix.npy"),
        "conservation_freq_q1_tsv": str(out_dir / "conservation_freq_q1.tsv"),
        "conservation_freq_q2_tsv": str(out_dir / "conservation_freq_q2.tsv"),
        "alignment_freq_q1_tsv": str(out_dir / "alignment_freq_q1.tsv"),
        "alignment_freq_q2_tsv": str(out_dir / "alignment_freq_q2.tsv"),
        "query1_search_tsv": str(q1_search_out),
        "query2_search_tsv": str(q2_search_out),
        "q1_interacting_accessions_tsv": str(out_dir / "q1_interacting_accessions.tsv"),
        "q2_interacting_accessions_tsv": str(out_dir / "q2_interacting_accessions.tsv"),
        "pairable_pairs_tsv": str(out_dir / "pairable_pairs.tsv"),
        "pairable_pairs_sequence_backed_tsv": str(out_dir / "pairable_pairs_sequence_backed.tsv"),
        "query1_conservation_profile_tsv": str(out_dir / "query1_conservation_profile.tsv"),
        "query2_conservation_profile_tsv": str(out_dir / "query2_conservation_profile.tsv"),
        "conservation_summary_json": str(out_dir / "conservation_summary.json"),
    }
    if write_heatmap_png:
        outputs["conservation_heatmap_png"] = str(out_dir / "conservation_heatmap.png")

    summary.update(
        {
            "status": "no_evidence",
            "error": None,
            "no_evidence_reason": NO_EVIDENCE_REASON,
            "paired_rows_used": 0,
            "pairs_missing_after_alignment": 0,
            "weak_msa_warning": True,
            "paired_msa_path": None,
            "paired_ssa_path": None,
            "conservation_matrix_shape": [q1_len, q2_len],
            "outputs": outputs,
        }
    )
    (out_dir / "conservation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def write_heatmap(path: Path, matrix: np.ndarray, title: str, xlabel: str, ylabel: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap="YlGnBu", origin="upper", aspect="auto")
    plt.colorbar()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def alignment_rows_to_array(trimmed_alignment: Dict[str, str], ordered_ids: List[str]) -> np.ndarray:
    rows = [list(trimmed_alignment[record_id]) for record_id in ordered_ids if record_id in trimmed_alignment]
    if not rows:
        return np.empty((0, 0), dtype="<U1")
    return np.array(rows, dtype="<U1")


def build_paired_rows_from_trimmed_msas(
    interacting_pairs: Iterable[Tuple[str, str]],
    left_rows: Dict[str, str],
    right_rows: Dict[str, str],
) -> tuple[List[Tuple[str, str, str]], int]:
    paired_rows: List[Tuple[str, str, str]] = []
    missing_pairs = 0
    for acc1, acc2 in interacting_pairs:
        left = left_rows.get(acc1)
        right = right_rows.get(acc2)
        if left is None or right is None:
            missing_pairs += 1
            continue
        paired_rows.append((acc1, acc2, left + right))
    return paired_rows, missing_pairs


def main() -> int:
    args = parse_args()
    pair_defaults = pair_dataset_defaults(args.pair_dataset)
    if args.pairs is None:
        args.pairs = pair_defaults["pairs"]
    if args.pairs_meta is None:
        args.pairs_meta = pair_defaults["pairs_meta"]
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    q1_header, q1_seq = read_single_fasta(args.query1)
    q2_header, q2_seq = read_single_fasta(args.query2)
    interaction_mode_used = resolve_interaction_mode(args.interaction_mode, q1_seq, q2_seq)
    if args.verbose:
        print(f"[INFO] Query1: {q1_header} (length {len(q1_seq)})")
        print(f"[INFO] Query2: {q2_header} (length {len(q2_seq)})")

    q1_search_out = args.query1_search_tsv
    q2_search_out = args.query2_search_tsv
    q1_hits = load_resolved_hits_tsv(q1_search_out)
    q2_hits = load_resolved_hits_tsv(q2_search_out)
    write_detected_accessions(out_dir / "q1_detected_accessions.tsv", q1_hits)
    write_detected_accessions(out_dir / "q2_detected_accessions.tsv", q2_hits)

    adjacency, pair_support_by_key, pair_graph_stats = load_pair_graph(
        pairs_path=args.pairs,
        pairs_meta_path=args.pairs_meta,
        skip_self_pairs=(interaction_mode_used == "heteromer"),
    )
    q1_selected_hits, q2_selected_hits, interacting_pairs = select_interacting_homologs(
        q1_hits,
        q2_hits,
        adjacency,
        pair_support_by_key=pair_support_by_key,
    )

    primary_pairable_pairs_found = len(interacting_pairs)
    pair_universe_used = args.pair_dataset
    write_pairable_pairs(out_dir / "pairable_pairs.tsv", interacting_pairs, pair_support_by_key=pair_support_by_key)

    wanted_sequence_ids = {
        hit.sequence_id for hit in q1_selected_hits.values()
    } | {
        hit.sequence_id for hit in q2_selected_hits.values()
    }
    selected_sequences = load_sequences_from_sources([args.sequence_fasta], wanted_sequence_ids)
    missing_sequence_ids = sorted(wanted_sequence_ids - set(selected_sequences))

    q1_sequence_backed_hits = {
        accession: hit
        for accession, hit in q1_selected_hits.items()
        if hit.sequence_id in selected_sequences
    }
    q2_sequence_backed_hits = {
        accession: hit
        for accession, hit in q2_selected_hits.items()
        if hit.sequence_id in selected_sequences
    }
    sequence_backed_pairs = [
        (acc1, acc2)
        for acc1, acc2 in interacting_pairs
        if acc1 in q1_sequence_backed_hits and acc2 in q2_sequence_backed_hits
    ]
    write_detected_accessions(out_dir / "q1_interacting_accessions.tsv", q1_sequence_backed_hits)
    write_detected_accessions(out_dir / "q2_interacting_accessions.tsv", q2_sequence_backed_hits)
    write_pairable_pairs(
        out_dir / "pairable_pairs_sequence_backed.tsv",
        sequence_backed_pairs,
        pair_support_by_key=pair_support_by_key,
    )

    summary = {
        "query1_header": q1_header,
        "query2_header": q2_header,
        "query1_length": len(q1_seq),
        "query2_length": len(q2_seq),
        "pair_dataset": args.pair_dataset,
        "pairs_file": str(args.pairs),
        "pairs_meta_file": str(args.pairs_meta),
        "pair_universe_used": pair_universe_used,
        "pair_support_meta_used": pair_graph_stats["metadata_used"],
        "pair_universe_rows_loaded": pair_graph_stats["input_pair_rows"],
        "pair_universe_rows_with_metadata": pair_graph_stats["metadata_pair_rows"],
        "pair_universe_rows_retained": pair_graph_stats["retained_pair_rows"],
        "pair_universe_rows_filtered": pair_graph_stats["filtered_pair_rows"],
        "pair_universe_rows_missing_metadata": pair_graph_stats["pairs_missing_metadata"],
        "interaction_mode_requested": args.interaction_mode,
        "interaction_mode_used": interaction_mode_used,
        "self_pairs_skipped": interaction_mode_used == "heteromer",
        "shared_search_logic": args.shared_search_mode,
        "sequence_fasta": str(args.sequence_fasta),
        "q1_homolog_accessions": len(q1_hits),
        "q2_homolog_accessions": len(q2_hits),
        "primary_pairable_pairs_found": primary_pairable_pairs_found,
        "pairable_pairs_found": len(sequence_backed_pairs),
        "conservation_mode": "per_chain_famsa_trimmed_query_gap_columns",
        "q1_interacting_homolog_accessions": len(q1_sequence_backed_hits),
        "q2_interacting_homolog_accessions": len(q2_sequence_backed_hits),
        "selected_homolog_accessions_missing_sequences": len(missing_sequence_ids),
        "missing_sequence_ids_sample": missing_sequence_ids[:50],
        "query_seed_row_excluded_from_conservation_scoring": True,
        "query1_search_tsv": str(q1_search_out),
        "query2_search_tsv": str(q2_search_out),
    }

    if not sequence_backed_pairs or not q1_sequence_backed_hits or not q2_sequence_backed_hits:
        if args.allow_no_evidence:
            write_no_evidence_outputs(
                out_dir=out_dir,
                q1_header=q1_header,
                q2_header=q2_header,
                q1_seq=q1_seq,
                q2_seq=q2_seq,
                q1_search_out=q1_search_out,
                q2_search_out=q2_search_out,
                summary=summary,
                write_heatmap_png=not args.no_heatmap,
            )
            if args.verbose:
                print(f"[WARN] {NO_EVIDENCE_REASON}")
                print("[INFO] Wrote zero-filled conservation outputs and continued because --allow-no-evidence was set.")
            return 0
        summary["error"] = NO_EVIDENCE_REASON
        (out_dir / "conservation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        raise SystemExit(NO_EVIDENCE_REASON)

    q1_accessions = sorted(q1_sequence_backed_hits)
    q2_accessions = sorted(q2_sequence_backed_hits)
    chain1_raw_fasta = out_dir / "chain1_raw.fasta"
    chain2_raw_fasta = out_dir / "chain2_raw.fasta"
    write_sequence_fasta(chain1_raw_fasta, q1_header, q1_seq, q1_accessions, q1_sequence_backed_hits, selected_sequences)
    write_sequence_fasta(chain2_raw_fasta, q2_header, q2_seq, q2_accessions, q2_sequence_backed_hits, selected_sequences)

    chain1_aligned_fasta = out_dir / "chain1_famsa.fasta"
    chain2_aligned_fasta = out_dir / "chain2_famsa.fasta"
    run_famsa(args.famsa_bin, chain1_raw_fasta, chain1_aligned_fasta, args.threads)
    run_famsa(args.famsa_bin, chain2_raw_fasta, chain2_aligned_fasta, args.threads)

    chain1_alignment = parse_alignment_fasta(chain1_aligned_fasta)
    chain2_alignment = parse_alignment_fasta(chain2_aligned_fasta)
    chain1_trimmed, chain1_trimmed_length = trim_query_gap_columns(chain1_alignment, q1_header, q1_seq)
    chain2_trimmed, chain2_trimmed_length = trim_query_gap_columns(chain2_alignment, q2_header, q2_seq)

    chain1_trimmed_fasta = out_dir / "chain1_trimmed.fasta"
    chain2_trimmed_fasta = out_dir / "chain2_trimmed.fasta"
    write_alignment_fasta(chain1_trimmed_fasta, chain1_trimmed, [q1_header, *q1_accessions])
    write_alignment_fasta(chain2_trimmed_fasta, chain2_trimmed, [q2_header, *q2_accessions])

    q1_homolog_array = alignment_rows_to_array(chain1_trimmed, q1_accessions)
    q2_homolog_array = alignment_rows_to_array(chain2_trimmed, q2_accessions)
    cons1, align1 = compute_chain_conservation_profile(q1_seq, q1_homolog_array)
    cons2, align2 = compute_chain_conservation_profile(q2_seq, q2_homolog_array)
    conservation_matrix = build_conservation_matrix(cons1, cons2, align1, align2)

    paired_rows, pairs_missing_after_alignment = build_paired_rows_from_trimmed_msas(
        sequence_backed_pairs,
        chain1_trimmed,
        chain2_trimmed,
    )
    _paired_alignment_array, paired_msa_path = write_paired_msa(out_dir, q1_seq, q2_seq, paired_rows)
    paired_ssa_path = write_ssa(out_dir, q1_seq, q2_seq)

    save_matrix_tsv(out_dir / "conservation_matrix.tsv", conservation_matrix)
    save_matrix_npy(out_dir / "conservation_matrix.npy", conservation_matrix)
    np.savetxt(out_dir / "conservation_freq_q1.tsv", cons1, fmt="%.6f", delimiter="\t")
    np.savetxt(out_dir / "conservation_freq_q2.tsv", cons2, fmt="%.6f", delimiter="\t")
    np.savetxt(out_dir / "alignment_freq_q1.tsv", align1, fmt="%.6f", delimiter="\t")
    np.savetxt(out_dir / "alignment_freq_q2.tsv", align2, fmt="%.6f", delimiter="\t")
    write_profile_tsv(out_dir / "query1_conservation_profile.tsv", q1_seq, cons1, align1)
    write_profile_tsv(out_dir / "query2_conservation_profile.tsv", q2_seq, cons2, align2)

    if not args.no_heatmap:
        write_heatmap(
            out_dir / "conservation_heatmap.png",
            conservation_matrix,
            "Conservation Heatmap",
            q2_header,
            q1_header,
        )

    summary.update(
        {
            "status": "ok",
            "error": None,
            "no_evidence_reason": None,
            "chain1_raw_msa_depth": 1 + len(q1_accessions),
            "chain2_raw_msa_depth": 1 + len(q2_accessions),
            "chain1_trimmed_length": chain1_trimmed_length,
            "chain2_trimmed_length": chain2_trimmed_length,
            "paired_rows_used": len(paired_rows),
            "pairs_missing_after_alignment": pairs_missing_after_alignment,
            "weak_msa_warning": min(len(q1_accessions), len(q2_accessions)) < 20,
            "paired_msa_path": str(paired_msa_path),
            "paired_ssa_path": str(paired_ssa_path),
            "conservation_matrix_shape": list(conservation_matrix.shape),
            "outputs": {
                "conservation_matrix_tsv": str(out_dir / "conservation_matrix.tsv"),
                "conservation_matrix_npy": str(out_dir / "conservation_matrix.npy"),
                "conservation_freq_q1_tsv": str(out_dir / "conservation_freq_q1.tsv"),
                "conservation_freq_q2_tsv": str(out_dir / "conservation_freq_q2.tsv"),
                "alignment_freq_q1_tsv": str(out_dir / "alignment_freq_q1.tsv"),
                "alignment_freq_q2_tsv": str(out_dir / "alignment_freq_q2.tsv"),
                "paired_msa": str(paired_msa_path),
                "paired_ssa": str(paired_ssa_path),
                "query1_search_tsv": str(q1_search_out),
                "query2_search_tsv": str(q2_search_out),
                "q1_interacting_accessions_tsv": str(out_dir / "q1_interacting_accessions.tsv"),
                "q2_interacting_accessions_tsv": str(out_dir / "q2_interacting_accessions.tsv"),
                "pairable_pairs_tsv": str(out_dir / "pairable_pairs.tsv"),
                "pairable_pairs_sequence_backed_tsv": str(out_dir / "pairable_pairs_sequence_backed.tsv"),
                "query1_conservation_profile_tsv": str(out_dir / "query1_conservation_profile.tsv"),
                "query2_conservation_profile_tsv": str(out_dir / "query2_conservation_profile.tsv"),
                "chain1_raw_fasta": str(chain1_raw_fasta),
                "chain2_raw_fasta": str(chain2_raw_fasta),
                "chain1_famsa_fasta": str(chain1_aligned_fasta),
                "chain2_famsa_fasta": str(chain2_aligned_fasta),
                "chain1_trimmed_fasta": str(chain1_trimmed_fasta),
                "chain2_trimmed_fasta": str(chain2_trimmed_fasta),
                "conservation_summary_json": str(out_dir / "conservation_summary.json"),
            },
        }
    )
    (out_dir / "conservation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.verbose:
        print(f"[INFO] Reusing shared homolog-search TSVs: {q1_search_out} and {q2_search_out}")
        print(f"[INFO] Homolog accessions: q1={len(q1_hits)} q2={len(q2_hits)}")
        print(f"[INFO] Interaction-supported homolog pairs: {len(sequence_backed_pairs)}")
        print(
            "[INFO] Unique interacting homolog rows used for conservation: "
            f"q1={len(q1_accessions)} q2={len(q2_accessions)}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
