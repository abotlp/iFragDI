#!/usr/bin/env python3
"""
Classical iFrag using BLAST.

This script implements only the classical iFrag branch:
- BLAST each query against a template sequence database
- map hits onto interacting template pairs
- make fragment-pair evidence explicit within each template interaction
- collapse redundancy after candidate template interactions are formed
- score residue pairs by the proportion of nonredundant template pairs that support them
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np

try:
    from Bio.Align import PairwiseAligner as _BioPairwiseAligner
except ImportError:  # pragma: no cover - optional dependency
    _BioPairwiseAligner = None


BLAST_OUTFMT = "6 qseqid sseqid pident evalue bitscore length qstart qend sstart send qseq sseq"
UNIPROT_BASE_RE = (
    r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9][A-Z0-9]{3}[0-9]|[A-Z0-9]{10})$"
)
UNIPROT_BASE_RX = re.compile(UNIPROT_BASE_RE)
TRAILING_DASH_NUMBER_RX = re.compile(r"-\d+$")
TEMPLATE_DATASET_CHOICES = ("intact_biogrid", "intact_biogrid_string")


@dataclass(frozen=True)
class Hit:
    query: str
    target: str
    template_acc: str
    pident: float
    evalue: float
    bitscore: float
    alnlen: int
    qstart: int
    qend: int
    tstart: int
    tend: int
    qseq: str
    sseq: str
    covered_q_positions: Tuple[int, ...]
    query_coverage: float


@dataclass(frozen=True)
class FragmentPairMatch:
    left_hit: Hit
    right_hit: Hit
    supported_cells: frozenset[int]
    combined_bitscore: float
    combined_evalue: float


@dataclass(frozen=True)
class TemplateInteractionMatch:
    left_acc: str
    right_acc: str
    left_hits: Tuple[Hit, ...]
    right_hits: Tuple[Hit, ...]
    fragment_pairs: Tuple[FragmentPairMatch, ...]
    supported_cells: frozenset[int]
    best_combined_bitscore: float
    best_combined_evalue: float

    @property
    def fragment_pair_count(self) -> int:
        return len(self.fragment_pairs)


@dataclass(frozen=True)
class TemplatePairLoadResult:
    pairs: Tuple[Tuple[str, str], ...]
    total_nonempty_rows: int
    valid_rows: int
    filtered_out_rows: int


def dataset_resource_defaults(dataset_name: str) -> Dict[str, Path]:
    return {
        "pairs": Path("data") / "datasets" / dataset_name / f"{dataset_name}.final.tsv",
        "blast_db": Path("data") / "db" / f"blast_templates_{dataset_name}" / "templates_db",
        "template_fasta": Path("data") / "interaction_templates" / dataset_name / "templates.fasta",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classical BLAST-based iFrag for one query protein pair.")
    parser.add_argument("--query1", required=True, type=Path, help="Single-sequence FASTA for protein 1.")
    parser.add_argument("--query2", required=True, type=Path, help="Single-sequence FASTA for protein 2.")
    parser.add_argument(
        "--template-dataset",
        choices=TEMPLATE_DATASET_CHOICES,
        default="intact_biogrid_string",
        help=(
            "Template resource set for iFrag. "
            "'intact_biogrid' is the curated core; "
            "'intact_biogrid_string' is the large STRING-expanded universe."
        ),
    )
    parser.add_argument(
        "--pairs",
        type=Path,
        default=None,
        help=(
            "Optional override for the template interaction table. If omitted, "
            "the file is resolved from --template-dataset."
        ),
    )
    parser.add_argument(
        "--blast-db",
        type=Path,
        default=None,
        help=(
            "Optional override for the BLAST protein database prefix. If omitted, "
            "the DB is resolved from --template-dataset."
        ),
    )
    parser.add_argument(
        "--template-fasta",
        type=Path,
        default=None,
        help=(
            "Optional override for the template FASTA used to fetch full protein "
            "sequences for template redundancy filtering. If omitted, the "
            "path is inferred from --template-dataset or --pairs."
        ),
    )
    parser.add_argument(
        "--pair-method-substring",
        action="append",
        default=[],
        help=(
            "Case-insensitive substring filter applied to detection_method in the "
            "template pair file. Can be passed multiple times. Useful for keeping "
            "only selected assay/method classes."
        ),
    )
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory.")
    parser.add_argument("--blast-bin", default="blastp", help="blastp executable.")
    parser.add_argument(
        "--redundancy-mode",
        choices=("cdhit_cluster_pair", "greedy_identity"),
        default="cdhit_cluster_pair",
        help=(
            "How redundant template interactions are collapsed. "
            "'cdhit_cluster_pair' is the paper-faithful mode: cluster template proteins "
            "with CD-HIT at --identity-threshold, then keep one representative per "
            "nonredundant cluster pair. "
            "'greedy_identity' keeps the legacy query-specific pairwise identity pruning."
        ),
    )
    parser.add_argument(
        "--cdhit-bin",
        default="cd-hit",
        help=(
            "cd-hit executable used when --redundancy-mode=cdhit_cluster_pair. "
            "Pass a full path if the binary is not already on PATH."
        ),
    )
    parser.add_argument(
        "--identity-threshold",
        type=float,
        default=40.0,
        help=(
            "Protein-level identity threshold used for template nonredundancy. "
            "With CD-HIT mode this is the protein clustering cutoff; with greedy mode "
            "this is the legacy pairwise identity cutoff."
        ),
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--evalue", type=float, default=0.01)
    parser.add_argument("--max-target-seqs", type=int, default=100000)
    parser.add_argument("--min-pident", type=float, default=0.0)
    parser.add_argument("--min-aln-len", type=int, default=1)
    parser.add_argument("--min-cov1", type=float, default=0.0)
    parser.add_argument("--max-cov1", type=float, default=1.0)
    parser.add_argument("--min-cov2", type=float, default=0.0)
    parser.add_argument("--max-cov2", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None, help="Optional maximum number of hits kept per query.")
    parser.add_argument("--heatmap", action="store_true", help="Write ifrag_heatmap.png.")
    args = parser.parse_args()

    if args.threads <= 0:
        raise SystemExit("--threads must be > 0")
    if args.max_target_seqs <= 0:
        raise SystemExit("--max-target-seqs must be > 0")
    if args.min_aln_len <= 0:
        raise SystemExit("--min-aln-len must be > 0")
    if not (0.0 <= args.identity_threshold <= 100.0):
        raise SystemExit("--identity-threshold must be in [0, 100]")
    return args


def read_single_fasta(path: Path) -> Tuple[str, str]:
    opener = gzip.open if str(path).endswith(".gz") else open
    records: List[Tuple[str, str]] = []
    header = None
    seq_chunks: List[str] = []
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq_chunks)))
                header = line[1:].strip() or "query"
                seq_chunks = []
            else:
                if header is None:
                    raise ValueError(f"{path}: sequence data found before FASTA header")
                seq_chunks.append(line)
    if header is not None:
        records.append((header, "".join(seq_chunks)))

    if len(records) != 1:
        raise ValueError(f"{path}: expected exactly one FASTA record, found {len(records)}")

    header, raw_sequence = records[0]
    sequence = raw_sequence.replace(" ", "").replace("\t", "").upper()
    if not header or not sequence:
        raise ValueError(f"{path}: expected a non-empty single-sequence FASTA")
    return header.split()[0], sequence


def canonicalize_accession(raw: str) -> Optional[str]:
    token = raw.strip()
    if not token:
        return None
    token = token.split()[0]
    if token.lower().startswith("uniprotkb:"):
        token = token.split(":", 1)[1].strip()
    if token.lower().startswith(("sp|", "tr|")) and "|" in token:
        parts = token.split("|")
        if len(parts) >= 2:
            token = parts[1]
    elif "|" in token:
        token = token.split("|", 1)[0]
    token = token.upper()
    match = TRAILING_DASH_NUMBER_RX.search(token)
    base = token[: match.start()] if match else token
    if not base or not UNIPROT_BASE_RX.match(base):
        return None
    return base


def run_blastp(
    blast_bin: str,
    query_fasta: Path,
    db_prefix: Path,
    out_tsv: Path,
    threads: int,
    evalue: float,
    max_target_seqs: int,
) -> None:
    cmd = [
        blast_bin,
        "-query",
        str(query_fasta),
        "-db",
        str(db_prefix),
        "-outfmt",
        BLAST_OUTFMT,
        "-evalue",
        str(evalue),
        "-max_target_seqs",
        str(max_target_seqs),
        "-num_threads",
        str(threads),
        "-out",
        str(out_tsv),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "BLAST search failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )


def covered_query_positions(qseq: str, sseq: str, qstart: int, qend: int, qlen: int) -> Tuple[int, ...]:
    if len(qseq) != len(sseq):
        return tuple()
    qpos = qstart
    qstep = 1 if qend >= qstart else -1
    covered: Set[int] = set()
    for q_char, s_char in zip(qseq, sseq):
        if q_char != "-":
            current_q = qpos
            qpos += qstep
        else:
            current_q = None
        if q_char != "-" and s_char != "-" and current_q is not None and 1 <= current_q <= qlen:
            covered.add(current_q)
    return tuple(sorted(covered))


def parse_blast_hits(
    tsv_path: Path,
    query_len: int,
    min_pident: float,
    min_aln_len: int,
    min_cov: float,
    max_cov: float,
    top_k: Optional[int],
) -> List[Hit]:
    hits: List[Hit] = []
    with tsv_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n\r")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 12:
                continue
            try:
                pident = float(parts[2])
                evalue = float(parts[3])
                bitscore = float(parts[4])
                alnlen = int(parts[5])
                qstart = int(parts[6])
                qend = int(parts[7])
                tstart = int(parts[8])
                tend = int(parts[9])
            except ValueError:
                continue
            qseq, sseq = parts[10], parts[11]
            if alnlen < min_aln_len or pident < min_pident:
                continue
            template_acc = canonicalize_accession(parts[1])
            if not template_acc:
                continue
            covered = covered_query_positions(qseq, sseq, qstart, qend, query_len)
            if not covered:
                continue
            query_coverage = len(covered) / float(query_len)
            if query_coverage < min_cov or query_coverage > max_cov:
                continue
            hits.append(
                Hit(
                    query=parts[0],
                    target=parts[1],
                    template_acc=template_acc,
                    pident=pident,
                    evalue=evalue,
                    bitscore=bitscore,
                    alnlen=alnlen,
                    qstart=qstart,
                    qend=qend,
                    tstart=tstart,
                    tend=tend,
                    qseq=qseq,
                    sseq=sseq,
                    covered_q_positions=covered,
                    query_coverage=query_coverage,
                )
            )
    hits.sort(key=lambda hit: (hit.evalue, -hit.pident, -hit.alnlen))
    if top_k is not None and top_k > 0:
        hits = hits[:top_k]
    return hits


def group_hits_by_template(hits: Sequence[Hit]) -> Dict[str, List[Hit]]:
    grouped: Dict[str, List[Hit]] = {}
    for hit in hits:
        grouped.setdefault(hit.template_acc, []).append(hit)
    return grouped


def load_template_pairs(path: Path, method_substrings: Sequence[str]) -> TemplatePairLoadResult:
    pairs_set: Set[Tuple[str, str]] = set()
    valid_rows = 0
    total_nonempty_rows = 0
    filtered_out_rows = 0
    require_method_filter = bool(method_substrings)
    method_substrings_lower = [token.lower() for token in method_substrings]
    saw_detection_method_column = False
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n\r")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            left_header = parts[0].strip().lower()
            right_header = parts[1].strip().lower()
            if (
                left_header in {"acca", "protein_1"}
                and right_header in {"accb", "protein_2"}
            ):
                continue
            total_nonempty_rows += 1
            detection_method = None
            if len(parts) >= 3:
                saw_detection_method_column = True
                detection_method = parts[2].strip().lower()
            if require_method_filter:
                if detection_method is None:
                    raise ValueError(
                        f"{path}: --pair-method-substring requires a pair file with a detection_method column"
                    )
                if not any(token in detection_method for token in method_substrings_lower):
                    filtered_out_rows += 1
                    continue
            acc_a = canonicalize_accession(parts[0])
            acc_b = canonicalize_accession(parts[1])
            if not acc_a or not acc_b:
                continue
            pair = (acc_a, acc_b) if acc_a < acc_b else (acc_b, acc_a)
            pairs_set.add(pair)
            valid_rows += 1
    if require_method_filter and not saw_detection_method_column:
        raise ValueError(
            f"{path}: --pair-method-substring requires a pair file with a detection_method column"
        )
    return TemplatePairLoadResult(
        pairs=tuple(sorted(pairs_set)),
        total_nonempty_rows=total_nonempty_rows,
        valid_rows=valid_rows,
        filtered_out_rows=filtered_out_rows,
    )


def infer_template_fasta_path(args: argparse.Namespace) -> Path:
    if args.template_fasta is not None:
        return args.template_fasta
    dataset_defaults = dataset_resource_defaults(args.template_dataset)
    dataset_template_fasta = dataset_defaults["template_fasta"]
    if dataset_template_fasta.exists():
        return dataset_template_fasta
    inferred = args.pairs.parent / "templates.fasta"
    if inferred.exists():
        return inferred
    raise FileNotFoundError(
        "Could not infer template FASTA path from --pairs. Pass --template-fasta explicitly."
    )


def iter_fasta_records(path: Path) -> Iterator[Tuple[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        header = None
        seq_chunks: List[str] = []
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_chunks)
                header = line[1:].strip()
                seq_chunks = []
            else:
                if header is None:
                    raise ValueError(f"{path}: sequence data found before FASTA header")
                seq_chunks.append(line)
        if header is not None:
            yield header, "".join(seq_chunks)


def load_template_sequences(path: Path, wanted: Set[str]) -> Dict[str, str]:
    sequences: Dict[str, str] = {}
    for header, seq in iter_fasta_records(path):
        acc = canonicalize_accession(header)
        if acc and acc in wanted:
            sequences[acc] = seq.upper()
            if len(sequences) == len(wanted):
                break
    return sequences


def build_fragment_pair_match(left_hit: Hit, right_hit: Hit, q2_len: int) -> FragmentPairMatch:
    supported_cells: Set[int] = set()
    for i in left_hit.covered_q_positions:
        base = (i - 1) * q2_len
        for j in right_hit.covered_q_positions:
            supported_cells.add(base + (j - 1))
    return FragmentPairMatch(
        left_hit=left_hit,
        right_hit=right_hit,
        supported_cells=frozenset(supported_cells),
        combined_bitscore=left_hit.bitscore + right_hit.bitscore,
        combined_evalue=left_hit.evalue * right_hit.evalue,
    )


def build_template_interaction_match(
    left_acc: str,
    right_acc: str,
    left_hits: Sequence[Hit],
    right_hits: Sequence[Hit],
    q2_len: int,
) -> TemplateInteractionMatch:
    supported_cells: Set[int] = set()
    fragment_pairs: List[FragmentPairMatch] = []
    best_combined_bitscore = float("-inf")
    best_combined_evalue = float("inf")
    for left_hit in left_hits:
        for right_hit in right_hits:
            fragment_pair = build_fragment_pair_match(left_hit, right_hit, q2_len)
            fragment_pairs.append(fragment_pair)
            supported_cells.update(fragment_pair.supported_cells)
            best_combined_bitscore = max(best_combined_bitscore, fragment_pair.combined_bitscore)
            best_combined_evalue = min(best_combined_evalue, fragment_pair.combined_evalue)
    return TemplateInteractionMatch(
        left_acc=left_acc,
        right_acc=right_acc,
        left_hits=tuple(left_hits),
        right_hits=tuple(right_hits),
        fragment_pairs=tuple(fragment_pairs),
        supported_cells=frozenset(supported_cells),
        best_combined_bitscore=best_combined_bitscore,
        best_combined_evalue=best_combined_evalue,
    )


def safe_neglog10(value: float) -> float:
    if value <= 0.0:
        return 300.0
    return -math.log10(value)


def representative_rank_key(candidate: TemplateInteractionMatch) -> Tuple[float, float, int, str, str]:
    return (
        candidate.best_combined_bitscore,
        safe_neglog10(candidate.best_combined_evalue),
        len(candidate.supported_cells),
        candidate.left_acc,
        candidate.right_acc,
    )


def protein_identity_method_description() -> str:
    if _BioPairwiseAligner is not None:
        return "Biopython PairwiseAligner global alignment with matches/max(len(seq1), len(seq2))"
    return "stdlib affine-gap global alignment fallback with matches/max(len(seq1), len(seq2))"


def cdhit_word_length(identity_threshold: float) -> int:
    threshold_fraction = identity_threshold / 100.0
    if threshold_fraction >= 0.7:
        return 5
    if threshold_fraction >= 0.6:
        return 4
    if threshold_fraction >= 0.5:
        return 3
    if threshold_fraction >= 0.4:
        return 2
    raise ValueError(
        "CD-HIT protein clustering supports thresholds >= 40%. "
        "Use --redundancy-mode greedy_identity for lower thresholds."
    )


def write_accession_fasta(path: Path, sequences: Dict[str, str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for accession in sorted(sequences):
            sequence = sequences[accession]
            handle.write(f">{accession}\n")
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start : start + 80] + "\n")


def parse_cdhit_clusters(path: Path) -> Dict[str, str]:
    cluster_by_accession: Dict[str, str] = {}
    current_cluster = None
    member_re = re.compile(r">([^\.]+)\.\.\.")
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">Cluster "):
                current_cluster = line[1:].replace(" ", "_")
                continue
            if current_cluster is None:
                continue
            match = member_re.search(line)
            if not match:
                continue
            accession = canonicalize_accession(match.group(1))
            if accession:
                cluster_by_accession[accession] = current_cluster
    return cluster_by_accession


def cluster_sequences_with_cdhit(
    sequences: Dict[str, str],
    cdhit_bin: str,
    identity_threshold: float,
    threads: int,
) -> Tuple[Dict[str, str], int]:
    if not sequences:
        return {}, 0
    if len(sequences) == 1:
        only_accession = next(iter(sequences))
        return {only_accession: "Cluster_0"}, 0

    word_length = cdhit_word_length(identity_threshold)
    threshold_fraction = identity_threshold / 100.0
    tmp_dir = tempfile.mkdtemp(prefix="ifrag_cdhit_")
    tmp_path = Path(tmp_dir)
    input_fasta = tmp_path / "candidate_templates.fasta"
    output_fasta = tmp_path / "candidate_templates.cdhit"
    write_accession_fasta(input_fasta, sequences)
    cmd = [
        cdhit_bin,
        "-i",
        str(input_fasta),
        "-o",
        str(output_fasta),
        "-c",
        f"{threshold_fraction:.3f}",
        "-n",
        str(word_length),
        "-T",
        str(threads),
        "-M",
        "0",
        "-d",
        "0",
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except FileNotFoundError as exc:
        shutil.rmtree(tmp_path, ignore_errors=True)
        raise RuntimeError(
            f"CD-HIT executable not found: {cdhit_bin}. "
            "Load the CD-HIT module or pass --cdhit-bin with a full path."
        ) from exc
    if result.returncode != 0:
        raise RuntimeError(
            "CD-HIT clustering failed.\n"
            f"Exit code: {result.returncode}\n"
            f"Sequence count: {len(sequences)}\n"
            f"Temporary directory kept at: {tmp_path}\n"
            f"Input FASTA: {input_fasta}\n"
            f"Output prefix: {output_fasta}\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
    cluster_file = Path(str(output_fasta) + ".clstr")
    cluster_by_accession = parse_cdhit_clusters(cluster_file)
    shutil.rmtree(tmp_path, ignore_errors=True)
    missing = sorted(set(sequences) - set(cluster_by_accession))
    if missing:
        raise RuntimeError(
            "CD-HIT finished but did not report clusters for all candidate template proteins. "
            f"Missing accessions (sample): {missing[:20]}"
        )
    return cluster_by_accession, word_length


def global_identity_percent_affine_fallback(seq1: str, seq2: str) -> float:
    n = len(seq1)
    m = len(seq2)
    if n == 0 or m == 0:
        return 0.0

    neg_inf = float("-inf")
    match_score = 1.0
    mismatch_score = 0.0
    gap_open = -1.0
    gap_extend = -0.5

    score_m = [[neg_inf] * (m + 1) for _ in range(n + 1)]
    score_x = [[neg_inf] * (m + 1) for _ in range(n + 1)]
    score_y = [[neg_inf] * (m + 1) for _ in range(n + 1)]
    trace_m = [[0] * (m + 1) for _ in range(n + 1)]
    trace_x = [[0] * (m + 1) for _ in range(n + 1)]
    trace_y = [[0] * (m + 1) for _ in range(n + 1)]

    score_m[0][0] = 0.0
    for i in range(1, n + 1):
        score_x[i][0] = gap_open if i == 1 else score_x[i - 1][0] + gap_extend
        trace_x[i][0] = 0 if i == 1 else 1
    for j in range(1, m + 1):
        score_y[0][j] = gap_open if j == 1 else score_y[0][j - 1] + gap_extend
        trace_y[0][j] = 0 if j == 1 else 2

    for i in range(1, n + 1):
        aa = seq1[i - 1]
        for j in range(1, m + 1):
            bb = seq2[j - 1]
            substitution = match_score if aa == bb else mismatch_score

            prev_m_scores = (score_m[i - 1][j - 1], score_x[i - 1][j - 1], score_y[i - 1][j - 1])
            prev_m_state = max(range(3), key=lambda idx: prev_m_scores[idx])
            score_m[i][j] = prev_m_scores[prev_m_state] + substitution
            trace_m[i][j] = prev_m_state

            open_x = score_m[i - 1][j] + gap_open
            extend_x = score_x[i - 1][j] + gap_extend
            if open_x >= extend_x:
                score_x[i][j] = open_x
                trace_x[i][j] = 0
            else:
                score_x[i][j] = extend_x
                trace_x[i][j] = 1

            open_y = score_m[i][j - 1] + gap_open
            extend_y = score_y[i][j - 1] + gap_extend
            if open_y >= extend_y:
                score_y[i][j] = open_y
                trace_y[i][j] = 0
            else:
                score_y[i][j] = extend_y
                trace_y[i][j] = 2

    end_scores = (score_m[n][m], score_x[n][m], score_y[n][m])
    state = max(range(3), key=lambda idx: end_scores[idx])
    i = n
    j = m
    matches = 0

    while i > 0 or j > 0:
        if state == 0:
            prev_state = trace_m[i][j]
            i -= 1
            j -= 1
            if i >= 0 and j >= 0 and seq1[i] == seq2[j]:
                matches += 1
            state = prev_state
        elif state == 1:
            prev_state = trace_x[i][j]
            i -= 1
            state = prev_state
        else:
            prev_state = trace_y[i][j]
            j -= 1
            state = prev_state

    denom = max(n, m)
    return (100.0 * matches / float(denom)) if denom > 0 else 0.0


def global_identity_percent(seq1: str, seq2: str) -> float:
    if not seq1 or not seq2:
        return 0.0
    if _BioPairwiseAligner is None:
        return global_identity_percent_affine_fallback(seq1, seq2)
    aligner = _BioPairwiseAligner(mode="global")
    aligner.match_score = 1.0
    aligner.mismatch_score = 0.0
    aligner.open_gap_score = -1.0
    aligner.extend_gap_score = -0.5
    alignment = aligner.align(seq1, seq2)[0]
    matches = 0
    for (start1, end1), (start2, end2) in zip(alignment.aligned[0], alignment.aligned[1]):
        matches += sum(aa == bb for aa, bb in zip(seq1[start1:end1], seq2[start2:end2]))
    denom = max(len(seq1), len(seq2))
    return (100.0 * matches / float(denom)) if denom > 0 else 0.0


def identities_pass(
    acc1: str,
    acc2: str,
    sequences: Dict[str, str],
    cache: Dict[Tuple[str, str], float],
    threshold: float,
) -> bool:
    key = (acc1, acc2) if acc1 <= acc2 else (acc2, acc1)
    if key not in cache:
        seq1 = sequences.get(acc1)
        seq2 = sequences.get(acc2)
        if seq1 is None or seq2 is None:
            cache[key] = 0.0
        else:
            cache[key] = global_identity_percent(seq1, seq2)
    return cache[key] > threshold


def interactions_redundant(
    first: TemplateInteractionMatch,
    second: TemplateInteractionMatch,
    sequences: Dict[str, str],
    cache: Dict[Tuple[str, str], float],
    threshold: float,
) -> bool:
    same_orientation = identities_pass(
        first.left_acc, second.left_acc, sequences, cache, threshold
    ) and identities_pass(first.right_acc, second.right_acc, sequences, cache, threshold)
    if same_orientation:
        return True
    return identities_pass(
        first.left_acc, second.right_acc, sequences, cache, threshold
    ) and identities_pass(first.right_acc, second.left_acc, sequences, cache, threshold)


def prune_redundant_candidates_greedy(
    candidates: Sequence[TemplateInteractionMatch],
    sequences: Dict[str, str],
    threshold: float,
) -> Tuple[List[TemplateInteractionMatch], Dict[Tuple[str, str], float], int]:
    cache: Dict[Tuple[str, str], float] = {}
    retained: List[TemplateInteractionMatch] = []
    pruned = 0
    ranked_candidates = sorted(candidates, key=representative_rank_key, reverse=True)
    for candidate in ranked_candidates:
        if any(
            interactions_redundant(candidate, kept, sequences, cache, threshold)
            for kept in retained
        ):
            pruned += 1
            continue
        retained.append(candidate)
    return retained, cache, pruned


def cluster_pair_key(candidate: TemplateInteractionMatch, cluster_by_accession: Dict[str, str]) -> Tuple[str, str]:
    left_cluster = cluster_by_accession[candidate.left_acc]
    right_cluster = cluster_by_accession[candidate.right_acc]
    if left_cluster <= right_cluster:
        return left_cluster, right_cluster
    return right_cluster, left_cluster


def prune_redundant_candidates_by_cluster_pair(
    candidates: Sequence[TemplateInteractionMatch],
    cluster_by_accession: Dict[str, str],
) -> Tuple[List[TemplateInteractionMatch], int, int]:
    retained: List[TemplateInteractionMatch] = []
    seen_cluster_pairs: Set[Tuple[str, str]] = set()
    pruned = 0
    ranked_candidates = sorted(candidates, key=representative_rank_key, reverse=True)
    for candidate in ranked_candidates:
        key = cluster_pair_key(candidate, cluster_by_accession)
        if key in seen_cluster_pairs:
            pruned += 1
            continue
        seen_cluster_pairs.add(key)
        retained.append(candidate)
    return retained, pruned, len(seen_cluster_pairs)


def write_matrix_tsv(path: Path, matrix: np.ndarray) -> None:
    np.savetxt(path, matrix, fmt="%.8g", delimiter="\t")


def write_fragment_pairs_tsv(path: Path, interactions: Sequence[TemplateInteractionMatch]) -> int:
    rows_written = 0
    with path.open("w", encoding="utf-8") as handle:
        handle.write(
            "template_left_acc\ttemplate_right_acc\tfragment_pair_index\tleft_qstart\tleft_qend\t"
            "right_qstart\tright_qend\tleft_query_coverage\tright_query_coverage\tleft_pident\t"
            "right_pident\tleft_bitscore\tright_bitscore\tleft_evalue\tright_evalue\t"
            "combined_bitscore\tcombined_evalue\tleft_covered_residues\tright_covered_residues\t"
            "supported_cell_count\n"
        )
        for interaction in interactions:
            for idx, fragment_pair in enumerate(interaction.fragment_pairs, start=1):
                left_hit = fragment_pair.left_hit
                right_hit = fragment_pair.right_hit
                handle.write(
                    f"{interaction.left_acc}\t{interaction.right_acc}\t{idx}\t"
                    f"{left_hit.qstart}\t{left_hit.qend}\t"
                    f"{right_hit.qstart}\t{right_hit.qend}\t"
                    f"{left_hit.query_coverage:.8g}\t{right_hit.query_coverage:.8g}\t"
                    f"{left_hit.pident:.8g}\t{right_hit.pident:.8g}\t"
                    f"{left_hit.bitscore:.8g}\t{right_hit.bitscore:.8g}\t"
                    f"{left_hit.evalue:.8g}\t{right_hit.evalue:.8g}\t"
                    f"{fragment_pair.combined_bitscore:.8g}\t{fragment_pair.combined_evalue:.8g}\t"
                    f"{len(left_hit.covered_q_positions)}\t{len(right_hit.covered_q_positions)}\t"
                    f"{len(fragment_pair.supported_cells)}\n"
                )
                rows_written += 1
    return rows_written


def write_template_interactions_tsv(path: Path, interactions: Sequence[TemplateInteractionMatch]) -> int:
    rows_written = 0
    with path.open("w", encoding="utf-8") as handle:
        handle.write(
            "template_left_acc\ttemplate_right_acc\tleft_hit_count\tright_hit_count\t"
            "fragment_pair_count\tsupported_cell_count\tbest_combined_bitscore\tbest_combined_evalue\n"
        )
        for interaction in interactions:
            handle.write(
                f"{interaction.left_acc}\t{interaction.right_acc}\t"
                f"{len(interaction.left_hits)}\t{len(interaction.right_hits)}\t"
                f"{interaction.fragment_pair_count}\t{len(interaction.supported_cells)}\t"
                f"{interaction.best_combined_bitscore:.8g}\t{interaction.best_combined_evalue:.8g}\n"
            )
            rows_written += 1
    return rows_written


def write_top_pairs(path: Path, matrix: np.ndarray, vote_counts: np.ndarray) -> int:
    coords = np.argwhere(matrix > 0.0)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("res1\tres2\tscore\tvotes\n")
        if coords.size == 0:
            return 0
        scores = matrix[coords[:, 0], coords[:, 1]]
        order = np.argsort(scores)[::-1]
        for idx in order:
            i0, j0 = coords[idx]
            handle.write(
                f"{i0 + 1}\t{j0 + 1}\t{matrix[i0, j0]:.8g}\t{int(vote_counts[i0, j0])}\n"
            )
    return len(coords)


def maybe_write_heatmap(path: Path, matrix: np.ndarray, title: str, label: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Requested --heatmap but matplotlib is not available") from exc
    plt.figure(figsize=(8, 6), dpi=150)
    plt.imshow(matrix, aspect="auto", origin="upper", cmap="viridis")
    plt.colorbar(label=label)
    plt.xlabel("Query 2 residue")
    plt.ylabel("Query 1 residue")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    dataset_defaults = dataset_resource_defaults(args.template_dataset)
    if args.pairs is None:
        args.pairs = dataset_defaults["pairs"]
    if args.blast_db is None:
        args.blast_db = dataset_defaults["blast_db"]
    if args.template_fasta is None and dataset_defaults["template_fasta"].exists():
        args.template_fasta = dataset_defaults["template_fasta"]

    template_fasta = infer_template_fasta_path(args)
    if not args.pairs.exists():
        raise FileNotFoundError(f"Template pair file not found: {args.pairs}")

    q1_id, q1_seq = read_single_fasta(args.query1)
    q2_id, q2_seq = read_single_fasta(args.query2)
    q1_len = len(q1_seq)
    q2_len = len(q2_seq)

    q1_out = args.out_dir / "q1.blast.tsv"
    q2_out = args.out_dir / "q2.blast.tsv"

    run_blastp(
        blast_bin=args.blast_bin,
        query_fasta=args.query1,
        db_prefix=args.blast_db,
        out_tsv=q1_out,
        threads=args.threads,
        evalue=args.evalue,
        max_target_seqs=args.max_target_seqs,
    )
    run_blastp(
        blast_bin=args.blast_bin,
        query_fasta=args.query2,
        db_prefix=args.blast_db,
        out_tsv=q2_out,
        threads=args.threads,
        evalue=args.evalue,
        max_target_seqs=args.max_target_seqs,
    )

    hits1 = parse_blast_hits(
        tsv_path=q1_out,
        query_len=q1_len,
        min_pident=args.min_pident,
        min_aln_len=args.min_aln_len,
        min_cov=args.min_cov1,
        max_cov=args.max_cov1,
        top_k=args.top_k,
    )
    hits2 = parse_blast_hits(
        tsv_path=q2_out,
        query_len=q2_len,
        min_pident=args.min_pident,
        min_aln_len=args.min_aln_len,
        min_cov=args.min_cov2,
        max_cov=args.max_cov2,
        top_k=args.top_k,
    )
    hits1_by_acc = group_hits_by_template(hits1)
    hits2_by_acc = group_hits_by_template(hits2)

    pair_load = load_template_pairs(args.pairs, args.pair_method_substring)
    template_pairs = list(pair_load.pairs)
    pair_rows_loaded = pair_load.valid_rows

    candidate_member_pairs: Set[Tuple[str, str]] = set()
    candidate_fragment_pairs = 0
    candidate_interactions: List[TemplateInteractionMatch] = []

    for acc_a, acc_b in template_pairs:
        left_hits = hits1_by_acc.get(acc_a)
        right_hits = hits2_by_acc.get(acc_b)
        if left_hits and right_hits:
            candidate_member_pairs.add((acc_a, acc_b))
            candidate = build_template_interaction_match(acc_a, acc_b, left_hits, right_hits, q2_len)
            candidate_fragment_pairs += candidate.fragment_pair_count
            candidate_interactions.append(candidate)

        left_hits = hits1_by_acc.get(acc_b)
        right_hits = hits2_by_acc.get(acc_a)
        if left_hits and right_hits:
            candidate_member_pairs.add((acc_a, acc_b))
            candidate = build_template_interaction_match(acc_b, acc_a, left_hits, right_hits, q2_len)
            candidate_fragment_pairs += candidate.fragment_pair_count
            candidate_interactions.append(candidate)

    candidate_accessions = {
        acc for candidate in candidate_interactions for acc in (candidate.left_acc, candidate.right_acc)
    }
    template_sequences = load_template_sequences(template_fasta, candidate_accessions)
    missing_candidate_sequences = sorted(candidate_accessions - set(template_sequences))
    redundancy_summary = {
        "mode": args.redundancy_mode,
        "identity_threshold_percent": args.identity_threshold,
        "candidate_template_proteins_with_sequences": len(template_sequences),
        "candidate_template_proteins_missing_sequences": len(missing_candidate_sequences),
        "missing_sequence_accessions_sample": missing_candidate_sequences[:20],
    }
    if args.redundancy_mode == "cdhit_cluster_pair":
        cluster_by_accession, cdhit_word = cluster_sequences_with_cdhit(
            template_sequences,
            args.cdhit_bin,
            args.identity_threshold,
            args.threads,
        )
        for accession in missing_candidate_sequences:
            cluster_by_accession[accession] = f"MissingSequence_{accession}"
        retained_candidates, pruned_candidates, retained_cluster_pairs = (
            prune_redundant_candidates_by_cluster_pair(candidate_interactions, cluster_by_accession)
        )
        redundancy_summary.update(
            {
                "cdhit_bin": args.cdhit_bin,
                "cdhit_word_length": cdhit_word,
                "candidate_template_protein_clusters": len(set(cluster_by_accession.values())),
                "retained_cluster_pairs": retained_cluster_pairs,
            }
        )
    else:
        retained_candidates, identity_cache, pruned_candidates = prune_redundant_candidates_greedy(
            candidate_interactions,
            template_sequences,
            args.identity_threshold,
        )
        redundancy_summary.update(
            {
                "protein_identity_method": protein_identity_method_description(),
                "pairwise_identity_comparisons_cached": len(identity_cache),
            }
        )
    retained_fragment_pairs = sum(candidate.fragment_pair_count for candidate in retained_candidates)

    denominator = len(retained_candidates)
    vote_counts = np.zeros((q1_len, q2_len), dtype=np.uint32)
    if denominator > 0:
        for candidate in retained_candidates:
            for index in candidate.supported_cells:
                i0 = index // q2_len
                j0 = index % q2_len
                vote_counts[i0, j0] += 1

    ifrag_matrix = (
        vote_counts.astype(np.float64) / float(denominator)
        if denominator > 0
        else np.zeros((q1_len, q2_len), dtype=np.float64)
    )

    matrix_npy = args.out_dir / "ifrag_matrix.npy"
    matrix_tsv = args.out_dir / "ifrag_matrix.tsv"
    top_pairs_tsv = args.out_dir / "ifrag_top_pairs.tsv"
    template_interactions_tsv = args.out_dir / "ifrag_template_interactions.tsv"
    fragment_pairs_tsv = args.out_dir / "ifrag_fragment_pairs.tsv"
    summary_json = args.out_dir / "ifrag_summary.json"
    heatmap_png = args.out_dir / "ifrag_heatmap.png"

    np.save(matrix_npy, ifrag_matrix)
    write_matrix_tsv(matrix_tsv, ifrag_matrix)
    retained_template_rows = write_template_interactions_tsv(template_interactions_tsv, retained_candidates)
    retained_fragment_rows = write_fragment_pairs_tsv(fragment_pairs_tsv, retained_candidates)
    top_pair_rows = write_top_pairs(top_pairs_tsv, ifrag_matrix, vote_counts)
    if args.heatmap:
        maybe_write_heatmap(heatmap_png, ifrag_matrix, "Classical iFrag matrix", "iFrag score")

    summary = {
        "method": "classical_ifrag_blast",
        "description": (
            "BLAST-based classical iFrag branch using explicit fragment-pair matches nested inside "
            "template interactions, followed by template interaction redundancy pruning."
        ),
        "query_ids": {"q1": q1_id, "q2": q2_id},
        "query_lengths": {"q1": q1_len, "q2": q2_len},
        "template_pairs_total_rows_loaded": pair_rows_loaded,
        "template_pairs_total_nonempty_rows": pair_load.total_nonempty_rows,
        "template_pairs_unique_loaded": len(template_pairs),
        "template_pairs_filtered_out_by_method": pair_load.filtered_out_rows,
        "template_dataset": args.template_dataset,
        "template_pair_method_substring": list(args.pair_method_substring),
        "template_proteins_with_hits_q1": len(hits1_by_acc),
        "template_proteins_with_hits_q2": len(hits2_by_acc),
        "blast_parameters": {
            "blast_bin": args.blast_bin,
            "blast_db": str(args.blast_db),
            "template_fasta": str(template_fasta),
            "pairs_file": str(args.pairs),
            "threads": args.threads,
            "evalue": args.evalue,
            "max_target_seqs": args.max_target_seqs,
            "outfmt": BLAST_OUTFMT,
        },
        "hit_filters": {
            "min_pident": args.min_pident,
            "min_aln_len": args.min_aln_len,
            "min_cov1": args.min_cov1,
            "max_cov1": args.max_cov1,
            "min_cov2": args.min_cov2,
            "max_cov2": args.max_cov2,
            "top_k": args.top_k,
        },
        "candidate_member_template_pairs_found": len(candidate_member_pairs),
        "candidate_template_interactions_found": len(candidate_interactions),
        "candidate_fragment_pairs": candidate_fragment_pairs,
        "candidate_member_hit_combinations": candidate_fragment_pairs,
        "classical_scoring_unit": {
            "fragment_pair_definition": "one BLAST HSP from query1 paired with one BLAST HSP from query2",
            "template_interaction_vote": (
                "one retained nonredundant template interaction contributes one vote to each cell in "
                "the union of its fragment-pair-supported cells"
            ),
        },
        "redundancy_filter": {
            **redundancy_summary,
            "candidate_interactions_retained": len(retained_candidates),
            "candidate_interactions_pruned": pruned_candidates,
            "retained_fragment_pairs": retained_fragment_pairs,
        },
        "vote_denominator_N": denominator,
        "fraction_nonzero_cells": float(np.count_nonzero(ifrag_matrix > 0.0)) / float(q1_len * q2_len),
        "outputs": {
            "q1_blast_tsv": str(q1_out),
            "q2_blast_tsv": str(q2_out),
            "ifrag_matrix_npy": str(matrix_npy),
            "ifrag_matrix_tsv": str(matrix_tsv),
            "ifrag_top_pairs_tsv": str(top_pairs_tsv),
            "ifrag_template_interactions_tsv": str(template_interactions_tsv),
            "ifrag_fragment_pairs_tsv": str(fragment_pairs_tsv),
            "ifrag_summary_json": str(summary_json),
            "ifrag_heatmap_png": str(heatmap_png) if args.heatmap else None,
        },
        "rows_written": {
            "ifrag_top_pairs": top_pair_rows,
            "ifrag_template_interactions": retained_template_rows,
            "ifrag_fragment_pairs": retained_fragment_rows,
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
