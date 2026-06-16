#!/usr/bin/env python3
from __future__ import annotations

import csv
import gzip
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


UNIPROT_BASE_RE = (
    r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9][A-Z0-9]{3}[0-9]|[A-Z0-9]{10})$"
)
UNIPROT_BASE_RX = re.compile(UNIPROT_BASE_RE)
TRAILING_DASH_NUMBER_RX = re.compile(r"-\d+$")
OX_RX = re.compile(r"\bOX=(\d+)\b")
TAXID_RX = re.compile(r"\b(?:taxid|TaxID)[:=](\d+)\b")
MISSING_TAXID_TOKENS = {"", "0", "00", "000", "0000", "na", "n/a", "none", "null", "-"}
MMSEQS_ALIGNMENT_OUTFMT = "query,target,pident,evalue,bits,qstart,qend,tstart,tend,qaln,taln,theader"
HOMOLOG_SEARCH_MODE_CHOICES = (
    "template_iterative",
    "template_single_pass",
)


@dataclass(frozen=True)
class ResolvedMmseqsHit:
    accession: str
    sequence_id: str
    search_tier: str
    taxid: str | None
    evalue: float
    bitscore: float
    aligned_query_positions: int
    pident: float
    row: str


def project_root() -> Path:
    return Path(__file__).resolve().parent


def default_template_fasta(dataset_name: str) -> Path:
    return project_root() / "data" / "interaction_templates" / dataset_name / "templates.fasta"


def default_template_proteins(dataset_name: str) -> Path:
    return project_root() / "data" / "interaction_templates" / dataset_name / "proteins.final.tsv"


def default_template_mmseqs_db(dataset_name: str) -> Path:
    return project_root() / "data" / "db" / f"mmseqs_templates_{dataset_name}" / "templates_db"


def mmseqs_db_exists(prefix: Path) -> bool:
    if prefix.exists():
        return True
    suffixes = (
        ".dbtype",
        ".index",
        ".lookup",
        ".source",
    )
    return any(Path(f"{prefix}{suffix}").exists() for suffix in suffixes)


def open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8", errors="replace") if path.suffix == ".gz" else path.open(
        "r", encoding="utf-8", errors="replace"
    )


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


def parse_taxid_field(text: str) -> str | None:
    for token in re.split(r"[;, ]+", text.strip()):
        value = token.strip()
        if not value or value.lower() in MISSING_TAXID_TOKENS:
            continue
        if value.isdigit():
            return None if int(value) == 0 else value
    return None


def parse_taxid_from_header(text: str) -> str | None:
    match = OX_RX.search(text)
    if match:
        value = match.group(1)
        return None if int(value) == 0 else value
    match = TAXID_RX.search(text)
    if match:
        value = match.group(1)
        return None if int(value) == 0 else value
    return None


def build_query_space_row(query_length: int, qstart: int, q_aln: str, t_aln: str) -> str:
    if len(q_aln) != len(t_aln):
        raise ValueError("Aligned query/target strings have different lengths")
    row = ["-"] * query_length
    qpos = qstart - 1
    for qchar, schar in zip(q_aln.upper(), t_aln.upper()):
        if qchar == "-":
            continue
        if qpos < 0 or qpos >= query_length:
            raise ValueError("HSP maps outside the query length")
        row[qpos] = "-" if schar == "-" else schar
        qpos += 1
    return "".join(row)


def better_hit_tuple(evalue: float, bitscore: float, aligned_query_positions: int, pident: float, key: str) -> tuple:
    return (evalue, -bitscore, -aligned_query_positions, -pident, key)


def summarize_hits_by_tier(hits: Dict[str, ResolvedMmseqsHit]) -> Dict[str, int]:
    counts = Counter(hit.search_tier for hit in hits.values())
    return dict(sorted(counts.items()))


def parse_template_mmseqs_hits(path: Path, query_length: int) -> tuple[Dict[str, ResolvedMmseqsHit], dict[str, object]]:
    resolved: Dict[str, ResolvedMmseqsHit] = {}
    stats = Counter()
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.lower().startswith("query\t"):
                continue
            parts = line.split("\t")
            if len(parts) != 12:
                raise RuntimeError(f"{path}: expected 12 MMseqs tabular columns, found {len(parts)}")
            (
                _query,
                target,
                pident,
                evalue,
                bits,
                qstart,
                _qend,
                _tstart,
                _tend,
                qaln,
                taln,
                theader,
            ) = parts
            target_id = (theader.split()[0] if theader.strip() else target).strip()
            accession = (
                canonicalize_accession(theader)
                or canonicalize_accession(target_id)
                or canonicalize_accession(target)
            )
            if accession is None:
                stats["rows_without_accession"] += 1
                continue
            hit = ResolvedMmseqsHit(
                accession=accession,
                sequence_id=accession,
                search_tier="template_db",
                taxid=parse_taxid_from_header(theader),
                evalue=float(evalue),
                bitscore=float(bits),
                aligned_query_positions=sum(1 for char in qaln if char != "-"),
                pident=float(pident),
                row=build_query_space_row(query_length, int(qstart), qaln, taln),
            )
            current = resolved.get(accession)
            if current is None or better_hit_tuple(
                hit.evalue,
                hit.bitscore,
                hit.aligned_query_positions,
                hit.pident,
                hit.accession,
            ) < better_hit_tuple(
                current.evalue,
                current.bitscore,
                current.aligned_query_positions,
                current.pident,
                current.accession,
            ):
                resolved[accession] = hit
                stats["best_hits_retained"] += 1
            else:
                stats["weaker_duplicate_hits"] += 1
    summary: dict[str, object] = dict(sorted(stats.items()))
    summary["resolved_accessions"] = len(resolved)
    summary["hits_by_search_tier"] = summarize_hits_by_tier(resolved)
    return resolved, summary


def write_resolved_hits_tsv(path: Path, hits: Dict[str, ResolvedMmseqsHit]) -> None:
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
                "row",
            ]
        )
        for accession in sorted(hits):
            hit = hits[accession]
            writer.writerow(
                [
                    hit.accession,
                    hit.sequence_id,
                    hit.search_tier,
                    hit.taxid or "",
                    f"{hit.evalue:.6g}",
                    f"{hit.bitscore:.6g}",
                    hit.aligned_query_positions,
                    f"{hit.pident:.6g}",
                    hit.row,
                ]
            )


def load_resolved_hits_tsv(path: Path) -> Dict[str, ResolvedMmseqsHit]:
    hits: Dict[str, ResolvedMmseqsHit] = {}
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        expected = {
            "accession",
            "sequence_id",
            "search_tier",
            "taxid",
            "evalue",
            "bitscore",
            "aligned_query_positions",
            "pident",
            "row",
        }
        if reader.fieldnames is None or not expected.issubset(set(reader.fieldnames)):
            raise RuntimeError(f"{path}: not a resolved-hits TSV")
        for row in reader:
            accession = canonicalize_accession(row.get("accession") or "")
            sequence_id = (row.get("sequence_id") or "").strip()
            if accession is None or not sequence_id:
                continue
            hit = ResolvedMmseqsHit(
                accession=accession,
                sequence_id=sequence_id,
                search_tier=(row.get("search_tier") or "").strip() or "template_db",
                taxid=(row.get("taxid") or "").strip() or None,
                evalue=float(row.get("evalue") or 0.0),
                bitscore=float(row.get("bitscore") or 0.0),
                aligned_query_positions=int(row.get("aligned_query_positions") or 0),
                pident=float(row.get("pident") or 0.0),
                row=(row.get("row") or "").strip(),
            )
            hits[accession] = hit
    return hits


def normalize_homolog_search_mode(search_mode: str) -> str:
    if search_mode not in HOMOLOG_SEARCH_MODE_CHOICES:
        raise ValueError(
            f"Unsupported homolog search mode {search_mode!r}; "
            f"expected one of {', '.join(HOMOLOG_SEARCH_MODE_CHOICES)}."
        )
    return search_mode


def run_mmseqs_command(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            "MMseqs command failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )


def load_template_taxids(path: Path) -> Dict[str, str]:
    taxids: Dict[str, str] = {}
    if not path.exists():
        return taxids
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"protein_id", "taxids"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise RuntimeError(f"{path}: expected proteins.final.tsv columns protein_id and taxids")
        for row in reader:
            accession = canonicalize_accession(row.get("protein_id") or "")
            if accession is None or accession in taxids:
                continue
            taxid = parse_taxid_field(row.get("taxids") or "")
            if taxid is not None:
                taxids[accession] = taxid
    return taxids


def enrich_hits_with_template_taxids(
    hits: Dict[str, ResolvedMmseqsHit],
    accession_taxids: Dict[str, str],
) -> Dict[str, ResolvedMmseqsHit]:
    enriched: Dict[str, ResolvedMmseqsHit] = {}
    for accession, hit in hits.items():
        enriched[accession] = ResolvedMmseqsHit(
            accession=hit.accession,
            sequence_id=hit.sequence_id,
            search_tier=hit.search_tier,
            taxid=hit.taxid or accession_taxids.get(accession),
            evalue=hit.evalue,
            bitscore=hit.bitscore,
            aligned_query_positions=hit.aligned_query_positions,
            pident=hit.pident,
            row=hit.row,
        )
    return enriched


def run_template_mmseqs_search(
    *,
    mmseqs_bin: str,
    query_fasta: Path,
    query_length: int,
    out_tsv: Path,
    work_dir: Path,
    template_db: Path,
    template_proteins: Path,
    threads: int = 8,
    max_hits: int = 100000,
    sensitivity: float = 7.5,
    evalue: float | None = None,
    stage1_iterations: int = 4,
    search_mode: str = "template_iterative",
) -> dict[str, object]:
    normalized_mode = normalize_homolog_search_mode(search_mode)

    work_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = work_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    query_db = work_dir / "query_db"
    stage1_result_db = work_dir / "template_stage1_result_db"
    result_db = work_dir / "template_result_db"
    raw_tsv = work_dir / "template_raw.tsv"

    run_mmseqs_command([mmseqs_bin, "createdb", str(query_fasta), str(query_db)])

    if normalized_mode == "template_iterative":
        result_db = stage1_result_db
        search_cmd = [
            mmseqs_bin,
            "search",
            str(query_db),
            str(template_db),
            str(result_db),
            str(tmp_dir),
            "--threads",
            str(threads),
            "-s",
            str(sensitivity),
            "--max-seq-id",
            "1.0",
            "--num-iterations",
            str(stage1_iterations),
            "-a",
        ]
    else:
        search_cmd = [
            mmseqs_bin,
            "search",
            str(query_db),
            str(template_db),
            str(result_db),
            str(tmp_dir),
            "--threads",
            str(threads),
            "-s",
            str(sensitivity),
            "--max-seq-id",
            "1.0",
            "--max-seqs",
            str(max_hits),
            "-a",
        ]
    if evalue is not None:
        search_cmd.extend(["-e", str(evalue)])
    run_mmseqs_command(search_cmd)

    run_mmseqs_command(
        [
            mmseqs_bin,
            "convertalis",
            str(query_db),
            str(template_db),
            str(result_db),
            str(raw_tsv),
            "--format-output",
            MMSEQS_ALIGNMENT_OUTFMT,
        ]
    )

    resolved_hits, resolve_summary = parse_template_mmseqs_hits(raw_tsv, query_length)
    accession_taxids = load_template_taxids(template_proteins)
    resolved_hits = enrich_hits_with_template_taxids(resolved_hits, accession_taxids)
    write_resolved_hits_tsv(out_tsv, resolved_hits)

    resolve_summary = dict(resolve_summary)
    resolve_summary["template_taxids_loaded"] = len(accession_taxids)

    return {
        "query_fasta": str(query_fasta),
        "query_length": query_length,
        "search_mode": normalized_mode,
        "template_db": str(template_db),
        "template_proteins": str(template_proteins),
        "stage1_iterations": stage1_iterations if normalized_mode == "template_iterative" else None,
        "max_hits": max_hits,
        "sensitivity": sensitivity,
        "evalue": evalue,
        "resolved_source": normalized_mode,
        "resolved_accessions": len(resolved_hits),
        "resolve_summary": resolve_summary,
        "outputs": {
            "resolved_tsv": str(out_tsv),
            "template_raw_tsv": str(raw_tsv),
        },
    }
