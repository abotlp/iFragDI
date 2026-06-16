#!/usr/bin/env python3
"""Build a clean IntAct + BioGRID + STRING physical PPI universe.

This script is intentionally standalone and uses only the Python standard
library. It reads the raw source archives directly, filters for physical
protein-protein interactions, maps interactors to canonical UniProt base
accessions, normalizes pairs as undirected, and writes deduplicated outputs.

Outputs:
  - intact_biogrid_string.final.tsv
      One unique undirected pair per row with aggregated methods and sources.
  - intact_biogrid_string.evidence.final.tsv
      One deduplicated row per (protein_1, protein_2, detection_method,
      detection_id, source) tuple.
  - template_pairs.final.tsv
      One deduplicated undirected pair per row.
  - template_pairs.meta.final.tsv
      Per-pair aggregated provenance and species metadata.
  - proteins.final.tsv
      Unique proteins retained in the universe.
  - build_summary.tsv
      Build counters by source and drop reason.

Biological defaults:
  - All organisms kept.
  - Same-species and cross-species interactions kept.
  - Self-interactions kept.
  - Pairs are undirected (canonical order = lexicographic sort).
  - Isoforms collapsed to UniProt base accessions.
  - IntAct: protein-protein rows only, negatives removed, obvious genetic/
    non-physical interaction types removed, expanded complex rows kept by
    default (disable with --drop-intact-expanded).
  - BioGRID: Experimental System Type == physical only.
  - STRING: parses the local physical links file you actually have. Since the
    file does not contain assay-level methods, STRING evidence is represented
    as positive evidence channels (experimental, database, textmining, and
    transferred variants if present in the header).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import re
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import Iterable, Iterator, Sequence


UNIPROT_6_RE = re.compile(
    r"^[OPQ][0-9][A-Z0-9]{3}[0-9]$"
    r"|^[A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9]$"
)
UNIPROT_10_RE = re.compile(
    r"^[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){2}$"
)
MI_ID_RE = re.compile(r"MI:\d{4}")
MI_LABEL_RE = re.compile(r"\((.+)\)$")
TAXID_RE = re.compile(r"taxid:(-?\d+)")
NON_PHYSICAL_INTACT_TERMS = (
    "genetic interaction",
    "synthetic lethality",
    "synthetic rescue",
    "synthetic growth defect",
    "synthetic haploinsufficiency",
    "dosage growth defect",
    "dosage lethality",
    "dosage rescue",
    "epistasis",
    "suppression",
    "phenotypic enhancement",
    "phenotypic suppression",
)

# STRING alias sources in priority order. The local aliases file definitely has
# UniProt_AC and Ensembl_UniProt; BLAST_UniProt_AC is kept as a harmless extra
# fallback in case the file varies across releases.
STRING_ALIAS_PRIORITY = {
    "UniProt_AC": 0,
    "Ensembl_UniProt": 1,
    "BLAST_UniProt_AC": 2,
}

STRING_DETAILED_HEADER = [
    "protein1",
    "protein2",
    "experimental",
    "database",
    "textmining",
    "combined_score",
]

STRING_FULL_HEADER = [
    "protein1",
    "protein2",
    "experimental",
    "experimental_transferred",
    "database",
    "database_transferred",
    "textmining",
    "textmining_transferred",
    "combined_score",
]

STRING_CHANNEL_SPECS = {
    "experimental": ("STRING experimental channel", "string:experimental"),
    "experimental_transferred": (
        "STRING experimental transferred channel",
        "string:experimental_transferred",
    ),
    "database": ("STRING database channel", "string:database"),
    "database_transferred": (
        "STRING database transferred channel",
        "string:database_transferred",
    ),
    "textmining": ("STRING textmining channel", "string:textmining"),
    "textmining_transferred": (
        "STRING textmining transferred channel",
        "string:textmining_transferred",
    ),
}


@dataclass(frozen=True, order=True)
class EvidenceRow:
    protein_1: str
    protein_2: str
    detection_method: str
    detection_id: str
    source: str


@dataclass
class PairMeta:
    sources: set[str] = field(default_factory=set)
    detection_methods: set[str] = field(default_factory=set)
    detection_ids: set[str] = field(default_factory=set)
    support_count: int = 0
    taxids_1: set[int] = field(default_factory=set)
    taxids_2: set[int] = field(default_factory=set)
    has_same_species_support: bool = False
    has_cross_species_support: bool = False
    has_self_interaction_support: bool = False


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[2]
    default_raw = default_root / "data" / "raw"
    default_out = default_root / "data" / "datasets" / "intact_biogrid_string"

    parser = argparse.ArgumentParser(
        description="Build a clean IntAct + BioGRID + STRING PPI universe.",
    )
    parser.add_argument(
        "--intact-zip",
        type=Path,
        default=default_raw / "intact.zip",
        help="Path to IntAct zip archive.",
    )
    parser.add_argument(
        "--biogrid-zip",
        type=Path,
        default=default_raw / "BIOGRID-SYSTEM-5.0.256.tab3.zip",
        help="Path to BioGRID SYSTEM zip archive.",
    )
    parser.add_argument(
        "--string-aliases-gz",
        type=Path,
        default=default_raw / "protein.aliases.v12.0.txt.gz",
        help="Path to STRING aliases file.",
    )
    parser.add_argument(
        "--string-links-gz",
        type=Path,
        default=default_raw / "protein.physical.links.detailed.v12.0.txt.gz",
        help="Path to STRING physical links file.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=default_out,
        help="Directory for output tables.",
    )
    parser.add_argument(
        "--drop-intact-expanded",
        action="store_true",
        help=(
            "Drop IntAct rows with non-empty expansion methods "
            "(for example spoke-expanded complex evidence)."
        ),
    )
    parser.add_argument(
        "--min-string-combined-score",
        type=int,
        default=400,
        help=(
            "Minimum STRING combined_score to keep a pair. Default 400. "
            "Use 0 to keep every row in the local physical links file."
        ),
    )
    parser.add_argument(
        "--min-string-experimental",
        type=int,
        default=0,
        help=(
            "Optional minimum STRING experimental channel score. Default 0. "
            "Use >0 only if you want to require some direct experimental "
            "support from STRING."
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=500000,
        help=(
            "Print a progress update to stderr every N parsed rows per source. "
            "Use 0 to disable."
        ),
    )
    return parser.parse_args(argv)


def die(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def warn(message: str) -> None:
    print(f"WARNING: {message}", file=sys.stderr)


def ensure_exists(path: Path) -> None:
    if not path.exists():
        die(f"Required file not found: {path}")


def iter_zip_rows(zip_path: Path, member_name: str) -> Iterator[list[str]]:
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member_name) as handle:
            text_handle = (line.decode("utf-8", errors="replace") for line in handle)
            reader = csv.reader(text_handle, delimiter="\t")
            yield from reader


def find_intact_member(zip_path: Path) -> str:
    with zipfile.ZipFile(zip_path) as zf:
        candidates = [
            name
            for name in sorted(zf.namelist())
            if name.endswith(".txt")
            and "readme" not in name.lower()
            and "schema" not in name.lower()
            and "negative" not in name.lower()
        ]
    if not candidates:
        die(f"No IntAct interaction text file found inside {zip_path}")
    if len(candidates) > 1:
        warn(
            f"Multiple IntAct text members found in {zip_path}; using first: "
            f"{candidates[0]} ; others={candidates[1:]}"
        )
    return candidates[0]


def iter_biogrid_rows(zip_path: Path) -> Iterator[list[str]]:
    with zipfile.ZipFile(zip_path) as zf:
        for member_name in sorted(zf.namelist()):
            lower_name = member_name.lower()
            if not member_name.endswith(".txt"):
                continue
            if any(
                token in lower_name
                for token in ("readme", "release", "changelog", "license", "note")
            ):
                continue
            with zf.open(member_name) as handle:
                text_handle = (
                    line.decode("utf-8", errors="replace") for line in handle
                )
                reader = csv.reader(text_handle, delimiter="\t")
                for row_index, row in enumerate(reader):
                    if row_index == 0:
                        continue
                    yield row


def maybe_report_progress(
    source: str,
    stats: Counter[str],
    every: int,
) -> None:
    if every <= 0:
        return
    total = stats.get(f"{source}_total_rows", 0)
    if total == 0 or total % every != 0:
        return

    kept_rows = stats.get(f"{source}_kept_source_rows", 0)
    duplicate_rows = stats.get(f"{source}_duplicate_source_rows", 0)
    kept_evidence = stats.get(f"{source}_kept_evidence_rows", 0)
    dropped = sum(
        value
        for key, value in stats.items()
        if key.startswith(f"{source}_drop_")
    )
    print(
        (
            f"[{source}] processed={total:,} kept_rows={kept_rows:,} "
            f"duplicate_rows={duplicate_rows:,} dropped={dropped:,} "
            f"kept_evidence={kept_evidence:,}"
        ),
        file=sys.stderr,
    )


def is_uniprot_accession(value: str) -> bool:
    return bool(UNIPROT_6_RE.match(value) or UNIPROT_10_RE.match(value))


def normalize_uniprot_accession(raw: str) -> str | None:
    value = raw.strip()
    if not value or value == "-":
        return None
    if ":" in value:
        namespace, token = value.split(":", 1)
        if namespace.lower() != "uniprotkb":
            return None
        value = token
    value = value.split(".", 1)[0].upper()
    base = value.split("-", 1)[0]
    if is_uniprot_accession(base):
        return base
    return None


def unique_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def extract_uniprot_candidates(field: str) -> list[str]:
    candidates: list[str] = []
    for token in field.split("|"):
        candidate = normalize_uniprot_accession(token)
        if candidate:
            candidates.append(candidate)
    return unique_preserve_order(candidates)


def choose_single_accession(primary_fields: Sequence[str]) -> tuple[str | None, str]:
    for field in primary_fields:
        unique = extract_uniprot_candidates(field)
        if unique:
            if len(unique) == 1:
                return unique[0], "ok"
            return None, "ambiguous"
    return None, "missing"


def parse_taxid(field: str) -> int | None:
    match = TAXID_RE.search(field)
    if not match:
        return None
    value = int(match.group(1))
    return value if value > 0 else None


def parse_positive_int(field: str) -> int | None:
    try:
        value = int(field)
    except ValueError:
        return None
    return value if value > 0 else None


def parse_mi_term_pairs(field: str) -> list[tuple[str, str]]:
    if not field or field == "-":
        return [("-", "-")]
    terms: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for token in field.split("|"):
        token = token.strip()
        if not token or token == "-":
            continue
        id_match = MI_ID_RE.search(token)
        label_match = MI_LABEL_RE.search(token)
        label = label_match.group(1).strip() if label_match else token
        term_id = id_match.group(0) if id_match else token
        pair = (label, term_id)
        if pair not in seen:
            seen.add(pair)
            terms.append(pair)
    return terms if terms else [("-", "-")]


def has_nonphysical_intact_interaction_type(field: str) -> bool:
    lowered = field.lower()
    if "mi:0208" in lowered:
        return True
    return any(term in lowered for term in NON_PHYSICAL_INTACT_TERMS)


def is_protein_interactor(field: str) -> bool:
    if not field or field == "-":
        return False
    for token in field.split("|"):
        token = token.strip().lower()
        if "(protein)" in token or token.endswith("protein"):
            return True
    return False


def normalize_pair(
    protein_a: str,
    protein_b: str,
    taxid_a: int | None,
    taxid_b: int | None,
) -> tuple[str, str, int | None, int | None]:
    if protein_a <= protein_b:
        return protein_a, protein_b, taxid_a, taxid_b
    return protein_b, protein_a, taxid_b, taxid_a


def slugify_biogrid_method(method: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", method.strip()).strip("_")
    return f"biogrid:{slug}" if slug else "biogrid:unknown"


def update_pair_meta(
    pair_meta: dict[tuple[str, str], PairMeta],
    evidence: EvidenceRow,
    taxid_1: int | None,
    taxid_2: int | None,
) -> None:
    meta = pair_meta[(evidence.protein_1, evidence.protein_2)]
    meta.sources.add(evidence.source)
    if evidence.detection_method and evidence.detection_method != "-":
        meta.detection_methods.add(evidence.detection_method)
    if evidence.detection_id and evidence.detection_id != "-":
        meta.detection_ids.add(evidence.detection_id)
    meta.support_count += 1
    if taxid_1 is not None:
        meta.taxids_1.add(taxid_1)
    if taxid_2 is not None:
        meta.taxids_2.add(taxid_2)
    if taxid_1 is not None and taxid_2 is not None:
        if taxid_1 == taxid_2:
            meta.has_same_species_support = True
        else:
            meta.has_cross_species_support = True
    if evidence.protein_1 == evidence.protein_2:
        meta.has_self_interaction_support = True


def update_proteins(
    proteins: dict[str, dict[str, set[object]]],
    protein: str,
    taxid: int | None,
    source: str,
) -> None:
    entry = proteins[protein]
    entry["sources"].add(source)
    if taxid is not None:
        entry["taxids"].add(taxid)


def build_intact(
    intact_zip: Path,
    evidence_rows: set[EvidenceRow],
    pair_meta: dict[tuple[str, str], PairMeta],
    proteins: dict[str, dict[str, set[object]]],
    stats: Counter[str],
    drop_expanded: bool,
    progress_every: int,
) -> None:
    print("[intact] Parsing ...", file=sys.stderr)
    intact_member = find_intact_member(intact_zip)
    row_iter = iter_zip_rows(intact_zip, intact_member)
    first_row = next(row_iter, None)
    if first_row is None:
        die(f"{intact_zip} does not contain any rows in {intact_member}")
    has_header = bool(
        first_row
        and first_row[0]
        and first_row[0].lstrip("#").startswith("ID(s) interactor A")
    )
    rows = row_iter if has_header else chain([first_row], row_iter)

    for row in rows:
        stats["intact_total_rows"] += 1
        maybe_report_progress("intact", stats, progress_every)
        if len(row) < 36:
            stats["intact_drop_short_row"] += 1
            continue

        id_a = row[0]
        id_b = row[1]
        alt_a = row[2]
        alt_b = row[3]
        detection_field = row[6]
        taxid_a = parse_taxid(row[9])
        taxid_b = parse_taxid(row[10])
        interaction_type_field = row[11]
        expansion_field = row[15]
        type_a = row[20]
        type_b = row[21]
        negative_field = row[35].strip().lower()

        if negative_field == "true":
            stats["intact_drop_negative"] += 1
            continue
        if has_nonphysical_intact_interaction_type(interaction_type_field):
            stats["intact_drop_nonphysical_interaction_type"] += 1
            continue
        if drop_expanded and expansion_field and expansion_field != "-":
            stats["intact_drop_expanded_complex"] += 1
            continue
        if not is_protein_interactor(type_a) or not is_protein_interactor(type_b):
            stats["intact_drop_nonprotein"] += 1
            continue

        protein_a, status_a = choose_single_accession((id_a, alt_a))
        protein_b, status_b = choose_single_accession((id_b, alt_b))
        if protein_a is None:
            stats[f"intact_drop_map_a_{status_a}"] += 1
            continue
        if protein_b is None:
            stats[f"intact_drop_map_b_{status_b}"] += 1
            continue
        protein_1, protein_2, taxid_1, taxid_2 = normalize_pair(
            protein_a,
            protein_b,
            taxid_a,
            taxid_b,
        )
        method_pairs = parse_mi_term_pairs(detection_field)
        kept_any_evidence = False
        for detection_method, detection_id in method_pairs:
            evidence = EvidenceRow(
                protein_1=protein_1,
                protein_2=protein_2,
                detection_method=detection_method,
                detection_id=detection_id,
                source="IntAct",
            )
            if evidence in evidence_rows:
                stats["intact_duplicate_evidence_rows"] += 1
                continue
            evidence_rows.add(evidence)
            update_pair_meta(pair_meta, evidence, taxid_1, taxid_2)
            kept_any_evidence = True
            stats["intact_kept_evidence_rows"] += 1

        if kept_any_evidence:
            update_proteins(proteins, protein_1, taxid_1, "IntAct")
            update_proteins(proteins, protein_2, taxid_2, "IntAct")
            stats["intact_kept_source_rows"] += 1
        else:
            stats["intact_duplicate_source_rows"] += 1

    print(
        f"[intact] done. total={stats['intact_total_rows']:,} "
        f"kept_evidence={stats['intact_kept_evidence_rows']:,}",
        file=sys.stderr,
    )


def choose_biogrid_accession(
    swissprot_field: str,
    trembl_field: str,
) -> tuple[str | None, str]:
    swiss = extract_uniprot_candidates(swissprot_field)
    if len(swiss) == 1:
        return swiss[0], "ok"
    if len(swiss) > 1:
        return None, "ambiguous_swissprot"

    trembl = extract_uniprot_candidates(trembl_field)
    if len(trembl) == 1:
        return trembl[0], "ok"
    if len(trembl) > 1:
        return None, "ambiguous_trembl"
    return None, "missing"


def build_biogrid(
    biogrid_zip: Path,
    evidence_rows: set[EvidenceRow],
    pair_meta: dict[tuple[str, str], PairMeta],
    proteins: dict[str, dict[str, set[object]]],
    stats: Counter[str],
    progress_every: int,
) -> None:
    print("[biogrid] Parsing ...", file=sys.stderr)
    for row in iter_biogrid_rows(biogrid_zip):
        stats["biogrid_total_rows"] += 1
        maybe_report_progress("biogrid", stats, progress_every)
        if len(row) < 29:
            stats["biogrid_drop_short_row"] += 1
            continue

        experimental_system = row[11].strip()
        experimental_system_type = row[12].strip().lower()
        taxid_a = parse_positive_int(row[15])
        taxid_b = parse_positive_int(row[16])
        swissprot_a = row[23]
        trembl_a = row[24]
        swissprot_b = row[26]
        trembl_b = row[27]

        if experimental_system_type != "physical":
            stats["biogrid_drop_nonphysical"] += 1
            continue

        protein_a, status_a = choose_biogrid_accession(swissprot_a, trembl_a)
        protein_b, status_b = choose_biogrid_accession(swissprot_b, trembl_b)
        if protein_a is None:
            stats[f"biogrid_drop_map_a_{status_a}"] += 1
            continue
        if protein_b is None:
            stats[f"biogrid_drop_map_b_{status_b}"] += 1
            continue
        protein_1, protein_2, taxid_1, taxid_2 = normalize_pair(
            protein_a,
            protein_b,
            taxid_a,
            taxid_b,
        )
        evidence = EvidenceRow(
            protein_1=protein_1,
            protein_2=protein_2,
            detection_method=experimental_system or "-",
            detection_id=slugify_biogrid_method(experimental_system),
            source="BioGRID",
        )
        if evidence in evidence_rows:
            stats["biogrid_duplicate_evidence_rows"] += 1
            stats["biogrid_duplicate_source_rows"] += 1
            continue
        evidence_rows.add(evidence)
        update_pair_meta(pair_meta, evidence, taxid_1, taxid_2)
        update_proteins(proteins, protein_1, taxid_1, "BioGRID")
        update_proteins(proteins, protein_2, taxid_2, "BioGRID")
        stats["biogrid_kept_evidence_rows"] += 1
        stats["biogrid_kept_source_rows"] += 1

    print(
        f"[biogrid] done. total={stats['biogrid_total_rows']:,} "
        f"kept_evidence={stats['biogrid_kept_evidence_rows']:,}",
        file=sys.stderr,
    )


def build_string_to_uniprot(
    aliases_gz: Path,
    progress_every: int,
) -> dict[str, str]:
    print("[string-aliases] Streaming aliases ...", file=sys.stderr)
    best: dict[str, tuple[int, str | None]] = {}
    total_lines = 0

    with gzip.open(aliases_gz, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            total_lines += 1
            if progress_every > 0 and total_lines % progress_every == 0:
                print(
                    f"[string-aliases] processed={total_lines:,} "
                    f"tracked_ids={len(best):,}",
                    file=sys.stderr,
                )
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue

            string_id, alias, source = parts[0], parts[1], parts[2]
            priority = STRING_ALIAS_PRIORITY.get(source)
            if priority is None:
                continue

            alias_base = alias.strip().upper().split("-", 1)[0]
            if not is_uniprot_accession(alias_base):
                continue

            current = best.get(string_id)
            if current is None:
                best[string_id] = (priority, alias_base)
                continue

            current_priority, current_accession = current
            if priority < current_priority:
                best[string_id] = (priority, alias_base)
            elif priority == current_priority and current_accession != alias_base:
                best[string_id] = (priority, None)

    resolved = {
        string_id: accession
        for string_id, (_, accession) in best.items()
        if accession is not None
    }
    print(
        f"[string-aliases] done. lines={total_lines:,} "
        f"resolved_ids={len(resolved):,}",
        file=sys.stderr,
    )
    return resolved


def parse_string_header(header_cols: list[str]) -> dict[str, int]:
    if header_cols == STRING_DETAILED_HEADER:
        return {
            "protein1": 0,
            "protein2": 1,
            "experimental": 2,
            "database": 3,
            "textmining": 4,
            "combined_score": 5,
        }
    if header_cols == STRING_FULL_HEADER:
        return {
            "protein1": 0,
            "protein2": 1,
            "experimental": 2,
            "experimental_transferred": 3,
            "database": 4,
            "database_transferred": 5,
            "textmining": 6,
            "textmining_transferred": 7,
            "combined_score": 8,
        }
    die(
        "Unsupported STRING links header. Expected either "
        f"{STRING_DETAILED_HEADER} or {STRING_FULL_HEADER}, got {header_cols}"
    )


def taxid_from_string_id(string_id: str) -> int | None:
    prefix, _, _ = string_id.partition(".")
    return parse_positive_int(prefix)


def positive_string_channels(
    parts: list[str],
    column_map: dict[str, int],
) -> list[tuple[str, str]]:
    channels: list[tuple[str, str]] = []
    for channel_name, (method, method_id) in STRING_CHANNEL_SPECS.items():
        index = column_map.get(channel_name)
        if index is None:
            continue
        try:
            value = int(parts[index])
        except ValueError:
            continue
        if value > 0:
            channels.append((method, method_id))
    if channels:
        return channels
    return [("STRING physical link", "string:physical_link")]


def build_string(
    links_gz: Path,
    string_to_uniprot: dict[str, str],
    evidence_rows: set[EvidenceRow],
    pair_meta: dict[tuple[str, str], PairMeta],
    proteins: dict[str, dict[str, set[object]]],
    stats: Counter[str],
    min_combined_score: int,
    min_experimental: int,
    progress_every: int,
) -> None:
    print("[string] Parsing ...", file=sys.stderr)
    with gzip.open(links_gz, "rt", encoding="utf-8", errors="replace") as handle:
        header_line = handle.readline()
        if not header_line:
            die(f"{links_gz} is empty")
        header_cols = header_line.strip().split()
        column_map = parse_string_header(header_cols)

        for line in handle:
            stats["string_total_rows"] += 1
            maybe_report_progress("string", stats, progress_every)

            parts = line.rstrip("\n").split()
            min_len = max(column_map.values()) + 1
            if len(parts) < min_len:
                stats["string_drop_short_row"] += 1
                continue

            string_id_a = parts[column_map["protein1"]]
            string_id_b = parts[column_map["protein2"]]

            try:
                experimental = int(parts[column_map["experimental"]])
                combined_score = int(parts[column_map["combined_score"]])
            except ValueError:
                stats["string_drop_parse_error"] += 1
                continue

            if experimental < min_experimental:
                stats["string_drop_below_min_experimental"] += 1
                continue
            if combined_score < min_combined_score:
                stats["string_drop_below_min_combined_score"] += 1
                continue

            protein_a = string_to_uniprot.get(string_id_a)
            protein_b = string_to_uniprot.get(string_id_b)
            if protein_a is None:
                stats["string_drop_map_a_missing"] += 1
                continue
            if protein_b is None:
                stats["string_drop_map_b_missing"] += 1
                continue

            taxid_a = taxid_from_string_id(string_id_a)
            taxid_b = taxid_from_string_id(string_id_b)
            protein_1, protein_2, taxid_1, taxid_2 = normalize_pair(
                protein_a,
                protein_b,
                taxid_a,
                taxid_b,
            )

            kept_any_evidence = False
            for detection_method, detection_id in positive_string_channels(
                parts, column_map
            ):
                evidence = EvidenceRow(
                    protein_1=protein_1,
                    protein_2=protein_2,
                    detection_method=detection_method,
                    detection_id=detection_id,
                    source="STRING",
                )
                if evidence in evidence_rows:
                    stats["string_duplicate_evidence_rows"] += 1
                    continue
                evidence_rows.add(evidence)
                update_pair_meta(pair_meta, evidence, taxid_1, taxid_2)
                kept_any_evidence = True
                stats["string_kept_evidence_rows"] += 1

            if kept_any_evidence:
                update_proteins(proteins, protein_1, taxid_1, "STRING")
                update_proteins(proteins, protein_2, taxid_2, "STRING")
                stats["string_kept_source_rows"] += 1
            else:
                stats["string_duplicate_source_rows"] += 1

    print(
        f"[string] done. total={stats['string_total_rows']:,} "
        f"kept_evidence={stats['string_kept_evidence_rows']:,}",
        file=sys.stderr,
    )


def write_tsv(path: Path, header: Sequence[str], rows: Iterable[Sequence[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def join_sorted(values: Iterable[object]) -> str:
    rendered = [str(value) for value in values]
    return ";".join(sorted(rendered)) if rendered else "-"


def join_sorted_ints(values: Iterable[int]) -> str:
    rendered = [str(value) for value in sorted(values)]
    return ";".join(rendered) if rendered else "-"


def build_summary_rows(
    stats: Counter[str],
    pair_meta: dict[tuple[str, str], PairMeta],
) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []

    def add(section: str, metric: str, value: object) -> None:
        rows.append((section, metric, str(value)))

    for source in ("intact", "biogrid", "string"):
        total = stats.get(f"{source}_total_rows", 0)
        kept_rows = stats.get(f"{source}_kept_source_rows", 0)
        duplicate_rows = stats.get(f"{source}_duplicate_source_rows", 0)
        kept_evidence = stats.get(f"{source}_kept_evidence_rows", 0)
        duplicate_evidence = stats.get(f"{source}_duplicate_evidence_rows", 0)
        drop_items = sorted(
            (
                key.removeprefix(f"{source}_drop_"),
                value,
            )
            for key, value in stats.items()
            if key.startswith(f"{source}_drop_")
        )
        dropped = sum(value for _, value in drop_items)
        accounted = kept_rows + duplicate_rows + dropped

        add(source, "total_rows", total)
        add(source, "kept_source_rows", kept_rows)
        add(source, "duplicate_source_rows", duplicate_rows)
        add(source, "kept_evidence_rows", kept_evidence)
        add(source, "duplicate_evidence_rows", duplicate_evidence)
        add(source, "discarded_rows_total", dropped)
        add(source, "rows_accounted_for", accounted)
        add(
            source,
            "kept_source_fraction",
            f"{(kept_rows / total):.6f}" if total else "0.000000",
        )
        add(
            source,
            "duplicate_source_fraction",
            f"{(duplicate_rows / total):.6f}" if total else "0.000000",
        )
        add(
            source,
            "discarded_fraction",
            f"{(dropped / total):.6f}" if total else "0.000000",
        )
        for metric, value in drop_items:
            add(source, f"drop_{metric}", value)

    source_combo_counts: Counter[str] = Counter()
    self_pairs = 0
    same_species_pairs = 0
    cross_species_pairs = 0

    for meta in pair_meta.values():
        source_combo_counts[join_sorted(meta.sources)] += 1
        if meta.has_self_interaction_support:
            self_pairs += 1
        if meta.has_same_species_support:
            same_species_pairs += 1
        if meta.has_cross_species_support:
            cross_species_pairs += 1

    add("merged", "unique_evidence_rows", stats.get("final_unique_evidence_rows", 0))
    add("merged", "unique_pairs", stats.get("final_unique_pairs", 0))
    add("merged", "unique_proteins", stats.get("final_unique_proteins", 0))
    for combo, count in sorted(source_combo_counts.items()):
        add("merged", f"pairs_source_{combo.replace(';', '_and_')}", count)
    add("merged", "self_interaction_pairs", self_pairs)
    add("merged", "same_species_pairs", same_species_pairs)
    add("merged", "cross_species_pairs", cross_species_pairs)

    return rows


def write_outputs(
    out_dir: Path,
    evidence_rows: set[EvidenceRow],
    pair_meta: dict[tuple[str, str], PairMeta],
    proteins: dict[str, dict[str, set[object]]],
    stats: Counter[str],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    final_pairs_path = out_dir / "intact_biogrid_string.final.tsv"
    evidence_path = out_dir / "intact_biogrid_string.evidence.final.tsv"
    pairs_path = out_dir / "template_pairs.final.tsv"
    pairs_meta_path = out_dir / "template_pairs.meta.final.tsv"
    proteins_path = out_dir / "proteins.final.tsv"
    summary_path = out_dir / "build_summary.tsv"

    sorted_evidence = sorted(evidence_rows)
    write_tsv(
        evidence_path,
        ("protein_1", "protein_2", "detection_method", "detection_id", "source"),
        (
            (
                row.protein_1,
                row.protein_2,
                row.detection_method,
                row.detection_id,
                row.source,
            )
            for row in sorted_evidence
        ),
    )

    sorted_pairs = sorted(pair_meta)
    write_tsv(
        final_pairs_path,
        ("protein_1", "protein_2", "detection_method", "detection_id", "source"),
        (
            (
                protein_1,
                protein_2,
                join_sorted(meta.detection_methods),
                join_sorted(meta.detection_ids),
                join_sorted(meta.sources),
            )
            for protein_1, protein_2 in sorted_pairs
            for meta in [pair_meta[(protein_1, protein_2)]]
        ),
    )

    write_tsv(
        pairs_path,
        ("protein_1", "protein_2"),
        sorted_pairs,
    )

    write_tsv(
        pairs_meta_path,
        (
            "protein_1",
            "protein_2",
            "sources",
            "detection_methods",
            "detection_ids",
            "support_count",
            "taxids_1",
            "taxids_2",
            "has_same_species_support",
            "has_cross_species_support",
            "has_self_interaction_support",
        ),
        (
            (
                protein_1,
                protein_2,
                join_sorted(meta.sources),
                join_sorted(meta.detection_methods),
                join_sorted(meta.detection_ids),
                meta.support_count,
                join_sorted_ints(meta.taxids_1),
                join_sorted_ints(meta.taxids_2),
                str(meta.has_same_species_support).lower(),
                str(meta.has_cross_species_support).lower(),
                str(meta.has_self_interaction_support).lower(),
            )
            for protein_1, protein_2 in sorted_pairs
            for meta in [pair_meta[(protein_1, protein_2)]]
        ),
    )

    write_tsv(
        proteins_path,
        ("protein_id", "sources", "taxids"),
        (
            (
                protein_id,
                join_sorted(entry["sources"]),
                join_sorted_ints(entry["taxids"]),
            )
            for protein_id, entry in sorted(proteins.items())
        ),
    )

    write_tsv(
        summary_path,
        ("section", "metric", "value"),
        build_summary_rows(stats, pair_meta),
    )
    return summary_path


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    ensure_exists(args.intact_zip)
    ensure_exists(args.biogrid_zip)
    ensure_exists(args.string_aliases_gz)
    ensure_exists(args.string_links_gz)

    evidence_rows: set[EvidenceRow] = set()
    pair_meta: dict[tuple[str, str], PairMeta] = defaultdict(PairMeta)
    proteins: dict[str, dict[str, set[object]]] = defaultdict(
        lambda: {"sources": set(), "taxids": set()}
    )
    stats: Counter[str] = Counter()

    build_intact(
        intact_zip=args.intact_zip,
        evidence_rows=evidence_rows,
        pair_meta=pair_meta,
        proteins=proteins,
        stats=stats,
        drop_expanded=args.drop_intact_expanded,
        progress_every=args.progress_every,
    )
    build_biogrid(
        biogrid_zip=args.biogrid_zip,
        evidence_rows=evidence_rows,
        pair_meta=pair_meta,
        proteins=proteins,
        stats=stats,
        progress_every=args.progress_every,
    )

    string_to_uniprot = build_string_to_uniprot(
        aliases_gz=args.string_aliases_gz,
        progress_every=args.progress_every,
    )
    build_string(
        links_gz=args.string_links_gz,
        string_to_uniprot=string_to_uniprot,
        evidence_rows=evidence_rows,
        pair_meta=pair_meta,
        proteins=proteins,
        stats=stats,
        min_combined_score=args.min_string_combined_score,
        min_experimental=args.min_string_experimental,
        progress_every=args.progress_every,
    )
    del string_to_uniprot

    stats["final_unique_evidence_rows"] = len(evidence_rows)
    stats["final_unique_pairs"] = len(pair_meta)
    stats["final_unique_proteins"] = len(proteins)

    summary_path = write_outputs(
        out_dir=args.out_dir,
        evidence_rows=evidence_rows,
        pair_meta=pair_meta,
        proteins=proteins,
        stats=stats,
    )

    print(f"Wrote IntAct + BioGRID + STRING dataset to: {args.out_dir}")
    print(f"Unique evidence rows: {len(evidence_rows)}")
    print(f"Unique pairs: {len(pair_meta)}")
    print(f"Unique proteins: {len(proteins)}")
    print(f"Coverage summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
