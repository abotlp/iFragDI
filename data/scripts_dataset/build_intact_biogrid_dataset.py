#!/usr/bin/env python3
"""Build a clean IntAct + BioGRID physical PPI universe.

This script is intentionally standalone and uses only the Python standard
library. It reads the raw source archives directly, filters for physical
protein-protein interactions, maps interactors to canonical UniProt base
accessions, normalizes pairs as undirected, and writes deduplicated outputs.

Outputs:
  - intact_biogrid.final.tsv
      One unique undirected pair per row with aggregated methods and sources.
  - intact_biogrid.evidence.final.tsv
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
"""

from __future__ import annotations

import argparse
import csv
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
    default_out = default_root / "data" / "datasets" / "intact_biogrid"

    parser = argparse.ArgumentParser(
        description="Build a clean IntAct + BioGRID PPI universe.",
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
        if label_match:
            label = label_match.group(1).strip()
        else:
            label = token
        if id_match:
            term_id = id_match.group(0)
        else:
            term_id = token
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
    proteins: dict[str, dict[str, set[str | int]]],
    stats: Counter[str],
    drop_expanded: bool,
    progress_every: int,
) -> None:
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
    proteins: dict[str, dict[str, set[str | int]]],
    stats: Counter[str],
    progress_every: int,
) -> None:
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

    for source in ("intact", "biogrid"):
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

    intact_only = 0
    biogrid_only = 0
    both_sources = 0
    self_pairs = 0
    same_species_pairs = 0
    cross_species_pairs = 0

    for meta in pair_meta.values():
        if meta.sources == {"IntAct"}:
            intact_only += 1
        elif meta.sources == {"BioGRID"}:
            biogrid_only += 1
        elif meta.sources == {"IntAct", "BioGRID"}:
            both_sources += 1
        else:
            add("merged", "unexpected_source_set", join_sorted(meta.sources))

        if meta.has_self_interaction_support:
            self_pairs += 1
        if meta.has_same_species_support:
            same_species_pairs += 1
        if meta.has_cross_species_support:
            cross_species_pairs += 1

    add("merged", "unique_evidence_rows", stats.get("final_unique_evidence_rows", 0))
    add("merged", "unique_pairs", stats.get("final_unique_pairs", 0))
    add("merged", "unique_proteins", stats.get("final_unique_proteins", 0))
    add("merged", "pairs_intact_only", intact_only)
    add("merged", "pairs_biogrid_only", biogrid_only)
    add("merged", "pairs_supported_by_both", both_sources)
    add("merged", "self_interaction_pairs", self_pairs)
    add("merged", "same_species_pairs", same_species_pairs)
    add("merged", "cross_species_pairs", cross_species_pairs)

    return rows


def write_outputs(
    out_dir: Path,
    evidence_rows: set[EvidenceRow],
    pair_meta: dict[tuple[str, str], PairMeta],
    proteins: dict[str, dict[str, set[str | int]]],
    stats: Counter[str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    final_pairs_path = out_dir / "intact_biogrid.final.tsv"
    evidence_path = out_dir / "intact_biogrid.evidence.final.tsv"
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


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    ensure_exists(args.intact_zip)
    ensure_exists(args.biogrid_zip)

    evidence_rows: set[EvidenceRow] = set()
    pair_meta: dict[tuple[str, str], PairMeta] = defaultdict(PairMeta)
    proteins: dict[str, dict[str, set[str | int]]] = defaultdict(
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

    stats["final_unique_evidence_rows"] = len(evidence_rows)
    stats["final_unique_pairs"] = len(pair_meta)
    stats["final_unique_proteins"] = len(proteins)

    write_outputs(
        out_dir=args.out_dir,
        evidence_rows=evidence_rows,
        pair_meta=pair_meta,
        proteins=proteins,
        stats=stats,
    )

    print(f"Wrote IntAct + BioGRID dataset to: {args.out_dir}")
    print(f"Unique evidence rows: {len(evidence_rows)}")
    print(f"Unique pairs: {len(pair_meta)}")
    print(f"Unique proteins: {len(proteins)}")
    print(f"Coverage summary: {args.out_dir / 'build_summary.tsv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
