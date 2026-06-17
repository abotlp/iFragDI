#!/usr/bin/env python3
"""Build a BM5 Phase 1 residue-level training table.

This script only merges existing Phase 1 outputs. It does not rerun iFragDI,
native labeling, docking, raDI, MMseqs, FAMSA, or freeSASA.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


TARGET_COLUMNS = ("interface_3p9A", "interface_5A", "interface_8A")
PRIMARY_TARGET = "interface_5A"
SECONDARY_TARGETS = ("interface_3p9A", "interface_8A")
KEY_COLUMNS = ("chainpair_id", "query_side", "score_residue_index")
EXPECTED_ZERO_5A_BOTH_SIDE_CASES = ("BM5CP00234", "BM5CP00238", "BM5CP00318")

FEATURE_COLUMNS = (
    "final_score",
    "patch_score",
    "ifrag_strength",
    "ifrag_specificity",
    "ifrag_component",
    "conservation_strength",
    "conservation_component",
    "radi_anchor",
    "radi_component",
    "blastpdb_anchor",
    "blastpdb_component",
)

IDENTITY_ALIASES = {
    "pdb_chain": "score_pdb_chain",
    "pdb_residue_id": "score_pdb_residue_id",
    "aa": "score_aa",
    "pdb_resname": "score_pdb_resname",
}

SUMMARY_METADATA_COLUMNS = (
    "completed",
    "planned_output_dir",
    "ifrag_template_interactions",
    "ifrag_fraction_nonzero",
    "conservation_status",
    "conservation_pairable_pairs",
    "paired_rows_used",
    "weak_msa_warning",
    "radi_summary_exists",
    "radi_interchain_pairs_retained",
    "radi_matrix_nonzero",
    "radi_matrix_max",
    "anchor_matrix_nonzero",
    "anchor_matrix_max",
    "q1_final_nonzero",
    "q1_final_max",
    "q2_final_nonzero",
    "q2_final_max",
    "strict_restraint_lines",
    "loose_restraint_lines",
    "evidence_class",
)

BRANCH_IDENTITY_PAIRS = (
    ("score_aa", "aa"),
    ("score_pdb_chain", "pdb_chain"),
    ("score_pdb_residue_id", "pdb_residue_id"),
    ("score_pdb_resname", "pdb_resname"),
    ("score_pdb_residue_label", "pdb_residue_label"),
)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Merge BM5 Phase 1 native interface labels with existing iFragDI "
            "residue-level feature outputs."
        )
    )
    parser.add_argument(
        "--labels-tsv",
        default="benchmark/labels/bm5_phase1_native_interface.labels.tsv",
        help="Native interface label TSV. Defaults to the Phase 1 label file.",
    )
    parser.add_argument(
        "--phase1-summary-tsv",
        default="benchmark/logs/bm5_phase1_summary.tsv",
        help="Phase 1 per-case summary TSV.",
    )
    parser.add_argument(
        "--outputs-root",
        default="benchmark/bm5_ifragdi_runs",
        help="Directory containing per-chainpair iFragDI output directories.",
    )
    parser.add_argument(
        "--out-tsv",
        default="benchmark/labels/bm5_phase1_training_table.tsv",
        help="Output residue-level training table TSV.",
    )
    parser.add_argument(
        "--out-summary-json",
        default="benchmark/labels/bm5_phase1_training_table.summary.json",
        help="Output summary JSON.",
    )
    parser.set_defaults(project_root=project_root)
    return parser.parse_args()


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(2)


def resolve_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def read_tsv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.exists():
        fail(f"Required input file not found: {path}")
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            fail(f"TSV has no header: {path}")
        rows = [dict(row) for row in reader]
    return rows, list(reader.fieldnames)


def read_optional_tsv(path: Path) -> tuple[list[dict[str, str]], list[str]] | None:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            fail(f"TSV has no header: {path}")
        rows = [dict(row) for row in reader]
    return rows, list(reader.fieldnames)


def require_columns(fields: Iterable[str], required: Iterable[str], path: Path) -> None:
    present = set(fields)
    missing = [column for column in required if column not in present]
    if missing:
        fail(f"{path} is missing required columns: {', '.join(missing)}")


def row_key(row: dict[str, str]) -> tuple[str, str, str]:
    return tuple((row.get(column) or "").strip() for column in KEY_COLUMNS)  # type: ignore[return-value]


def check_duplicate_label_keys(rows: list[dict[str, str]]) -> dict[str, object]:
    counts = Counter(row_key(row) for row in rows)
    duplicates = [(key, count) for key, count in counts.items() if count > 1]
    if duplicates:
        examples = [
            {"chainpair_id": key[0], "query_side": key[1], "score_residue_index": key[2], "count": count}
            for key, count in duplicates[:20]
        ]
        fail(
            "Duplicate label key rows found for "
            f"{', '.join(KEY_COLUMNS)}. Examples: {json.dumps(examples, indent=2)}"
        )
    return {"status": "passed", "key_columns": list(KEY_COLUMNS), "duplicate_count": 0}


def validate_binary_targets(rows: list[dict[str, str]]) -> tuple[int, dict[str, dict[str, int]]]:
    missing_label_count = 0
    invalid_examples: list[dict[str, str]] = []
    counts: dict[str, dict[str, int]] = {}
    for target in TARGET_COLUMNS:
        target_counts = {"positive": 0, "negative": 0}
        for row in rows:
            value = (row.get(target) or "").strip()
            if value == "":
                missing_label_count += 1
                invalid_examples.append({**{column: row.get(column, "") for column in KEY_COLUMNS}, "target": target, "value": value})
                continue
            if value not in {"0", "1"}:
                invalid_examples.append({**{column: row.get(column, "") for column in KEY_COLUMNS}, "target": target, "value": value})
                continue
            if value == "1":
                target_counts["positive"] += 1
            else:
                target_counts["negative"] += 1
        counts[target] = target_counts
    if invalid_examples:
        fail(
            "Target columns must contain only binary 0/1 values. "
            f"Invalid or missing examples: {json.dumps(invalid_examples[:20], indent=2)}"
        )
    return missing_label_count, counts


def values_equal(left: str, right: str) -> bool:
    left_text = (left or "").strip()
    right_text = (right or "").strip()
    if left_text == right_text:
        return True
    if left_text == "" or right_text == "":
        return False
    try:
        return abs(float(left_text) - float(right_text)) <= 1e-9
    except ValueError:
        return False


def index_by_case_side(rows: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[((row.get("chainpair_id") or "").strip(), (row.get("query_side") or "").strip())].append(row)
    return dict(grouped)


def side_to_score_file_stem(query_side: str) -> str | None:
    if query_side == "query1":
        return "query1"
    if query_side == "query2":
        return "query2"
    return None


def verify_branch_scores(
    grouped_rows: dict[tuple[str, str], list[dict[str, str]]],
    outputs_root: Path,
) -> dict[str, object]:
    diagnostics: dict[str, object] = {
        "feature_source": "labels score_* columns; branch score files are used for verification only",
        "branch_score_files_checked": 0,
        "missing_branch_score_files": 0,
        "missing_branch_score_files_sample": [],
        "label_rows_verified_against_branch": 0,
        "label_rows_missing_in_branch": 0,
        "label_rows_missing_in_branch_sample": [],
        "extra_branch_rows_not_in_labels": 0,
        "extra_branch_rows_sample": [],
        "branch_duplicate_residue_indices": 0,
        "branch_duplicate_residue_indices_sample": [],
        "branch_value_mismatches": 0,
        "branch_value_mismatches_sample": [],
    }
    missing_files: list[str] = []
    missing_rows: list[dict[str, str]] = []
    extra_rows: list[dict[str, str]] = []
    duplicate_branch_keys: list[dict[str, str]] = []
    mismatches: list[dict[str, str]] = []

    for (chainpair_id, query_side), label_rows in sorted(grouped_rows.items()):
        stem = side_to_score_file_stem(query_side)
        if stem is None:
            missing_files.append(f"{chainpair_id}/{query_side}_branch_scores.tsv")
            continue
        branch_path = outputs_root / chainpair_id / f"{stem}_branch_scores.tsv"
        loaded = read_optional_tsv(branch_path)
        if loaded is None:
            missing_files.append(str(branch_path))
            continue
        diagnostics["branch_score_files_checked"] = int(diagnostics["branch_score_files_checked"]) + 1
        branch_rows, branch_fields = loaded
        require_columns(branch_fields, ("residue_index",), branch_path)
        branch_counts = Counter((row.get("residue_index") or "").strip() for row in branch_rows)
        for residue_index, count in branch_counts.items():
            if count > 1:
                duplicate_branch_keys.append(
                    {"chainpair_id": chainpair_id, "query_side": query_side, "residue_index": residue_index, "count": str(count)}
                )
        branch_by_index = {
            (row.get("residue_index") or "").strip(): row
            for row in branch_rows
            if branch_counts[(row.get("residue_index") or "").strip()] == 1
        }
        label_indices = {(row.get("score_residue_index") or "").strip() for row in label_rows}
        for row in label_rows:
            residue_index = (row.get("score_residue_index") or "").strip()
            branch_row = branch_by_index.get(residue_index)
            if branch_row is None:
                missing_rows.append(
                    {"chainpair_id": chainpair_id, "query_side": query_side, "score_residue_index": residue_index}
                )
                continue
            diagnostics["label_rows_verified_against_branch"] = int(diagnostics["label_rows_verified_against_branch"]) + 1
            for label_column, branch_column in BRANCH_IDENTITY_PAIRS:
                if label_column in row and branch_column in branch_row and not values_equal(row[label_column], branch_row[branch_column]):
                    mismatches.append(
                        {
                            "chainpair_id": chainpair_id,
                            "query_side": query_side,
                            "score_residue_index": residue_index,
                            "column": label_column,
                            "label_value": row.get(label_column, ""),
                            "branch_value": branch_row.get(branch_column, ""),
                        }
                    )
            for feature in FEATURE_COLUMNS:
                label_column = f"score_{feature}"
                if label_column in row and feature in branch_row and not values_equal(row[label_column], branch_row[feature]):
                    mismatches.append(
                        {
                            "chainpair_id": chainpair_id,
                            "query_side": query_side,
                            "score_residue_index": residue_index,
                            "column": label_column,
                            "label_value": row.get(label_column, ""),
                            "branch_value": branch_row.get(feature, ""),
                        }
                    )
        for residue_index in sorted(set(branch_by_index) - label_indices):
            extra_rows.append({"chainpair_id": chainpair_id, "query_side": query_side, "residue_index": residue_index})

    diagnostics["missing_branch_score_files"] = len(missing_files)
    diagnostics["missing_branch_score_files_sample"] = missing_files[:50]
    diagnostics["label_rows_missing_in_branch"] = len(missing_rows)
    diagnostics["label_rows_missing_in_branch_sample"] = missing_rows[:50]
    diagnostics["extra_branch_rows_not_in_labels"] = len(extra_rows)
    diagnostics["extra_branch_rows_sample"] = extra_rows[:50]
    diagnostics["branch_duplicate_residue_indices"] = len(duplicate_branch_keys)
    diagnostics["branch_duplicate_residue_indices_sample"] = duplicate_branch_keys[:50]
    diagnostics["branch_value_mismatches"] = len(mismatches)
    diagnostics["branch_value_mismatches_sample"] = mismatches[:50]
    if int(diagnostics["branch_score_files_checked"]) == 0:
        fail(f"No branch score files were found under --outputs-root {outputs_root}")
    return diagnostics


def merge_optional_residue_scores(
    grouped_rows: dict[tuple[str, str], list[dict[str, str]]],
    outputs_root: Path,
) -> dict[str, object]:
    diagnostics: dict[str, object] = {
        "residue_score_files_checked": 0,
        "missing_residue_score_files": 0,
        "missing_residue_score_files_sample": [],
        "label_rows_matched_to_residue_scores": 0,
        "label_rows_missing_residue_scores": 0,
        "label_rows_missing_residue_scores_sample": [],
        "extra_residue_score_rows_not_in_labels": 0,
        "extra_residue_score_rows_sample": [],
        "residue_score_duplicate_indices": 0,
        "residue_score_duplicate_indices_sample": [],
    }
    missing_files: list[str] = []
    missing_rows: list[dict[str, str]] = []
    extra_rows: list[dict[str, str]] = []
    duplicate_keys: list[dict[str, str]] = []

    for (chainpair_id, query_side), label_rows in sorted(grouped_rows.items()):
        stem = side_to_score_file_stem(query_side)
        if stem is None:
            missing_files.append(f"{chainpair_id}/{query_side}_residue_scores.tsv")
            continue
        score_path = outputs_root / chainpair_id / f"{stem}_residue_scores.tsv"
        loaded = read_optional_tsv(score_path)
        if loaded is None:
            missing_files.append(str(score_path))
            continue
        diagnostics["residue_score_files_checked"] = int(diagnostics["residue_score_files_checked"]) + 1
        score_rows, score_fields = loaded
        require_columns(score_fields, ("residue_index", "residue_score"), score_path)
        score_counts = Counter((row.get("residue_index") or "").strip() for row in score_rows)
        for residue_index, count in score_counts.items():
            if count > 1:
                duplicate_keys.append(
                    {"chainpair_id": chainpair_id, "query_side": query_side, "residue_index": residue_index, "count": str(count)}
                )
        scores_by_index = {
            (row.get("residue_index") or "").strip(): row
            for row in score_rows
            if score_counts[(row.get("residue_index") or "").strip()] == 1
        }
        label_indices = {(row.get("score_residue_index") or "").strip() for row in label_rows}
        for row in label_rows:
            residue_index = (row.get("score_residue_index") or "").strip()
            score_row = scores_by_index.get(residue_index)
            if score_row is None:
                missing_rows.append(
                    {"chainpair_id": chainpair_id, "query_side": query_side, "score_residue_index": residue_index}
                )
                row["residue_score"] = ""
                continue
            row["residue_score"] = score_row.get("residue_score", "")
            diagnostics["label_rows_matched_to_residue_scores"] = int(diagnostics["label_rows_matched_to_residue_scores"]) + 1
        for residue_index in sorted(set(scores_by_index) - label_indices):
            extra_rows.append({"chainpair_id": chainpair_id, "query_side": query_side, "residue_index": residue_index})

    diagnostics["missing_residue_score_files"] = len(missing_files)
    diagnostics["missing_residue_score_files_sample"] = missing_files[:50]
    diagnostics["label_rows_missing_residue_scores"] = len(missing_rows)
    diagnostics["label_rows_missing_residue_scores_sample"] = missing_rows[:50]
    diagnostics["extra_residue_score_rows_not_in_labels"] = len(extra_rows)
    diagnostics["extra_residue_score_rows_sample"] = extra_rows[:50]
    diagnostics["residue_score_duplicate_indices"] = len(duplicate_keys)
    diagnostics["residue_score_duplicate_indices_sample"] = duplicate_keys[:50]
    return diagnostics


def build_summary_by_case(summary_rows: list[dict[str, str]]) -> tuple[dict[str, dict[str, str]], dict[str, object]]:
    by_case: dict[str, dict[str, str]] = {}
    duplicates: list[str] = []
    for row in summary_rows:
        chainpair_id = (row.get("chainpair_id") or "").strip()
        if not chainpair_id:
            continue
        if chainpair_id in by_case:
            duplicates.append(chainpair_id)
            continue
        by_case[chainpair_id] = row
    diagnostics = {
        "summary_rows": len(summary_rows),
        "summary_cases": len(by_case),
        "duplicate_chainpair_ids": sorted(set(duplicates)),
    }
    return by_case, diagnostics


def add_backbone_columns(
    rows: list[dict[str, str]],
    summary_by_case: dict[str, dict[str, str]],
    summary_fields: list[str],
) -> tuple[list[str], dict[str, object]]:
    missing_summary_cases: set[str] = set()
    feature_columns_missing_in_labels = [f"score_{feature}" for feature in FEATURE_COLUMNS if f"score_{feature}" not in rows[0]]
    metadata_fields = [
        field
        for field in SUMMARY_METADATA_COLUMNS
        if field in summary_fields and field not in rows[0] and field != "chainpair_id"
    ]

    for row in rows:
        chainpair_id = (row.get("chainpair_id") or "").strip()
        row["case_id"] = chainpair_id
        for alias, source in IDENTITY_ALIASES.items():
            row[alias] = row.get(source, "")
        for feature in FEATURE_COLUMNS:
            source = f"score_{feature}"
            row[feature] = row.get(source, "")
        row.setdefault("residue_score", "")
        summary_row = summary_by_case.get(chainpair_id)
        if summary_row is None:
            missing_summary_cases.add(chainpair_id)
            for field in metadata_fields:
                row[field] = ""
        else:
            for field in metadata_fields:
                row[field] = summary_row.get(field, "")

    primary_fields = [
        "chainpair_id",
        "case_id",
        "query_side",
        "query_role",
        "score_residue_index",
        "pdb_chain",
        "pdb_residue_id",
        "aa",
        "pdb_resname",
    ]
    original_fields = list(rows[0])
    output_fields: list[str] = []
    for field in [*primary_fields, *original_fields, *FEATURE_COLUMNS, "residue_score", *metadata_fields]:
        if field not in output_fields:
            output_fields.append(field)

    diagnostics = {
        "summary_metadata_columns_added": metadata_fields,
        "missing_summary_cases": sorted(missing_summary_cases),
        "feature_columns_missing_in_labels": feature_columns_missing_in_labels,
    }
    return output_fields, diagnostics


def zero_positive_5a_diagnostics(rows: list[dict[str, str]]) -> dict[str, object]:
    grouped: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        chainpair_id = row.get("chainpair_id", "")
        query_side = row.get("query_side", "")
        if row.get(PRIMARY_TARGET, "").strip() == "1":
            grouped[chainpair_id][query_side] += 1
        else:
            grouped[chainpair_id].setdefault(query_side, 0)

    zero_sides: list[dict[str, object]] = []
    both_side_cases: list[str] = []
    for chainpair_id, side_counts in sorted(grouped.items()):
        zero_side_names = sorted(side for side, count in side_counts.items() if count == 0)
        if zero_side_names:
            zero_sides.append(
                {
                    "chainpair_id": chainpair_id,
                    "zero_positive_sides": zero_side_names,
                    "side_positive_counts": dict(sorted(side_counts.items())),
                }
            )
        if {"query1", "query2"}.issubset(side_counts) and side_counts["query1"] == 0 and side_counts["query2"] == 0:
            both_side_cases.append(chainpair_id)

    expected = set(EXPECTED_ZERO_5A_BOTH_SIDE_CASES)
    observed = set(both_side_cases)
    return {
        "zero_5A_positive_either_side": zero_sides,
        "zero_5A_positive_either_side_cases": [entry["chainpair_id"] for entry in zero_sides],
        "zero_5A_positive_both_sides_cases": both_side_cases,
        "expected_zero_5A_positive_both_sides_cases": list(EXPECTED_ZERO_5A_BOTH_SIDE_CASES),
        "expected_zero_5A_cases_observed": sorted(expected & observed),
        "expected_zero_5A_cases_missing": sorted(expected - observed),
        "unexpected_zero_5A_both_sides_cases": sorted(observed - expected),
    }


def count_missing_final_score(rows: list[dict[str, str]]) -> int:
    return sum(1 for row in rows if (row.get("final_score") or "").strip() == "")


def write_tsv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def print_counter(title: str, counter: Counter[str]) -> None:
    print(title)
    if not counter:
        print("  none")
        return
    for key, count in sorted(counter.items()):
        label = key if key != "" else "<missing>"
        print(f"  {label}: {count}")


def print_validation(summary: dict[str, object]) -> None:
    print("BM5 Phase 1 training table written")
    print(f"  scored residue rows: {summary['scored_residue_rows']}")
    print(f"  label rows: {summary['label_row_count']}")
    print(f"  merged rows: {summary['merged_row_count']}")
    print(f"  represented cases: {summary['represented_case_count']}")
    target_counts = summary["target_counts"]
    assert isinstance(target_counts, dict)
    for target in TARGET_COLUMNS:
        counts = target_counts[target]
        assert isinstance(counts, dict)
        print(f"  {target}: positive={counts['positive']} negative={counts['negative']}")
    print(f"  rows with missing labels: {summary['missing_label_count']}")
    print(f"  rows with missing final_score: {summary['missing_final_score_count']}")
    print_counter("counts by evidence_class:", Counter(summary["evidence_class_counts"]))  # type: ignore[arg-type]
    print_counter("counts by query_side:", Counter(summary["query_side_counts"]))  # type: ignore[arg-type]
    zero_diag = summary["zero_5A_positive_cases"]
    assert isinstance(zero_diag, dict)
    either_side = zero_diag["zero_5A_positive_either_side"]
    assert isinstance(either_side, list)
    print("cases with zero positives for interface_5A on either side:")
    if not either_side:
        print("  none")
    else:
        for entry in either_side:
            assert isinstance(entry, dict)
            sides = ",".join(entry["zero_positive_sides"])  # type: ignore[index]
            counts = entry["side_positive_counts"]
            print(f"  {entry['chainpair_id']}: zero_sides={sides} counts={counts}")


def main() -> int:
    args = parse_args()
    project_root: Path = args.project_root
    labels_path = resolve_path(project_root, args.labels_tsv)
    phase1_summary_path = resolve_path(project_root, args.phase1_summary_tsv)
    outputs_root = resolve_path(project_root, args.outputs_root)
    out_tsv = resolve_path(project_root, args.out_tsv)
    out_summary_json = resolve_path(project_root, args.out_summary_json)

    labels, label_fields = read_tsv(labels_path)
    summary_rows, summary_fields = read_tsv(phase1_summary_path)
    if not labels:
        fail(f"Label file contains no rows: {labels_path}")
    if not summary_rows:
        fail(f"Phase 1 summary contains no rows: {phase1_summary_path}")
    require_columns(label_fields, (*KEY_COLUMNS, "query_role", *TARGET_COLUMNS), labels_path)
    require_columns(summary_fields, ("chainpair_id",), phase1_summary_path)
    if not outputs_root.exists():
        fail(f"--outputs-root not found: {outputs_root}")

    duplicate_key_check = check_duplicate_label_keys(labels)
    missing_label_count, target_counts = validate_binary_targets(labels)
    summary_by_case, summary_table_diagnostics = build_summary_by_case(summary_rows)
    grouped_rows = index_by_case_side(labels)
    branch_diagnostics = verify_branch_scores(grouped_rows, outputs_root)
    residue_score_diagnostics = merge_optional_residue_scores(grouped_rows, outputs_root)
    output_fields, merge_diagnostics = add_backbone_columns(labels, summary_by_case, summary_fields)

    query_side_counts = Counter(row.get("query_side", "") for row in labels)
    evidence_class_counts = Counter(row.get("evidence_class", "") for row in labels)
    zero_5a_diagnostics = zero_positive_5a_diagnostics(labels)
    missing_final_score_count = count_missing_final_score(labels)

    summary: dict[str, object] = {
        "input_paths": {
            "labels_tsv": str(labels_path),
            "phase1_summary_tsv": str(phase1_summary_path),
            "outputs_root": str(outputs_root),
        },
        "output_path": str(out_tsv),
        "summary_json_path": str(out_summary_json),
        "label_row_count": len(labels),
        "scored_residue_rows": len(labels),
        "merged_row_count": len(labels),
        "represented_case_count": len({row.get("chainpair_id", "") for row in labels}),
        "query_side_counts": dict(sorted(query_side_counts.items())),
        "evidence_class_counts": dict(sorted(evidence_class_counts.items())),
        "primary_target": PRIMARY_TARGET,
        "secondary_targets": list(SECONDARY_TARGETS),
        "target_counts": target_counts,
        "missing_label_count": missing_label_count,
        "missing_final_score_count": missing_final_score_count,
        "duplicate_key_check": duplicate_key_check,
        "branch_score_verification": branch_diagnostics,
        "optional_residue_score_availability": residue_score_diagnostics,
        "summary_table_diagnostics": summary_table_diagnostics,
        "merge_diagnostics": merge_diagnostics,
        "zero_5A_positive_cases": zero_5a_diagnostics,
        "notes": [
            "Labels TSV is the row backbone.",
            "case_id is copied from chainpair_id.",
            "score_* feature columns in the labels file are copied to unprefixed feature columns.",
            "Branch score files are used for verification only; their columns are not duplicated.",
            "Missing optional residue-score files are warnings, not fatal errors.",
            "All-zero/no-evidence cases are retained.",
        ],
    }

    write_tsv(out_tsv, labels, output_fields)
    write_json(out_summary_json, summary)
    print_validation(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
