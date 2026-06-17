#!/usr/bin/env python3
"""Evaluate BM5 Phase 1 residue-level scores before ML training.

This script reads the already-built training table and computes classical
ranking metrics. It does not rerun Phase 1, start docking comparisons, or train
an ML model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


PRIMARY_TARGET = "interface_5A"
SECONDARY_TARGETS = ("interface_3p9A", "interface_8A")
TARGET_COLUMNS = ("interface_5A", "interface_3p9A", "interface_8A")
PRIMARY_SCORE = "final_score"
SCORE_COLUMNS = (
    "final_score",
    "patch_score",
    "ifrag_component",
    "conservation_component",
    "radi_component",
    "ifrag_strength",
    "ifrag_specificity",
    "conservation_strength",
    "radi_anchor",
)
BLASTPDB_COLUMNS = ("blastpdb_anchor", "blastpdb_component")
GROUP_COLUMNS = ("chainpair_id", "query_side")

TOP_K_SPECS = (
    ("top1", "fixed", 1),
    ("top5", "fixed", 5),
    ("top10", "fixed", 10),
    ("top20", "fixed", 20),
    ("top_L10", "fraction", 10),
    ("top_L5", "fraction", 5),
)

OUTPUT_COLUMNS = (
    "target",
    "score_column",
    "aggregation_level",
    "aggregation_value",
    "row_count",
    "positive_count",
    "negative_count",
    "group_count",
    "groups_with_positive",
    "zero_positive_groups",
    "constant_score_groups",
    "missing_score_groups",
    "evaluated_groups",
    "roc_auc",
    "auprc",
    "top1_precision",
    "top1_recall",
    "top1_enrichment",
    "top5_precision",
    "top5_recall",
    "top5_enrichment",
    "top10_precision",
    "top10_recall",
    "top10_enrichment",
    "top20_precision",
    "top20_recall",
    "top20_enrichment",
    "top_L10_precision",
    "top_L10_recall",
    "top_L10_enrichment",
    "top_L5_precision",
    "top_L5_recall",
    "top_L5_enrichment",
)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Evaluate BM5 Phase 1 residue-score recovery from the merged training table."
    )
    parser.add_argument(
        "--training-table-tsv",
        default="benchmark/labels/bm5_phase1_training_table.tsv",
        help="Merged Phase 1 residue-level training table.",
    )
    parser.add_argument(
        "--out-tsv",
        default="benchmark/labels/bm5_phase1_residue_score_eval.tsv",
        help="Output evaluation TSV.",
    )
    parser.add_argument(
        "--out-summary-json",
        default="benchmark/labels/bm5_phase1_residue_score_eval.summary.json",
        help="Output evaluation summary JSON.",
    )
    parser.set_defaults(project_root=project_root)
    return parser.parse_args()


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(2)


def resolve_path(project_root: Path, path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else project_root / path


def read_tsv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.exists():
        fail(f"Required input file not found: {path}")
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            fail(f"TSV has no header: {path}")
        rows = [dict(row) for row in reader]
    if not rows:
        fail(f"Input table contains no rows: {path}")
    return rows, list(reader.fieldnames)


def require_columns(fields: Iterable[str], required: Iterable[str], path: Path) -> None:
    present = set(fields)
    missing = [column for column in required if column not in present]
    if missing:
        fail(f"{path} is missing required columns: {', '.join(missing)}")


def parse_float(value: str | None) -> float | None:
    text = (value or "").strip()
    if text == "":
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def parse_binary(value: str | None) -> int | None:
    text = (value or "").strip()
    if text == "0":
        return 0
    if text == "1":
        return 1
    return None


def parse_residue_index(row: dict[str, str]) -> int:
    value = (row.get("score_residue_index") or row.get("residue_index") or "").strip()
    try:
        return int(value)
    except ValueError:
        return 10**12


def validate_targets(rows: list[dict[str, str]], path: Path) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    invalid_examples: list[dict[str, str]] = []
    for target in TARGET_COLUMNS:
        positive = 0
        negative = 0
        for row in rows:
            value = parse_binary(row.get(target))
            if value is None:
                invalid_examples.append(
                    {
                        "chainpair_id": row.get("chainpair_id", ""),
                        "query_side": row.get("query_side", ""),
                        "score_residue_index": row.get("score_residue_index", ""),
                        "target": target,
                        "value": row.get(target, ""),
                    }
                )
                continue
            if value == 1:
                positive += 1
            else:
                negative += 1
        counts[target] = {"positive": positive, "negative": negative}
    if invalid_examples:
        fail(
            f"{path} contains non-binary or missing target values. "
            f"Examples: {json.dumps(invalid_examples[:20], indent=2)}"
        )
    return counts


def present_score_columns(fields: list[str]) -> list[str]:
    present = [column for column in SCORE_COLUMNS if column in fields]
    if PRIMARY_SCORE not in present:
        fail(f"Primary score column not found in training table: {PRIMARY_SCORE}")
    return present


def nonzero_counts(rows: list[dict[str, str]], columns: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for column in columns:
        count = 0
        for row in rows:
            value = parse_float(row.get(column))
            if value is not None and value != 0.0:
                count += 1
        counts[column] = count
    return counts


def group_key(row: dict[str, str]) -> tuple[str, str]:
    return ((row.get("chainpair_id") or "").strip(), (row.get("query_side") or "").strip())


def rows_by_group(rows: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[group_key(row)].append(row)
    return dict(grouped)


def roc_auc_rank_based(labels_and_scores: list[tuple[int, float]]) -> float | None:
    positive_count = sum(1 for label, _score in labels_and_scores if label == 1)
    negative_count = len(labels_and_scores) - positive_count
    if positive_count == 0 or negative_count == 0:
        return None

    ordered = sorted(labels_and_scores, key=lambda item: item[1])
    rank_sum_positive = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        score = ordered[index][1]
        while end < len(ordered) and ordered[end][1] == score:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        positives_in_tie = sum(1 for label, _score in ordered[index:end] if label == 1)
        rank_sum_positive += average_rank * positives_in_tie
        index = end

    numerator = rank_sum_positive - (positive_count * (positive_count + 1) / 2.0)
    return numerator / float(positive_count * negative_count)


def average_precision_conservative(labels_and_scores: list[tuple[int, float]]) -> float | None:
    positive_count = sum(1 for label, _score in labels_and_scores if label == 1)
    if positive_count == 0:
        return None

    # Conservative tie behavior: for identical scores, negatives are ranked
    # before positives, lowering AP when the score cannot distinguish them.
    ordered = sorted(labels_and_scores, key=lambda item: (-item[1], item[0]))
    hits = 0
    precision_sum = 0.0
    for rank, (label, _score) in enumerate(ordered, start=1):
        if label != 1:
            continue
        hits += 1
        precision_sum += hits / float(rank)
    return precision_sum / float(positive_count)


def fixed_or_fraction_k(label: str, mode: str, value: int, group_length: int) -> int:
    if mode == "fixed":
        return max(1, min(value, group_length))
    if label == "top_L10":
        return max(1, math.ceil(group_length / float(value)))
    if label == "top_L5":
        return max(1, math.ceil(group_length / float(value)))
    raise ValueError(f"Unsupported top-k spec: {label}")


def initialize_top_sums() -> dict[str, dict[str, float]]:
    return {label: {"precision": 0.0, "recall": 0.0, "enrichment": 0.0} for label, _mode, _value in TOP_K_SPECS}


def top_k_metrics_for_groups(
    grouped: dict[tuple[str, str], list[dict[str, str]]],
    *,
    target: str,
    score_column: str,
) -> dict[str, object]:
    group_count = len(grouped)
    zero_positive_groups = 0
    constant_score_groups = 0
    missing_score_groups = 0
    evaluated_groups = 0
    groups_with_positive = 0
    top_sums = initialize_top_sums()

    for _key, group_rows in grouped.items():
        positive_count = sum(1 for row in group_rows if parse_binary(row.get(target)) == 1)
        if positive_count == 0:
            zero_positive_groups += 1
            continue
        groups_with_positive += 1

        parsed_rows: list[tuple[float, int, int]] = []
        has_missing_score = False
        for row_index, row in enumerate(group_rows):
            score = parse_float(row.get(score_column))
            if score is None:
                has_missing_score = True
                break
            parsed_rows.append((score, parse_residue_index(row), row_index))
        if has_missing_score or not parsed_rows:
            missing_score_groups += 1
            continue

        evaluated_groups += 1
        unique_scores = {score for score, _residue_index, _row_index in parsed_rows}
        if len(unique_scores) <= 1:
            constant_score_groups += 1

        ordered_indices = [
            row_index
            for _score, _residue_index, row_index in sorted(
                parsed_rows,
                key=lambda item: (-item[0], item[1], item[2]),
            )
        ]
        group_length = len(group_rows)
        positive_fraction = positive_count / float(group_length)

        for label, mode, value in TOP_K_SPECS:
            k = fixed_or_fraction_k(label, mode, value, group_length)
            selected = ordered_indices[:k]
            true_positive_at_k = sum(1 for row_index in selected if parse_binary(group_rows[row_index].get(target)) == 1)
            precision = true_positive_at_k / float(k)
            recall = true_positive_at_k / float(positive_count)
            enrichment = precision / positive_fraction if positive_fraction > 0 else None
            top_sums[label]["precision"] += precision
            top_sums[label]["recall"] += recall
            top_sums[label]["enrichment"] += enrichment if enrichment is not None else 0.0

    averaged: dict[str, float | None] = {}
    denominator = float(evaluated_groups) if evaluated_groups > 0 else None
    for label in top_sums:
        for metric_name, value in top_sums[label].items():
            averaged[f"{label}_{metric_name}"] = value / denominator if denominator else None

    return {
        "group_count": group_count,
        "groups_with_positive": groups_with_positive,
        "zero_positive_groups": zero_positive_groups,
        "constant_score_groups": constant_score_groups,
        "missing_score_groups": missing_score_groups,
        "evaluated_groups": evaluated_groups,
        **averaged,
    }


def row_level_metrics(rows: list[dict[str, str]], *, target: str, score_column: str) -> dict[str, object]:
    labels_and_scores: list[tuple[int, float]] = []
    positive_count = 0
    negative_count = 0
    for row in rows:
        label = parse_binary(row.get(target))
        if label is None:
            continue
        if label == 1:
            positive_count += 1
        else:
            negative_count += 1
        score = parse_float(row.get(score_column))
        if score is not None:
            labels_and_scores.append((label, score))

    return {
        "row_count": len(rows),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "roc_auc": roc_auc_rank_based(labels_and_scores),
        "auprc": average_precision_conservative(labels_and_scores),
    }


def format_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def make_eval_row(
    rows: list[dict[str, str]],
    *,
    target: str,
    score_column: str,
    aggregation_level: str,
    aggregation_value: str,
) -> dict[str, str]:
    row_metrics = row_level_metrics(rows, target=target, score_column=score_column)
    group_metrics = top_k_metrics_for_groups(rows_by_group(rows), target=target, score_column=score_column)
    payload: dict[str, object] = {
        "target": target,
        "score_column": score_column,
        "aggregation_level": aggregation_level,
        "aggregation_value": aggregation_value,
        **row_metrics,
        **group_metrics,
    }
    return {column: format_value(payload.get(column)) for column in OUTPUT_COLUMNS}


def aggregation_subsets(rows: list[dict[str, str]]) -> list[tuple[str, str, list[dict[str, str]]]]:
    subsets: list[tuple[str, str, list[dict[str, str]]]] = [("all", "all", rows)]
    evidence_values = sorted({(row.get("evidence_class") or "").strip() for row in rows})
    for evidence_class in evidence_values:
        subsets.append(
            (
                "evidence_class",
                evidence_class if evidence_class else "<missing>",
                [row for row in rows if (row.get("evidence_class") or "").strip() == evidence_class],
            )
        )
    query_sides = sorted({(row.get("query_side") or "").strip() for row in rows})
    for query_side in query_sides:
        subsets.append(
            (
                "query_side",
                query_side if query_side else "<missing>",
                [row for row in rows if (row.get("query_side") or "").strip() == query_side],
            )
        )
    return subsets


def build_evaluation_rows(rows: list[dict[str, str]], score_columns: list[str]) -> list[dict[str, str]]:
    output_rows: list[dict[str, str]] = []
    for target in TARGET_COLUMNS:
        for score_column in score_columns:
            for aggregation_level, aggregation_value, subset_rows in aggregation_subsets(rows):
                output_rows.append(
                    make_eval_row(
                        subset_rows,
                        target=target,
                        score_column=score_column,
                        aggregation_level=aggregation_level,
                        aggregation_value=aggregation_value,
                    )
                )
    return output_rows


def write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=list(OUTPUT_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def metric_as_float(row: dict[str, str], column: str) -> float | None:
    return parse_float(row.get(column))


def best_methods(
    eval_rows: list[dict[str, str]],
    *,
    target: str,
    metric: str,
    limit: int = 10,
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for row in eval_rows:
        if row.get("target") != target:
            continue
        if row.get("aggregation_level") != "all" or row.get("aggregation_value") != "all":
            continue
        value = metric_as_float(row, metric)
        if value is None:
            continue
        candidates.append(
            {
                "score_column": row.get("score_column", ""),
                metric: value,
                "roc_auc": metric_as_float(row, "roc_auc"),
                "auprc": metric_as_float(row, "auprc"),
                "top_L10_recall": metric_as_float(row, "top_L10_recall"),
                "evaluated_groups": int(row.get("evaluated_groups") or 0),
                "constant_score_groups": int(row.get("constant_score_groups") or 0),
            }
        )
    candidates.sort(key=lambda item: (-(item.get(metric) or 0.0), str(item.get("score_column", ""))))
    return candidates[:limit]


def diagnostic_group_counts(eval_rows: list[dict[str, str]]) -> dict[str, dict[str, int]]:
    diagnostics: dict[str, dict[str, int]] = {}
    for row in eval_rows:
        if row.get("aggregation_level") != "all" or row.get("aggregation_value") != "all":
            continue
        key = f"{row.get('target')}::{row.get('score_column')}"
        diagnostics[key] = {
            "group_count": int(row.get("group_count") or 0),
            "groups_with_positive": int(row.get("groups_with_positive") or 0),
            "zero_positive_groups": int(row.get("zero_positive_groups") or 0),
            "constant_score_groups": int(row.get("constant_score_groups") or 0),
            "missing_score_groups": int(row.get("missing_score_groups") or 0),
            "evaluated_groups": int(row.get("evaluated_groups") or 0),
        }
    return diagnostics


def target_counts(rows: list[dict[str, str]]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for target in TARGET_COLUMNS:
        positive = sum(1 for row in rows if parse_binary(row.get(target)) == 1)
        negative = sum(1 for row in rows if parse_binary(row.get(target)) == 0)
        counts[target] = {"positive": positive, "negative": negative}
    return counts


def zero_positive_both_side_cases(rows: list[dict[str, str]], target: str = PRIMARY_TARGET) -> list[dict[str, object]]:
    side_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    seen_sides: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        chainpair_id = row.get("chainpair_id", "")
        query_side = row.get("query_side", "")
        seen_sides[chainpair_id].add(query_side)
        if parse_binary(row.get(target)) == 1:
            side_counts[chainpair_id][query_side] += 1
        else:
            side_counts[chainpair_id].setdefault(query_side, 0)

    cases: list[dict[str, object]] = []
    for chainpair_id in sorted(seen_sides):
        if not {"query1", "query2"}.issubset(seen_sides[chainpair_id]):
            continue
        if side_counts[chainpair_id]["query1"] == 0 and side_counts[chainpair_id]["query2"] == 0:
            cases.append(
                {
                    "chainpair_id": chainpair_id,
                    "target": target,
                    "side_positive_counts": dict(sorted(side_counts[chainpair_id].items())),
                }
            )
    return cases


def print_top_methods(title: str, methods: list[dict[str, object]], metric: str) -> None:
    print(title)
    if not methods:
        print("  none")
        return
    for entry in methods[:5]:
        value = entry.get(metric)
        text = "" if value is None else f"{float(value):.6g}"
        print(f"  {entry['score_column']}: {text}")


def print_validation(summary: dict[str, object]) -> None:
    print("BM5 Phase 1 residue-score evaluation written")
    print(f"  loaded rows: {summary['row_count']}")
    print(f"  represented chainpairs: {summary['represented_chainpair_count']}")
    print(f"  represented groups: {summary['represented_group_count']}")
    print("target positive counts:")
    target_summary = summary["target_counts"]
    assert isinstance(target_summary, dict)
    for target in TARGET_COLUMNS:
        counts = target_summary[target]
        assert isinstance(counts, dict)
        print(f"  {target}: positive={counts['positive']} negative={counts['negative']}")
    print("score nonzero counts:")
    score_counts = summary["score_column_nonzero_counts"]
    assert isinstance(score_counts, dict)
    for score_column, count in sorted(score_counts.items()):
        print(f"  {score_column}: {count}")
    print_top_methods(
        "top methods for interface_5A by AUPRC:",
        summary["best_methods_by_auprc_interface_5A"],  # type: ignore[arg-type]
        "auprc",
    )
    print_top_methods(
        "top methods for interface_5A by top_L10_recall:",
        summary["best_methods_by_top_L10_recall_interface_5A"],  # type: ignore[arg-type]
        "top_L10_recall",
    )


def main() -> int:
    args = parse_args()
    project_root: Path = args.project_root
    input_path = resolve_path(project_root, args.training_table_tsv)
    out_tsv = resolve_path(project_root, args.out_tsv)
    out_summary_json = resolve_path(project_root, args.out_summary_json)

    rows, fields = read_tsv(input_path)
    require_columns(fields, ("chainpair_id", "query_side", "score_residue_index", *TARGET_COLUMNS), input_path)
    validate_targets(rows, input_path)
    score_columns = present_score_columns(fields)
    blastpdb_present = [column for column in BLASTPDB_COLUMNS if column in fields]

    eval_rows = build_evaluation_rows(rows, score_columns)
    write_tsv(out_tsv, eval_rows)

    represented_chainpairs = {row.get("chainpair_id", "") for row in rows}
    represented_groups = {group_key(row) for row in rows}
    score_nonzero = nonzero_counts(rows, [*score_columns, *blastpdb_present])
    summary: dict[str, object] = {
        "input_path": str(input_path),
        "output_path": str(out_tsv),
        "summary_json_path": str(out_summary_json),
        "row_count": len(rows),
        "represented_chainpair_count": len(represented_chainpairs),
        "represented_group_count": len(represented_groups),
        "targets": {
            "primary": PRIMARY_TARGET,
            "secondary": list(SECONDARY_TARGETS),
        },
        "target_counts": target_counts(rows),
        "score_columns_evaluated": score_columns,
        "score_column_nonzero_counts": score_nonzero,
        "blastpdb_columns_reported_not_evaluated": blastpdb_present,
        "best_methods_by_auprc_interface_5A": best_methods(eval_rows, target=PRIMARY_TARGET, metric="auprc"),
        "best_methods_by_top_L10_recall_interface_5A": best_methods(
            eval_rows,
            target=PRIMARY_TARGET,
            metric="top_L10_recall",
        ),
        "skipped_diagnostic_group_counts": diagnostic_group_counts(eval_rows),
        "zero_positive_both_side_cases": zero_positive_both_side_cases(rows),
        "metric_notes": {
            "roc_auc": "Rank-based Mann-Whitney calculation with average ranks for tied scores.",
            "auprc": "Average precision after sorting descending score; tied scores use conservative ordering with negatives before positives.",
            "top_k": "Rows are sorted descending score with deterministic tie break by lower score_residue_index.",
            "zero_positive_groups": "Excluded from top-k recovery averages and counted in diagnostics.",
            "constant_score_groups": "Included in top-k diagnostics using deterministic residue-index ordering.",
        },
    }
    write_json(out_summary_json, summary)
    print_validation(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
