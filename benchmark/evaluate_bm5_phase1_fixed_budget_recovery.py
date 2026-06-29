#!/usr/bin/env python3
"""
Evaluate BM5 Phase 1 residue scores with fixed-budget interface recovery metrics.

This script answers the docking-oriented question:

    If each method is allowed to nominate the same number of residues, which
    method selects more true interface residues and fewer false positives?

It complements global ranking metrics such as AUPRC/ROC-AUC with fixed-budget
precision/recall/F1 at budgets such as top 15 residues, top 5% of residues, and
top N where N is the native interface size.

Default inputs:
    benchmark/labels/bm5_phase1_training_table.tsv
    benchmark/labels/bm5_phase1_patch_structure_features.tsv
    benchmark/labels/bm5_phase1_structure_ml_logreg.predictions.tsv
    benchmark/labels/bm5_phase1_structure_ablation_logreg.predictions.tsv

Default outputs:
    benchmark/labels/bm5_phase1_fixed_budget_recovery.tsv
    benchmark/labels/bm5_phase1_fixed_budget_recovery.per_group.tsv
    benchmark/labels/bm5_phase1_fixed_budget_recovery.interface_size_distribution.tsv
    benchmark/labels/bm5_phase1_fixed_budget_recovery.summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

KEY_COLUMNS = ["chainpair_id", "query_side", "score_residue_index"]
DEFAULT_BASE_METHODS = [
    "final_score",
    "patch_score",
    "ifrag_component",
    "ifrag_strength",
    "ifrag_specificity",
    "conservation_component",
    "conservation_strength",
    "radi_component",
    "radi_anchor",
    "blastpdb_anchor",
]
SURFACE_FLAG = "struct_surface_flag_rsa_ge_0p20"
RSA_COLUMN = "struct_rsa_rel"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate fixed-budget residue-level interface recovery for BM5 Phase 1."
    )
    parser.add_argument(
        "--training-table",
        default="benchmark/labels/bm5_phase1_training_table.tsv",
        help="Base training table with residue labels and original scores.",
    )
    parser.add_argument(
        "--structure-feature-table",
        default="benchmark/labels/bm5_phase1_patch_structure_features.tsv",
        help="Structure feature table containing RSA/surface flags.",
    )
    parser.add_argument(
        "--structure-predictions",
        default="benchmark/labels/bm5_phase1_structure_ml_logreg.predictions.tsv",
        help="Structure ML prediction table. Ignored if missing unless --require-predictions is set.",
    )
    parser.add_argument(
        "--ablation-predictions",
        default="benchmark/labels/bm5_phase1_structure_ablation_logreg.predictions.tsv",
        help="Structure ablation prediction table. Ignored if missing unless --require-predictions is set.",
    )
    parser.add_argument(
        "--out-prefix",
        default="benchmark/labels/bm5_phase1_fixed_budget_recovery",
        help="Output prefix.",
    )
    parser.add_argument("--target", default="interface_5A")
    parser.add_argument(
        "--rsa-surface-threshold",
        type=float,
        default=0.20,
        help="Fallback RSA threshold for surface-only evaluation if surface flag is missing.",
    )
    parser.add_argument(
        "--fixed-budgets",
        default="5,10,15,20",
        help="Comma-separated fixed residue-count budgets.",
    )
    parser.add_argument(
        "--percent-budgets",
        default="3,5,10",
        help="Comma-separated percentage budgets, interpreted as percent of candidate residues.",
    )
    parser.add_argument(
        "--require-predictions",
        action="store_true",
        help="Fail if prediction files are missing instead of ignoring them.",
    )
    return parser.parse_args()


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(2)


def read_table(path: Path, required: bool = True) -> Optional[pd.DataFrame]:
    if not path.exists():
        if required:
            fail(f"Missing required input file: {path}")
        return None
    return pd.read_csv(path, sep="\t", low_memory=False)


def parse_int_list(value: str) -> List[int]:
    out: List[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        parsed = int(item)
        if parsed <= 0:
            fail(f"Budget values must be positive integers: {value}")
        out.append(parsed)
    if not out:
        fail(f"No valid integer budgets parsed from: {value}")
    return out


def parse_float_list(value: str) -> List[float]:
    out: List[float] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        parsed = float(item)
        if parsed <= 0:
            fail(f"Percentage budget values must be positive: {value}")
        out.append(parsed)
    if not out:
        fail(f"No valid percentage budgets parsed from: {value}")
    return out


def numeric(series: pd.Series, fill: Optional[float] = None) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    if fill is not None:
        out = out.fillna(fill)
    return out


def ensure_required_columns(df: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        fail(f"{label} is missing required columns: {missing}")


def deduplicate_score_columns(columns: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for col in columns:
        if col in seen:
            continue
        seen.add(col)
        out.append(col)
    return out


def merge_score_table(base: pd.DataFrame, other: Optional[pd.DataFrame], source_label: str) -> Tuple[pd.DataFrame, List[str]]:
    if other is None:
        return base, []
    ensure_required_columns(other, KEY_COLUMNS, source_label)
    score_cols = [c for c in other.columns if c.startswith("score__")]
    if not score_cols:
        return base, []
    keep = KEY_COLUMNS + score_cols
    sub = other[keep].copy()
    if sub.duplicated(KEY_COLUMNS).any():
        dup_n = int(sub.duplicated(KEY_COLUMNS).sum())
        fail(f"{source_label} has duplicate residue keys: {dup_n}")
    new_cols: List[str] = []
    rename: Dict[str, str] = {}
    for col in score_cols:
        if col not in base.columns:
            new_cols.append(col)
            continue
        # Duplicate baseline columns can appear in multiple prediction files. Keep the first.
        if col.startswith("score__baseline_"):
            continue
        new_name = f"{col}__from_{source_label}"
        while new_name in base.columns or new_name in sub.columns:
            new_name += "_dup"
        rename[col] = new_name
        new_cols.append(new_name)
    if rename:
        sub = sub.rename(columns=rename)
    merge_cols = KEY_COLUMNS + [c for c in new_cols if c in sub.columns]
    if len(merge_cols) == len(KEY_COLUMNS):
        return base, []
    merged = base.merge(sub[merge_cols], on=KEY_COLUMNS, how="left", validate="one_to_one")
    return merged, [c for c in merge_cols if c not in KEY_COLUMNS]


def merge_surface_features(base: pd.DataFrame, structure_features: Optional[pd.DataFrame], rsa_threshold: float) -> pd.DataFrame:
    if structure_features is None:
        base["surface_candidate"] = True
        base["surface_source"] = "missing_structure_table_all_residues"
        return base
    ensure_required_columns(structure_features, KEY_COLUMNS, "structure feature table")
    cols = [c for c in [SURFACE_FLAG, RSA_COLUMN, "struct_freesasa_found", "struct_dssp_found"] if c in structure_features.columns]
    if not cols:
        base["surface_candidate"] = True
        base["surface_source"] = "missing_surface_columns_all_residues"
        return base
    sub = structure_features[KEY_COLUMNS + cols].copy()
    if sub.duplicated(KEY_COLUMNS).any():
        dup_n = int(sub.duplicated(KEY_COLUMNS).sum())
        fail(f"structure feature table has duplicate residue keys: {dup_n}")
    for col in cols:
        if col in base.columns:
            base = base.drop(columns=[col])
    out = base.merge(sub, on=KEY_COLUMNS, how="left", validate="one_to_one")
    if SURFACE_FLAG in out.columns:
        out["surface_candidate"] = numeric(out[SURFACE_FLAG], fill=0).gt(0)
        out["surface_source"] = SURFACE_FLAG
    elif RSA_COLUMN in out.columns:
        out["surface_candidate"] = numeric(out[RSA_COLUMN], fill=np.nan).ge(rsa_threshold).fillna(False)
        out["surface_source"] = f"{RSA_COLUMN}_ge_{rsa_threshold}"
    else:
        out["surface_candidate"] = True
        out["surface_source"] = "missing_surface_columns_all_residues"
    return out


def add_hybrid_scores(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    created: List[str] = []
    full = "score__logreg_patch_conservation_ifrag_patch_radi_structure"
    reduced = "score__logreg_patch_conservation_ifrag_patch_structure"
    old = "score__logreg_patch_conservation_ifrag_patch_radi"
    if full in df.columns and reduced in df.columns:
        df["hybrid_min_full_reduced"] = np.minimum(numeric(df[full]), numeric(df[reduced]))
        df["hybrid_mean_full_reduced"] = (numeric(df[full]) + numeric(df[reduced])) / 2.0
        created.extend(["hybrid_min_full_reduced", "hybrid_mean_full_reduced"])
        if old in df.columns:
            df["hybrid_mean_full_reduced_sequence"] = (numeric(df[full]) + numeric(df[reduced]) + numeric(df[old])) / 3.0
            created.append("hybrid_mean_full_reduced_sequence")
    return df, created


def discover_methods(df: pd.DataFrame, extra_hybrids: Sequence[str]) -> Dict[str, str]:
    methods: Dict[str, str] = {}
    for col in DEFAULT_BASE_METHODS:
        if col in df.columns and pd.api.types.is_numeric_dtype(pd.to_numeric(df[col], errors="coerce")):
            methods[col] = col
    for col in df.columns:
        if col.startswith("score__"):
            methods[col.removeprefix("score__")] = col
    for col in extra_hybrids:
        if col in df.columns:
            methods[col] = col
    return methods


def subset_masks(df: pd.DataFrame) -> Dict[str, pd.Series]:
    masks: Dict[str, pd.Series] = {"all_rows": pd.Series(True, index=df.index)}
    if "primary_candidate_group" in df.columns:
        masks["primary_ml_set"] = numeric(df["primary_candidate_group"], fill=0).gt(0)
    if "evidence_class" in df.columns:
        # Useful for diagnostics; no more than all classes present in BM5 Phase 1.
        for value in sorted(v for v in df["evidence_class"].dropna().unique()):
            masks[f"evidence_class__{value}"] = df["evidence_class"].eq(value)
    return masks


def budget_definitions(fixed_budgets: Sequence[int], percent_budgets: Sequence[float]) -> List[Tuple[str, str, float]]:
    budgets: List[Tuple[str, str, float]] = []
    for k in fixed_budgets:
        budgets.append((f"top_{k}", "fixed", float(k)))
    for pct in percent_budgets:
        label = str(pct).replace(".", "p").rstrip("0").rstrip("p")
        budgets.append((f"top_{label}pct", "percent", pct))
    budgets.append(("top_trueN_chain", "trueN_chain", np.nan))
    budgets.append(("top_trueN_universe", "trueN_universe", np.nan))
    return budgets


def compute_budget_size(
    budget_kind: str,
    budget_value: float,
    candidate_count: int,
    chain_true_count: int,
    universe_true_count: int,
) -> int:
    if candidate_count <= 0:
        return 0
    if budget_kind == "fixed":
        return min(candidate_count, int(budget_value))
    if budget_kind == "percent":
        return min(candidate_count, max(1, int(math.ceil(candidate_count * budget_value / 100.0))))
    if budget_kind == "trueN_chain":
        return min(candidate_count, max(0, chain_true_count))
    if budget_kind == "trueN_universe":
        return min(candidate_count, max(0, universe_true_count))
    raise ValueError(f"Unknown budget kind: {budget_kind}")


def safe_div(numer: float, denom: float) -> float:
    if denom == 0:
        return float("nan")
    return float(numer) / float(denom)


def f1_score(precision: float, recall: float) -> float:
    if math.isnan(precision) or math.isnan(recall) or precision + recall == 0:
        return float("nan")
    return 2.0 * precision * recall / (precision + recall)


def summarize_group_sizes(df: pd.DataFrame, subset_name: str, target: str) -> pd.DataFrame:
    rows = []
    for (chainpair_id, query_side), g in df.groupby(["chainpair_id", "query_side"], sort=False):
        true_count = int(numeric(g[target], fill=0).gt(0).sum())
        rows.append(
            {
                "subset": subset_name,
                "chainpair_id": chainpair_id,
                "query_side": query_side,
                "residue_count": int(len(g)),
                "interface_residue_count": true_count,
                "surface_residue_count": int(g["surface_candidate"].sum()) if "surface_candidate" in g.columns else int(len(g)),
                "surface_interface_residue_count": int((numeric(g[target], fill=0).gt(0) & g.get("surface_candidate", True)).sum())
                if "surface_candidate" in g.columns
                else true_count,
            }
        )
    return pd.DataFrame(rows)


def summarize_distribution(size_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    bins = [
        ("zero", 0, 0),
        ("1_to_5", 1, 5),
        ("6_to_10", 6, 10),
        ("11_to_15", 11, 15),
        ("16_to_20", 16, 20),
        ("21_to_30", 21, 30),
        ("gt_30", 31, None),
    ]
    for subset_name, g in size_df.groupby("subset", sort=False):
        counts = g["interface_residue_count"].astype(int)
        row = {
            "subset": subset_name,
            "groups": int(len(g)),
            "groups_with_interface": int((counts > 0).sum()),
            "mean_interface_residues": float(counts.mean()) if len(counts) else float("nan"),
            "median_interface_residues": float(counts.median()) if len(counts) else float("nan"),
            "q25_interface_residues": float(counts.quantile(0.25)) if len(counts) else float("nan"),
            "q75_interface_residues": float(counts.quantile(0.75)) if len(counts) else float("nan"),
            "min_interface_residues": int(counts.min()) if len(counts) else 0,
            "max_interface_residues": int(counts.max()) if len(counts) else 0,
        }
        for label, lo, hi in bins:
            if hi is None:
                row[f"groups_{label}"] = int((counts >= lo).sum())
            else:
                row[f"groups_{label}"] = int(((counts >= lo) & (counts <= hi)).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate(
    df: pd.DataFrame,
    methods: Mapping[str, str],
    target: str,
    budgets: Sequence[Tuple[str, str, float]],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    result_rows: List[Dict[str, object]] = []
    per_group_rows: List[Dict[str, object]] = []
    size_rows: List[pd.DataFrame] = []

    all_subset_masks = subset_masks(df)
    for subset_name, mask in all_subset_masks.items():
        subset_df = df.loc[mask].copy()
        if subset_df.empty:
            continue
        size_rows.append(summarize_group_sizes(subset_df, subset_name, target))
        full_true_counts = {
            key: int(numeric(g[target], fill=0).gt(0).sum())
            for key, g in subset_df.groupby(["chainpair_id", "query_side"], sort=False)
        }
        # Only groups with a true native interface are informative for residue-recovery metrics.
        interface_keys = {key for key, count in full_true_counts.items() if count > 0}
        if not interface_keys:
            continue

        surface_modes = {
            "all_residues": pd.Series(True, index=subset_df.index),
            "surface_only": subset_df["surface_candidate"].fillna(False).astype(bool)
            if "surface_candidate" in subset_df.columns
            else pd.Series(True, index=subset_df.index),
        }

        for surface_mode, surface_mask in surface_modes.items():
            universe = subset_df.loc[surface_mask].copy()
            if universe.empty:
                continue
            grouped_universe = dict(tuple(universe.groupby(["chainpair_id", "query_side"], sort=False)))

            for method_name, score_col in methods.items():
                if score_col not in universe.columns:
                    continue
                for budget_name, budget_kind, budget_value in budgets:
                    totals = {
                        "groups_evaluated": 0,
                        "groups_with_selected_residues": 0,
                        "candidate_residues": 0,
                        "selected_residues": 0,
                        "true_interface_residues_in_chain": 0,
                        "true_interface_residues_in_universe": 0,
                        "true_positives": 0,
                        "false_positives": 0,
                        "false_negatives_universe": 0,
                        "false_negatives_vs_chain": 0,
                    }
                    group_tp: List[int] = []
                    group_fp: List[int] = []
                    group_precision: List[float] = []
                    group_recall: List[float] = []
                    for key in interface_keys:
                        chain_true_count = full_true_counts[key]
                        g = grouped_universe.get(key)
                        if g is None or g.empty:
                            candidate_count = 0
                            universe_true_count = 0
                            k = 0
                            selected = g
                            tp = 0
                            fp = 0
                        else:
                            candidate_count = int(len(g))
                            y = numeric(g[target], fill=0).gt(0)
                            universe_true_count = int(y.sum())
                            k = compute_budget_size(
                                budget_kind=budget_kind,
                                budget_value=budget_value,
                                candidate_count=candidate_count,
                                chain_true_count=chain_true_count,
                                universe_true_count=universe_true_count,
                            )
                            work = g.copy()
                            work["_score"] = numeric(work[score_col]).fillna(-np.inf)
                            work["_tie_index"] = numeric(work["score_residue_index"], fill=np.inf)
                            if k > 0:
                                selected = work.sort_values(
                                    ["_score", "_tie_index"],
                                    ascending=[False, True],
                                    kind="mergesort",
                                ).head(k)
                            else:
                                selected = work.iloc[0:0]
                            selected_y = numeric(selected[target], fill=0).gt(0) if selected is not None else pd.Series(dtype=bool)
                            tp = int(selected_y.sum())
                            fp = int(len(selected) - tp)

                        fn_universe = max(0, universe_true_count - tp)
                        fn_chain = max(0, chain_true_count - tp)
                        selected_count = int(len(selected)) if selected is not None else 0
                        precision = safe_div(tp, selected_count)
                        recall = safe_div(tp, universe_true_count)
                        recall_vs_chain = safe_div(tp, chain_true_count)
                        f1 = f1_score(precision, recall)
                        f1_vs_chain = f1_score(precision, recall_vs_chain)

                        selected_indices = ""
                        selected_true_indices = ""
                        selected_false_indices = ""
                        if selected is not None and selected_count > 0:
                            idx_values = selected["score_residue_index"].astype(str).tolist()
                            selected_indices = ",".join(idx_values)
                            selected_true_indices = ",".join(selected.loc[numeric(selected[target], fill=0).gt(0), "score_residue_index"].astype(str).tolist())
                            selected_false_indices = ",".join(selected.loc[~numeric(selected[target], fill=0).gt(0), "score_residue_index"].astype(str).tolist())

                        per_group_rows.append(
                            {
                                "subset": subset_name,
                                "surface_mode": surface_mode,
                                "method": method_name,
                                "score_column": score_col,
                                "budget": budget_name,
                                "budget_kind": budget_kind,
                                "budget_value": budget_value,
                                "chainpair_id": key[0],
                                "query_side": key[1],
                                "candidate_residues": candidate_count,
                                "selected_residues": selected_count,
                                "true_interface_residues_in_chain": chain_true_count,
                                "true_interface_residues_in_universe": universe_true_count,
                                "true_positives": tp,
                                "false_positives": fp,
                                "false_negatives_universe": fn_universe,
                                "false_negatives_vs_chain": fn_chain,
                                "precision": precision,
                                "recall": recall,
                                "recall_vs_chain": recall_vs_chain,
                                "f1": f1,
                                "f1_vs_chain": f1_vs_chain,
                                "selected_residue_indices": selected_indices,
                                "selected_true_positive_indices": selected_true_indices,
                                "selected_false_positive_indices": selected_false_indices,
                            }
                        )

                        totals["groups_evaluated"] += 1
                        totals["groups_with_selected_residues"] += int(selected_count > 0)
                        totals["candidate_residues"] += candidate_count
                        totals["selected_residues"] += selected_count
                        totals["true_interface_residues_in_chain"] += chain_true_count
                        totals["true_interface_residues_in_universe"] += universe_true_count
                        totals["true_positives"] += tp
                        totals["false_positives"] += fp
                        totals["false_negatives_universe"] += fn_universe
                        totals["false_negatives_vs_chain"] += fn_chain
                        group_tp.append(tp)
                        group_fp.append(fp)
                        if not math.isnan(precision):
                            group_precision.append(precision)
                        if not math.isnan(recall):
                            group_recall.append(recall)

                    precision_total = safe_div(totals["true_positives"], totals["selected_residues"])
                    recall_total = safe_div(totals["true_positives"], totals["true_interface_residues_in_universe"])
                    recall_vs_chain_total = safe_div(totals["true_positives"], totals["true_interface_residues_in_chain"])
                    result_rows.append(
                        {
                            "subset": subset_name,
                            "surface_mode": surface_mode,
                            "method": method_name,
                            "score_column": score_col,
                            "budget": budget_name,
                            "budget_kind": budget_kind,
                            "budget_value": budget_value,
                            **totals,
                            "precision": precision_total,
                            "recall": recall_total,
                            "recall_vs_chain": recall_vs_chain_total,
                            "f1": f1_score(precision_total, recall_total),
                            "f1_vs_chain": f1_score(precision_total, recall_vs_chain_total),
                            "mean_TP_per_chain": float(np.mean(group_tp)) if group_tp else float("nan"),
                            "mean_FP_per_chain": float(np.mean(group_fp)) if group_fp else float("nan"),
                            "median_TP_per_chain": float(np.median(group_tp)) if group_tp else float("nan"),
                            "median_FP_per_chain": float(np.median(group_fp)) if group_fp else float("nan"),
                            "mean_precision_per_chain": float(np.mean(group_precision)) if group_precision else float("nan"),
                            "mean_recall_per_chain": float(np.mean(group_recall)) if group_recall else float("nan"),
                        }
                    )

    size_df = pd.concat(size_rows, ignore_index=True) if size_rows else pd.DataFrame()
    return pd.DataFrame(result_rows), pd.DataFrame(per_group_rows), size_df


def json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if np.isnan(obj):
            return None
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return str(obj)


def main() -> None:
    args = parse_args()
    training_path = Path(args.training_table)
    structure_feature_path = Path(args.structure_feature_table)
    structure_pred_path = Path(args.structure_predictions)
    ablation_pred_path = Path(args.ablation_predictions)
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    fixed_budgets = parse_int_list(args.fixed_budgets)
    percent_budgets = parse_float_list(args.percent_budgets)
    budgets = budget_definitions(fixed_budgets, percent_budgets)

    base = read_table(training_path, required=True)
    assert base is not None
    ensure_required_columns(base, KEY_COLUMNS + [args.target], "training table")
    if base.duplicated(KEY_COLUMNS).any():
        dup_n = int(base.duplicated(KEY_COLUMNS).sum())
        fail(f"training table has duplicate residue keys: {dup_n}")

    structure_features = read_table(structure_feature_path, required=False)
    base = merge_surface_features(base, structure_features, args.rsa_surface_threshold)

    added_prediction_columns: Dict[str, List[str]] = {}
    structure_predictions = read_table(structure_pred_path, required=args.require_predictions)
    base, added = merge_score_table(base, structure_predictions, "structure_ml")
    added_prediction_columns["structure_ml"] = added

    ablation_predictions = read_table(ablation_pred_path, required=args.require_predictions)
    base, added = merge_score_table(base, ablation_predictions, "structure_ablation")
    added_prediction_columns["structure_ablation"] = added

    base, hybrid_cols = add_hybrid_scores(base)
    methods = discover_methods(base, hybrid_cols)
    if not methods:
        fail("No score methods were discovered for fixed-budget evaluation.")

    # Coerce target to binary once for robust downstream arithmetic.
    base[args.target] = numeric(base[args.target], fill=0).gt(0).astype(int)

    results, per_group, size_df = evaluate(base, methods, args.target, budgets)
    distribution = summarize_distribution(size_df) if not size_df.empty else pd.DataFrame()

    results_path = f"{out_prefix}.tsv"
    per_group_path = f"{out_prefix}.per_group.tsv"
    size_path = f"{out_prefix}.interface_size_distribution.tsv"
    summary_path = f"{out_prefix}.summary.json"

    results.to_csv(results_path, sep="\t", index=False)
    per_group.to_csv(per_group_path, sep="\t", index=False)
    distribution.to_csv(size_path, sep="\t", index=False)

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": Path(__file__).name,
        "training_table": str(training_path),
        "structure_feature_table": str(structure_feature_path),
        "structure_predictions": str(structure_pred_path),
        "ablation_predictions": str(ablation_pred_path),
        "out_prefix": str(out_prefix),
        "target": args.target,
        "fixed_budgets": fixed_budgets,
        "percent_budgets": percent_budgets,
        "budget_names": [b[0] for b in budgets],
        "rows_loaded": int(len(base)),
        "chainpairs": int(base["chainpair_id"].nunique()),
        "query_side_groups": int(base.groupby(["chainpair_id", "query_side"]).ngroups),
        "target_positive_rows": int(base[args.target].sum()),
        "surface_candidate_rows": int(base["surface_candidate"].sum()) if "surface_candidate" in base.columns else None,
        "surface_source": str(base["surface_source"].iloc[0]) if "surface_source" in base.columns and len(base) else None,
        "methods_evaluated": list(methods.keys()),
        "method_count": int(len(methods)),
        "prediction_columns_added": added_prediction_columns,
        "hybrid_columns_created": hybrid_cols,
        "outputs": {
            "results": results_path,
            "per_group": per_group_path,
            "interface_size_distribution": size_path,
            "summary": summary_path,
        },
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, default=json_default)

    print("BM5 Phase 1 fixed-budget residue recovery written")
    print(f"  rows loaded: {len(base)}")
    print(f"  methods evaluated: {len(methods)}")
    print(f"  budgets: {', '.join([b[0] for b in budgets])}")
    print(f"  outputs:")
    print(f"    results: {results_path}")
    print(f"    per_group: {per_group_path}")
    print(f"    interface_size_distribution: {size_path}")
    print(f"    summary: {summary_path}")

    # Print a compact first-look table for the exact question that motivated this script.
    focus_methods = [
        "conservation_component",
        "ifrag_component",
        "radi_component",
        "final_score",
        "hybrid_min_full_reduced",
        "logreg_patch_conservation_ifrag_patch_radi_structure",
        "ablate_full_all_safe",
        "ablate_no_ifrag_clean_drop_composite",
        "ablate_no_conservation_clean_drop_composite",
        "ablate_no_radi_all",
        "ablate_no_structure",
    ]
    focus = results[
        results["subset"].eq("primary_ml_set")
        & results["surface_mode"].isin(["all_residues", "surface_only"])
        & results["budget"].isin(["top_15", "top_5pct", "top_trueN_chain"])
        & results["method"].isin(focus_methods)
    ].copy()
    if not focus.empty:
        cols = [
            "surface_mode",
            "budget",
            "method",
            "true_positives",
            "false_positives",
            "precision",
            "recall_vs_chain",
            "f1_vs_chain",
            "mean_TP_per_chain",
            "mean_FP_per_chain",
        ]
        print("\nFirst-look primary fixed-budget metrics:")
        print(
            focus[cols]
            .sort_values(["surface_mode", "budget", "precision"], ascending=[True, True, False], kind="mergesort")
            .to_string(index=False)
        )

    if not distribution.empty:
        print("\nInterface-size distribution:")
        print(distribution.to_string(index=False))


if __name__ == "__main__":
    main()
