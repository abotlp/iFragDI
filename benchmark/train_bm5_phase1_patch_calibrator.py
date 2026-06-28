#!/usr/bin/env python3
"""
Train BM5 Phase 1 patch/window-aware residue calibrators for iFragDI.

This script consumes the patch/window feature table produced by
benchmark/build_bm5_phase1_patch_features.py and compares residue-level
baselines against leakage-safe logistic-regression feature sets.

Main question:
    Do local patch/window features improve docking-useful top-L/10 and top-L/5
    recovery of true interface residues, beyond conservation-only, current
    final_score, and the first residue-level logistic baseline?

Default input:
    benchmark/labels/bm5_phase1_patch_features.tsv
    benchmark/labels/bm5_phase1_patch_features.feature_manifest.tsv

Default outputs with --out-prefix benchmark/labels/bm5_phase1_patch_ml_logreg:
    *.predictions.tsv
    *.metrics.tsv
    *.group_metrics.tsv
    *.best_models.tsv
    *.coefficients.tsv
    *.coefficients_mean.tsv
    *.summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


NONCONTACTING_CONTROLS_DEFAULT = ("BM5CP00234", "BM5CP00238", "BM5CP00318")

IDENTIFIER_COLUMNS = [
    "chainpair_id",
    "case_id",
    "query_side",
    "query_role",
    "score_residue_index",
    "pdb_chain",
    "pdb_residue_id",
    "aa",
    "pdb_resname",
    "evidence_class",
]

BASELINE_SCORE_COLUMNS = [
    "final_score",
    "conservation_component",
    "conservation_strength",
    "ifrag_strength",
    "ifrag_component",
    "patch_score",
    "radi_component",
]

LEAKAGE_GUARD_COLUMNS = {
    "group_positive_count_target",
    "min_partner_atom_distance_A",
    "nearest_partner_residue_label",
}

LEAKAGE_PREFIXES = (
    "interface_",
    "near_interface_",
    "bound_",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train BM5 Phase 1 patch/window-aware residue calibrators from the "
            "patch-feature table."
        )
    )
    parser.add_argument(
        "--feature-table",
        default="benchmark/labels/bm5_phase1_patch_features.tsv",
        help="Patch/window-aware residue feature table.",
    )
    parser.add_argument(
        "--feature-manifest",
        default="benchmark/labels/bm5_phase1_patch_features.feature_manifest.tsv",
        help="Feature manifest produced by build_bm5_phase1_patch_features.py.",
    )
    parser.add_argument(
        "--out-prefix",
        default="benchmark/labels/bm5_phase1_patch_ml_logreg",
        help="Output prefix.",
    )
    parser.add_argument(
        "--target",
        default="interface_5A",
        help="Binary residue target column. Default: interface_5A.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of group-safe cross-validation folds. Default: 5.",
    )
    parser.add_argument(
        "--noncontacting-controls",
        default=",".join(NONCONTACTING_CONTROLS_DEFAULT),
        help="Comma-separated chainpair_id values treated as explicit noncontacting controls.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=2000,
        help="Maximum LogisticRegression iterations. Default: 2000.",
    )
    parser.add_argument(
        "--regularization-c",
        type=float,
        default=1.0,
        help="Inverse regularization strength for LogisticRegression. Default: 1.0.",
    )
    parser.add_argument(
        "--soft-target",
        default=None,
        help=(
            "Optional secondary soft/window target to evaluate, e.g. "
            "near_interface_5A_window_5. This is evaluation-only unless --train-target is set."
        ),
    )
    parser.add_argument(
        "--train-target",
        default=None,
        help=(
            "Optional target used for fitting ML models. Defaults to --target. "
            "Use cautiously for soft-target experiments."
        ),
    )
    return parser.parse_args()


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(2)


def warn(message: str) -> None:
    print(f"WARNING: {message}", file=sys.stderr)


def parse_controls(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def to_numeric_series(series: pd.Series, fill: Optional[float] = None) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    if fill is not None:
        out = out.fillna(fill)
    return out


def as_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    lowered = series.astype(str).str.strip().str.lower()
    return lowered.isin({"true", "t", "1", "yes", "y"})


def safe_div(numer: float, denom: float) -> float:
    if denom == 0 or not np.isfinite(denom):
        return float("nan")
    return float(numer / denom)


def safe_metric(metric_name: str, y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    ok = np.isfinite(y_score)
    y_true = y_true[ok]
    y_score = y_score[ok]
    if len(y_true) == 0:
        return float("nan")
    positives = int(y_true.sum())
    negatives = int(len(y_true) - positives)
    if metric_name == "auprc":
        if positives == 0:
            return float("nan")
        return float(average_precision_score(y_true, y_score))
    if metric_name == "roc_auc":
        if positives == 0 or negatives == 0:
            return float("nan")
        return float(roc_auc_score(y_true, y_score))
    raise ValueError(f"Unknown metric: {metric_name}")


def deterministic_topk(group: pd.DataFrame, score_col: str, divisor: int) -> pd.DataFrame:
    n = len(group)
    k = max(1, int(math.ceil(n / float(divisor))))
    work = group.copy()
    work["_score_for_rank"] = to_numeric_series(work[score_col], fill=-np.inf)
    if "score_residue_index" in work.columns:
        work["_residue_index_for_rank"] = to_numeric_series(work["score_residue_index"], fill=np.inf)
    else:
        work["_residue_index_for_rank"] = np.arange(n)
    work["_row_order_for_rank"] = np.arange(n)
    ranked = work.sort_values(
        ["_score_for_rank", "_residue_index_for_rank", "_row_order_for_rank"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    return ranked.head(k)


def topk_metrics(
    df: pd.DataFrame,
    score_col: str,
    target_col: str,
    group_cols: Sequence[str] = ("chainpair_id", "query_side"),
) -> Dict[str, float]:
    records = []
    totals = {
        "top_L10_selected": 0,
        "top_L10_recovered": 0,
        "top_L5_selected": 0,
        "top_L5_recovered": 0,
        "positives": 0,
        "groups_with_positive": 0,
    }

    for _, group in df.groupby(list(group_cols), sort=False, dropna=False):
        y = to_numeric_series(group[target_col], fill=0).clip(lower=0, upper=1).astype(int)
        positives = int(y.sum())
        if positives <= 0:
            continue
        totals["groups_with_positive"] += 1
        totals["positives"] += positives

        row = {
            "chainpair_id": str(group["chainpair_id"].iloc[0]),
            "query_side": str(group["query_side"].iloc[0]),
            "n_rows": int(len(group)),
            "n_positive": positives,
        }

        for label, divisor in (("top_L10", 10), ("top_L5", 5)):
            top = deterministic_topk(group, score_col, divisor)
            selected = int(len(top))
            recovered = int(to_numeric_series(top[target_col], fill=0).clip(lower=0, upper=1).sum())
            totals[f"{label}_selected"] += selected
            totals[f"{label}_recovered"] += recovered
            row[f"{label}_selected"] = selected
            row[f"{label}_recovered"] = recovered
            row[f"{label}_recall"] = safe_div(recovered, positives)
            row[f"{label}_precision"] = safe_div(recovered, selected)
            base_rate = safe_div(positives, len(group))
            row[f"{label}_enrichment"] = safe_div(row[f"{label}_precision"], base_rate)
        records.append(row)

    if not records:
        return {
            "top_L10_recall": float("nan"),
            "top_L10_precision": float("nan"),
            "top_L10_enrichment": float("nan"),
            "top_L5_recall": float("nan"),
            "top_L5_precision": float("nan"),
            "top_L5_enrichment": float("nan"),
            "top_groups_with_positive": 0,
        }

    group_df = pd.DataFrame(records)
    return {
        "top_L10_recall": safe_div(totals["top_L10_recovered"], totals["positives"]),
        "top_L10_precision": safe_div(totals["top_L10_recovered"], totals["top_L10_selected"]),
        "top_L10_enrichment": safe_div(
            safe_div(totals["top_L10_recovered"], totals["top_L10_selected"]),
            safe_div(totals["positives"], int(df.shape[0])),
        ),
        "top_L5_recall": safe_div(totals["top_L5_recovered"], totals["positives"]),
        "top_L5_precision": safe_div(totals["top_L5_recovered"], totals["top_L5_selected"]),
        "top_L5_enrichment": safe_div(
            safe_div(totals["top_L5_recovered"], totals["top_L5_selected"]),
            safe_div(totals["positives"], int(df.shape[0])),
        ),
        "top_L10_recall_group_mean": float(group_df["top_L10_recall"].mean()),
        "top_L10_precision_group_mean": float(group_df["top_L10_precision"].mean()),
        "top_L5_recall_group_mean": float(group_df["top_L5_recall"].mean()),
        "top_L5_precision_group_mean": float(group_df["top_L5_precision"].mean()),
        "top_groups_with_positive": int(totals["groups_with_positive"]),
    }


def group_level_rows(
    df: pd.DataFrame,
    score_col: str,
    target_col: str,
    model: str,
    subset: str,
    group_cols: Sequence[str] = ("chainpair_id", "query_side"),
) -> List[Dict[str, object]]:
    rows = []
    for _, group in df.groupby(list(group_cols), sort=False, dropna=False):
        y = to_numeric_series(group[target_col], fill=0).clip(lower=0, upper=1).astype(int)
        positives = int(y.sum())
        if positives <= 0:
            continue
        record: Dict[str, object] = {
            "subset": subset,
            "model": model,
            "chainpair_id": str(group["chainpair_id"].iloc[0]),
            "query_side": str(group["query_side"].iloc[0]),
            "n_rows": int(len(group)),
            "n_positive": positives,
        }
        for label, divisor in (("top_L10", 10), ("top_L5", 5)):
            top = deterministic_topk(group, score_col, divisor)
            selected = int(len(top))
            recovered = int(to_numeric_series(top[target_col], fill=0).clip(lower=0, upper=1).sum())
            record[f"{label}_selected"] = selected
            record[f"{label}_recovered"] = recovered
            record[f"{label}_recall"] = safe_div(recovered, positives)
            record[f"{label}_precision"] = safe_div(recovered, selected)
        rows.append(record)
    return rows


def base_feature_safe_columns(df: pd.DataFrame, manifest: pd.DataFrame) -> List[str]:
    if "column" not in manifest.columns or "leakage_status" not in manifest.columns:
        fail("Feature manifest must contain columns: column, leakage_status")

    safe_status = {"feature_safe"}
    candidates = manifest.loc[manifest["leakage_status"].isin(safe_status), "column"].astype(str).tolist()

    safe = []
    for col in candidates:
        if col not in df.columns:
            continue
        if col in LEAKAGE_GUARD_COLUMNS:
            continue
        if any(col.startswith(prefix) for prefix in LEAKAGE_PREFIXES):
            continue
        if col.startswith("diagnostic_final_score"):
            continue
        safe.append(col)

    # Remove duplicates while preserving order.
    seen = set()
    out = []
    for col in safe:
        if col not in seen:
            seen.add(col)
            out.append(col)
    return out


def columns_with_prefixes(columns: Sequence[str], prefixes: Sequence[str]) -> List[str]:
    return [col for col in columns if any(col.startswith(prefix) for prefix in prefixes)]


def existing_numeric_columns(df: pd.DataFrame, columns: Sequence[str]) -> List[str]:
    out = []
    seen = set()
    for col in columns:
        if col in seen or col not in df.columns:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().any():
            out.append(col)
            seen.add(col)
    return out


def build_feature_sets(df: pd.DataFrame, manifest: pd.DataFrame) -> Dict[str, List[str]]:
    safe_cols = base_feature_safe_columns(df, manifest)

    original_core = [
        "conservation_component",
        "conservation_strength",
        "ifrag_strength",
        "ifrag_specificity",
        "ifrag_component",
        "patch_score",
    ]
    original_core_radi = original_core + ["radi_anchor", "radi_component"]

    conservation_patch = columns_with_prefixes(
        safe_cols,
        ("conservation_component_", "conservation_strength_"),
    ) + ["conservation_component", "conservation_strength"]

    ifrag_patch = columns_with_prefixes(
        safe_cols,
        ("ifrag_component_", "ifrag_strength_", "ifrag_specificity_"),
    ) + ["ifrag_component", "ifrag_strength", "ifrag_specificity"]

    patch_score_patch = columns_with_prefixes(safe_cols, ("patch_score_",)) + ["patch_score"]

    radi_neighborhood = columns_with_prefixes(
        safe_cols,
        ("radi_anchor_", "radi_component_"),
    ) + ["radi_anchor", "radi_component"]

    feature_sets = {
        "logreg_original_core": original_core,
        "logreg_original_core_radi": original_core_radi,
        "logreg_patch_conservation": conservation_patch,
        "logreg_patch_ifrag": ifrag_patch,
        "logreg_patch_conservation_ifrag": conservation_patch + ifrag_patch,
        "logreg_patch_conservation_ifrag_patch": conservation_patch + ifrag_patch + patch_score_patch,
        "logreg_patch_conservation_ifrag_patch_radi": conservation_patch + ifrag_patch + patch_score_patch + radi_neighborhood,
    }

    return {name: existing_numeric_columns(df, cols) for name, cols in feature_sets.items()}


def add_required_annotations(df: pd.DataFrame, target: str, controls: Sequence[str]) -> pd.DataFrame:
    out = df.copy()

    if "group_key" not in out.columns:
        out["group_key"] = out["chainpair_id"].astype(str) + "||" + out["query_side"].astype(str)

    if "is_noncontacting_control" not in out.columns:
        out["is_noncontacting_control"] = out["chainpair_id"].astype(str).isin(set(controls))
    else:
        out["is_noncontacting_control"] = as_bool_series(out["is_noncontacting_control"])

    if "is_no_evidence_completed" not in out.columns:
        if "evidence_class" in out.columns:
            out["is_no_evidence_completed"] = out["evidence_class"].astype(str).eq("no_evidence_completed")
        else:
            out["is_no_evidence_completed"] = False
    else:
        out["is_no_evidence_completed"] = as_bool_series(out["is_no_evidence_completed"])

    y = to_numeric_series(out[target], fill=0).clip(lower=0, upper=1).astype(int)
    out[target] = y

    group_pos = y.groupby(out["group_key"]).transform("sum")
    out["group_positive_count_target_runtime"] = group_pos.astype(float)

    if "primary_candidate_group" in out.columns:
        # Preserve the builder's primary mask, but normalize it to bool/int.
        out["primary_candidate_group"] = to_numeric_series(out["primary_candidate_group"], fill=0).gt(0).astype(int)
    else:
        out["primary_candidate_group"] = (
            group_pos.gt(0) & ~out["is_no_evidence_completed"] & ~out["is_noncontacting_control"]
        ).astype(int)

    if "weak_msa_warning_bool" not in out.columns:
        if "weak_msa_warning" in out.columns:
            out["weak_msa_warning_bool"] = as_bool_series(out["weak_msa_warning"]).astype(int)
        else:
            out["weak_msa_warning_bool"] = 0

    return out


def define_subsets(df: pd.DataFrame) -> Dict[str, pd.Series]:
    subsets: Dict[str, pd.Series] = {}
    true = pd.Series(True, index=df.index)
    subsets["all_rows"] = true
    subsets["primary_ml_set"] = to_numeric_series(df["primary_candidate_group"], fill=0).gt(0)
    subsets["no_evidence_completed"] = as_bool_series(df["is_no_evidence_completed"])
    subsets["noncontacting_controls"] = as_bool_series(df["is_noncontacting_control"])

    group_key = df["group_key"].astype(str)

    if "radi_anchor" in df.columns:
        has_anchor = to_numeric_series(df["radi_anchor"], fill=0).gt(0).groupby(group_key).transform("any")
        subsets["groups_with_radi_anchor"] = has_anchor.astype(bool)
    if "radi_anchor_win5_count" in df.columns:
        has_anchor_win = to_numeric_series(df["radi_anchor_win5_count"], fill=0).gt(0).groupby(group_key).transform("any")
        subsets["groups_with_radi_anchor_window5"] = has_anchor_win.astype(bool)

    if "radi_interchain_pairs_retained" in df.columns:
        has_pairs = to_numeric_series(df["radi_interchain_pairs_retained"], fill=0).gt(0).groupby(group_key).transform("any")
        subsets["groups_with_radi_pairs"] = has_pairs.astype(bool)

    if "paired_rows_used" in df.columns:
        paired_group = to_numeric_series(df["paired_rows_used"], fill=0).groupby(group_key).transform("max")
        q75 = paired_group.groupby(group_key).first().quantile(0.75)
        subsets["groups_high_paired_rows_q75"] = paired_group.ge(q75)

    weak = to_numeric_series(df["weak_msa_warning_bool"], fill=0).gt(0)
    subsets["groups_weak_msa_warning_true"] = weak
    subsets["groups_weak_msa_warning_false"] = ~weak

    if "evidence_class" in df.columns:
        for evidence_class in sorted(df["evidence_class"].fillna("missing").astype(str).unique()):
            safe_name = evidence_class.replace(" ", "_").replace("/", "_")
            subsets[f"evidence_class={safe_name}"] = df["evidence_class"].fillna("missing").astype(str).eq(evidence_class)

    return subsets


def make_pipeline(c_value: float, max_iter: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    C=c_value,
                    penalty="l2",
                    solver="liblinear",
                    class_weight="balanced",
                    max_iter=max_iter,
                    random_state=1,
                ),
            ),
        ]
    )


def train_cv_predictions(
    df: pd.DataFrame,
    feature_sets: Mapping[str, Sequence[str]],
    target_col: str,
    folds: int,
    c_value: float,
    max_iter: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    chainpairs = df["chainpair_id"].astype(str)
    unique_chainpairs = chainpairs.drop_duplicates().to_numpy()
    n_splits = min(int(folds), len(unique_chainpairs))
    if n_splits < 2:
        fail("Need at least two chainpair groups for cross-validation.")

    gkf = GroupKFold(n_splits=n_splits)
    primary_mask = to_numeric_series(df["primary_candidate_group"], fill=0).gt(0)
    y_train_target = to_numeric_series(df[target_col], fill=0).clip(lower=0, upper=1).astype(int)

    predictions = pd.DataFrame(index=df.index)
    coefficient_rows: List[Dict[str, object]] = []
    fold_summaries: List[Dict[str, object]] = []

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(df, y_train_target, groups=chainpairs), start=1):
        train_index = df.index[train_idx]
        test_index = df.index[test_idx]

        fit_mask = primary_mask.loc[train_index]
        fit_index = train_index[fit_mask.to_numpy()]
        y_fit = y_train_target.loc[fit_index]

        positives = int(y_fit.sum())
        negatives = int(len(y_fit) - positives)
        fold_summaries.append(
            {
                "fold": fold_idx,
                "train_primary_rows": int(len(fit_index)),
                "train_positive": positives,
                "train_negative": negatives,
                "test_rows": int(len(test_index)),
                "test_chainpairs": int(chainpairs.loc[test_index].nunique()),
            }
        )

        if positives == 0 or negatives == 0:
            warn(f"Skipping fold {fold_idx}: train set lacks both classes.")
            continue

        for model_name, features in feature_sets.items():
            if not features:
                warn(f"Skipping {model_name}: no available features.")
                continue

            pipeline = make_pipeline(c_value=c_value, max_iter=max_iter)
            x_fit = df.loc[fit_index, list(features)]
            x_test = df.loc[test_index, list(features)]

            pipeline.fit(x_fit, y_fit)
            predictions.loc[test_index, model_name] = pipeline.predict_proba(x_test)[:, 1]

            logreg = pipeline.named_steps["logreg"]
            coef = logreg.coef_[0]
            for feature, value in zip(features, coef):
                coefficient_rows.append(
                    {
                        "model": model_name,
                        "fold": fold_idx,
                        "feature": feature,
                        "coefficient": float(value),
                    }
                )
            coefficient_rows.append(
                {
                    "model": model_name,
                    "fold": fold_idx,
                    "feature": "intercept",
                    "coefficient": float(logreg.intercept_[0]),
                }
            )

    diagnostics = {
        "n_splits": n_splits,
        "fold_summaries": fold_summaries,
    }
    return predictions, pd.DataFrame(coefficient_rows), diagnostics


def evaluate_models(
    df: pd.DataFrame,
    model_score_cols: Mapping[str, str],
    target_col: str,
    subsets: Mapping[str, pd.Series],
    include_group_rows: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: List[Dict[str, object]] = []
    group_rows: List[Dict[str, object]] = []

    for subset_name, mask in subsets.items():
        mask = mask.reindex(df.index).fillna(False).astype(bool)
        sub = df.loc[mask].copy()
        if sub.empty:
            continue

        y = to_numeric_series(sub[target_col], fill=0).clip(lower=0, upper=1).astype(int).to_numpy()
        n_positive = int(y.sum())
        n_negative = int(len(y) - n_positive)

        for model_name, score_col in model_score_cols.items():
            if score_col not in sub.columns:
                continue
            scores = to_numeric_series(sub[score_col], fill=np.nan).to_numpy()

            row: Dict[str, object] = {
                "subset": subset_name,
                "model": model_name,
                "target": target_col,
                "n_rows": int(len(sub)),
                "n_positive": n_positive,
                "n_negative": n_negative,
                "auprc": safe_metric("auprc", y, scores),
                "roc_auc": safe_metric("roc_auc", y, scores),
            }
            row.update(topk_metrics(sub, score_col, target_col))
            metric_rows.append(row)

            if include_group_rows:
                group_rows.extend(group_level_rows(sub, score_col, target_col, model_name, subset_name))

    return pd.DataFrame(metric_rows), pd.DataFrame(group_rows)


def add_delta_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    out = metrics.copy()
    for metric in ["auprc", "top_L10_recall", "top_L5_recall", "top_L10_precision", "top_L5_precision"]:
        out[f"delta_{metric}_vs_conservation"] = np.nan
        out[f"delta_{metric}_vs_final_score"] = np.nan
        out[f"delta_{metric}_vs_original_core"] = np.nan

    key_cols = ["subset", "target"]
    lookup = out.set_index(key_cols + ["model"])

    for idx, row in out.iterrows():
        key = (row["subset"], row["target"])
        for metric in ["auprc", "top_L10_recall", "top_L5_recall", "top_L10_precision", "top_L5_precision"]:
            value = row.get(metric, np.nan)
            for baseline_model, suffix in [
                ("baseline_conservation_component", "conservation"),
                ("baseline_final_score", "final_score"),
                ("logreg_original_core", "original_core"),
            ]:
                baseline_key = key + (baseline_model,)
                if baseline_key in lookup.index:
                    baseline_value = lookup.loc[baseline_key, metric]
                    if np.isfinite(value) and np.isfinite(baseline_value):
                        out.loc[idx, f"delta_{metric}_vs_{suffix}"] = float(value - baseline_value)
    return out


def summarize_best_models(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (subset, target), group in metrics.groupby(["subset", "target"], dropna=False):
        for metric in ["auprc", "roc_auc", "top_L10_recall", "top_L5_recall", "top_L10_precision", "top_L5_precision"]:
            valid = group[np.isfinite(pd.to_numeric(group[metric], errors="coerce"))].copy()
            if valid.empty:
                continue
            best = valid.sort_values(metric, ascending=False, kind="mergesort").iloc[0]
            rows.append(
                {
                    "subset": subset,
                    "target": target,
                    "selection_metric": metric,
                    "best_model": best["model"],
                    "best_value": float(best[metric]),
                    "n_rows": int(best["n_rows"]),
                    "n_positive": int(best["n_positive"]),
                }
            )
    return pd.DataFrame(rows)


def coefficient_means(coefficients: pd.DataFrame) -> pd.DataFrame:
    if coefficients.empty:
        return pd.DataFrame(columns=["model", "feature", "coefficient_mean", "coefficient_sd", "n_folds"])
    grouped = coefficients.groupby(["model", "feature"], dropna=False)["coefficient"]
    out = grouped.agg(["mean", "std", "count"]).reset_index()
    out = out.rename(
        columns={
            "mean": "coefficient_mean",
            "std": "coefficient_sd",
            "count": "n_folds",
        }
    )
    return out.sort_values(["model", "coefficient_mean"], ascending=[True, False], kind="mergesort")


def json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if pd.isna(obj):
        return None
    raise TypeError(f"Object of type {type(obj)!r} is not JSON serializable")


def main() -> None:
    args = parse_args()
    controls = parse_controls(args.noncontacting_controls)

    feature_table = Path(args.feature_table)
    manifest_path = Path(args.feature_manifest)
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    if not feature_table.exists():
        fail(f"Feature table does not exist: {feature_table}")
    if not manifest_path.exists():
        fail(f"Feature manifest does not exist: {manifest_path}")

    df = pd.read_csv(feature_table, sep="\t", low_memory=False)
    manifest = pd.read_csv(manifest_path, sep="\t", low_memory=False)

    for col in ["chainpair_id", "query_side", args.target]:
        if col not in df.columns:
            fail(f"Missing required feature-table column: {col}")

    train_target = args.train_target or args.target
    if train_target not in df.columns:
        fail(f"Training target column does not exist: {train_target}")

    df = add_required_annotations(df, target=args.target, controls=controls)
    if train_target != args.target:
        df[train_target] = to_numeric_series(df[train_target], fill=0).clip(lower=0, upper=1).astype(int)

    feature_sets = build_feature_sets(df, manifest)
    feature_sets = {name: cols for name, cols in feature_sets.items() if cols}

    if not feature_sets:
        fail("No non-empty feature sets were constructed.")

    baseline_score_cols = {}
    for col in BASELINE_SCORE_COLUMNS:
        if col in df.columns:
            score_col = f"score__baseline_{col}"
            df[score_col] = to_numeric_series(df[col], fill=0.0)
            baseline_score_cols[f"baseline_{col}"] = score_col

    pred_df, coefficients, cv_diag = train_cv_predictions(
        df=df,
        feature_sets=feature_sets,
        target_col=train_target,
        folds=args.folds,
        c_value=args.regularization_c,
        max_iter=args.max_iter,
    )

    ml_score_cols = {}
    for model_name in feature_sets:
        if model_name in pred_df.columns:
            score_col = f"score__{model_name}"
            df[score_col] = pred_df[model_name]
            ml_score_cols[model_name] = score_col

    model_score_cols = {**baseline_score_cols, **ml_score_cols}
    subsets = define_subsets(df)

    metrics, group_metrics = evaluate_models(
        df=df,
        model_score_cols=model_score_cols,
        target_col=args.target,
        subsets=subsets,
        include_group_rows=True,
    )
    metrics = add_delta_metrics(metrics)
    best_models = summarize_best_models(metrics)
    coef_mean = coefficient_means(coefficients)

    prediction_columns = []
    for col in IDENTIFIER_COLUMNS + [
        args.target,
        train_target,
        "group_key",
        "primary_candidate_group",
        "is_no_evidence_completed",
        "is_noncontacting_control",
        "weak_msa_warning_bool",
    ]:
        if col in df.columns and col not in prediction_columns:
            prediction_columns.append(col)
    prediction_columns += list(model_score_cols.values())

    predictions_out = df[prediction_columns].copy()

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": Path(__file__).name,
        "feature_table": str(feature_table),
        "feature_manifest": str(manifest_path),
        "out_prefix": str(out_prefix),
        "target": args.target,
        "train_target": train_target,
        "folds_requested": args.folds,
        "folds_used": cv_diag["n_splits"],
        "regularization_c": args.regularization_c,
        "max_iter": args.max_iter,
        "n_rows": int(len(df)),
        "n_chainpairs": int(df["chainpair_id"].nunique()),
        "n_query_side_groups": int(df["group_key"].nunique()),
        "target_positive_count": int(to_numeric_series(df[args.target], fill=0).sum()),
        "primary_rows": int(to_numeric_series(df["primary_candidate_group"], fill=0).gt(0).sum()),
        "model_count": int(len(model_score_cols)),
        "baseline_models": list(baseline_score_cols.keys()),
        "ml_models": list(ml_score_cols.keys()),
        "feature_set_sizes": {name: len(cols) for name, cols in feature_sets.items()},
        "leakage_guard_columns_excluded_even_if_manifest_safe": sorted(LEAKAGE_GUARD_COLUMNS),
        "cv_diagnostics": cv_diag,
        "outputs": {
            "predictions": f"{out_prefix}.predictions.tsv",
            "metrics": f"{out_prefix}.metrics.tsv",
            "group_metrics": f"{out_prefix}.group_metrics.tsv",
            "best_models": f"{out_prefix}.best_models.tsv",
            "coefficients": f"{out_prefix}.coefficients.tsv",
            "coefficients_mean": f"{out_prefix}.coefficients_mean.tsv",
            "summary": f"{out_prefix}.summary.json",
        },
    }

    predictions_out.to_csv(f"{out_prefix}.predictions.tsv", sep="\t", index=False)
    metrics.to_csv(f"{out_prefix}.metrics.tsv", sep="\t", index=False)
    group_metrics.to_csv(f"{out_prefix}.group_metrics.tsv", sep="\t", index=False)
    best_models.to_csv(f"{out_prefix}.best_models.tsv", sep="\t", index=False)
    coefficients.to_csv(f"{out_prefix}.coefficients.tsv", sep="\t", index=False)
    coef_mean.to_csv(f"{out_prefix}.coefficients_mean.tsv", sep="\t", index=False)
    with open(f"{out_prefix}.summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, default=json_default)

    print("BM5 Phase 1 patch/window ML benchmark written")
    print(f"  input rows: {len(df)}")
    print(f"  chainpairs: {df['chainpair_id'].nunique()}")
    print(f"  query-side groups: {df['group_key'].nunique()}")
    print(f"  target: {args.target}")
    print(f"  train target: {train_target}")
    print(f"  target positives: {summary['target_positive_count']}")
    print(f"  primary rows: {summary['primary_rows']}")
    print(f"  folds used: {summary['folds_used']}")
    print("  feature sets:")
    for name, cols in feature_sets.items():
        print(f"    {name}: {len(cols)}")
    print("  outputs:")
    for label, path in summary["outputs"].items():
        print(f"    {label}: {path}")

    primary = metrics[(metrics["subset"].eq("primary_ml_set")) & (metrics["target"].eq(args.target))]
    if not primary.empty:
        cols = ["model", "auprc", "roc_auc", "top_L10_recall", "top_L10_precision", "top_L5_recall", "top_L5_precision"]
        print("\nPrimary ML-set metrics:")
        print(primary[cols].sort_values("auprc", ascending=False, kind="mergesort").to_string(index=False))


if __name__ == "__main__":
    main()
