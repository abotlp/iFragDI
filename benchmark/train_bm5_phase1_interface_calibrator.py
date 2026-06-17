#!/usr/bin/env python3
"""
Train and evaluate a first BM5 Phase 1 iFragDI residue-interface calibrator.

This script is intentionally an experimental benchmark script. It does not rerun
iFragDI, does not modify the core pipeline, and does not require generated
benchmark outputs to be committed.

Main purpose:
    Learn a leakage-safe, interpretable residue-level interface-likelihood score
    from iFragDI component features, and compare it against conservation alone
    and the current manual final_score.

Default input:
    benchmark/labels/bm5_phase1_training_table.tsv

Default outputs, using --out-prefix benchmark/labels/bm5_phase1_ml_logreg:
    benchmark/labels/bm5_phase1_ml_logreg.predictions.tsv
    benchmark/labels/bm5_phase1_ml_logreg.metrics.tsv
    benchmark/labels/bm5_phase1_ml_logreg.group_metrics.tsv
    benchmark/labels/bm5_phase1_ml_logreg.diagnostics.tsv
    benchmark/labels/bm5_phase1_ml_logreg.best_models.tsv
    benchmark/labels/bm5_phase1_ml_logreg.coefficients.tsv
    benchmark/labels/bm5_phase1_ml_logreg.coefficients_mean.tsv
    benchmark/labels/bm5_phase1_ml_logreg.summary.json

Design choices:
    * Primary target: interface_5A.
    * Group-safe cross-validation by chainpair_id.
    * Primary ML set excludes no_evidence_completed and explicit noncontacting
      controls, and requires at least one positive residue per chainpair/query_side.
    * final_score is a baseline only, never an input feature.
    * Logistic-regression ablations are used first for interpretability.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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
    "score_rank",
    "evidence_class",
]

LEAKAGE_COLUMNS = [
    "final_score",  # baseline only, not a feature
    "interface_3p9A",
    "interface_5A",
    "interface_8A",
    "min_partner_atom_distance_A",
    "nearest_partner_residue_label",
    "bound_pdb",
    "bound_chain",
    "bound_sequence_index",
    "bound_residue_label",
    "bound_residue_id",
    "bound_resname",
]

BASELINE_SCORE_COLUMNS = {
    "baseline_conservation_component": "conservation_component",
    "baseline_final_score": "final_score",
    "baseline_patch_score": "patch_score",
    "baseline_ifrag_component": "ifrag_component",
    "baseline_radi_component": "radi_component",
}

CORE_NUMERIC_COLUMNS = [
    "conservation_component",
    "conservation_strength",
    "ifrag_component",
    "ifrag_strength",
    "ifrag_specificity",
    "patch_score",
    "radi_component",
    "radi_anchor",
]

EXPECTED_EVIDENCE_INPUT_COLUMNS = [
    "paired_rows_used",
    "weak_msa_warning",
    "radi_interchain_pairs_retained",
    "radi_matrix_nonzero",
    "anchor_matrix_nonzero",
    "radi_matrix_max",
    "anchor_matrix_max",
    "ifrag_fraction_nonzero",
]

EVIDENCE_NUMERIC_COLUMNS = [
    "paired_rows_used",
    "radi_interchain_pairs_retained",
    "radi_matrix_nonzero",
    "anchor_matrix_nonzero",
    "radi_matrix_max",
    "anchor_matrix_max",
    "ifrag_fraction_nonzero",
    "weak_msa_warning_bool",
    "log1p_paired_rows_used",
]

INTERACTION_COLUMNS = [
    "inter_ifrag_x_conservation",
    "inter_patch_x_conservation",
    "inter_radi_anchor_x_patch",
    "inter_radi_anchor_x_conservation",
    "inter_radi_component_x_log1p_paired_rows",
]

LOGREG_MODEL_SPECS = {
    "logreg_conservation_only": {
        "numeric": ["conservation_component"],
        "categorical": [],
        "description": "Regularized logistic regression using conservation_component only.",
    },
    "logreg_conservation_ifrag": {
        "numeric": [
            "conservation_component",
            "conservation_strength",
            "ifrag_component",
            "ifrag_strength",
            "ifrag_specificity",
        ],
        "categorical": [],
        "description": "Conservation plus iFrag component/strength/specificity features.",
    },
    "logreg_conservation_ifrag_patch": {
        "numeric": [
            "conservation_component",
            "conservation_strength",
            "ifrag_component",
            "ifrag_strength",
            "ifrag_specificity",
            "patch_score",
        ],
        "categorical": [],
        "description": "Conservation plus iFrag plus patch_score.",
    },
    "logreg_conservation_ifrag_patch_radi": {
        "numeric": [
            "conservation_component",
            "conservation_strength",
            "ifrag_component",
            "ifrag_strength",
            "ifrag_specificity",
            "patch_score",
            "radi_component",
            "radi_anchor",
        ],
        "categorical": [],
        "description": "Conservation plus iFrag plus patch_score plus raDI features.",
    },
    "logreg_plus_evidence_quality": {
        "numeric": [
            "conservation_component",
            "conservation_strength",
            "ifrag_component",
            "ifrag_strength",
            "ifrag_specificity",
            "patch_score",
            "radi_component",
            "radi_anchor",
            *EVIDENCE_NUMERIC_COLUMNS,
        ],
        "categorical": ["evidence_class"],
        "description": "Core features plus evidence-quality numeric features and evidence_class.",
    },
    "logreg_plus_interactions": {
        "numeric": [
            "conservation_component",
            "conservation_strength",
            "ifrag_component",
            "ifrag_strength",
            "ifrag_specificity",
            "patch_score",
            "radi_component",
            "radi_anchor",
            *EVIDENCE_NUMERIC_COLUMNS,
            *INTERACTION_COLUMNS,
        ],
        "categorical": ["evidence_class"],
        "description": "Evidence-quality model plus explicit biological interaction terms.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train/evaluate a BM5 Phase 1 residue-level iFragDI interface "
            "calibrator using leakage-safe GroupKFold by chainpair_id."
        )
    )
    parser.add_argument(
        "--training-table",
        default="benchmark/labels/bm5_phase1_training_table.tsv",
        help="Merged BM5 Phase 1 residue feature/label table.",
    )
    parser.add_argument(
        "--out-prefix",
        default="benchmark/labels/bm5_phase1_ml_logreg",
        help="Output prefix for predictions, metrics, diagnostics, and summary.",
    )
    parser.add_argument(
        "--target",
        default="interface_5A",
        help="Binary target column. Default: interface_5A.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of GroupKFold folds. Reduced automatically if too high.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=13,
        help="Random seed passed to deterministic-compatible sklearn estimators.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=5000,
        help="Maximum iterations for LogisticRegression.",
    )
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Inverse regularization strength for LogisticRegression.",
    )
    parser.add_argument(
        "--noncontacting-controls",
        default=",".join(NONCONTACTING_CONTROLS_DEFAULT),
        help=(
            "Comma-separated chainpair_id values treated as explicit "
            "noncontacting diagnostic controls."
        ),
    )
    parser.add_argument(
        "--primary-include-no-evidence",
        action="store_true",
        help=(
            "Include no_evidence_completed rows in the primary ML training set. "
            "Default is to keep them diagnostic only."
        ),
    )
    parser.add_argument(
        "--primary-include-noncontacting-controls",
        action="store_true",
        help=(
            "Include explicit noncontacting controls in the primary ML training set. "
            "Default is to keep them diagnostic only."
        ),
    )
    return parser.parse_args()


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(2)


def warn(message: str) -> None:
    print(f"WARNING: {message}", file=sys.stderr)


def as_bool_series(series: pd.Series) -> pd.Series:
    """Convert mixed bool/string/numeric values into boolean values."""
    if series.dtype == bool:
        return series.fillna(False)
    lowered = series.astype(str).str.strip().str.lower()
    truthy = {"true", "t", "1", "yes", "y"}
    return lowered.isin(truthy)


def to_numeric_inplace(df: pd.DataFrame, columns: Sequence[str]) -> None:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def require_columns(df: pd.DataFrame, columns: Sequence[str], context: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        fail(f"Missing required {context} columns: {', '.join(missing)}")


def add_derived_features(df: pd.DataFrame) -> None:
    """Add leakage-safe derived features from prediction-side features only."""
    if "weak_msa_warning" in df.columns:
        df["weak_msa_warning_bool"] = as_bool_series(df["weak_msa_warning"]).astype(int)
    else:
        df["weak_msa_warning_bool"] = 0

    if "paired_rows_used" not in df.columns:
        df["paired_rows_used"] = 0.0

    df["log1p_paired_rows_used"] = np.log1p(
        pd.to_numeric(df["paired_rows_used"], errors="coerce").fillna(0.0).clip(lower=0.0)
    )

    for col in set(CORE_NUMERIC_COLUMNS + EVIDENCE_NUMERIC_COLUMNS):
        if col not in df.columns:
            df[col] = 0.0

    to_numeric_inplace(df, list(set(CORE_NUMERIC_COLUMNS + EVIDENCE_NUMERIC_COLUMNS)))

    df["inter_ifrag_x_conservation"] = (
        df["ifrag_component"].fillna(0.0) * df["conservation_component"].fillna(0.0)
    )
    df["inter_patch_x_conservation"] = (
        df["patch_score"].fillna(0.0) * df["conservation_component"].fillna(0.0)
    )
    df["inter_radi_anchor_x_patch"] = (
        df["radi_anchor"].fillna(0.0) * df["patch_score"].fillna(0.0)
    )
    df["inter_radi_anchor_x_conservation"] = (
        df["radi_anchor"].fillna(0.0) * df["conservation_component"].fillna(0.0)
    )
    df["inter_radi_component_x_log1p_paired_rows"] = (
        df["radi_component"].fillna(0.0) * df["log1p_paired_rows_used"].fillna(0.0)
    )


def add_group_annotations(
    df: pd.DataFrame,
    target: str,
    noncontacting_controls: Sequence[str],
    primary_include_no_evidence: bool,
    primary_include_noncontacting: bool,
) -> pd.DataFrame:
    """Add group-level metadata and primary/diagnostic masks."""
    df["group_key"] = df["chainpair_id"].astype(str) + "||" + df["query_side"].astype(str)

    target_numeric = pd.to_numeric(df[target], errors="coerce")
    df[target] = target_numeric
    valid_target = target_numeric.isin([0, 1])
    df["valid_target"] = valid_target

    group_target_sum = df.loc[valid_target].groupby("group_key")[target].sum()
    df["group_positive_count"] = df["group_key"].map(group_target_sum).fillna(0).astype(float)
    df["group_has_positive"] = df["group_positive_count"] > 0

    df["is_noncontacting_control"] = df["chainpair_id"].astype(str).isin(set(noncontacting_controls))
    df["is_no_evidence_completed"] = df["evidence_class"].astype(str).eq("no_evidence_completed")

    # Group-level raDI/evidence annotations. These are used to define whole-group
    # diagnostic subsets rather than selecting only anchor-positive residues.
    group_stats = df.groupby("group_key").agg(
        group_has_radi_anchor=("radi_anchor", lambda s: bool(pd.to_numeric(s, errors="coerce").fillna(0).gt(0).any())),
        group_has_radi_pairs=(
            "radi_interchain_pairs_retained",
            lambda s: bool(pd.to_numeric(s, errors="coerce").fillna(0).gt(0).any()),
        ),
        group_paired_rows_used=("paired_rows_used", lambda s: float(pd.to_numeric(s, errors="coerce").fillna(0).max())),
        group_weak_msa_warning=("weak_msa_warning_bool", lambda s: bool(pd.to_numeric(s, errors="coerce").fillna(0).gt(0).any())),
    )

    for col in group_stats.columns:
        df[col] = df["group_key"].map(group_stats[col])

    primary_candidate = valid_target & df["group_has_positive"]

    if not primary_include_no_evidence:
        primary_candidate &= ~df["is_no_evidence_completed"]

    if not primary_include_noncontacting:
        primary_candidate &= ~df["is_noncontacting_control"]

    df["primary_ml_set"] = primary_candidate

    primary_group_rows = group_stats.loc[
        group_stats.index.isin(df.loc[df["primary_ml_set"], "group_key"].unique())
    ]
    paired_values = primary_group_rows["group_paired_rows_used"].astype(float)
    nonzero_paired = paired_values[paired_values > 0]
    if len(nonzero_paired) > 0:
        high_threshold = float(nonzero_paired.quantile(0.75))
    else:
        high_threshold = float("nan")

    df["high_paired_rows_q75_threshold"] = high_threshold
    if math.isnan(high_threshold):
        df["group_high_paired_rows_q75"] = False
    else:
        df["group_high_paired_rows_q75"] = df["group_paired_rows_used"].astype(float) >= high_threshold

    return group_stats


def compute_group_balanced_class_weights(y: pd.Series, group_keys: pd.Series) -> np.ndarray:
    """
    Compute sample weights so each chainpair/query_side group contributes equal
    total weight, then balance positive and negative classes under those weights.
    """
    y_int = y.astype(int).to_numpy()
    group_counts = group_keys.value_counts()
    base = group_keys.map(lambda g: 1.0 / float(group_counts[g])).astype(float).to_numpy()

    # Balance classes after group balancing.
    weighted_sum_by_class = {}
    for cls in (0, 1):
        weighted_sum_by_class[cls] = float(base[y_int == cls].sum())

    multipliers = {}
    total_base = float(base.sum())
    for cls in (0, 1):
        if weighted_sum_by_class[cls] > 0:
            multipliers[cls] = total_base / (2.0 * weighted_sum_by_class[cls])
        else:
            multipliers[cls] = 1.0

    weights = np.array([base[i] * multipliers[int(y_int[i])] for i in range(len(y_int))], dtype=float)

    mean_weight = float(np.nanmean(weights)) if len(weights) else 1.0
    if mean_weight > 0:
        weights = weights / mean_weight
    return weights


def make_onehot_encoder() -> OneHotEncoder:
    """Return a dense OneHotEncoder compatible with old and new sklearn."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor(numeric_features: Sequence[str], categorical_features: Sequence[str]) -> ColumnTransformer:
    transformers = []
    if numeric_features:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="constant", fill_value=0.0)),
                        ("scale", StandardScaler()),
                    ]
                ),
                list(numeric_features),
            )
        )
    if categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
                        ("onehot", make_onehot_encoder()),
                    ]
                ),
                list(categorical_features),
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)


def make_logreg_pipeline(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
    C: float,
    max_iter: int,
    random_seed: int,
) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", make_preprocessor(numeric_features, categorical_features)),
            (
                "logreg",
                LogisticRegression(
                    C=C,
                    solver="lbfgs",
                    max_iter=max_iter,
                    random_state=random_seed,
                ),
            ),
        ]
    )


def get_feature_names_from_pipeline(pipe: Pipeline) -> List[str]:
    preprocess = pipe.named_steps["preprocess"]
    try:
        return list(preprocess.get_feature_names_out())
    except Exception:
        # Fallback for older/newer sklearn edge cases.
        names = []
        for name, transformer, cols in preprocess.transformers_:
            if name == "remainder":
                continue
            if hasattr(transformer, "get_feature_names_out"):
                try:
                    names.extend(list(transformer.get_feature_names_out(cols)))
                    continue
                except Exception:
                    pass
            names.extend(list(cols))
        return names


def safe_metrics(y_true: np.ndarray, score: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y_true)
    s = np.asarray(score, dtype=float)
    ok = np.isfinite(s) & np.isfinite(y)
    y = y[ok].astype(int)
    s = s[ok]

    n = int(len(y))
    pos = int(np.sum(y == 1))
    neg = int(np.sum(y == 0))
    unique_scores = int(len(np.unique(s))) if n else 0

    out: Dict[str, float] = {
        "n_rows_scored": n,
        "n_positive": pos,
        "n_negative": neg,
        "score_min": float(np.nanmin(s)) if n else np.nan,
        "score_max": float(np.nanmax(s)) if n else np.nan,
        "score_mean": float(np.nanmean(s)) if n else np.nan,
        "constant_score": bool(unique_scores <= 1) if n else True,
        "auprc": np.nan,
        "roc_auc": np.nan,
        "brier": np.nan,
        "mcc_0p5": np.nan,
        "best_f1": np.nan,
        "best_f1_threshold": np.nan,
    }

    if n == 0:
        return out

    clipped = np.clip(s, 0.0, 1.0)
    try:
        out["brier"] = float(brier_score_loss(y, clipped))
    except Exception:
        out["brier"] = np.nan

    if pos > 0:
        try:
            out["auprc"] = float(average_precision_score(y, s))
        except Exception:
            out["auprc"] = np.nan

    if pos > 0 and neg > 0:
        try:
            out["roc_auc"] = float(roc_auc_score(y, s))
        except Exception:
            out["roc_auc"] = np.nan

        try:
            pred = (s >= 0.5).astype(int)
            out["mcc_0p5"] = float(matthews_corrcoef(y, pred))
        except Exception:
            out["mcc_0p5"] = np.nan

        try:
            precision, recall, thresholds = precision_recall_curve(y, s)
            denom = precision + recall
            f1 = np.divide(2.0 * precision * recall, denom, out=np.zeros_like(denom), where=denom > 0)
            best_idx = int(np.nanargmax(f1))
            out["best_f1"] = float(f1[best_idx])
            if best_idx == 0:
                out["best_f1_threshold"] = float("-inf")
            elif best_idx - 1 < len(thresholds):
                out["best_f1_threshold"] = float(thresholds[best_idx - 1])
        except Exception:
            out["best_f1"] = np.nan
            out["best_f1_threshold"] = np.nan

    return out


def safe_nanmean(values: pd.Series) -> float:
    """Return a NaN-safe mean without emitting RuntimeWarning on all-NaN slices."""
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric[np.isfinite(numeric)]
    if numeric.empty:
        return float("nan")
    return float(numeric.mean())


def topk_metrics_for_group(group: pd.DataFrame, score_col: str, target: str) -> Dict[str, float]:
    cols = [score_col, target]
    has_score_residue_index = "score_residue_index" in group.columns
    if has_score_residue_index:
        cols.append("score_residue_index")

    valid = group[cols].copy()
    # Deterministic fallback from the original group row order. This is used
    # when score_residue_index is absent, and also for rows where it cannot be
    # parsed numerically.
    valid["_fallback_tie_order"] = np.arange(len(valid), dtype=float)

    valid[score_col] = pd.to_numeric(valid[score_col], errors="coerce")
    valid[target] = pd.to_numeric(valid[target], errors="coerce")
    if has_score_residue_index:
        valid["score_residue_index"] = pd.to_numeric(valid["score_residue_index"], errors="coerce")
        valid["_topk_tie_order"] = valid["score_residue_index"].fillna(valid["_fallback_tie_order"])
    else:
        valid["score_residue_index"] = valid["_fallback_tie_order"]
        valid["_topk_tie_order"] = valid["_fallback_tie_order"]

    valid = valid[valid[score_col].notna() & valid[target].isin([0, 1])]

    n = int(len(valid))
    positives = int(valid[target].sum())
    result: Dict[str, float] = {
        "n_residues": n,
        "n_positive": positives,
        "positive_fraction": float(positives / n) if n else np.nan,
    }

    if n == 0:
        for label in ("L10", "L5"):
            result[f"top_{label}_n"] = 0
            result[f"top_{label}_hits"] = 0
            result[f"top_{label}_precision"] = np.nan
            result[f"top_{label}_recall"] = np.nan
            result[f"top_{label}_enrichment"] = np.nan
        return result

    ranked = valid.sort_values(
        [score_col, "_topk_tie_order"],
        ascending=[False, True],
        kind="mergesort",
    )

    for label, divisor in (("L10", 10.0), ("L5", 5.0)):
        k = max(1, int(math.ceil(n / divisor)))
        top = ranked.head(k)
        hits = int(top[target].sum())
        precision = float(hits / k) if k else np.nan
        recall = float(hits / positives) if positives > 0 else np.nan
        random_precision = float(positives / n) if n else np.nan
        enrichment = float(precision / random_precision) if random_precision and random_precision > 0 else np.nan

        result[f"top_{label}_n"] = k
        result[f"top_{label}_hits"] = hits
        result[f"top_{label}_precision"] = precision
        result[f"top_{label}_recall"] = recall
        result[f"top_{label}_enrichment"] = enrichment

    return result


def build_subset_masks(df: pd.DataFrame) -> Dict[str, pd.Series]:
    masks: Dict[str, pd.Series] = {}
    masks["all_rows"] = pd.Series(True, index=df.index)
    masks["primary_ml_set"] = df["primary_ml_set"].astype(bool)
    masks["diagnostic_no_evidence_completed"] = df["is_no_evidence_completed"].astype(bool)
    masks["diagnostic_noncontacting_controls"] = df["is_noncontacting_control"].astype(bool)
    masks["diagnostic_not_primary"] = ~df["primary_ml_set"].astype(bool)
    masks["groups_with_radi_anchor"] = df["group_has_radi_anchor"].astype(bool)
    masks["groups_with_radi_pairs"] = df["group_has_radi_pairs"].astype(bool)
    masks["groups_high_paired_rows_q75"] = df["group_high_paired_rows_q75"].astype(bool)
    masks["groups_weak_msa_warning_true"] = df["group_weak_msa_warning"].astype(bool)
    masks["groups_weak_msa_warning_false"] = ~df["group_weak_msa_warning"].astype(bool)

    for ev in sorted(df["evidence_class"].dropna().astype(str).unique()):
        clean = ev.replace(" ", "_").replace("/", "_")
        masks[f"evidence_class={clean}"] = df["evidence_class"].astype(str).eq(ev)

    return masks


def make_diagnostics_table(df: pd.DataFrame, subset_masks: Mapping[str, pd.Series], target: str) -> pd.DataFrame:
    rows = []
    for subset_name, mask in subset_masks.items():
        sub = df.loc[mask & df["valid_target"]].copy()
        n_rows = int(len(sub))
        groups = sub["group_key"].nunique() if n_rows else 0
        chainpairs = sub["chainpair_id"].nunique() if n_rows else 0
        positives = int(sub[target].sum()) if n_rows else 0
        negatives = int(n_rows - positives)
        rows.append(
            {
                "subset": subset_name,
                "n_rows": n_rows,
                "n_chainpairs": int(chainpairs),
                "n_query_side_groups": int(groups),
                "n_positive": positives,
                "n_negative": negatives,
                "positive_fraction": float(positives / n_rows) if n_rows else np.nan,
                "mean_paired_rows_used": float(pd.to_numeric(sub.get("paired_rows_used", pd.Series(dtype=float)), errors="coerce").mean()) if n_rows else np.nan,
                "max_paired_rows_used": float(pd.to_numeric(sub.get("paired_rows_used", pd.Series(dtype=float)), errors="coerce").max()) if n_rows else np.nan,
                "radi_anchor_positive_rows": int(pd.to_numeric(sub.get("radi_anchor", pd.Series(dtype=float)), errors="coerce").fillna(0).gt(0).sum()) if n_rows else 0,
                "radi_pairs_positive_rows": int(pd.to_numeric(sub.get("radi_interchain_pairs_retained", pd.Series(dtype=float)), errors="coerce").fillna(0).gt(0).sum()) if n_rows else 0,
            }
        )
    return pd.DataFrame(rows)


def summarize_topk_by_subset(
    group_metrics: pd.DataFrame,
    df: pd.DataFrame,
    subset_masks: Mapping[str, pd.Series],
    model_name: str,
) -> Dict[str, Dict[str, float]]:
    """
    Summarize top-k metrics for a model over each diagnostic subset.

    Only whole groups represented in a subset are considered. For row-level masks,
    this uses groups that have at least one row in the subset, then reads the group
    metric computed on the full group.
    """
    output: Dict[str, Dict[str, float]] = {}
    gm = group_metrics[group_metrics["model"] == model_name].copy()
    gm = gm.set_index("group_key", drop=False)

    for subset_name, mask in subset_masks.items():
        group_keys = sorted(df.loc[mask, "group_key"].dropna().astype(str).unique())
        sub_gm = gm.loc[gm.index.intersection(group_keys)].copy()
        if sub_gm.empty:
            output[subset_name] = {
                "n_topk_groups": 0,
                "top_L10_recall_mean": np.nan,
                "top_L10_precision_mean": np.nan,
                "top_L10_enrichment_mean": np.nan,
                "top_L5_recall_mean": np.nan,
                "top_L5_precision_mean": np.nan,
                "top_L5_enrichment_mean": np.nan,
            }
            continue

        # Recall/enrichment are undefined for zero-positive groups. Use a
        # helper so all-NaN diagnostic subsets do not emit noisy RuntimeWarnings.
        output[subset_name] = {
            "n_topk_groups": int(len(sub_gm)),
            "n_topk_positive_groups": int(sub_gm["n_positive"].gt(0).sum()),
            "top_L10_recall_mean": safe_nanmean(sub_gm["top_L10_recall"]),
            "top_L10_precision_mean": safe_nanmean(sub_gm["top_L10_precision"]),
            "top_L10_enrichment_mean": safe_nanmean(sub_gm["top_L10_enrichment"]),
            "top_L5_recall_mean": safe_nanmean(sub_gm["top_L5_recall"]),
            "top_L5_precision_mean": safe_nanmean(sub_gm["top_L5_precision"]),
            "top_L5_enrichment_mean": safe_nanmean(sub_gm["top_L5_enrichment"]),
        }

    return output


def train_oof_logreg_models(
    df: pd.DataFrame,
    args: argparse.Namespace,
    model_specs: Mapping[str, Mapping[str, Sequence[str]]],
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], Dict[str, object]]:
    """
    Train logistic-regression ablations with GroupKFold and return:
      * df with pred_<model> columns
      * coefficients dataframe
      * prediction column names
      * CV summary dictionary
    """
    primary = df["primary_ml_set"].astype(bool) & df["valid_target"].astype(bool)
    primary_groups = sorted(df.loc[primary, "chainpair_id"].astype(str).unique())
    all_groups = sorted(df.loc[df["valid_target"], "chainpair_id"].astype(str).unique())

    if len(primary_groups) < 2:
        fail("Fewer than two primary chainpair_id groups are available for GroupKFold.")

    n_splits = min(int(args.folds), len(primary_groups))
    if n_splits < 2:
        fail("At least two folds are required.")
    if n_splits != int(args.folds):
        warn(f"Reducing folds from {args.folds} to {n_splits} because only {len(primary_groups)} primary groups exist.")

    # Use GroupKFold on all valid rows so diagnostic-only groups also receive
    # out-of-fold predictions. Training rows are still restricted to primary_ml_set.
    valid_idx = df.index[df["valid_target"].astype(bool)]
    valid_df = df.loc[valid_idx].copy()
    groups = valid_df["chainpair_id"].astype(str)

    cv = GroupKFold(n_splits=n_splits)
    fold_assignments = pd.Series(index=df.index, data=np.nan, dtype=float)

    coeff_rows = []
    prediction_columns = []
    fold_summaries = []

    for model_name in model_specs:
        pred_col = f"pred_{model_name}"
        df[pred_col] = np.nan
        prediction_columns.append(pred_col)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        for fold_idx, (train_pos, test_pos) in enumerate(cv.split(valid_df, valid_df[args.target], groups=groups), start=1):
            raw_train_idx = valid_df.index[train_pos]
            test_idx = valid_df.index[test_pos]

            train_mask = df.index.isin(raw_train_idx) & df["primary_ml_set"].astype(bool) & df["valid_target"].astype(bool)
            train_idx = df.index[train_mask]
            y_train = df.loc[train_idx, args.target].astype(int)

            fold_assignments.loc[test_idx] = fold_idx

            train_groups = sorted(df.loc[train_idx, "chainpair_id"].astype(str).unique())
            test_groups = sorted(df.loc[test_idx, "chainpair_id"].astype(str).unique())
            fold_summaries.append(
                {
                    "fold": fold_idx,
                    "n_train_rows_primary": int(len(train_idx)),
                    "n_train_groups_primary": int(len(train_groups)),
                    "n_test_rows_all": int(len(test_idx)),
                    "n_test_groups_all": int(len(test_groups)),
                    "train_positive": int(y_train.sum()) if len(y_train) else 0,
                    "train_negative": int(len(y_train) - y_train.sum()) if len(y_train) else 0,
                }
            )

            if len(y_train.unique()) < 2:
                warn(f"Fold {fold_idx} has only one class in primary training rows; model predictions skipped for this fold.")
                continue

            sample_weight = compute_group_balanced_class_weights(
                y_train,
                df.loc[train_idx, "group_key"].astype(str),
            )

            for model_name, spec in model_specs.items():
                numeric_features = list(spec["numeric"])
                categorical_features = list(spec["categorical"])

                required = numeric_features + categorical_features
                missing = [col for col in required if col not in df.columns]
                if missing:
                    fail(f"Model {model_name} references missing columns: {', '.join(missing)}")

                pipe = make_logreg_pipeline(
                    numeric_features=numeric_features,
                    categorical_features=categorical_features,
                    C=float(args.C),
                    max_iter=int(args.max_iter),
                    random_seed=int(args.random_seed),
                )
                pipe.fit(
                    df.loc[train_idx, required],
                    y_train,
                    logreg__sample_weight=sample_weight,
                )

                pred_col = f"pred_{model_name}"
                probabilities = pipe.predict_proba(df.loc[test_idx, required])[:, 1]
                df.loc[test_idx, pred_col] = probabilities

                feature_names = get_feature_names_from_pipeline(pipe)
                coefs = pipe.named_steps["logreg"].coef_.ravel()
                intercept = float(pipe.named_steps["logreg"].intercept_[0])
                coeff_rows.append(
                    {
                        "model": model_name,
                        "fold": fold_idx,
                        "feature": "__intercept__",
                        "coefficient": intercept,
                    }
                )
                for feature, coef in zip(feature_names, coefs):
                    coeff_rows.append(
                        {
                            "model": model_name,
                            "fold": fold_idx,
                            "feature": feature,
                            "coefficient": float(coef),
                        }
                    )

    df["cv_fold"] = fold_assignments

    coeff_df = pd.DataFrame(coeff_rows)
    if not coeff_df.empty:
        summary_rows = []
        for (model, feature), sub in coeff_df.groupby(["model", "feature"], dropna=False):
            summary_rows.append(
                {
                    "model": model,
                    "fold": "mean",
                    "feature": feature,
                    "coefficient": float(sub["coefficient"].mean()),
                    "coefficient_sd": float(sub["coefficient"].std(ddof=1)) if len(sub) > 1 else 0.0,
                    "n_folds": int(len(sub)),
                }
            )
        coeff_df = pd.concat([coeff_df, pd.DataFrame(summary_rows)], ignore_index=True, sort=False)

    cv_summary = {
        "requested_folds": int(args.folds),
        "used_folds": int(n_splits),
        "primary_chainpair_group_count": int(len(primary_groups)),
        "all_valid_chainpair_group_count": int(len(all_groups)),
        "folds": fold_summaries,
    }

    return df, coeff_df, prediction_columns, cv_summary


def add_baseline_prediction_columns(df: pd.DataFrame) -> List[str]:
    prediction_cols = []
    for baseline_name, source_col in BASELINE_SCORE_COLUMNS.items():
        require_columns(df, [source_col], f"baseline score for {baseline_name}")
        out_col = f"score_{baseline_name}"
        df[out_col] = pd.to_numeric(df[source_col], errors="coerce")
        prediction_cols.append(out_col)
    return prediction_cols


def make_group_metrics(df: pd.DataFrame, score_columns: Mapping[str, str], target: str) -> pd.DataFrame:
    rows = []
    group_meta_cols = [
        "chainpair_id",
        "query_side",
        "evidence_class",
        "primary_ml_set",
        "is_no_evidence_completed",
        "is_noncontacting_control",
        "group_has_radi_anchor",
        "group_has_radi_pairs",
        "group_paired_rows_used",
        "group_weak_msa_warning",
        "group_high_paired_rows_q75",
    ]

    for model_name, score_col in score_columns.items():
        for group_key, group in df.groupby("group_key", sort=True):
            metrics = topk_metrics_for_group(group, score_col, target)
            meta = {"group_key": group_key, "model": model_name}
            for col in group_meta_cols:
                if col in group.columns:
                    vals = group[col].dropna().unique()
                    meta[col] = vals[0] if len(vals) else np.nan
            rows.append({**meta, **metrics})
    return pd.DataFrame(rows)


def make_metrics_table(
    df: pd.DataFrame,
    score_columns: Mapping[str, str],
    group_metrics: pd.DataFrame,
    subset_masks: Mapping[str, pd.Series],
    target: str,
) -> pd.DataFrame:
    topk_by_model = {
        model_name: summarize_topk_by_subset(group_metrics, df, subset_masks, model_name)
        for model_name in score_columns
    }

    rows = []
    for subset_name, mask in subset_masks.items():
        sub = df.loc[mask & df["valid_target"]].copy()
        y = sub[target].astype(int).to_numpy() if not sub.empty else np.array([], dtype=int)

        for model_name, score_col in score_columns.items():
            score = sub[score_col].to_numpy(dtype=float) if not sub.empty and score_col in sub.columns else np.array([], dtype=float)
            base = safe_metrics(y, score)
            top = topk_by_model.get(model_name, {}).get(subset_name, {})
            rows.append(
                {
                    "subset": subset_name,
                    "model": model_name,
                    **base,
                    **top,
                }
            )

    metrics = add_metric_delta_columns(pd.DataFrame(rows))
    preferred_cols = [
        "subset",
        "model",
        "n_rows_scored",
        "n_positive",
        "n_negative",
        "auprc",
        "roc_auc",
        "brier",
        "mcc_0p5",
        "best_f1",
        "best_f1_threshold",
        "n_topk_groups",
        "n_topk_positive_groups",
        "top_L10_recall_mean",
        "top_L10_precision_mean",
        "top_L10_enrichment_mean",
        "top_L5_recall_mean",
        "top_L5_precision_mean",
        "top_L5_enrichment_mean",
        "delta_auprc_vs_conservation_component",
        "delta_auprc_vs_final_score",
        "delta_auprc_vs_logreg_conservation_ifrag_patch",
        "delta_top_L10_recall_mean_vs_conservation_component",
        "delta_top_L10_recall_mean_vs_final_score",
        "delta_top_L10_recall_mean_vs_logreg_conservation_ifrag_patch",
        "delta_top_L5_recall_mean_vs_conservation_component",
        "delta_top_L5_recall_mean_vs_final_score",
        "delta_top_L5_recall_mean_vs_logreg_conservation_ifrag_patch",
        "score_min",
        "score_max",
        "score_mean",
        "constant_score",
    ]
    existing = [col for col in preferred_cols if col in metrics.columns]
    rest = [col for col in metrics.columns if col not in existing]
    return metrics[existing + rest]


def add_metric_delta_columns(metrics: pd.DataFrame) -> pd.DataFrame:
    """Add per-subset delta columns against key baseline/reference models."""
    metrics = metrics.copy()
    reference_models = {
        "conservation_component": "baseline_conservation_component",
        "final_score": "baseline_final_score",
        "logreg_conservation_ifrag_patch": "logreg_conservation_ifrag_patch",
    }
    metric_cols = [
        "auprc",
        "roc_auc",
        "top_L10_recall_mean",
        "top_L10_precision_mean",
        "top_L10_enrichment_mean",
        "top_L5_recall_mean",
        "top_L5_precision_mean",
        "top_L5_enrichment_mean",
    ]

    for ref_label in reference_models:
        for metric_col in metric_cols:
            if metric_col in metrics.columns:
                metrics[f"delta_{metric_col}_vs_{ref_label}"] = np.nan

    for _, sub in metrics.groupby("subset", sort=False):
        subset_idx = sub.index
        for ref_label, ref_model in reference_models.items():
            ref_rows = sub[sub["model"].eq(ref_model)]
            if ref_rows.empty:
                continue
            ref_row = ref_rows.iloc[0]
            for metric_col in metric_cols:
                if metric_col not in metrics.columns:
                    continue
                ref_value = ref_row.get(metric_col, np.nan)
                if pd.isna(ref_value):
                    continue
                metrics.loc[subset_idx, f"delta_{metric_col}_vs_{ref_label}"] = (
                    pd.to_numeric(metrics.loc[subset_idx, metric_col], errors="coerce") - float(ref_value)
                )

    return metrics


def make_best_models_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """Return the best model per subset under several ranking metrics."""
    rows = []
    ranking_metrics = [
        "auprc",
        "top_L10_recall_mean",
        "top_L5_recall_mean",
        "top_L10_precision_mean",
        "top_L5_precision_mean",
    ]
    context_cols = [
        "n_rows_scored",
        "n_positive",
        "n_negative",
        "auprc",
        "roc_auc",
        "top_L10_recall_mean",
        "top_L10_precision_mean",
        "top_L5_recall_mean",
        "top_L5_precision_mean",
        "delta_auprc_vs_conservation_component",
        "delta_auprc_vs_final_score",
        "delta_auprc_vs_logreg_conservation_ifrag_patch",
        "delta_top_L10_recall_mean_vs_conservation_component",
        "delta_top_L5_recall_mean_vs_conservation_component",
    ]

    for subset_name, sub in metrics.groupby("subset", sort=False):
        for ranking_metric in ranking_metrics:
            if ranking_metric not in sub.columns:
                continue
            ranked = sub[pd.to_numeric(sub[ranking_metric], errors="coerce").notna()].copy()
            if ranked.empty:
                continue
            ranked[ranking_metric] = pd.to_numeric(ranked[ranking_metric], errors="coerce")
            ranked = ranked.sort_values(
                by=[ranking_metric, "auprc", "model"],
                ascending=[False, False, True],
                na_position="last",
            )
            best = ranked.iloc[0]
            row = {
                "subset": subset_name,
                "selection_metric": ranking_metric,
                "best_model": best["model"],
                "best_value": float(best[ranking_metric]),
            }
            for col in context_cols:
                if col in best.index:
                    row[col] = best[col]
            rows.append(row)

    return pd.DataFrame(rows)


def prepare_predictions_output(df: pd.DataFrame, score_columns: Mapping[str, str], target: str) -> pd.DataFrame:
    output_cols = []
    for col in IDENTIFIER_COLUMNS:
        if col in df.columns:
            output_cols.append(col)

    extra_cols = [
        target,
        "interface_3p9A",
        "interface_8A",
        "primary_ml_set",
        "is_no_evidence_completed",
        "is_noncontacting_control",
        "group_key",
        "group_positive_count",
        "cv_fold",
    ]
    output_cols.extend([col for col in extra_cols if col in df.columns and col not in output_cols])
    output_cols.extend([col for col in score_columns.values() if col in df.columns and col not in output_cols])

    return df[output_cols].copy()


def json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if np.isnan(obj):
            return None
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if pd.isna(obj):
        return None
    raise TypeError(f"Object of type {type(obj)!r} is not JSON serializable")


def main() -> None:
    args = parse_args()

    training_table = Path(args.training_table)
    if not training_table.exists():
        fail(f"Training table does not exist: {training_table}")

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(training_table, sep="\t", low_memory=False)

    require_columns(
        df,
        ["chainpair_id", "query_side", "evidence_class", args.target],
        "training table",
    )

    # Ensure required feature/baseline columns exist before doing any expensive work.
    required_feature_cols = set()
    for spec in LOGREG_MODEL_SPECS.values():
        required_feature_cols.update(spec["numeric"])
        required_feature_cols.update(spec["categorical"])
    # Derived columns are created below and should not be required in the input.
    input_required = sorted(
        required_feature_cols
        - set(EVIDENCE_NUMERIC_COLUMNS)
        - set(INTERACTION_COLUMNS)
        - {"weak_msa_warning_bool", "log1p_paired_rows_used"}
    )
    require_columns(df, input_required, "input feature")
    require_columns(df, list(BASELINE_SCORE_COLUMNS.values()), "baseline")

    if "final_score" in required_feature_cols:
        fail("final_score must never be listed as an input feature. Use it only as a baseline.")

    forbidden_as_feature = sorted(set(required_feature_cols).intersection(LEAKAGE_COLUMNS))
    if forbidden_as_feature:
        fail(f"Potential leakage columns were included as features: {', '.join(forbidden_as_feature)}")

    df["evidence_class"] = df["evidence_class"].fillna("missing").astype(str)

    missing_evidence_inputs = [
        col for col in EXPECTED_EVIDENCE_INPUT_COLUMNS if col not in df.columns
    ]
    if missing_evidence_inputs:
        warn(
            "Missing expected evidence-quality input columns; they will be "
            "zero-filled where needed: " + ", ".join(missing_evidence_inputs)
        )

    all_numeric = sorted(
        set(CORE_NUMERIC_COLUMNS)
        | set(EVIDENCE_NUMERIC_COLUMNS)
        | set(INTERACTION_COLUMNS)
        | set(BASELINE_SCORE_COLUMNS.values())
        | {args.target}
    )
    to_numeric_inplace(df, all_numeric)

    add_derived_features(df)

    noncontacting_controls = [
        item.strip() for item in str(args.noncontacting_controls).split(",") if item.strip()
    ]

    group_stats = add_group_annotations(
        df,
        target=args.target,
        noncontacting_controls=noncontacting_controls,
        primary_include_no_evidence=bool(args.primary_include_no_evidence),
        primary_include_noncontacting=bool(args.primary_include_noncontacting_controls),
    )

    primary = df["primary_ml_set"].astype(bool) & df["valid_target"].astype(bool)
    primary_rows = int(primary.sum())
    primary_groups = int(df.loc[primary, "group_key"].nunique())
    primary_chainpairs = int(df.loc[primary, "chainpair_id"].nunique())
    primary_pos = int(df.loc[primary, args.target].sum()) if primary_rows else 0
    primary_neg = int(primary_rows - primary_pos)

    if primary_rows == 0:
        fail("Primary ML set is empty after applying masks.")
    if primary_pos == 0 or primary_neg == 0:
        fail(
            "Primary ML set must contain both positive and negative residues. "
            f"Observed positive={primary_pos}, negative={primary_neg}."
        )

    baseline_prediction_cols = add_baseline_prediction_columns(df)

    df, coeff_df, logreg_prediction_cols, cv_summary = train_oof_logreg_models(
        df=df,
        args=args,
        model_specs=LOGREG_MODEL_SPECS,
    )

    # Model-name -> score column map.
    score_columns: Dict[str, str] = {}
    for baseline_col in baseline_prediction_cols:
        model_name = baseline_col.replace("score_", "")
        score_columns[model_name] = baseline_col
    for pred_col in logreg_prediction_cols:
        model_name = pred_col.replace("pred_", "")
        score_columns[model_name] = pred_col

    subset_masks = build_subset_masks(df)
    diagnostics_df = make_diagnostics_table(df, subset_masks, args.target)
    group_metrics_df = make_group_metrics(df, score_columns, args.target)
    metrics_df = make_metrics_table(df, score_columns, group_metrics_df, subset_masks, args.target)
    best_models_df = make_best_models_table(metrics_df)
    predictions_df = prepare_predictions_output(df, score_columns, args.target)

    predictions_path = Path(f"{out_prefix}.predictions.tsv")
    metrics_path = Path(f"{out_prefix}.metrics.tsv")
    group_metrics_path = Path(f"{out_prefix}.group_metrics.tsv")
    diagnostics_path = Path(f"{out_prefix}.diagnostics.tsv")
    best_models_path = Path(f"{out_prefix}.best_models.tsv")
    coefficients_path = Path(f"{out_prefix}.coefficients.tsv")
    coefficients_mean_path = Path(f"{out_prefix}.coefficients_mean.tsv")
    summary_path = Path(f"{out_prefix}.summary.json")

    predictions_df.to_csv(predictions_path, sep="\t", index=False)
    metrics_df.to_csv(metrics_path, sep="\t", index=False)
    group_metrics_df.to_csv(group_metrics_path, sep="\t", index=False)
    diagnostics_df.to_csv(diagnostics_path, sep="\t", index=False)
    best_models_df.to_csv(best_models_path, sep="\t", index=False)
    coeff_df.to_csv(coefficients_path, sep="\t", index=False)
    coeff_mean_df = coeff_df[coeff_df["fold"].astype(str).eq("mean")].copy() if not coeff_df.empty else coeff_df.copy()
    coeff_mean_df.to_csv(coefficients_mean_path, sep="\t", index=False)

    primary_metrics = metrics_df[metrics_df["subset"].eq("primary_ml_set")].copy()
    primary_metrics = primary_metrics.sort_values(
        by=["auprc", "top_L10_recall_mean"],
        ascending=[False, False],
        na_position="last",
    )

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": Path(__file__).name,
        "training_table": str(training_table),
        "out_prefix": str(out_prefix),
        "target": args.target,
        "n_input_rows": int(len(df)),
        "n_valid_target_rows": int(df["valid_target"].sum()),
        "n_chainpairs": int(df["chainpair_id"].nunique()),
        "n_query_side_groups": int(df["group_key"].nunique()),
        "primary_ml_set": {
            "n_rows": primary_rows,
            "n_chainpairs": primary_chainpairs,
            "n_query_side_groups": primary_groups,
            "n_positive": primary_pos,
            "n_negative": primary_neg,
            "positive_fraction": float(primary_pos / primary_rows) if primary_rows else None,
            "excluded_no_evidence_completed": not bool(args.primary_include_no_evidence),
            "excluded_noncontacting_controls": not bool(args.primary_include_noncontacting_controls),
            "noncontacting_controls": noncontacting_controls,
        },
        "cv": cv_summary,
        "baseline_score_columns": BASELINE_SCORE_COLUMNS,
        "logreg_model_specs": LOGREG_MODEL_SPECS,
        "leakage_columns_excluded_from_features": LEAKAGE_COLUMNS,
        "outputs": {
            "predictions": str(predictions_path),
            "metrics": str(metrics_path),
            "group_metrics": str(group_metrics_path),
            "diagnostics": str(diagnostics_path),
            "best_models": str(best_models_path),
            "coefficients": str(coefficients_path),
            "coefficients_mean": str(coefficients_mean_path),
            "summary": str(summary_path),
        },
        "top_primary_models_by_auprc": primary_metrics.head(10).replace({np.nan: None}).to_dict(orient="records"),
        "best_models_by_subset": best_models_df.replace({np.nan: None}).to_dict(orient="records"),
        "package_versions": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }

    try:
        import sklearn

        summary["package_versions"]["scikit_learn"] = sklearn.__version__
    except Exception:
        pass

    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, default=json_default)

    print("BM5 Phase 1 interface calibrator benchmark written")
    print(f"  input rows: {len(df)}")
    print(f"  target: {args.target}")
    print(f"  primary rows: {primary_rows}")
    print(f"  primary chainpairs: {primary_chainpairs}")
    print(f"  primary query-side groups: {primary_groups}")
    print(f"  primary positives/negatives: {primary_pos}/{primary_neg}")
    print(f"  folds used: {cv_summary['used_folds']}")
    print("  outputs:")
    print(f"    predictions:   {predictions_path}")
    print(f"    metrics:       {metrics_path}")
    print(f"    group metrics: {group_metrics_path}")
    print(f"    diagnostics:   {diagnostics_path}")
    print(f"    best models:   {best_models_path}")
    print(f"    coefficients:  {coefficients_path}")
    print(f"    coeff means:   {coefficients_mean_path}")
    print(f"    summary:       {summary_path}")

    print("\nTop primary_ml_set models by AUPRC:")
    cols_to_show = [
        "model",
        "n_rows_scored",
        "n_positive",
        "auprc",
        "roc_auc",
        "top_L10_recall_mean",
        "top_L10_precision_mean",
        "top_L5_recall_mean",
        "top_L5_precision_mean",
        "delta_auprc_vs_conservation_component",
        "delta_auprc_vs_final_score",
        "delta_auprc_vs_logreg_conservation_ifrag_patch",
    ]
    show = primary_metrics[[col for col in cols_to_show if col in primary_metrics.columns]].head(10)
    if show.empty:
        print("  No primary metrics available.")
    else:
        print(show.to_string(index=False))


if __name__ == "__main__":
    main()
