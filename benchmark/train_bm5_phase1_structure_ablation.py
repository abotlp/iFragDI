#!/usr/bin/env python3
"""
Train BM5 Phase 1 structure-aware evidence-block ablation models.

This script asks whether the structure-aware iFragDI score is genuinely using
multiple evidence sources or is mostly driven by conservation + surface exposure.
It consumes the patch + structure feature table and trains group-safe logistic
regression models with systematic evidence-block removals.

Default input:
    benchmark/labels/bm5_phase1_patch_structure_features.tsv
    benchmark/labels/bm5_phase1_patch_structure_features.feature_manifest.tsv

Default outputs with --out-prefix benchmark/labels/bm5_phase1_structure_ablation_logreg:
    *.predictions.tsv
    *.metrics.tsv
    *.group_metrics.tsv
    *.best_models.tsv
    *.coefficients.tsv
    *.coefficients_mean.tsv
    *.feature_sets.tsv
    *.summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Set

import numpy as np
import pandas as pd

try:
    import train_bm5_phase1_patch_calibrator as patch
except Exception as exc:  # pragma: no cover
    print(f"ERROR: could not import patch calibrator helpers: {exc}", file=sys.stderr)
    raise SystemExit(2)


STRUCTURE_CATEGORICAL_COLUMNS = {"struct_ss8", "struct_ss3"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train BM5 Phase 1 structure-aware evidence-block ablation models."
    )
    parser.add_argument(
        "--feature-table",
        default="benchmark/labels/bm5_phase1_patch_structure_features.tsv",
        help="Patch + structure residue feature table.",
    )
    parser.add_argument(
        "--feature-manifest",
        default="benchmark/labels/bm5_phase1_patch_structure_features.feature_manifest.tsv",
        help="Feature manifest produced by build_bm5_phase1_structure_features.py.",
    )
    parser.add_argument(
        "--out-prefix",
        default="benchmark/labels/bm5_phase1_structure_ablation_logreg",
        help="Output prefix.",
    )
    parser.add_argument("--target", default="interface_5A")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--noncontacting-controls",
        default=",".join(patch.NONCONTACTING_CONTROLS_DEFAULT),
        help="Comma-separated chainpair_id values treated as explicit noncontacting controls.",
    )
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--regularization-c", type=float, default=1.0)
    parser.add_argument(
        "--train-target",
        default=None,
        help="Optional target used for fitting ML models. Defaults to --target.",
    )
    return parser.parse_args()


def unique_keep_order(columns: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for col in columns:
        if col not in seen:
            seen.add(col)
            out.append(col)
    return out


def numeric_existing(df: pd.DataFrame, columns: Sequence[str]) -> List[str]:
    return patch.existing_numeric_columns(
        df,
        [c for c in unique_keep_order(columns) if c not in STRUCTURE_CATEGORICAL_COLUMNS],
    )


def has_radi(col: str) -> bool:
    return "radi" in col.lower()


def has_ifrag(col: str) -> bool:
    return "ifrag" in col.lower()


def has_conservation(col: str) -> bool:
    return col.startswith("conservation_") or "conservation" in col.lower()


def has_structure(col: str) -> bool:
    return col.startswith("struct_") or col.endswith("_x_struct_rsa") or "_x_struct_" in col


def has_composite_patch_score(col: str) -> bool:
    return col.startswith("patch_score")


def direct_radi(col: str) -> bool:
    if not has_radi(col):
        return False
    lowered = col.lower()
    contextual_markers = ("_x_conservation", "_x_ifrag", "_x_patch", "_x_struct")
    return not any(marker in lowered for marker in contextual_markers)


def block_labels(col: str) -> List[str]:
    labels = []
    if has_conservation(col):
        labels.append("conservation")
    if has_ifrag(col):
        labels.append("ifrag")
    if has_radi(col):
        labels.append("radi")
    if has_structure(col):
        labels.append("structure")
    if has_composite_patch_score(col):
        labels.append("composite_patch_score")
    if not labels:
        labels.append("other_safe_numeric")
    return labels


def build_ablation_feature_sets(df: pd.DataFrame, manifest: pd.DataFrame) -> Dict[str, List[str]]:
    safe_cols = patch.base_feature_safe_columns(df, manifest)
    safe_cols = [c for c in safe_cols if c not in STRUCTURE_CATEGORICAL_COLUMNS]
    safe_cols = numeric_existing(df, safe_cols)

    conservation = [c for c in safe_cols if has_conservation(c)]
    ifrag = [c for c in safe_cols if has_ifrag(c)]
    radi_all = [c for c in safe_cols if has_radi(c)]
    radi_direct = [c for c in safe_cols if direct_radi(c)]
    structure_all = [c for c in safe_cols if has_structure(c)]
    structure_base = [c for c in safe_cols if c.startswith("struct_") and c not in STRUCTURE_CATEGORICAL_COLUMNS]
    composite_patch = [c for c in safe_cols if has_composite_patch_score(c)]

    feature_sets = {
        # Production-like and clean full models.
        "ablate_full_all_safe": safe_cols,
        "ablate_full_no_composite_patch_score": [c for c in safe_cols if not has_composite_patch_score(c)],

        # Single-block removal tests.
        "ablate_no_radi_all": [c for c in safe_cols if not has_radi(c)],
        "ablate_no_direct_radi_keep_contextual_radi": [c for c in safe_cols if not direct_radi(c)],
        "ablate_no_ifrag_named_keep_composite": [c for c in safe_cols if not has_ifrag(c)],
        "ablate_no_ifrag_clean_drop_composite": [c for c in safe_cols if not has_ifrag(c) and not has_composite_patch_score(c)],
        "ablate_no_conservation_named_keep_composite": [c for c in safe_cols if not has_conservation(c)],
        "ablate_no_conservation_clean_drop_composite": [c for c in safe_cols if not has_conservation(c) and not has_composite_patch_score(c)],
        "ablate_no_structure": [c for c in safe_cols if not has_structure(c)],
        "ablate_no_composite_patch_score": [c for c in safe_cols if not has_composite_patch_score(c)],

        # Ingredient-only controls.
        "ablate_conservation_only": conservation,
        "ablate_ifrag_only": ifrag,
        "ablate_radi_direct_only": radi_direct,
        "ablate_radi_all_contextual_only": radi_all,
        "ablate_structure_base_only": structure_base,
        "ablate_structure_all_including_interactions": structure_all,

        # Biologically interpretable combinations.
        "ablate_conservation_structure": unique_keep_order(conservation + structure_base + [c for c in safe_cols if has_conservation(c) and has_structure(c)]),
        "ablate_ifrag_conservation_structure_no_radi": [c for c in safe_cols if not has_radi(c) and not has_composite_patch_score(c)],
        "ablate_ifrag_conservation_radi_no_structure": [c for c in safe_cols if not has_structure(c)],
        "ablate_structure_plus_contextual_radi_no_direct_radi": [c for c in safe_cols if has_structure(c) or (has_radi(c) and not direct_radi(c))],
    }

    return {name: numeric_existing(df, cols) for name, cols in feature_sets.items() if numeric_existing(df, cols)}


def make_feature_set_table(feature_sets: Mapping[str, Sequence[str]]) -> pd.DataFrame:
    rows = []
    for model, features in feature_sets.items():
        for feature in features:
            rows.append(
                {
                    "model": model,
                    "feature": feature,
                    "blocks": ",".join(block_labels(feature)),
                    "is_radi": int(has_radi(feature)),
                    "is_direct_radi": int(direct_radi(feature)),
                    "is_ifrag": int(has_ifrag(feature)),
                    "is_conservation": int(has_conservation(feature)),
                    "is_structure": int(has_structure(feature)),
                    "is_composite_patch_score": int(has_composite_patch_score(feature)),
                }
            )
    return pd.DataFrame(rows)


def model_block_summary(feature_sets: Mapping[str, Sequence[str]]) -> Dict[str, Dict[str, int]]:
    summary: Dict[str, Dict[str, int]] = {}
    for model, features in feature_sets.items():
        summary[model] = {
            "n_features": len(features),
            "n_conservation": sum(has_conservation(c) for c in features),
            "n_ifrag": sum(has_ifrag(c) for c in features),
            "n_radi_all": sum(has_radi(c) for c in features),
            "n_radi_direct": sum(direct_radi(c) for c in features),
            "n_structure": sum(has_structure(c) for c in features),
            "n_composite_patch_score": sum(has_composite_patch_score(c) for c in features),
        }
    return summary


def main() -> None:
    args = parse_args()
    controls = patch.parse_controls(args.noncontacting_controls)

    feature_table = Path(args.feature_table)
    manifest_path = Path(args.feature_manifest)
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    if not feature_table.exists():
        patch.fail(f"Feature table does not exist: {feature_table}")
    if not manifest_path.exists():
        patch.fail(f"Feature manifest does not exist: {manifest_path}")

    df = pd.read_csv(feature_table, sep="\t", low_memory=False)
    manifest = pd.read_csv(manifest_path, sep="\t", low_memory=False)

    for col in ["chainpair_id", "query_side", args.target]:
        if col not in df.columns:
            patch.fail(f"Missing required feature-table column: {col}")

    train_target = args.train_target or args.target
    if train_target not in df.columns:
        patch.fail(f"Training target column does not exist: {train_target}")

    df = patch.add_required_annotations(df, target=args.target, controls=controls)
    if train_target != args.target:
        df[train_target] = patch.to_numeric_series(df[train_target], fill=0).clip(lower=0, upper=1).astype(int)

    feature_sets = build_ablation_feature_sets(df, manifest)
    if not feature_sets:
        patch.fail("No non-empty feature sets were constructed.")

    baseline_score_cols = {}
    for col in patch.BASELINE_SCORE_COLUMNS:
        if col in df.columns:
            score_col = f"score__baseline_{col}"
            df[score_col] = patch.to_numeric_series(df[col], fill=0.0)
            baseline_score_cols[f"baseline_{col}"] = score_col

    pred_df, coefficients, cv_diag = patch.train_cv_predictions(
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
    subsets = patch.define_subsets(df)
    metrics, group_metrics = patch.evaluate_models(
        df=df,
        model_score_cols=model_score_cols,
        target_col=args.target,
        subsets=subsets,
        include_group_rows=True,
    )
    metrics = patch.add_delta_metrics(metrics)
    best_models = patch.summarize_best_models(metrics)
    coef_mean = patch.coefficient_means(coefficients)
    feature_set_table = make_feature_set_table(feature_sets)

    prediction_columns = []
    for col in patch.IDENTIFIER_COLUMNS + [
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
        "target_positive_count": int(patch.to_numeric_series(df[args.target], fill=0).sum()),
        "primary_rows": int(patch.to_numeric_series(df["primary_candidate_group"], fill=0).gt(0).sum()),
        "model_count": int(len(model_score_cols)),
        "baseline_models": list(baseline_score_cols.keys()),
        "ml_models": list(ml_score_cols.keys()),
        "feature_set_sizes": {name: len(cols) for name, cols in feature_sets.items()},
        "model_block_summary": model_block_summary(feature_sets),
        "categorical_structure_columns_excluded": sorted(STRUCTURE_CATEGORICAL_COLUMNS),
        "notes": [
            "*_named_keep_composite models remove named evidence columns but keep patch_score composite features.",
            "*_clean_drop_composite models also remove patch_score* because patch_score may mix multiple evidence sources.",
            "ablate_no_direct_radi_keep_contextual_radi removes direct radi features but keeps contextual radi interaction terms.",
            "ablate_no_radi_all removes every feature whose name contains radi.",
        ],
        "cv_diagnostics": cv_diag,
        "outputs": {
            "predictions": f"{out_prefix}.predictions.tsv",
            "metrics": f"{out_prefix}.metrics.tsv",
            "group_metrics": f"{out_prefix}.group_metrics.tsv",
            "best_models": f"{out_prefix}.best_models.tsv",
            "coefficients": f"{out_prefix}.coefficients.tsv",
            "coefficients_mean": f"{out_prefix}.coefficients_mean.tsv",
            "feature_sets": f"{out_prefix}.feature_sets.tsv",
            "summary": f"{out_prefix}.summary.json",
        },
    }

    predictions_out.to_csv(f"{out_prefix}.predictions.tsv", sep="\t", index=False)
    metrics.to_csv(f"{out_prefix}.metrics.tsv", sep="\t", index=False)
    group_metrics.to_csv(f"{out_prefix}.group_metrics.tsv", sep="\t", index=False)
    best_models.to_csv(f"{out_prefix}.best_models.tsv", sep="\t", index=False)
    coefficients.to_csv(f"{out_prefix}.coefficients.tsv", sep="\t", index=False)
    coef_mean.to_csv(f"{out_prefix}.coefficients_mean.tsv", sep="\t", index=False)
    feature_set_table.to_csv(f"{out_prefix}.feature_sets.tsv", sep="\t", index=False)
    with open(f"{out_prefix}.summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, default=patch.json_default)

    print("BM5 Phase 1 structure-aware evidence-block ablation written")
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
        block_summary = summary["model_block_summary"][name]
        print(
            f"    {name}: {len(cols)} "
            f"(cons={block_summary['n_conservation']}, ifrag={block_summary['n_ifrag']}, "
            f"radi={block_summary['n_radi_all']}, direct_radi={block_summary['n_radi_direct']}, "
            f"structure={block_summary['n_structure']}, patch_score={block_summary['n_composite_patch_score']})"
        )
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
