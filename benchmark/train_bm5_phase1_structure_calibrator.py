#!/usr/bin/env python3
"""
Train BM5 Phase 1 structure-aware residue calibrators for iFragDI.

This script consumes the patch + structure feature table produced by
benchmark/build_bm5_phase1_structure_features.py and reuses the evaluation
machinery from train_bm5_phase1_patch_calibrator.py.

Main question:
    Do non-leaky monomer/query-structure features, especially SASA/RSA exposure,
    improve residue-level interface prioritization beyond patch/window evidence?

Default input:
    benchmark/labels/bm5_phase1_patch_structure_features.tsv
    benchmark/labels/bm5_phase1_patch_structure_features.feature_manifest.tsv

Default outputs with --out-prefix benchmark/labels/bm5_phase1_structure_ml_logreg:
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
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

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
        description="Train BM5 Phase 1 structure-aware residue calibrators."
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
        default="benchmark/labels/bm5_phase1_structure_ml_logreg",
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
        "--soft-target",
        default=None,
        help="Optional secondary soft/window target to evaluate. Reserved for compatibility; not used for fitting unless --train-target is set.",
    )
    parser.add_argument(
        "--train-target",
        default=None,
        help="Optional target used for fitting ML models. Defaults to --target.",
    )
    return parser.parse_args()


def unique_existing_numeric(df: pd.DataFrame, cols: Sequence[str]) -> List[str]:
    return patch.existing_numeric_columns(df, [c for c in cols if c not in STRUCTURE_CATEGORICAL_COLUMNS])


def with_prefix(columns: Sequence[str], prefixes: Sequence[str]) -> List[str]:
    return [col for col in columns if any(col.startswith(prefix) for prefix in prefixes)]


def structure_columns_by_type(safe_cols: Sequence[str]) -> Dict[str, List[str]]:
    struct_cols = [c for c in safe_cols if c.startswith("struct_") and c not in STRUCTURE_CATEGORICAL_COLUMNS]

    exposure = [
        c for c in struct_cols
        if c.startswith((
            "struct_sasa",
            "struct_rsa",
            "struct_surface",
            "struct_buried",
            "struct_intermediate_surface",
            "struct_freesasa_found",
        ))
    ]
    secondary = [
        c for c in struct_cols
        if c.startswith((
            "struct_dssp",
            "struct_ss3_helix",
            "struct_ss3_sheet",
            "struct_ss3_coil",
        ))
    ]
    aa = [c for c in struct_cols if c.startswith("struct_aa_")]
    return {
        "structure_exposure": exposure,
        "structure_secondary": secondary,
        "structure_aa": aa,
        "structure_all": struct_cols,
    }


def build_structure_feature_sets(df: pd.DataFrame, manifest: pd.DataFrame) -> Dict[str, List[str]]:
    safe_cols = patch.base_feature_safe_columns(df, manifest)

    original_core = [
        "conservation_component",
        "conservation_strength",
        "ifrag_strength",
        "ifrag_specificity",
        "ifrag_component",
        "patch_score",
    ]
    original_core_radi = original_core + ["radi_anchor", "radi_component"]

    conservation_patch = with_prefix(
        safe_cols,
        ("conservation_component_", "conservation_strength_"),
    ) + ["conservation_component", "conservation_strength"]

    ifrag_patch = with_prefix(
        safe_cols,
        ("ifrag_component_", "ifrag_strength_", "ifrag_specificity_"),
    ) + ["ifrag_component", "ifrag_strength", "ifrag_specificity"]

    patch_score_patch = with_prefix(safe_cols, ("patch_score_",)) + ["patch_score"]

    radi_neighborhood = with_prefix(
        safe_cols,
        ("radi_anchor_", "radi_component_"),
    ) + ["radi_anchor", "radi_component"]

    struct = structure_columns_by_type(safe_cols)
    structure_interactions = [c for c in safe_cols if c.endswith("_x_struct_rsa")]
    structure_all_plus_interactions = struct["structure_all"] + structure_interactions

    feature_sets = {
        "logreg_original_core": original_core,
        "logreg_original_core_radi": original_core_radi,
        "logreg_patch_conservation": conservation_patch,
        "logreg_patch_conservation_ifrag_patch_radi": conservation_patch + ifrag_patch + patch_score_patch + radi_neighborhood,
        "logreg_structure_exposure": struct["structure_exposure"],
        "logreg_structure_all": struct["structure_all"],
        "logreg_patch_conservation_structure": conservation_patch + structure_all_plus_interactions,
        "logreg_patch_conservation_ifrag_patch_structure": conservation_patch + ifrag_patch + patch_score_patch + structure_all_plus_interactions,
        "logreg_patch_conservation_ifrag_patch_radi_structure": conservation_patch + ifrag_patch + patch_score_patch + radi_neighborhood + structure_all_plus_interactions,
    }

    return {name: unique_existing_numeric(df, cols) for name, cols in feature_sets.items()}


def json_default(obj):
    return patch.json_default(obj)


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

    feature_sets = build_structure_feature_sets(df, manifest)
    feature_sets = {name: cols for name, cols in feature_sets.items() if cols}
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

    structure_feature_sizes = {name: len(cols) for name, cols in feature_sets.items() if "structure" in name}
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
        "structure_feature_set_sizes": structure_feature_sizes,
        "categorical_structure_columns_excluded": sorted(STRUCTURE_CATEGORICAL_COLUMNS),
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

    print("BM5 Phase 1 structure-aware ML benchmark written")
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
