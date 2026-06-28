#!/usr/bin/env python3
"""
Build BM5 Phase 1 patch/window-aware residue features for iFragDI ML.

This is a feature-building step, not a training step. It starts from the merged
BM5 Phase 1 residue training table and adds leakage-safe local/window features
that better match the biology of iFragDI:

  * iFrag is broad fragment/region evidence, not exact per-residue contact truth.
  * conservation is a broad functional/interface prior, not partner-contact proof.
  * raDI is sparse coevolutionary anchor-pair evidence and should be represented
    as local/nearby anchor support, not only as exact residue flags.

The script also writes soft/diagnostic label columns such as residues near a true
interface residue in sequence. These are label/diagnostic columns and should not
be used as prediction features unless explicitly training a soft-target model.

Default input:
    benchmark/labels/bm5_phase1_training_table.tsv

Default outputs with --out-prefix benchmark/labels/bm5_phase1_patch_features:
    benchmark/labels/bm5_phase1_patch_features.tsv
    benchmark/labels/bm5_phase1_patch_features.feature_manifest.tsv
    benchmark/labels/bm5_phase1_patch_features.summary.json
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


NONCONTACTING_CONTROLS_DEFAULT = ("BM5CP00234", "BM5CP00238", "BM5CP00318")

BASE_SCORE_COLUMNS = [
    "conservation_component",
    "conservation_strength",
    "ifrag_component",
    "ifrag_strength",
    "ifrag_specificity",
    "patch_score",
    "radi_component",
    "radi_anchor",
]

# These are kept as diagnostic/baseline-derived local summaries only. They are
# not recommended as input features for a model intended to replace final_score.
DIAGNOSTIC_SCORE_COLUMNS = ["final_score"]

EVIDENCE_COLUMNS = [
    "paired_rows_used",
    "weak_msa_warning",
    "radi_interchain_pairs_retained",
    "radi_matrix_nonzero",
    "anchor_matrix_nonzero",
    "radi_matrix_max",
    "anchor_matrix_max",
    "ifrag_fraction_nonzero",
]

TARGET_COLUMNS = ["interface_3p9A", "interface_5A", "interface_8A"]

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

LEAKAGE_OR_LABEL_PATTERNS = (
    "interface_",
    "near_interface_",
    "interface_window_",
    "min_partner_atom_distance_A",
    "nearest_partner_residue_label",
    "bound_",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build BM5 Phase 1 patch/window-aware residue features from the "
            "merged iFragDI residue training table."
        )
    )
    parser.add_argument(
        "--training-table",
        default="benchmark/labels/bm5_phase1_training_table.tsv",
        help="Merged BM5 Phase 1 residue feature/label table.",
    )
    parser.add_argument(
        "--out-prefix",
        default="benchmark/labels/bm5_phase1_patch_features",
        help="Output prefix for feature table, feature manifest, and summary JSON.",
    )
    parser.add_argument(
        "--target",
        default="interface_5A",
        help="Primary binary interface label used for diagnostics. Default: interface_5A.",
    )
    parser.add_argument(
        "--windows",
        default="3,5,10",
        help="Comma-separated sequence half-window sizes. Default: 3,5,10.",
    )
    parser.add_argument(
        "--top-quantile",
        type=float,
        default=0.90,
        help="Per-query-side quantile among positive scores used to define high-support segments. Default: 0.90.",
    )
    parser.add_argument(
        "--noncontacting-controls",
        default=",".join(NONCONTACTING_CONTROLS_DEFAULT),
        help="Comma-separated chainpair_id values treated as explicit noncontacting controls.",
    )
    parser.add_argument(
        "--include-diagnostic-final-score-features",
        action="store_true",
        help=(
            "Also build local/window diagnostic features from final_score. These are marked "
            "baseline_diagnostic and should not be used as input to a model that replaces final_score."
        ),
    )
    return parser.parse_args()


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(2)


def warn(message: str) -> None:
    print(f"WARNING: {message}", file=sys.stderr)


def parse_windows(text: str) -> List[int]:
    values: List[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            value = int(part)
        except ValueError:
            fail(f"Invalid window size {part!r}; expected comma-separated integers.")
        if value < 0:
            fail(f"Invalid window size {value}; windows must be nonnegative.")
        values.append(value)
    if not values:
        fail("At least one window size is required.")
    return sorted(set(values))


def parse_controls(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def require_columns(df: pd.DataFrame, columns: Sequence[str], context: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        fail(f"Missing required {context} columns: {', '.join(missing)}")


def as_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    lowered = series.astype(str).str.strip().str.lower()
    return lowered.isin({"true", "t", "1", "yes", "y"})


def to_numeric(df: pd.DataFrame, columns: Sequence[str], fill_missing: bool = False) -> None:
    for col in columns:
        if col not in df.columns:
            if fill_missing:
                df[col] = 0.0
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")


def sorted_group(df: pd.DataFrame) -> pd.DataFrame:
    """Return a deterministic order within each chainpair/query_side group."""
    out = df.copy()
    if "score_residue_index" in out.columns:
        out["_score_residue_index_numeric"] = pd.to_numeric(out["score_residue_index"], errors="coerce")
    else:
        out["_score_residue_index_numeric"] = np.nan
    out["_original_row_order"] = np.arange(len(out), dtype=int)
    out = out.sort_values(
        ["chainpair_id", "query_side", "_score_residue_index_numeric", "_original_row_order"],
        ascending=[True, True, True, True],
        kind="mergesort",
        na_position="last",
    )
    out["residue_order_in_group"] = out.groupby(["chainpair_id", "query_side"], sort=False).cumcount() + 1
    out["chain_length_in_table"] = out.groupby(["chainpair_id", "query_side"], sort=False)["residue_order_in_group"].transform("max")
    return out


def rolling_center(values: pd.Series, half_window: int, reducer: str) -> pd.Series:
    window = 2 * int(half_window) + 1
    numeric = pd.to_numeric(values, errors="coerce")
    roll = numeric.rolling(window=window, center=True, min_periods=1)
    if reducer == "mean":
        return roll.mean()
    if reducer == "max":
        return roll.max()
    if reducer == "sum":
        return roll.sum()
    raise ValueError(f"Unknown reducer: {reducer}")


def group_rank_percentile(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() == 0:
        return pd.Series(np.nan, index=values.index)
    # Higher score = higher percentile. NaN values are placed at the bottom.
    filled = numeric.fillna(numeric.min() - 1.0 if numeric.notna().any() else -1.0)
    return filled.rank(method="average", pct=True, ascending=True)


def positive_quantile_threshold(values: pd.Series, q: float) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    positive = numeric[numeric > 0]
    if positive.empty:
        return float("inf")
    return float(positive.quantile(q))


def distance_to_nearest_true(mask: pd.Series) -> pd.Series:
    arr = mask.fillna(False).astype(bool).to_numpy()
    n = len(arr)
    if n == 0:
        return pd.Series([], dtype=float, index=mask.index)
    true_positions = np.flatnonzero(arr)
    if len(true_positions) == 0:
        return pd.Series(np.full(n, np.nan), index=mask.index, dtype=float)

    positions = np.arange(n)
    # For BM5 chain lengths this simple vectorized distance matrix is small and
    # clearer than a two-pass implementation. Avoid it only for very long chains.
    if n * len(true_positions) <= 2_000_000:
        dist = np.min(np.abs(positions[:, None] - true_positions[None, :]), axis=1)
    else:
        # Memory-safe fallback.
        dist = np.full(n, np.inf)
        last = -np.inf
        for i in range(n):
            if arr[i]:
                last = i
            dist[i] = min(dist[i], i - last)
        last = np.inf
        for i in range(n - 1, -1, -1):
            if arr[i]:
                last = i
            dist[i] = min(dist[i], last - i)
    return pd.Series(dist.astype(float), index=mask.index)


def segment_lengths(mask: pd.Series) -> pd.Series:
    arr = mask.fillna(False).astype(bool).to_numpy()
    n = len(arr)
    out = np.zeros(n, dtype=int)
    i = 0
    while i < n:
        if not arr[i]:
            i += 1
            continue
        j = i
        while j < n and arr[j]:
            j += 1
        out[i:j] = j - i
        i = j
    return pd.Series(out, index=mask.index)


def add_window_features_for_group(
    group: pd.DataFrame,
    score_columns: Sequence[str],
    diagnostic_score_columns: Sequence[str],
    windows: Sequence[int],
    top_quantile: float,
    target_columns: Sequence[str],
) -> pd.DataFrame:
    """Add local sequence-window features for one chainpair/query_side group.

    New columns are accumulated in a dictionary and concatenated once. This keeps
    pandas from fragmenting the DataFrame and avoids PerformanceWarning spam.
    """
    g = group.copy()
    new_cols: Dict[str, pd.Series] = {}
    topq_label = int(top_quantile * 100)

    # Main score columns: leakage-safe prediction-side features.
    for col in score_columns:
        if col not in g.columns:
            g[col] = 0.0
        g[col] = pd.to_numeric(g[col], errors="coerce").fillna(0.0)
        new_cols[f"{col}_rank_pct_in_chain"] = group_rank_percentile(g[col])

        threshold = positive_quantile_threshold(g[col], top_quantile)
        high = g[col].gt(0) & g[col].ge(threshold)
        new_cols[f"{col}_topq{topq_label:02d}_flag"] = high.astype(int)
        new_cols[f"{col}_topq{topq_label:02d}_segment_len"] = segment_lengths(high).astype(int)
        new_cols[f"{col}_dist_to_topq{topq_label:02d}"] = distance_to_nearest_true(high)

        for w in windows:
            prefix = f"{col}_win{w}"
            new_cols[f"{prefix}_mean"] = rolling_center(g[col], w, "mean")
            new_cols[f"{prefix}_max"] = rolling_center(g[col], w, "max")
            new_cols[f"{prefix}_sum"] = rolling_center(g[col], w, "sum")

    # Diagnostic baseline/local summaries, marked separately in the manifest.
    for col in diagnostic_score_columns:
        if col not in g.columns:
            continue
        g[col] = pd.to_numeric(g[col], errors="coerce").fillna(0.0)
        new_cols[f"diagnostic_{col}_rank_pct_in_chain"] = group_rank_percentile(g[col])
        for w in windows:
            prefix = f"diagnostic_{col}_win{w}"
            new_cols[f"{prefix}_mean"] = rolling_center(g[col], w, "mean")
            new_cols[f"{prefix}_max"] = rolling_center(g[col], w, "max")

    # Combine once so cross-features can refer to the newly created window columns.
    if new_cols:
        g = pd.concat([g, pd.DataFrame(new_cols, index=g.index)], axis=1)

    radi_new: Dict[str, pd.Series] = {}

    # raDI-neighborhood features. These are derived only from prediction-side raDI
    # features and local sequence position.
    radi_anchor_positive = pd.to_numeric(g.get("radi_anchor", 0.0), errors="coerce").fillna(0.0).gt(0)
    radi_dist = distance_to_nearest_true(radi_anchor_positive)
    radi_new["radi_anchor_dist_nearest"] = radi_dist
    radi_new["radi_anchor_dist_nearest_filled"] = radi_dist.fillna(g["chain_length_in_table"] + 1)

    radi_anchor_numeric = radi_anchor_positive.astype(int)
    for w in windows:
        count_col = f"radi_anchor_win{w}_count"
        anchor_count = rolling_center(radi_anchor_numeric, w, "sum")
        radi_new[count_col] = anchor_count
        radi_new[f"radi_anchor_win{w}_has_anchor"] = anchor_count.gt(0).astype(int)

        # radi_component window max is already generated by the generic score loop
        # because radi_component is part of BASE_SCORE_COLUMNS. Do not add it here
        # again, otherwise pandas allows duplicate column names and silently changes
        # the output schema.

        # Contextualized raDI support: raDI anchors are only expected to help when
        # they fall inside a broader iFrag/conservation/patch-supported region.
        if f"patch_score_win{w}_max" in g.columns:
            radi_new[f"radi_anchor_win{w}_x_patch_max"] = anchor_count * g[f"patch_score_win{w}_max"].fillna(0.0)
        if f"conservation_component_win{w}_max" in g.columns:
            radi_new[f"radi_anchor_win{w}_x_conservation_max"] = anchor_count * g[f"conservation_component_win{w}_max"].fillna(0.0)
        if f"ifrag_strength_win{w}_max" in g.columns:
            radi_new[f"radi_anchor_win{w}_x_ifrag_strength_max"] = anchor_count * g[f"ifrag_strength_win{w}_max"].fillna(0.0)

    if radi_new:
        g = pd.concat([g, pd.DataFrame(radi_new, index=g.index)], axis=1)

    label_new: Dict[str, pd.Series] = {}

    # Soft/diagnostic labels derived from native interface labels. These should
    # be treated as labels/diagnostics, not input features.
    for target_col in target_columns:
        if target_col not in g.columns:
            continue
        target_numeric = pd.to_numeric(g[target_col], errors="coerce").fillna(0).clip(lower=0, upper=1)
        for w in windows:
            label_prefix = f"{target_col}_win{w}"
            label_new[f"near_{target_col}_window_{w}"] = rolling_center(target_numeric, w, "max").fillna(0).astype(int)
            label_new[f"{label_prefix}_positive_count"] = rolling_center(target_numeric, w, "sum").fillna(0).astype(int)

    if label_new:
        g = pd.concat([g, pd.DataFrame(label_new, index=g.index)], axis=1)

    return g.copy()


def build_feature_manifest(
    original_columns: Sequence[str],
    output_columns: Sequence[str],
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows = []
    original = set(original_columns)
    windows = parse_windows(args.windows)

    for col in output_columns:
        if col in original:
            role = "original"
            source = "input_table"
            leakage_status = "depends_on_column"
        elif col.startswith("near_interface_") or col.startswith("interface_") and "win" in col:
            role = "diagnostic_label"
            source = "native_interface_label_window"
            leakage_status = "label_do_not_use_as_feature"
        elif col.startswith("diagnostic_final_score"):
            role = "baseline_diagnostic"
            source = "final_score_local_summary"
            leakage_status = "baseline_only_not_replacement_feature"
        elif any(col.startswith(f"{score}_") for score in BASE_SCORE_COLUMNS):
            role = "patch_feature"
            source = "prediction_side_local_window_or_rank"
            leakage_status = "feature_safe"
        elif col.startswith("radi_anchor_") or col.startswith("radi_component_"):
            role = "radi_neighborhood_feature"
            source = "prediction_side_radi_local_window"
            leakage_status = "feature_safe"
        elif col.startswith("residue_order") or col.startswith("chain_length") or col.startswith("group_"):
            role = "group_metadata"
            source = "input_table_grouping"
            leakage_status = "feature_safe_or_metadata"
        elif col in {"is_noncontacting_control", "is_no_evidence_completed", "primary_candidate_group"}:
            role = "diagnostic_metadata"
            source = "case_metadata"
            leakage_status = "diagnostic_metadata"
        else:
            role = "derived"
            source = "derived_from_input_features"
            leakage_status = "review_before_model_use"

        if col.startswith("bound_") or col in {"min_partner_atom_distance_A", "nearest_partner_residue_label"}:
            leakage_status = "native_bound_metadata_do_not_use_as_feature"
        if col in TARGET_COLUMNS:
            role = "label"
            leakage_status = "label_do_not_use_as_feature"

        rows.append(
            {
                "column": col,
                "role": role,
                "source": source,
                "leakage_status": leakage_status,
                "windows_requested": ",".join(str(w) for w in windows),
                "top_quantile": args.top_quantile,
            }
        )

    return pd.DataFrame(rows)


def add_basic_annotations(df: pd.DataFrame, target: str, controls: Sequence[str]) -> None:
    df["group_key"] = df["chainpair_id"].astype(str) + "||" + df["query_side"].astype(str)
    if "evidence_class" in df.columns:
        df["is_no_evidence_completed"] = df["evidence_class"].astype(str).eq("no_evidence_completed")
    else:
        df["is_no_evidence_completed"] = False
    df["is_noncontacting_control"] = df["chainpair_id"].astype(str).isin(set(controls))

    if target in df.columns:
        target_numeric = pd.to_numeric(df[target], errors="coerce").fillna(0).clip(lower=0, upper=1)
        group_pos = target_numeric.groupby(df["group_key"]).transform("sum")
        df["group_positive_count_target"] = group_pos.astype(float)
        df["primary_candidate_group"] = (
            group_pos.gt(0) & ~df["is_no_evidence_completed"] & ~df["is_noncontacting_control"]
        ).astype(int)
    else:
        df["group_positive_count_target"] = np.nan
        df["primary_candidate_group"] = 0

    if "weak_msa_warning" in df.columns:
        df["weak_msa_warning_bool"] = as_bool_series(df["weak_msa_warning"]).astype(int)
    else:
        df["weak_msa_warning_bool"] = 0


def summarize_feature_table(
    df: pd.DataFrame,
    original_columns: Sequence[str],
    feature_manifest: pd.DataFrame,
    args: argparse.Namespace,
    warnings_list: Sequence[str],
) -> Dict[str, object]:
    derived_cols = [c for c in df.columns if c not in set(original_columns)]
    feature_safe_cols = feature_manifest.loc[
        feature_manifest["leakage_status"].isin(["feature_safe", "feature_safe_or_metadata"]), "column"
    ].tolist()
    label_cols = feature_manifest.loc[
        feature_manifest["leakage_status"].str.contains("label", na=False), "column"
    ].tolist()

    def nonzero_count(col: str) -> int:
        if col not in df.columns:
            return 0
        return int(pd.to_numeric(df[col], errors="coerce").fillna(0).ne(0).sum())

    highlighted = [
        "ifrag_strength_win5_max",
        "ifrag_strength_win5_mean",
        "patch_score_win5_max",
        "conservation_component_win5_max",
        "radi_anchor_win5_count",
        "radi_anchor_win5_x_patch_max",
        "near_interface_5A_window_5",
    ]

    subset_counts = {}
    if "primary_candidate_group" in df.columns:
        subset_counts["primary_candidate_group_rows"] = int(df["primary_candidate_group"].sum())
    if "is_no_evidence_completed" in df.columns:
        subset_counts["no_evidence_completed_rows"] = int(df["is_no_evidence_completed"].sum())
    if "is_noncontacting_control" in df.columns:
        subset_counts["noncontacting_control_rows"] = int(df["is_noncontacting_control"].sum())

    target = args.target
    target_positive = int(pd.to_numeric(df[target], errors="coerce").fillna(0).sum()) if target in df.columns else None

    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": Path(__file__).name,
        "training_table": args.training_table,
        "out_prefix": args.out_prefix,
        "target": args.target,
        "windows": parse_windows(args.windows),
        "top_quantile": args.top_quantile,
        "n_rows": int(len(df)),
        "n_columns_input": int(len(original_columns)),
        "n_columns_output": int(len(df.columns)),
        "n_derived_columns": int(len(derived_cols)),
        "n_feature_safe_columns": int(len(feature_safe_cols)),
        "n_label_or_diagnostic_label_columns": int(len(label_cols)),
        "n_chainpairs": int(df["chainpair_id"].nunique()) if "chainpair_id" in df.columns else None,
        "n_query_side_groups": int(df["group_key"].nunique()) if "group_key" in df.columns else None,
        "target_positive_count": target_positive,
        "subset_counts": subset_counts,
        "warnings": list(warnings_list),
        "highlighted_nonzero_counts": {col: nonzero_count(col) for col in highlighted if col in df.columns},
        "outputs": {
            "feature_table": f"{args.out_prefix}.tsv",
            "feature_manifest": f"{args.out_prefix}.feature_manifest.tsv",
            "summary": f"{args.out_prefix}.summary.json",
        },
    }


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
    windows = parse_windows(args.windows)
    controls = parse_controls(args.noncontacting_controls)
    warnings_list: List[str] = []

    training_table = Path(args.training_table)
    if not training_table.exists():
        fail(f"Training table does not exist: {training_table}")

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(training_table, sep="\t", low_memory=False)
    original_columns = list(df.columns)

    require_columns(df, ["chainpair_id", "query_side", args.target], "training table")

    missing_base = [col for col in BASE_SCORE_COLUMNS if col not in df.columns]
    if missing_base:
        msg = "Missing expected base score columns; zero-filling: " + ", ".join(missing_base)
        warn(msg)
        warnings_list.append(msg)
        for col in missing_base:
            df[col] = 0.0

    missing_evidence = [col for col in EVIDENCE_COLUMNS if col not in df.columns]
    if missing_evidence:
        msg = "Missing expected evidence columns; zero-filling where needed: " + ", ".join(missing_evidence)
        warn(msg)
        warnings_list.append(msg)
        for col in missing_evidence:
            df[col] = 0.0

    for target_col in TARGET_COLUMNS:
        if target_col not in df.columns:
            msg = f"Optional target/label column missing; skipping soft labels for {target_col}."
            warn(msg)
            warnings_list.append(msg)

    numeric_cols = list(set(BASE_SCORE_COLUMNS + DIAGNOSTIC_SCORE_COLUMNS + EVIDENCE_COLUMNS + TARGET_COLUMNS + ["score_residue_index"]))
    to_numeric(df, numeric_cols, fill_missing=False)

    if "evidence_class" not in df.columns:
        df["evidence_class"] = "missing"
    df["evidence_class"] = df["evidence_class"].fillna("missing").astype(str)

    add_basic_annotations(df, target=args.target, controls=controls)
    df = sorted_group(df)

    diagnostic_cols = DIAGNOSTIC_SCORE_COLUMNS if args.include_diagnostic_final_score_features else []
    available_targets = [col for col in TARGET_COLUMNS if col in df.columns]

    parts = []
    for _, group in df.groupby(["chainpair_id", "query_side"], sort=False, dropna=False):
        parts.append(
            add_window_features_for_group(
                group=group,
                score_columns=BASE_SCORE_COLUMNS,
                diagnostic_score_columns=diagnostic_cols,
                windows=windows,
                top_quantile=float(args.top_quantile),
                target_columns=available_targets,
            )
        )

    out_df = pd.concat(parts, axis=0, ignore_index=False).sort_values("_original_row_order", kind="mergesort")
    out_df = out_df.drop(columns=[c for c in ["_score_residue_index_numeric", "_original_row_order"] if c in out_df.columns])
    out_df = out_df.copy()

    manifest = build_feature_manifest(original_columns, list(out_df.columns), args)
    summary = summarize_feature_table(out_df, original_columns, manifest, args, warnings_list)

    feature_table_path = Path(f"{out_prefix}.tsv")
    manifest_path = Path(f"{out_prefix}.feature_manifest.tsv")
    summary_path = Path(f"{out_prefix}.summary.json")

    out_df.to_csv(feature_table_path, sep="\t", index=False)
    manifest.to_csv(manifest_path, sep="\t", index=False)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, default=json_default)

    print("BM5 Phase 1 patch/window feature table written")
    print(f"  input rows: {len(df)}")
    print(f"  output rows: {len(out_df)}")
    print(f"  input columns: {len(original_columns)}")
    print(f"  output columns: {len(out_df.columns)}")
    print(f"  derived columns: {len(out_df.columns) - len(original_columns)}")
    print(f"  windows: {','.join(str(w) for w in windows)}")
    print(f"  top quantile: {args.top_quantile}")
    print("  outputs:")
    print(f"    feature table:    {feature_table_path}")
    print(f"    feature manifest: {manifest_path}")
    print(f"    summary:          {summary_path}")

    highlighted = summary.get("highlighted_nonzero_counts", {})
    if highlighted:
        print("\nHighlighted nonzero counts:")
        for col, count in highlighted.items():
            print(f"  {col}: {count}")


if __name__ == "__main__":
    main()
