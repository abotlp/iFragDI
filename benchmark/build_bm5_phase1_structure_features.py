#!/usr/bin/env python3
"""
Build BM5 Phase 1 structure-aware residue features for iFragDI ML.

This script extends the patch/window feature table with non-leaky structural
features computed from query_pdb coordinates only:

  * FreeSASA residue SASA
  * relative solvent accessibility (RSA) using Tien/Wilke MaxASA values
  * surface/buried flags and local surface windows
  * DSSP secondary structure from mkdssp output
  * simple amino-acid physicochemical flags
  * evidence x RSA interaction terms

Leakage rule: never compute these input features from the bound complex. Bound
complex SASA would reveal interface burial.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import freesasa
except Exception as exc:  # pragma: no cover
    freesasa = None
    FREESASA_IMPORT_ERROR = exc
else:
    FREESASA_IMPORT_ERROR = None


# Tien/Wilke theoretical MaxASA values, Angstrom^2.
MAX_ASA_TIEN_THEORETICAL = {
    "A": 129.0, "R": 274.0, "N": 195.0, "D": 193.0, "C": 167.0,
    "Q": 223.0, "E": 225.0, "G": 104.0, "H": 224.0, "I": 197.0,
    "L": 201.0, "K": 236.0, "M": 224.0, "F": 240.0, "P": 159.0,
    "S": 155.0, "T": 172.0, "W": 285.0, "Y": 263.0, "V": 174.0,
}

AA3_TO_AA1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M",
}

DSSP_8_TO_3 = {
    "H": "helix", "G": "helix", "I": "helix",
    "E": "sheet", "B": "sheet",
    "T": "coil", "S": "coil", "C": "coil", "-": "coil", "": "coil",
}

HYDROPHOBIC = set("AVILMFWYC")
POLAR = set("STNQYCH")
POSITIVE = set("KRH")
NEGATIVE = set("DE")
AROMATIC = set("FYW")

EVIDENCE_COLUMNS_FOR_RSA = [
    "conservation_component",
    "conservation_strength",
    "ifrag_strength",
    "ifrag_specificity",
    "ifrag_component",
    "patch_score",
    "radi_anchor",
    "radi_component",
    "radi_anchor_win5_count",
    "radi_anchor_win5_x_conservation_max",
    "radi_anchor_win5_x_ifrag_strength_max",
    "radi_anchor_win5_x_patch_max",
]

StructureKey = Tuple[str, str]  # chain, residue id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build SASA/RSA/DSSP structure features for BM5 Phase 1 residue ML."
    )
    parser.add_argument("--feature-table", default="benchmark/labels/bm5_phase1_patch_features.tsv")
    parser.add_argument("--input-manifest", default=None)
    parser.add_argument("--out-prefix", default="benchmark/labels/bm5_phase1_patch_structure_features")
    parser.add_argument("--windows", default="3,5,10")
    parser.add_argument("--dssp-executable", default="mkdssp")
    parser.add_argument("--skip-dssp", action="store_true")
    parser.add_argument("--surface-threshold", type=float, default=0.20)
    parser.add_argument("--buried-threshold", type=float, default=0.05)
    return parser.parse_args()


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(2)


def parse_windows(text: str) -> List[int]:
    values: List[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            value = int(part)
        except ValueError:
            fail(f"Invalid window size: {part!r}")
        if value < 0:
            fail(f"Window size must be nonnegative: {value}")
        values.append(value)
    if not values:
        fail("At least one window size is required.")
    return sorted(set(values))


def norm_chain(value: object) -> str:
    if value is None or pd.isna(value):
        return "_"
    text = str(value).strip()
    return text if text else "_"


def norm_resid(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        head = text[:-2]
        if head.lstrip("-").isdigit():
            text = head
    return text.replace(" ", "")


def aa3_to_aa1(value: object) -> str:
    if value is None or pd.isna(value):
        return "X"
    text = str(value).strip().upper()
    if len(text) == 1:
        return text
    return AA3_TO_AA1.get(text, "X")


def safe_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def rsa_from_asa(asa: object, aa1: object) -> float:
    asa_float = safe_float(asa)
    aa = str(aa1).strip().upper()[:1]
    max_asa = MAX_ASA_TIEN_THEORETICAL.get(aa)
    if pd.isna(asa_float) or not max_asa:
        return float("nan")
    return asa_float / float(max_asa)


def ss8_to_ss3(value: object) -> str:
    ss = str(value).strip().upper()
    return DSSP_8_TO_3.get(ss, "coil")


def empty_record() -> Dict[str, object]:
    return {
        "struct_sasa_abs": np.nan,
        "struct_rsa_rel": np.nan,
        "struct_rsa_rel_clipped": np.nan,
        "struct_freesasa_found": 0,
        "struct_dssp_acc": np.nan,
        "struct_dssp_rsa_rel": np.nan,
        "struct_dssp_found": 0,
        "struct_ss8": "missing",
        "struct_ss3": "missing",
        "struct_ss3_helix_flag": 0,
        "struct_ss3_sheet_flag": 0,
        "struct_ss3_coil_flag": 0,
    }


def aa_features(aa: object) -> Dict[str, object]:
    aa1 = aa3_to_aa1(aa)
    charge = 1 if aa1 in POSITIVE else -1 if aa1 in NEGATIVE else 0
    return {
        "struct_aa_charge": charge,
        "struct_aa_hydrophobic_flag": int(aa1 in HYDROPHOBIC),
        "struct_aa_polar_flag": int(aa1 in POLAR),
        "struct_aa_aromatic_flag": int(aa1 in AROMATIC),
        "struct_aa_glycine_flag": int(aa1 == "G"),
        "struct_aa_proline_flag": int(aa1 == "P"),
    }


def rolling_center(values: pd.Series, half_window: int, reducer: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    roll = numeric.rolling(window=2 * int(half_window) + 1, center=True, min_periods=1)
    if reducer == "mean":
        return roll.mean()
    if reducer == "max":
        return roll.max()
    if reducer == "sum":
        return roll.sum()
    raise ValueError(reducer)


def find_executable(name_or_path: str) -> Optional[str]:
    p = Path(name_or_path)
    if p.exists() and p.is_file():
        return str(p)
    return shutil.which(name_or_path)


def parse_dssp_file(path: Path) -> Dict[StructureKey, Dict[str, object]]:
    records: Dict[StructureKey, Dict[str, object]] = {}
    lines = path.read_text(errors="replace").splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.startswith("  #") and "RESIDUE" in line and "STRUCTURE" in line:
            start = i + 1
            break
    if start is None:
        return records

    for line in lines[start:]:
        if len(line) < 38:
            continue
        aa = line[13].strip()
        if aa in {"!", "*"}:
            continue
        resnum = norm_resid(line[5:10])
        chain = norm_chain(line[11:12])
        if not resnum:
            continue
        ss8 = line[16].strip() or "C"
        acc = safe_float(line[34:38].strip())
        aa1 = aa.strip().upper()[:1] if aa.strip() else "X"
        ss3 = ss8_to_ss3(ss8)
        records[(chain, resnum)] = {
            "struct_dssp_acc": acc,
            "struct_dssp_rsa_rel": rsa_from_asa(acc, aa1),
            "struct_dssp_found": 1,
            "struct_ss8": ss8,
            "struct_ss3": ss3,
            "struct_ss3_helix_flag": int(ss3 == "helix"),
            "struct_ss3_sheet_flag": int(ss3 == "sheet"),
            "struct_ss3_coil_flag": int(ss3 == "coil"),
        }
    return records


def run_dssp(pdb_path: Path, dssp_executable: str, temp_dir: Path) -> Tuple[Dict[StructureKey, Dict[str, object]], str]:
    """Run mkdssp robustly across module versions.

    The Shiva DSSP/3.1.5 module exposes mkdssp 4.5.5. In Patricia's test,
    `mkdssp input.pdb output.dssp` produced parseable DSSP output, while
    `mkdssp -i input.pdb -o output.dssp` failed with 'unknown option'.
    Therefore positional invocation is attempted first.
    """
    digest = hashlib.sha1(str(pdb_path).encode("utf-8")).hexdigest()[:16]
    attempts = [
        [dssp_executable, str(pdb_path), str(temp_dir / f"{digest}.positional.dssp")],
        [dssp_executable, "-i", str(pdb_path), "-o", str(temp_dir / f"{digest}.dashio.dssp")],
    ]
    errors: List[str] = []
    for cmd in attempts:
        out_path = Path(cmd[-1])
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if out_path.exists():
            parsed = parse_dssp_file(out_path)
            if parsed:
                # Some mkdssp builds emit mmCIF/PDB warnings to stderr while still
                # producing valid DSSP output. Treat parseable output as success.
                return parsed, ""
        msg = (proc.stderr or proc.stdout or f"return_code={proc.returncode}").strip()
        errors.append(" ".join(cmd[:2]) + f": {msg}")
    return {}, " | ".join(errors)[:1000]


def freesasa_records(pdb_path: Path) -> Tuple[Dict[StructureKey, Dict[str, object]], str]:
    if freesasa is None:
        return {}, f"FreeSASA import failed: {FREESASA_IMPORT_ERROR}"
    records: Dict[StructureKey, Dict[str, object]] = {}
    try:
        structure = freesasa.Structure(str(pdb_path))
        result = freesasa.calc(structure)
        areas = result.residueAreas()
    except Exception as exc:
        return {}, str(exc)
    for chain, residues in areas.items():
        c = norm_chain(chain)
        for resnum, area in residues.items():
            records[(c, norm_resid(resnum))] = {
                "struct_sasa_abs": safe_float(getattr(area, "total", np.nan)),
                "struct_freesasa_found": 1,
            }
    return records, ""


def collect_structure_records(
    pdb_paths: Sequence[str],
    use_dssp: bool,
    dssp_executable: Optional[str],
) -> Tuple[Dict[Tuple[str, str, str], Dict[str, object]], List[Dict[str, object]]]:
    all_records: Dict[Tuple[str, str, str], Dict[str, object]] = {}
    logs: List[Dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="ifragdi_dssp_") as tmp:
        temp_dir = Path(tmp)
        for idx, pdb_text in enumerate(pdb_paths, start=1):
            pdb_path = Path(str(pdb_text))
            log: Dict[str, object] = {
                "query_pdb": str(pdb_text),
                "exists": pdb_path.exists(),
                "freesasa_records": 0,
                "dssp_records": 0,
                "freesasa_error": "",
                "dssp_error": "",
            }
            if not pdb_path.exists():
                log["freesasa_error"] = "missing_pdb"
                log["dssp_error"] = "missing_pdb"
                logs.append(log)
                continue

            fs_records, fs_error = freesasa_records(pdb_path)
            dssp_records: Dict[StructureKey, Dict[str, object]] = {}
            dssp_error = ""
            if use_dssp and dssp_executable:
                dssp_records, dssp_error = run_dssp(pdb_path, dssp_executable, temp_dir)

            log["freesasa_records"] = len(fs_records)
            log["dssp_records"] = len(dssp_records)
            log["freesasa_error"] = fs_error
            log["dssp_error"] = dssp_error

            for chain, resid in set(fs_records) | set(dssp_records):
                rec = empty_record()
                rec.update(fs_records.get((chain, resid), {}))
                rec.update(dssp_records.get((chain, resid), {}))
                all_records[(str(pdb_text), chain, resid)] = rec

            if idx % 50 == 0:
                print(f"Processed structures: {idx}/{len(pdb_paths)}", file=sys.stderr)
            logs.append(log)
    return all_records, logs


def lookup_record(
    records: Mapping[Tuple[str, str, str], Dict[str, object]],
    query_pdb: object,
    chains: Iterable[object],
    residue_id: object,
) -> Dict[str, object]:
    pdb_key = str(query_pdb)
    resid = norm_resid(residue_id)
    candidates: List[str] = []
    for c in chains:
        cc = norm_chain(c)
        if cc not in candidates:
            candidates.append(cc)
    if "_" not in candidates:
        candidates.append("_")
    for chain in candidates:
        key = (pdb_key, chain, resid)
        if key in records:
            return records[key].copy()
    return empty_record()


def add_local_windows(group: pd.DataFrame, windows: Sequence[int]) -> pd.DataFrame:
    g = group.copy()
    new: Dict[str, pd.Series] = {}
    rsa = pd.to_numeric(g["struct_rsa_rel_clipped"], errors="coerce")
    surface = pd.to_numeric(g["struct_surface_flag_rsa_ge_0p20"], errors="coerce").fillna(0)
    buried = pd.to_numeric(g["struct_buried_flag_rsa_lt_0p05"], errors="coerce").fillna(0)
    helix = pd.to_numeric(g["struct_ss3_helix_flag"], errors="coerce").fillna(0)
    sheet = pd.to_numeric(g["struct_ss3_sheet_flag"], errors="coerce").fillna(0)
    coil = pd.to_numeric(g["struct_ss3_coil_flag"], errors="coerce").fillna(0)
    for w in windows:
        new[f"struct_rsa_win{w}_mean"] = rolling_center(rsa, w, "mean")
        new[f"struct_rsa_win{w}_max"] = rolling_center(rsa, w, "max")
        new[f"struct_rsa_win{w}_sum"] = rolling_center(rsa, w, "sum")
        new[f"struct_surface_win{w}_count"] = rolling_center(surface, w, "sum")
        new[f"struct_buried_win{w}_count"] = rolling_center(buried, w, "sum")
        new[f"struct_ss3_helix_win{w}_count"] = rolling_center(helix, w, "sum")
        new[f"struct_ss3_sheet_win{w}_count"] = rolling_center(sheet, w, "sum")
        new[f"struct_ss3_coil_win{w}_count"] = rolling_center(coil, w, "sum")
    return pd.concat([g, pd.DataFrame(new, index=g.index)], axis=1).copy()


def add_structure_features(df: pd.DataFrame, records: Mapping[Tuple[str, str, str], Dict[str, object]], args: argparse.Namespace) -> pd.DataFrame:
    struct_rows: List[Dict[str, object]] = []
    for row in df[["query_pdb", "query_chain", "pdb_chain", "pdb_residue_id", "aa", "pdb_resname"]].itertuples(index=False):
        query_pdb, query_chain, pdb_chain, pdb_residue_id, aa, pdb_resname = row
        rec = lookup_record(records, query_pdb, [pdb_chain, query_chain], pdb_residue_id)
        aa1 = aa3_to_aa1(aa if not pd.isna(aa) else pdb_resname)
        if pd.isna(rec.get("struct_sasa_abs", np.nan)) and not pd.isna(rec.get("struct_dssp_acc", np.nan)):
            rec["struct_sasa_abs"] = rec.get("struct_dssp_acc", np.nan)
        rec["struct_rsa_rel"] = rsa_from_asa(rec.get("struct_sasa_abs", np.nan), aa1)
        rsa = rec["struct_rsa_rel"]
        rec["struct_rsa_rel_clipped"] = np.nan if pd.isna(rsa) else float(np.clip(rsa, 0.0, 1.0))
        rsa_clip = rec["struct_rsa_rel_clipped"]
        rec["struct_surface_flag_rsa_ge_0p20"] = int(pd.notna(rsa_clip) and rsa_clip >= args.surface_threshold)
        rec["struct_buried_flag_rsa_lt_0p05"] = int(pd.notna(rsa_clip) and rsa_clip < args.buried_threshold)
        rec["struct_intermediate_surface_flag"] = int(pd.notna(rsa_clip) and args.buried_threshold <= rsa_clip < args.surface_threshold)
        rec["struct_sasa_missing_flag"] = int(pd.isna(rsa_clip))
        rec.update(aa_features(aa1))
        struct_rows.append(rec)

    out = pd.concat([df, pd.DataFrame(struct_rows, index=df.index)], axis=1)
    rsa_fill = pd.to_numeric(out["struct_rsa_rel_clipped"], errors="coerce").fillna(0.0)
    for col in EVIDENCE_COLUMNS_FOR_RSA:
        if col in out.columns:
            out[f"{col}_x_struct_rsa"] = pd.to_numeric(out[col], errors="coerce").fillna(0.0) * rsa_fill

    out["_original_row_order"] = np.arange(len(out), dtype=int)
    out["_score_residue_index_numeric"] = pd.to_numeric(out.get("score_residue_index", np.nan), errors="coerce")
    sorted_df = out.sort_values(
        ["chainpair_id", "query_side", "_score_residue_index_numeric", "_original_row_order"],
        ascending=[True, True, True, True],
        kind="mergesort",
        na_position="last",
    )
    parts = [add_local_windows(g, parse_windows(args.windows)) for _, g in sorted_df.groupby(["chainpair_id", "query_side"], sort=False, dropna=False)]
    out2 = pd.concat(parts, axis=0).sort_values("_original_row_order", kind="mergesort")
    return out2.drop(columns=["_original_row_order", "_score_residue_index_numeric"], errors="ignore").copy()


def infer_manifest_path(feature_table: Path) -> Path:
    if feature_table.name.endswith(".tsv"):
        return feature_table.with_name(feature_table.name[:-4] + ".feature_manifest.tsv")
    return feature_table.with_suffix(".feature_manifest.tsv")


def build_manifest(input_manifest: Optional[Path], original_columns: Sequence[str], output_columns: Sequence[str], args: argparse.Namespace) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    used = set()
    if input_manifest and input_manifest.exists():
        old = pd.read_csv(input_manifest, sep="\t", low_memory=False)
        if "column" in old.columns:
            for _, row in old.iterrows():
                col = str(row["column"])
                if col in output_columns and col not in used:
                    rows.append(row.to_dict())
                    used.add(col)
    original_set = set(original_columns)
    for col in output_columns:
        if col in used:
            continue
        if col in original_set:
            role, source, leakage = "original", "input_feature_table", "depends_on_column"
        elif col in {"struct_ss8", "struct_ss3"}:
            role, source, leakage = "structure_feature_categorical", "query_pdb_dssp", "feature_safe_categorical"
        elif col.endswith("_x_struct_rsa"):
            role, source, leakage = "structure_interaction_feature", "prediction_feature_x_query_pdb_rsa", "feature_safe"
        elif col.startswith("struct_"):
            role, source, leakage = "structure_feature", "query_pdb_freesasa_or_dssp", "feature_safe"
        else:
            role, source, leakage = "derived", "derived_from_input_features", "review_before_model_use"
        rows.append({
            "column": col,
            "role": role,
            "source": source,
            "leakage_status": leakage,
            "windows_requested": args.windows,
            "surface_threshold": args.surface_threshold,
            "buried_threshold": args.buried_threshold,
        })
    return pd.DataFrame(rows)


def summarize(out_df: pd.DataFrame, original_columns: Sequence[str], logs: Sequence[Mapping[str, object]], args: argparse.Namespace) -> Dict[str, object]:
    def nonmissing(col: str) -> int:
        return int(pd.to_numeric(out_df[col], errors="coerce").notna().sum()) if col in out_df.columns else 0
    def nonzero(col: str) -> int:
        return int(pd.to_numeric(out_df[col], errors="coerce").fillna(0).ne(0).sum()) if col in out_df.columns else 0
    logs_df = pd.DataFrame(logs)
    dssp_errors = []
    fs_errors = []
    if not logs_df.empty:
        if "dssp_error" in logs_df.columns:
            dssp_errors = logs_df.loc[logs_df["dssp_error"].fillna("").astype(str).ne(""), ["query_pdb", "dssp_error"]].head(20).to_dict("records")
        if "freesasa_error" in logs_df.columns:
            fs_errors = logs_df.loc[logs_df["freesasa_error"].fillna("").astype(str).ne(""), ["query_pdb", "freesasa_error"]].head(20).to_dict("records")
    highlighted = {
        "struct_sasa_abs_nonmissing": nonmissing("struct_sasa_abs"),
        "struct_rsa_rel_nonmissing": nonmissing("struct_rsa_rel"),
        "struct_surface_flag_rsa_ge_0p20": nonzero("struct_surface_flag_rsa_ge_0p20"),
        "struct_buried_flag_rsa_lt_0p05": nonzero("struct_buried_flag_rsa_lt_0p05"),
        "struct_dssp_found": nonzero("struct_dssp_found"),
        "struct_ss3_helix_flag": nonzero("struct_ss3_helix_flag"),
        "struct_ss3_sheet_flag": nonzero("struct_ss3_sheet_flag"),
        "struct_ss3_coil_flag": nonzero("struct_ss3_coil_flag"),
        "conservation_component_x_struct_rsa": nonzero("conservation_component_x_struct_rsa"),
        "ifrag_strength_x_struct_rsa": nonzero("ifrag_strength_x_struct_rsa"),
        "patch_score_x_struct_rsa": nonzero("patch_score_x_struct_rsa"),
        "radi_anchor_win5_x_conservation_max_x_struct_rsa": nonzero("radi_anchor_win5_x_conservation_max_x_struct_rsa"),
    }
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": Path(__file__).name,
        "feature_table": args.feature_table,
        "out_prefix": args.out_prefix,
        "windows": parse_windows(args.windows),
        "surface_threshold": args.surface_threshold,
        "buried_threshold": args.buried_threshold,
        "dssp_executable": args.dssp_executable,
        "skip_dssp": bool(args.skip_dssp),
        "n_rows": int(len(out_df)),
        "n_columns_input": int(len(original_columns)),
        "n_columns_output": int(len(out_df.columns)),
        "n_derived_columns": int(len(out_df.columns) - len(original_columns)),
        "n_unique_query_pdb": int(out_df["query_pdb"].nunique()) if "query_pdb" in out_df.columns else None,
        "n_unique_query_pdb_chain": int(out_df[["query_pdb", "query_chain"]].drop_duplicates().shape[0]) if {"query_pdb", "query_chain"}.issubset(out_df.columns) else None,
        "structure_logs_count": int(len(logs)),
        "freesasa_error_examples": fs_errors,
        "dssp_error_examples": dssp_errors,
        "highlighted_counts": highlighted,
        "outputs": {
            "feature_table": f"{args.out_prefix}.tsv",
            "feature_manifest": f"{args.out_prefix}.feature_manifest.tsv",
            "summary": f"{args.out_prefix}.summary.json",
            "structure_logs": f"{args.out_prefix}.structure_logs.tsv",
        },
    }


def json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if pd.isna(obj):
        return None
    raise TypeError(f"Not JSON serializable: {type(obj)!r}")


def main() -> None:
    args = parse_args()
    feature_table = Path(args.feature_table)
    if not feature_table.exists():
        fail(f"Feature table does not exist: {feature_table}")
    if freesasa is None:
        fail(f"Could not import freesasa: {FREESASA_IMPORT_ERROR}")
    dssp_exe = None
    if not args.skip_dssp:
        dssp_exe = find_executable(args.dssp_executable)
        if not dssp_exe:
            fail(f"Could not find DSSP executable {args.dssp_executable!r}")

    df = pd.read_csv(feature_table, sep="\t", low_memory=False)
    required = ["chainpair_id", "query_side", "query_pdb", "query_chain", "pdb_chain", "pdb_residue_id", "aa", "pdb_resname"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        fail("Missing required columns: " + ", ".join(missing))
    original_columns = list(df.columns)
    pdb_paths = sorted(df["query_pdb"].dropna().astype(str).unique())

    records, logs = collect_structure_records(pdb_paths, use_dssp=not args.skip_dssp, dssp_executable=dssp_exe)
    out_df = add_structure_features(df, records, args)

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    input_manifest = Path(args.input_manifest) if args.input_manifest else infer_manifest_path(feature_table)
    manifest = build_manifest(input_manifest if input_manifest.exists() else None, original_columns, list(out_df.columns), args)
    summary = summarize(out_df, original_columns, logs, args)

    feature_table_out = Path(f"{out_prefix}.tsv")
    manifest_out = Path(f"{out_prefix}.feature_manifest.tsv")
    summary_out = Path(f"{out_prefix}.summary.json")
    logs_out = Path(f"{out_prefix}.structure_logs.tsv")
    out_df.to_csv(feature_table_out, sep="\t", index=False)
    manifest.to_csv(manifest_out, sep="\t", index=False)
    pd.DataFrame(logs).to_csv(logs_out, sep="\t", index=False)
    summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True, default=json_default) + "\n")

    print("BM5 Phase 1 structure feature table written")
    print(f"  input rows: {len(df)}")
    print(f"  output rows: {len(out_df)}")
    print(f"  input columns: {len(original_columns)}")
    print(f"  output columns: {len(out_df.columns)}")
    print(f"  derived columns: {len(out_df.columns) - len(original_columns)}")
    print(f"  unique query_pdb: {len(pdb_paths)}")
    print(f"  DSSP executable: {dssp_exe or 'skipped'}")
    print("  outputs:")
    print(f"    feature table:    {feature_table_out}")
    print(f"    feature manifest: {manifest_out}")
    print(f"    summary:          {summary_out}")
    print(f"    structure logs:   {logs_out}")
    print("\nHighlighted counts:")
    for key, value in summary["highlighted_counts"].items():
        print(f"  {key}: {value}")
    if summary["freesasa_error_examples"]:
        print("\nFreeSASA error examples:")
        for row in summary["freesasa_error_examples"][:5]:
            print(f"  {row}")
    if summary["dssp_error_examples"]:
        print("\nDSSP error examples:")
        for row in summary["dssp_error_examples"][:5]:
            print(f"  {row}")


if __name__ == "__main__":
    main()
