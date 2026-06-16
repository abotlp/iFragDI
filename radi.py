#!/usr/bin/env python3
"""
Run raDI on a prebuilt paired interacting-homolog alignment.

Biologically, this script does one job:
- take the paired alignment already chosen by radi_prepare.py
- run raDI on that same alignment
- export the top inter-chain anchor matrix
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run raDI on a prebuilt paired interacting-homolog alignment."
    )
    parser.add_argument("--paired-msa", required=True, type=Path, help="Paired MSA written by radi_prepare.py.")
    parser.add_argument("--paired-ssa", required=True, type=Path, help="Paired SSA written by radi_prepare.py.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory.")
    parser.add_argument("--radi-bin", default="tools/RADI/bin/raDI", help="raDI executable.")
    parser.add_argument("--ra", type=int, default=1, help="raDI alphabet / reduced alphabet mode. Defaults to 1.")
    parser.add_argument(
        "--max-radi-pairs",
        type=int,
        default=40,
        help="Maximum number of ranked inter-chain raDI pairs kept in the output matrix.",
    )
    parser.add_argument("--query1-label", default="query1")
    parser.add_argument("--query2-label", default="query2")
    parser.add_argument("--no-heatmap", action="store_true", help="Skip heatmap PNG output.")
    parser.add_argument("--verbose", action="store_true", help="Print progress information.")
    args = parser.parse_args()

    if args.max_radi_pairs <= 0:
        raise SystemExit("--max-radi-pairs must be > 0")
    if not args.paired_msa.exists():
        raise SystemExit(f"Paired MSA not found: {args.paired_msa}")
    if not args.paired_ssa.exists():
        raise SystemExit(f"Paired SSA not found: {args.paired_ssa}")
    return args


def infer_chain_lengths_from_ssa(path: Path) -> tuple[int, int]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    if len(lines) < 4 or lines[0] != ">Query" or lines[2] != ">Structure":
        raise RuntimeError(f"{path}: unexpected SSA format")
    structure = lines[3]
    l1 = len(structure) - len(structure.lstrip("H"))
    l2 = len(structure) - l1
    if l1 <= 0 or l2 <= 0:
        raise RuntimeError(f"{path}: could not infer chain lengths from SSA")
    return l1, l2


def run_radi(radi_bin: str, msa_path: Path, ssa_path: Path, ra: int) -> Path:
    msa_path = msa_path.resolve()
    ssa_path = ssa_path.resolve()
    prefix = msa_path.with_name(f"{msa_path.stem}_RADI").resolve()
    cmd = (
        "ulimit -s unlimited && "
        f"{shlex.quote(radi_bin)} "
        f"-msa {shlex.quote(str(msa_path))} "
        f"-ssa {shlex.quote(str(ssa_path))} "
        f"-ra {ra} "
        f"-o {shlex.quote(str(prefix))}"
    )
    result = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            "raDI failed.\n"
            f"Command: {cmd}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
    di_path = prefix.with_name(f"{prefix.name}_ra{ra}_DI.out")
    if not di_path.exists():
        raise RuntimeError(f"raDI finished but DI output file was not found: {di_path}")
    return di_path


def parse_di_triplet(
    raw_pos_a: str,
    raw_pos_b: str,
    raw_score: str,
    expected_len: int,
) -> tuple[int, int, float, str] | None:
    try:
        pos_a = int(raw_pos_a)
        pos_b = int(raw_pos_b)
        score = float(raw_score)
    except ValueError:
        return None
    if 1 <= pos_a <= expected_len and 1 <= pos_b <= expected_len:
        return pos_a - 1, pos_b - 1, score, "1_based"
    if 0 <= pos_a < expected_len and 0 <= pos_b < expected_len:
        return pos_a, pos_b, score, "0_based"
    return None


def parse_interchain_radi_pairs(di_path: Path, l1: int, l2: int) -> tuple[List[Tuple[int, int, float]], dict[str, object]]:
    expected_len = l1 + l2
    pair_to_score: Dict[Tuple[int, int], float] = {}
    parsed_line_count = 0
    schema_counts: Counter[str] = Counter()
    indexing_counts: Counter[str] = Counter()
    with di_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parsed_line_count += 1
            parts = line.split()
            layouts: List[Tuple[str, int, int, int]] = []
            if len(parts) >= 7:
                layouts.append(("cols_4_5_6", 4, 5, 6))
            if len(parts) >= 3:
                layouts.append(("cols_0_1_2", 0, 1, 2))
                layouts.append(("cols_last3", len(parts) - 3, len(parts) - 2, len(parts) - 1))

            parsed = None
            schema_used = None
            for schema_name, pos_a_idx, pos_b_idx, score_idx in layouts:
                if pos_a_idx == pos_b_idx or pos_a_idx == score_idx or pos_b_idx == score_idx:
                    continue
                parsed = parse_di_triplet(
                    parts[pos_a_idx],
                    parts[pos_b_idx],
                    parts[score_idx],
                    expected_len,
                )
                if parsed is not None:
                    schema_used = schema_name
                    break
            if parsed is None or schema_used is None:
                continue
            pos_a, pos_b, score, indexing_used = parsed
            schema_counts[schema_used] += 1
            indexing_counts[indexing_used] += 1
            if pos_a < l1 and pos_b >= l1:
                i = pos_a
                j = pos_b - l1
            elif pos_b < l1 and pos_a >= l1:
                i = pos_b
                j = pos_a - l1
            else:
                continue
            key = (i, j)
            if score > pair_to_score.get(key, float("-inf")):
                pair_to_score[key] = score
    pairs = sorted(
        ((i, j, score) for (i, j), score in pair_to_score.items()),
        key=lambda item: item[2],
        reverse=True,
    )
    diagnostics = {
        "parsed_line_count": parsed_line_count,
        "schema_counts": dict(schema_counts),
        "indexing_counts": dict(indexing_counts),
        "schema_used": schema_counts.most_common(1)[0][0] if schema_counts else None,
        "indexing_detected": indexing_counts.most_common(1)[0][0] if indexing_counts else None,
        "unique_interchain_pairs_retained": len(pairs),
    }
    return pairs, diagnostics


def build_radi_matrix(l1: int, l2: int, radi_pairs: Iterable[Tuple[int, int, float]]) -> np.ndarray:
    matrix = np.zeros((l1, l2), dtype=float)
    for i, j, score in radi_pairs:
        if score > matrix[i, j]:
            matrix[i, j] = score
    return matrix


def write_top_pairs(path: Path, radi_pairs: Iterable[Tuple[int, int, float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["rank", "query1_index_1based", "query2_index_1based", "di_score"])
        for rank, (i, j, score) in enumerate(radi_pairs, start=1):
            writer.writerow([rank, i + 1, j + 1, f"{score:.6f}"])


def save_matrix_tsv(path: Path, matrix: np.ndarray) -> None:
    np.savetxt(path, matrix, fmt="%.6f", delimiter="\t")


def save_matrix_npy(path: Path, matrix: np.ndarray) -> None:
    np.save(path, matrix)


def write_heatmap(path: Path, matrix: np.ndarray, title: str, xlabel: str, ylabel: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    plt.imshow(matrix, cmap="YlGnBu", origin="upper", aspect="auto")
    plt.colorbar()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    l1, l2 = infer_chain_lengths_from_ssa(args.paired_ssa)
    di_path = run_radi(args.radi_bin, args.paired_msa, args.paired_ssa, args.ra)
    all_radi_pairs, diagnostics = parse_interchain_radi_pairs(di_path, l1, l2)
    radi_matrix = build_radi_matrix(l1, l2, all_radi_pairs)
    top_radi_pairs = all_radi_pairs[: args.max_radi_pairs]
    radi_heatmap_matrix = build_radi_matrix(l1, l2, top_radi_pairs)

    save_matrix_tsv(out_dir / "paired_interchain_matrix.tsv", radi_matrix)
    save_matrix_npy(out_dir / "paired_interchain_matrix.npy", radi_matrix)
    write_top_pairs(out_dir / "radi_top_pairs.tsv", top_radi_pairs)

    if not args.no_heatmap:
        write_heatmap(
            out_dir / "radi_heatmap.png",
            radi_heatmap_matrix,
            f"raDI Heatmap (top {len(top_radi_pairs)} pairs)",
            args.query2_label,
            args.query1_label,
        )

    summary = {
        "paired_msa_path": str(args.paired_msa),
        "paired_ssa_path": str(args.paired_ssa),
        "query1_length": l1,
        "query2_length": l2,
        "radi_di_path": str(di_path),
        "radi_interchain_pairs_retained": len(all_radi_pairs),
        "radi_top_pairs_written": len(top_radi_pairs),
        "radi_heatmap_pairs_shown": len(top_radi_pairs),
        "radi_matrix_shape": list(radi_matrix.shape),
        "di_parse_diagnostics": diagnostics,
        "outputs": {
            "paired_interchain_matrix_tsv": str(out_dir / "paired_interchain_matrix.tsv"),
            "paired_interchain_matrix_npy": str(out_dir / "paired_interchain_matrix.npy"),
            "radi_top_pairs_tsv": str(out_dir / "radi_top_pairs.tsv"),
            "radi_heatmap_png": str(out_dir / "radi_heatmap.png") if not args.no_heatmap else None,
            "radi_summary_json": str(out_dir / "radi_summary.json"),
        },
    }
    (out_dir / "radi_summary.json").write_text(json.dumps(summary, indent=2))

    if args.verbose:
        print(f"[INFO] Query lengths from SSA: q1={l1} q2={l2}")
        print(f"[INFO] raDI inter-chain pairs retained: {len(all_radi_pairs)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
