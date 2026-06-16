#!/usr/bin/env python3
"""Build and optionally execute iFragDI smoke-test commands from a manifest.

This runner intentionally stays thin: it reads an existing chain-pair manifest,
constructs calls to combine_ifrag_radi.py with the agreed smoke-test settings,
and records exactly what it would run or did run.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import shlex
import subprocess
import sys
from pathlib import Path


REQUIRED_COLUMNS = {
    "chainpair_id",
    "receptor_unbound_pdb",
    "ligand_unbound_pdb",
    "query1_chain",
    "query2_chain",
    "planned_output_dir",
}

COMMAND_LOG_COLUMNS = [
    "timestamp",
    "mode",
    "chainpair_id",
    "status",
    "reason",
    "planned_output_dir",
    "stdout_log",
    "stderr_log",
    "returncode",
    "query1_chain_arg",
    "query2_chain_arg",
    "command",
]

TRUE_VALUES = {"1", "true", "yes", "y"}
FALSE_VALUES = {"0", "false", "no", "n"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create or execute canonical iFragDI smoke-test commands from a BM5 chain-pair manifest."
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to chain-pair manifest TSV, relative to the project root or absolute.",
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--only-chainpair-ids",
        help="Comma-separated chainpair_id values to run, for example BM5CP00019,BM5CP00237.",
    )
    selection.add_argument(
        "--all",
        action="store_true",
        help="Select every row in the manifest.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Print and log commands without executing combine_ifrag_radi.py.",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="Execute combine_ifrag_radi.py and write per-job stdout/stderr logs.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Do not skip rows whose planned output already has consensus_summary.json.",
    )
    return parser.parse_args()


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(2)


def project_root_from_script() -> Path:
    # This script lives in benchmark/, so its parent directory is the project root.
    return Path(__file__).resolve().parents[1]


def resolve_under_project(project_root: Path, path_text: str) -> Path:
    if path_text is None or path_text == "":
        fail("Encountered an empty required path in the manifest.")
    path = Path(path_text)
    if path.is_absolute():
        return path
    return project_root / path


def load_manifest(manifest_path: Path) -> list[dict[str, str]]:
    if not manifest_path.exists():
        fail(f"Manifest not found: {manifest_path}")

    with manifest_path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            fail(f"Manifest has no header: {manifest_path}")
        missing = sorted(REQUIRED_COLUMNS - set(reader.fieldnames))
        if missing:
            fail(f"Manifest is missing required columns: {', '.join(missing)}")
        rows = list(reader)

    if not rows:
        fail(f"Manifest contains no data rows: {manifest_path}")
    return rows


def requested_ids(text: str | None) -> list[str]:
    if text is None:
        return []
    ids = [item.strip() for item in text.split(",") if item.strip()]
    if not ids:
        fail("--only-chainpair-ids was provided but no IDs were parsed.")
    duplicates = sorted({item for item in ids if ids.count(item) > 1})
    if duplicates:
        fail(f"Duplicate chainpair IDs requested: {', '.join(duplicates)}")
    return ids


def select_rows(rows: list[dict[str, str]], args: argparse.Namespace) -> list[dict[str, str]]:
    if args.all:
        return rows

    ids = requested_ids(args.only_chainpair_ids)
    by_id = {row["chainpair_id"]: row for row in rows}
    missing = [chainpair_id for chainpair_id in ids if chainpair_id not in by_id]
    if missing:
        fail(f"Requested chainpair IDs not found in manifest: {', '.join(missing)}")
    return [by_id[chainpair_id] for chainpair_id in ids]


def validate_runnable(row: dict[str, str]) -> None:
    value = row.get("chainpair_runnable")
    if value is None or value == "":
        return
    normalized = value.strip().lower()
    if normalized in TRUE_VALUES:
        return
    if normalized in FALSE_VALUES:
        fail(f"{row['chainpair_id']} is marked chainpair_runnable={value!r}; refusing to run it.")
    fail(f"{row['chainpair_id']} has unrecognized chainpair_runnable value {value!r}.")


def chain_arg(value: str, column: str, chainpair_id: str) -> str:
    """Convert manifest chain placeholders to combine_ifrag_radi.py arguments.

    combine_ifrag_radi.py expects a blank PDB chain as the literal one-character
    string " ". The manifest uses "_" as the blank-chain placeholder, so this
    function performs that conversion deliberately instead of dropping the flag.
    """
    if value is None or value == "":
        fail(f"{chainpair_id} has an empty {column}; use '_' for a blank PDB chain.")
    if value == "_":
        return " "
    if len(value) != 1:
        fail(f"{chainpair_id} has {column}={value!r}; expected one character or '_'.")
    return value


def build_command(row: dict[str, str]) -> tuple[list[str], str, str]:
    chainpair_id = row["chainpair_id"]
    query1_chain = chain_arg(row["query1_chain"], "query1_chain", chainpair_id)
    query2_chain = chain_arg(row["query2_chain"], "query2_chain", chainpair_id)

    command = [
        "python3",
        "combine_ifrag_radi.py",
        "--query1-pdb",
        row["receptor_unbound_pdb"],
        "--query2-pdb",
        row["ligand_unbound_pdb"],
        "--query1-chain",
        query1_chain,
        "--query2-chain",
        query2_chain,
        "--combine-mode",
        "ifrag_conservation_radi",
        "--ifrag-template-dataset",
        "intact_biogrid",
        "--radi-pair-dataset",
        "intact_biogrid",
        "--homolog-search-mode",
        "template_iterative",
        "--radi-ra",
        "1",
        "--out-dir",
        row["planned_output_dir"],
    ]
    return command, query1_chain, query2_chain


def append_command_record(log_path: Path, record: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not log_path.exists() or log_path.stat().st_size == 0
    with log_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=COMMAND_LOG_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(record)


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def main() -> int:
    args = parse_args()
    project_root = project_root_from_script()
    manifest_path = resolve_under_project(project_root, args.manifest)
    command_log = project_root / "benchmark" / "logs" / "ifragdi_smoke_commands.tsv"
    smoke_log_dir = project_root / "benchmark" / "logs" / "ifragdi_smoke"

    rows = load_manifest(manifest_path)
    selected = select_rows(rows, args)
    mode = "execute" if args.execute else "dry_run"

    print(f"Project root: {project_root}")
    print(f"Manifest: {manifest_path}")
    print(f"Selected rows: {len(selected)}")
    print(f"Mode: {mode}")
    print(f"Command log: {command_log}")
    print(f"Per-job logs: {smoke_log_dir}")

    smoke_log_dir.mkdir(parents=True, exist_ok=True)
    failures = 0

    for row in selected:
        chainpair_id = row["chainpair_id"]
        validate_runnable(row)
        command, query1_chain, query2_chain = build_command(row)
        out_dir = resolve_under_project(project_root, row["planned_output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        done_marker = out_dir / "consensus_summary.json"
        stdout_log = smoke_log_dir / f"{chainpair_id}.stdout.log"
        stderr_log = smoke_log_dir / f"{chainpair_id}.stderr.log"
        command_text = shlex.join(command)

        status = "dry_run"
        reason = ""
        returncode = ""

        if done_marker.exists() and not args.force:
            status = "skipped_completed"
            reason = f"Found existing completion marker: {done_marker}"
            print(f"[SKIP] {chainpair_id}: {reason}")
        elif args.dry_run:
            print(f"[DRY-RUN] {chainpair_id}: {command_text}")
        else:
            print(f"[RUN] {chainpair_id}: {command_text}")
            with stdout_log.open("w") as out_handle, stderr_log.open("w") as err_handle:
                completed = subprocess.run(
                    command,
                    cwd=project_root,
                    stdout=out_handle,
                    stderr=err_handle,
                    text=True,
                    check=False,
                )
            returncode = str(completed.returncode)
            if completed.returncode == 0:
                status = "completed"
                print(f"[OK] {chainpair_id}: returncode 0")
            else:
                status = "failed"
                failures += 1
                print(f"[FAIL] {chainpair_id}: returncode {completed.returncode}", file=sys.stderr)

        append_command_record(
            command_log,
            {
                "timestamp": now_iso(),
                "mode": mode,
                "chainpair_id": chainpair_id,
                "status": status,
                "reason": reason,
                "planned_output_dir": row["planned_output_dir"],
                "stdout_log": str(stdout_log.relative_to(project_root)),
                "stderr_log": str(stderr_log.relative_to(project_root)),
                "returncode": returncode,
                "query1_chain_arg": "blank" if query1_chain == " " else query1_chain,
                "query2_chain_arg": "blank" if query2_chain == " " else query2_chain,
                "command": command_text,
            },
        )

    if failures:
        print(f"Finished with {failures} failed job(s).", file=sys.stderr)
        return 1

    print("Finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
