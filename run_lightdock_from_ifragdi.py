#!/usr/bin/env python3
"""
Run LightDock as a post-prediction stage from an iFragDI output directory.

This wrapper follows the official LightDock residue-restraints tutorial:
1. setup with residue restraints
2. simulation
3. generate conformations per swarm
4. cluster intra-swarm
5. rank
6. filter by restraint satisfaction
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable


RESTRAINT_MODE_TO_FILE = {
    "strict_active": "lightdock_restraints.strict_active.list",
    "strict": "lightdock_restraints.strict.list",
    "loose": "lightdock_restraints.loose.list",
    "query1_only": "lightdock_restraints.query1_only.strict.list",
    "query2_only": "lightdock_restraints.query2_only.strict.list",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run LightDock from a completed iFragDI prediction directory."
    )
    p.add_argument("--combine-out-dir", required=True, type=Path)
    p.add_argument(
        "--restraint-mode",
        choices=tuple(RESTRAINT_MODE_TO_FILE),
        default="strict_active",
        help="Which exported iFragDI LightDock restraint file to use.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="LightDock run directory. Defaults to <combine-out-dir>/lightdock_<restraint-mode>.",
    )
    p.add_argument(
        "--query1-pdb",
        type=Path,
        default=None,
        help="Optional receptor PDB override. If omitted, use query1.surface_input_chain_*.pdb from the combine output.",
    )
    p.add_argument(
        "--query2-pdb",
        type=Path,
        default=None,
        help="Optional ligand PDB override. If omitted, use query2.surface_input_chain_*.pdb from the combine output.",
    )
    p.add_argument("--setup-bin", default="lightdock3_setup.py")
    p.add_argument("--lightdock-bin", default="lightdock3.py")
    p.add_argument("--generate-bin", default="lgd_generate_conformations.py")
    p.add_argument("--cluster-bin", default="lgd_cluster_bsas.py")
    p.add_argument("--rank-bin", default="lgd_rank.py")
    p.add_argument("--filter-bin", default="lgd_filter_restraints.py")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--cores", type=int, default=8, help="Cores passed to LightDock simulation.")
    p.add_argument(
        "--post-cores",
        type=int,
        default=8,
        help="Parallel worker count for structure generation and clustering across swarms.",
    )
    p.add_argument("--glowworms", type=int, default=200)
    p.add_argument("--scoring", default="fastdfire")
    p.add_argument("--cutoff", type=float, default=5.0)
    p.add_argument("--fnat", type=float, default=0.4)
    p.add_argument("--noxt", action="store_true", default=True)
    p.add_argument("--noh", action="store_true", default=True)
    p.add_argument("--now", action="store_true", default=True)
    p.add_argument("--anm", action="store_true", default=True)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def require_file(path: Path, what: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{what} not found: {path}")
    return path


def pick_single(pattern: str, base_dir: Path, what: str) -> Path | None:
    matches = sorted(base_dir.glob(pattern))
    if not matches:
        return None
    if len(matches) > 1:
        raise RuntimeError(
            f"Found multiple {what} files in {base_dir}: {', '.join(str(m.name) for m in matches)}"
        )
    return matches[0]


def detect_first_protein_chain(path: Path) -> str:
    with path.open() as handle:
        for raw in handle:
            if not raw.startswith(("ATOM", "HETATM")):
                continue
            resname = raw[17:20].strip().upper()
            if len(resname) != 3:
                continue
            return raw[21:22]
    raise RuntimeError(f"Could not detect a protein chain in {path}")


def run_command(cmd: list[str], cwd: Path, log_path: Path | None = None) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if log_path is not None:
        log_path.write_text(
            f"$ {' '.join(cmd)}\n\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}\n",
            encoding="utf-8",
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit {result.returncode}) in {cwd}:\n"
            f"{' '.join(cmd)}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
    return result


def run_many(cmds: Iterable[tuple[list[str], Path, Path]], max_workers: int) -> None:
    items = list(cmds)
    if not items:
        return
    if max_workers <= 1 or len(items) == 1:
        for cmd, cwd, log_path in items:
            run_command(cmd, cwd=cwd, log_path=log_path)
        return
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(run_command, cmd, cwd, log_path): (cmd, cwd)
            for cmd, cwd, log_path in items
        }
        for fut in as_completed(futures):
            fut.result()


def main() -> int:
    args = parse_args()
    combine_out_dir = require_file(args.combine_out_dir / "consensus_summary.json", "consensus summary").parent
    restraint_file = require_file(
        combine_out_dir / RESTRAINT_MODE_TO_FILE[args.restraint_mode],
        f"LightDock restraint file for mode '{args.restraint_mode}'",
    )

    receptor_pdb = args.query1_pdb or pick_single("query1.surface_input_chain_*.pdb", combine_out_dir, "query1 surface-input PDB")
    ligand_pdb = args.query2_pdb or pick_single("query2.surface_input_chain_*.pdb", combine_out_dir, "query2 surface-input PDB")
    if receptor_pdb is None or ligand_pdb is None:
        raise RuntimeError(
            "Could not auto-detect chain-only receptor/ligand PDBs from the combine output. "
            "Pass --query1-pdb and --query2-pdb explicitly."
        )
    require_file(receptor_pdb, "receptor PDB")
    require_file(ligand_pdb, "ligand PDB")

    receptor_chain = detect_first_protein_chain(receptor_pdb)
    ligand_chain = detect_first_protein_chain(ligand_pdb)

    run_dir = args.out_dir or (combine_out_dir / f"lightdock_{args.restraint_mode}")
    if run_dir.exists():
        existing = list(run_dir.iterdir())
        if existing:
            raise RuntimeError(
                f"LightDock output directory already exists and is not empty: {run_dir}\n"
                "Choose a new --out-dir."
            )
    run_dir.mkdir(parents=True, exist_ok=True)

    receptor_copy = run_dir / "receptor.pdb"
    ligand_copy = run_dir / "ligand.pdb"
    restraints_copy = run_dir / "restraints.list"
    shutil.copyfile(receptor_pdb, receptor_copy)
    shutil.copyfile(ligand_pdb, ligand_copy)
    shutil.copyfile(restraint_file, restraints_copy)

    setup_cmd = [args.setup_bin, receptor_copy.name, ligand_copy.name]
    if args.noxt:
        setup_cmd.append("--noxt")
    if args.noh:
        setup_cmd.append("--noh")
    if args.now:
        setup_cmd.append("--now")
    if args.anm:
        setup_cmd.append("-anm")
    setup_cmd.extend(["-rst", restraints_copy.name])
    run_command(setup_cmd, cwd=run_dir, log_path=run_dir / "lightdock_setup.log")

    simulation_cmd = [
        args.lightdock_bin,
        "setup.json",
        str(args.steps),
        "-s",
        args.scoring,
        "-c",
        str(args.cores),
    ]
    run_command(simulation_cmd, cwd=run_dir, log_path=run_dir / "lightdock_simulation.log")

    swarm_dirs = sorted(path for path in run_dir.glob("swarm_*") if path.is_dir())
    if not swarm_dirs:
        raise RuntimeError(f"No swarm_* directories were produced in {run_dir}")

    gso_file = f"gso_{args.steps}.out"
    generate_cmds: list[tuple[list[str], Path, Path]] = []
    cluster_cmds: list[tuple[list[str], Path, Path]] = []
    for swarm_dir in swarm_dirs:
        generate_cmds.append(
            (
                [args.generate_bin, "../receptor.pdb", "../ligand.pdb", gso_file, str(args.glowworms)],
                swarm_dir,
                swarm_dir / "generate.log",
            )
        )
        cluster_cmds.append(
            (
                [args.cluster_bin, gso_file],
                swarm_dir,
                swarm_dir / "cluster.log",
            )
        )

    run_many(generate_cmds, max_workers=args.post_cores)
    run_many(cluster_cmds, max_workers=args.post_cores)

    rank_cmd = [args.rank_bin, str(len(swarm_dirs)), str(args.steps)]
    run_command(rank_cmd, cwd=run_dir, log_path=run_dir / "lightdock_rank.log")

    filter_cmd = [
        args.filter_bin,
        "--cutoff",
        str(args.cutoff),
        "--fnat",
        str(args.fnat),
        "rank_by_scoring.list",
        restraints_copy.name,
        receptor_chain.strip() or " ",
        ligand_chain.strip() or " ",
    ]
    run_command(filter_cmd, cwd=run_dir, log_path=run_dir / "lightdock_filter.log")

    summary = {
        "combine_out_dir": str(combine_out_dir),
        "restraint_mode": args.restraint_mode,
        "receptor_pdb": str(receptor_pdb),
        "ligand_pdb": str(ligand_pdb),
        "receptor_chain": receptor_chain,
        "ligand_chain": ligand_chain,
        "restraints_file": str(restraint_file),
        "run_dir": str(run_dir),
        "steps": args.steps,
        "cores": args.cores,
        "post_cores": args.post_cores,
        "glowworms": args.glowworms,
        "scoring": args.scoring,
        "cutoff": args.cutoff,
        "fnat": args.fnat,
        "swarm_count": len(swarm_dirs),
        "commands": {
            "setup": setup_cmd,
            "simulation": simulation_cmd,
            "rank": rank_cmd,
            "filter": filter_cmd,
        },
        "outputs": {
            "setup_json": str(run_dir / "setup.json"),
            "rank_by_scoring": str(run_dir / "rank_by_scoring.list"),
            "filtered_dir": str(run_dir / "filtered"),
            "rank_filtered": str(run_dir / "filtered" / "rank_filtered.list"),
            "setup_log": str(run_dir / "lightdock_setup.log"),
            "simulation_log": str(run_dir / "lightdock_simulation.log"),
            "rank_log": str(run_dir / "lightdock_rank.log"),
            "filter_log": str(run_dir / "lightdock_filter.log"),
        },
    }
    (run_dir / "lightdock_run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if args.verbose:
        print(f"[INFO] Receptor PDB: {receptor_pdb}")
        print(f"[INFO] Ligand PDB: {ligand_pdb}")
        print(f"[INFO] Restraints: {restraint_file}")
        print(f"[INFO] Swarms: {len(swarm_dirs)}")
        print(f"[INFO] LightDock run complete: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
