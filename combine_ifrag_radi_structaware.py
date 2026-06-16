#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from structure_features import rerank_with_structure_features
from template_mmseqs import HOMOLOG_SEARCH_MODE_CHOICES


AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "MSE": "M",
}
RADI_PAIR_DATASET_CHOICES = ("intact_biogrid", "intact_biogrid_string")


@dataclass(frozen=True)
class ResolvedQuery:
    label: str
    source_type: str
    sequence: str
    fasta_path: Path
    pdb_path: Path | None
    chain: str | None
    header: str
    pdb_residue_ids: list[tuple[str, str]] | None
    pdb_residue_labels: list[str] | None
    pdb_residue_coords: np.ndarray | None


def radi_pair_dataset_defaults(dataset_name: str) -> dict[str, Path]:
    return {
        "pairs": Path("data") / "datasets" / dataset_name / "template_pairs.final.tsv",
        "pairs_meta": Path("data") / "datasets" / dataset_name / "template_pairs.meta.final.tsv",
    }


def homolog_template_dataset_defaults(dataset_name: str) -> dict[str, Path]:
    return {
        "template_fasta": Path("data") / "interaction_templates" / dataset_name / "templates.fasta",
        "template_proteins": Path("data") / "interaction_templates" / dataset_name / "proteins.final.tsv",
        "template_mmseqs_db": Path("data") / "db" / f"mmseqs_templates_{dataset_name}" / "templates_db",
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Classical combine_ifrag_radi.py pipeline with optional structure-aware reranking of final residue scores."
    )
    p.add_argument("--query1-fasta", type=Path)
    p.add_argument("--query2-fasta", type=Path)
    p.add_argument("--query1-pdb", type=Path)
    p.add_argument("--query2-pdb", type=Path)
    p.add_argument("--query1-chain", default=None)
    p.add_argument("--query2-chain", default=None)
    p.add_argument(
        "--query1-structure-source",
        choices=("experimental", "alphafold_like", "auto"),
        default="auto",
        help=(
            "How to interpret query1 structural confidence. "
            "'experimental' disables B-factor-as-confidence logic, "
            "'alphafold_like' treats B-factors as pLDDT, and 'auto' keeps heuristic detection."
        ),
    )
    p.add_argument(
        "--query2-structure-source",
        choices=("experimental", "alphafold_like", "auto"),
        default="auto",
        help=(
            "How to interpret query2 structural confidence. "
            "'experimental' disables B-factor-as-confidence logic, "
            "'alphafold_like' treats B-factors as pLDDT, and 'auto' keeps heuristic detection."
        ),
    )
    p.add_argument(
        "--interaction-mode",
        choices=("heteromer", "homomer", "auto"),
        default="heteromer",
        help="Interpret query pairing as heteromer, homomer, or auto-resolve from identical query sequences.",
    )
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--threads", type=int, default=min(os.cpu_count() or 1, 8))
    p.add_argument(
        "--ifrag-template-dataset",
        choices=("intact_biogrid", "intact_biogrid_string"),
        default="intact_biogrid_string",
        help=(
            "Template dataset used by iFrag. "
            "'intact_biogrid' is the curated core; "
            "'intact_biogrid_string' is the STRING-expanded universe."
        ),
    )
    p.add_argument(
        "--ifrag-pairs",
        type=Path,
        default=None,
        help="Optional override for the iFrag pair table. If omitted, iFrag resolves it from --ifrag-template-dataset.",
    )
    p.add_argument(
        "--ifrag-blast-db",
        "--ifrag-mmseqs-db",
        dest="ifrag_blast_db",
        type=Path,
        default=None,
        help="BLAST database prefix for classical iFrag. --ifrag-mmseqs-db is accepted as a deprecated alias.",
    )
    p.add_argument(
        "--ifrag-template-fasta",
        type=Path,
        default=None,
        help="Optional override for the template FASTA used by iFrag for query-specific redundancy filtering.",
    )
    p.add_argument(
        "--ifrag-pair-method-substring",
        action="append",
        default=[],
        help=(
            "Case-insensitive substring filter applied to iFrag template-pair detection_method values. "
            "Can be passed multiple times."
        ),
    )
    p.add_argument("--ifrag-evalue", type=float, default=0.01)
    p.add_argument(
        "--ifrag-redundancy-mode",
        choices=("cdhit_cluster_pair", "greedy_identity"),
        default="cdhit_cluster_pair",
        help="Template nonredundancy mode forwarded to ifrags.py. Defaults to the CD-HIT cluster-pair path.",
    )
    p.add_argument(
        "--ifrag-cdhit-bin",
        default="cd-hit",
        help="cd-hit executable forwarded to ifrags.py when --ifrag-redundancy-mode cdhit_cluster_pair.",
    )
    p.add_argument(
        "--ifrag-identity-threshold",
        type=float,
        default=40.0,
        help="Template-protein identity threshold forwarded to ifrags.py.",
    )
    p.add_argument("--ifrag-max-target-seqs", type=int, default=100000)
    p.add_argument("--ifrag-min-pident", type=float, default=0.0)
    p.add_argument("--ifrag-min-aln-len", type=int, default=1)
    p.add_argument("--ifrag-min-cov1", type=float, default=0.0)
    p.add_argument("--ifrag-max-cov1", type=float, default=1.0)
    p.add_argument("--ifrag-min-cov2", type=float, default=0.0)
    p.add_argument("--ifrag-max-cov2", type=float, default=1.0)
    p.add_argument("--ifrag-top-k", type=int, default=None)
    p.add_argument(
        "--combine-mode",
        choices=("ifrag_radi", "conservation_radi", "ifrag_conservation", "ifrag_conservation_radi", "ifrag_blastpdb"),
        default="ifrag_conservation_radi",
        help=(
            "How to combine template-derived signals with raDI. "
            "'ifrag_radi' uses classical iFrag plus raDI. "
            "'conservation_radi' uses conservation plus raDI. "
            "'ifrag_conservation' uses iFrag plus conservation without raDI. "
            "'ifrag_conservation_radi' uses both template-derived branches plus raDI. "
            "'ifrag_blastpdb' uses classical iFrag plus optional blastPDB structural anchors without conservation or raDI."
        ),
    )
    p.add_argument(
        "--radi-pair-dataset",
        choices=RADI_PAIR_DATASET_CHOICES,
        default="intact_biogrid_string",
        help=(
            "Pair universe used by conservation.py and radi_prepare.py. "
            "'intact_biogrid' is the curated core; "
            "'intact_biogrid_string' is the STRING-expanded universe."
        ),
    )
    p.add_argument(
        "--radi-pairs",
        type=Path,
        default=None,
        help="Optional override for the homolog-side pair table. If omitted, it is resolved from --radi-pair-dataset.",
    )
    p.add_argument(
        "--radi-pairs-meta",
        type=Path,
        default=None,
        help="Optional override for the homolog-side pair metadata table. If omitted, it is resolved from --radi-pair-dataset.",
    )
    p.add_argument(
        "--radi-sequence-fasta",
        type=Path,
        default=None,
        help="Full-sequence FASTA used by conservation.py and radi_prepare.py before per-chain FAMSA. Defaults to the selected interaction-template FASTA.",
    )
    p.add_argument(
        "--radi-template-mmseqs-db",
        type=Path,
        default=None,
        help="MMseqs DB prefix used by the shared template-backed homolog-search stage. Defaults to the selected interaction-template dataset resource.",
    )
    p.add_argument(
        "--radi-template-proteins",
        type=Path,
        default=None,
        help="proteins.final.tsv for the selected interaction-template dataset, used to recover taxids for shared homolog hits.",
    )
    p.add_argument(
        "--homolog-search-mode",
        "--radi-homolog-search-mode",
        dest="homolog_search_mode",
        choices=HOMOLOG_SEARCH_MODE_CHOICES,
        default="template_iterative",
        help=(
            "Shared homolog-search mode for conservation and raDI. "
            "'template_iterative' runs one 4-iteration MMseqs search on the interaction-template DB; "
            "'template_single_pass' is a faster smoke-test mode."
        ),
    )
    p.add_argument("--radi-mmseqs-bin", default="mmseqs")
    p.add_argument("--radi-famsa-bin", default="famsa")
    p.add_argument(
        "--radi-stage1-iterations",
        "--radi-iterations",
        dest="radi_stage1_iterations",
        type=int,
        default=4,
        help="Number of MMseqs iterations used in template_iterative mode.",
    )
    p.add_argument("--radi-bin", default="tools/RADI/bin/raDI")
    p.add_argument(
        "--radi-ra",
        type=int,
        default=1,
        help="raDI alphabet / reduced alphabet mode forwarded to radi.py. Defaults to 1.",
    )
    p.add_argument(
        "--radi-evalue",
        type=float,
        default=None,
        help="Optional MMseqs E-value cutoff for the shared homolog search. If omitted, keep the MMseqs default like the original RADI workflow.",
    )
    p.add_argument(
        "--radi-max-hits",
        "--radi-max-seqs",
        dest="radi_max_hits",
        type=int,
        default=100000,
        help="Maximum homologs retained by the shared template-backed MMseqs search. --radi-max-seqs is accepted as a deprecated alias.",
    )
    p.add_argument("--radi-mmseqs-sensitivity", "--radi-sensitivity", dest="radi_mmseqs_sensitivity", type=float, default=7.5)
    p.add_argument(
        "--radi-min-trusted-paired-rows",
        type=int,
        default=20,
        help="If radi paired-row depth is below this, ignore radi anchors in ifrag_conservation_radi mode.",
    )
    p.add_argument(
        "--use-blastpdb",
        action="store_true",
        help="Run the experimental-PDB blastPDB branch and use it as optional structural anchor evidence.",
    )
    p.add_argument(
        "--blastpdb-cache-dir",
        type=Path,
        default=Path("data/cache/blastpdb"),
        help="Cache directory for blastPDB remote search results, downloaded assemblies, and extracted contacts.",
    )
    p.add_argument(
        "--blastpdb-blast-bin",
        default="blastp",
        help="blastp executable used by blastPDB for local chain mapping.",
    )
    p.add_argument(
        "--blastpdb-makeblastdb-bin",
        default="makeblastdb",
        help="makeblastdb executable used by blastPDB for temporary per-assembly chain databases.",
    )
    p.add_argument(
        "--blastpdb-top-assemblies",
        type=int,
        default=25,
        help="Maximum number of shared candidate biological assemblies kept by blastPDB after remote discovery.",
    )
    p.add_argument(
        "--blastpdb-sequence-search-identity-cutoff",
        type=float,
        default=0.30,
        help="RCSB sequence-search identity cutoff used by blastPDB candidate discovery.",
    )
    p.add_argument(
        "--blastpdb-sequence-search-evalue-cutoff",
        type=float,
        default=1.0,
        help="RCSB sequence-search E-value cutoff used by blastPDB candidate discovery.",
    )
    p.add_argument(
        "--blastpdb-local-blast-evalue",
        type=float,
        default=0.01,
        help="Local BLAST E-value used by blastPDB to remap query sequences onto candidate assembly chains.",
    )
    p.add_argument(
        "--blastpdb-local-blast-max-target-seqs",
        type=int,
        default=1000,
        help="Maximum local chain hits retained per query/assembly by blastPDB.",
    )
    p.add_argument(
        "--blastpdb-cbeta-threshold",
        type=float,
        default=12.0,
        help="C-beta contact threshold in angstroms used by blastPDB. Gly falls back to CA.",
    )
    p.add_argument(
        "--blastpdb-min-template-contacts",
        type=int,
        default=5,
        help="Minimum number of structural contacts required to keep an assembly chain pair as a blastPDB template.",
    )
    p.add_argument(
        "--blastpdb-min-trusted-templates",
        type=int,
        default=1,
        help="If fewer blastPDB templates survive contact transfer, ignore blastPDB anchors during residue scoring.",
    )
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--radi-top-pairs-consensus", type=int, default=40)
    p.add_argument(
        "--overlay-top-pairs",
        type=int,
        default=None,
        help="Optional number of raDI pairs to draw in the overlay. Defaults to --radi-top-pairs-consensus.",
    )
    p.add_argument(
        "--strict-active-residues-per-chain",
        type=int,
        default=4,
        help="Recommended high-confidence active residues per chain for the primary docking set.",
    )
    p.add_argument(
        "--strict-passive-residues-per-chain",
        type=int,
        default=4,
        help="Recommended high-confidence passive residues per chain for the primary docking set.",
    )
    p.add_argument(
        "--active-residues-per-chain",
        type=int,
        default=8,
        help="Broader active residues per chain for the loose docking set.",
    )
    p.add_argument(
        "--passive-residues-per-chain",
        type=int,
        default=8,
        help="Broader passive residues per chain for the loose docking set.",
    )
    p.add_argument(
        "--patch-residues-per-chain",
        type=int,
        default=16,
        help="Maximum number of residues per chain kept in the conservation-defined patch mask for ifrag_conservation_radi mode.",
    )
    p.add_argument(
        "--surface-rsa-threshold",
        type=float,
        default=20.0,
        help="Minimum residue RSA percentage kept by the general SASA surface prior when query PDB input is available.",
    )
    p.add_argument(
        "--structaware-mode",
        choices=("off", "rerank"),
        default="rerank",
        help="Apply structure-aware reranking on top of the classical residue score without changing the classical evidence branches.",
    )
    p.add_argument(
        "--structaware-confidence-mode",
        choices=("auto", "plddt_bfactor", "off"),
        default="auto",
        help="How to interpret PDB B-factors as confidence values for the structure-aware reranker.",
    )
    p.add_argument(
        "--structaware-hydrophobic-weight",
        type=float,
        default=0.02,
        help=(
            "Weight of the hydrophobic patchiness term in the structure-aware reranker. "
            "Set to 0 to disable it."
        ),
    )
    p.add_argument("--no-heatmap", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    if args.threads <= 0:
        raise SystemExit("--threads must be > 0")
    if args.ifrag_max_target_seqs <= 0:
        raise SystemExit("--ifrag-max-target-seqs must be > 0")
    if args.ifrag_min_aln_len <= 0:
        raise SystemExit("--ifrag-min-aln-len must be > 0")
    if not (0.0 <= args.ifrag_identity_threshold <= 100.0):
        raise SystemExit("--ifrag-identity-threshold must be in [0, 100]")
    if args.ifrag_top_k is not None and args.ifrag_top_k <= 0:
        raise SystemExit("--ifrag-top-k must be > 0")
    if args.top_k <= 0:
        raise SystemExit("--top-k must be > 0")
    if args.radi_top_pairs_consensus <= 0:
        raise SystemExit("--radi-top-pairs-consensus must be > 0")
    if args.overlay_top_pairs is not None and args.overlay_top_pairs <= 0:
        raise SystemExit("--overlay-top-pairs must be > 0")
    if args.strict_active_residues_per_chain <= 0:
        raise SystemExit("--strict-active-residues-per-chain must be > 0")
    if args.strict_passive_residues_per_chain < 0:
        raise SystemExit("--strict-passive-residues-per-chain must be >= 0")
    if args.active_residues_per_chain <= 0:
        raise SystemExit("--active-residues-per-chain must be > 0")
    if args.passive_residues_per_chain < 0:
        raise SystemExit("--passive-residues-per-chain must be >= 0")
    if args.patch_residues_per_chain <= 0:
        raise SystemExit("--patch-residues-per-chain must be > 0")
    if args.radi_stage1_iterations <= 0:
        raise SystemExit("--radi-stage1-iterations must be > 0")
    if args.radi_max_hits <= 0:
        raise SystemExit("--radi-max-hits must be > 0")
    if args.radi_min_trusted_paired_rows < 0:
        raise SystemExit("--radi-min-trusted-paired-rows must be >= 0")
    if args.blastpdb_top_assemblies <= 0:
        raise SystemExit("--blastpdb-top-assemblies must be > 0")
    if not (0.0 < args.blastpdb_sequence_search_identity_cutoff <= 1.0):
        raise SystemExit("--blastpdb-sequence-search-identity-cutoff must be in (0, 1]")
    if args.blastpdb_sequence_search_evalue_cutoff <= 0.0:
        raise SystemExit("--blastpdb-sequence-search-evalue-cutoff must be > 0")
    if args.blastpdb_local_blast_evalue <= 0.0:
        raise SystemExit("--blastpdb-local-blast-evalue must be > 0")
    if args.blastpdb_local_blast_max_target_seqs <= 0:
        raise SystemExit("--blastpdb-local-blast-max-target-seqs must be > 0")
    if args.blastpdb_cbeta_threshold <= 0.0:
        raise SystemExit("--blastpdb-cbeta-threshold must be > 0")
    if args.blastpdb_min_template_contacts <= 0:
        raise SystemExit("--blastpdb-min-template-contacts must be > 0")
    if args.blastpdb_min_trusted_templates < 0:
        raise SystemExit("--blastpdb-min-trusted-templates must be >= 0")
    if args.surface_rsa_threshold < 0.0:
        raise SystemExit("--surface-rsa-threshold must be >= 0")
    if args.structaware_hydrophobic_weight < 0.0:
        raise SystemExit("--structaware-hydrophobic-weight must be >= 0")
    radi_pair_defaults = radi_pair_dataset_defaults(args.radi_pair_dataset)
    template_defaults = homolog_template_dataset_defaults(args.radi_pair_dataset)
    if args.radi_pairs is None:
        args.radi_pairs = radi_pair_defaults["pairs"]
    if args.radi_pairs_meta is None:
        args.radi_pairs_meta = radi_pair_defaults["pairs_meta"]
    if args.radi_sequence_fasta is None:
        args.radi_sequence_fasta = template_defaults["template_fasta"]
    if args.radi_template_mmseqs_db is None:
        args.radi_template_mmseqs_db = template_defaults["template_mmseqs_db"]
    if args.radi_template_proteins is None:
        args.radi_template_proteins = template_defaults["template_proteins"]
    if not args.radi_sequence_fasta.exists():
        raise SystemExit(f"Template sequence FASTA not found: {args.radi_sequence_fasta}")
    if not args.radi_template_proteins.exists():
        raise SystemExit(f"Template proteins table not found: {args.radi_template_proteins}")
    validate_input_presence(args.query1_fasta, args.query1_pdb, "query1")
    validate_input_presence(args.query2_fasta, args.query2_pdb, "query2")
    return args


def validate_input_presence(fasta: Path | None, pdb: Path | None, label: str) -> None:
    if fasta is None and pdb is None:
        raise SystemExit(f"{label}: provide at least one of --{label}-fasta or --{label}-pdb")


def resolve_structure_confidence_mode(global_mode: str, structure_source: str) -> str:
    if global_mode == "off":
        return "off"
    if structure_source == "experimental":
        return "off"
    if structure_source == "alphafold_like":
        return "plddt_bfactor"
    return global_mode


def resolve_interaction_mode(requested_mode: str, q1_seq: str, q2_seq: str) -> str:
    if requested_mode == "auto":
        return "homomer" if q1_seq == q2_seq else "heteromer"
    if requested_mode == "homomer" and q1_seq != q2_seq:
        raise ValueError("homomer mode currently requires identical query sequences")
    return requested_mode


def run_command(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd),
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"Executable not found: {cmd[0]}") from exc
    return result


def parse_single_fasta(path: Path) -> tuple[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"FASTA file not found: {path}")

    opener = gzip.open if path.suffix == ".gz" else open
    records: list[tuple[str, str]] = []
    header: str | None = None
    chunks: list[str] = []
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(chunks)))
                header = line[1:].strip() or "query"
                chunks = []
            else:
                if header is None:
                    raise ValueError(f"{path}: malformed FASTA (sequence before header)")
                chunks.append("".join(line.split()))
    if header is not None:
        records.append((header, "".join(chunks)))

    if len(records) != 1:
        raise ValueError(f"{path}: expected exactly one FASTA record, found {len(records)}")
    seq = records[0][1].upper()
    if not seq:
        raise ValueError(f"{path}: empty sequence")
    if "-" in seq or "." in seq:
        raise ValueError(f"{path}: sequence must be ungapped")
    return records[0][0], seq


def parse_pdb_sequence(path: Path, chain_id: str | None) -> tuple[str, str, list[tuple[str, str]], list[str], np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"PDB file not found: {path}")

    chains: dict[str, list[dict[str, object]]] = {}
    seen_atoms: dict[str, set[tuple[str, str, str, str]]] = {}
    residue_lookup: dict[str, dict[tuple[str, str, str], int]] = {}

    with path.open() as handle:
        for raw in handle:
            if not raw.startswith(("ATOM", "HETATM")):
                continue
            altloc = raw[16:17]
            if altloc not in (" ", "A"):
                continue
            resname = raw[17:20].strip().upper()
            aa = AA3_TO_1.get(resname)
            if aa is None:
                continue
            chain = raw[21:22]
            resseq = raw[22:26].strip()
            icode = raw[26:27].strip()
            resid = (resseq, icode, resname)
            atom_name = raw[12:16].strip()
            try:
                x = float(raw[30:38])
                y = float(raw[38:46])
                z = float(raw[46:54])
            except ValueError:
                continue

            if chain not in chains:
                chains[chain] = []
                seen_atoms[chain] = set()
                residue_lookup[chain] = {}
            atom_key = (resseq, icode, resname, atom_name)
            if atom_key in seen_atoms[chain]:
                continue
            seen_atoms[chain].add(atom_key)

            idx = residue_lookup[chain].get(resid)
            if idx is None:
                idx = len(chains[chain])
                residue_lookup[chain][resid] = idx
                chains[chain].append(
                    {
                        "resid": resid,
                        "aa": aa,
                        "coords": [(x, y, z)],
                        "ca_coord": (x, y, z) if atom_name == "CA" else None,
                    }
                )
            else:
                residue_entry = chains[chain][idx]
                residue_entry["coords"].append((x, y, z))
                if atom_name == "CA":
                    residue_entry["ca_coord"] = (x, y, z)

    if not chains:
        raise ValueError(f"{path}: no protein residues found in ATOM records")

    selected_chain = choose_chain(chains, chain_id, path)
    residues = chains[selected_chain]
    sequence = "".join(str(entry["aa"]) for entry in residues)
    if not sequence:
        raise ValueError(f"{path}: selected chain '{selected_chain}' has empty sequence")
    residue_ids = []
    residue_labels = []
    coords = []
    for entry in residues:
        resseq, icode, resname = entry["resid"]  # type: ignore[misc]
        residue_id = f"{resseq}{icode or ''}"
        residue_ids.append((selected_chain, residue_id))
        residue_labels.append(f"{selected_chain}.{resname}.{residue_id}")
        ca_coord = entry["ca_coord"]
        if ca_coord is not None:
            coords.append(ca_coord)
        else:
            atom_coords = np.asarray(entry["coords"], dtype=float)
            coords.append(tuple(np.mean(atom_coords, axis=0)))
    header = f"{path.stem}_{selected_chain.strip() or 'blank'}"
    return header, sequence, residue_ids, residue_labels, np.asarray(coords, dtype=float)


def choose_chain(chains: dict[str, list[tuple[tuple[str, str, str], str]]], chain_id: str | None, path: Path) -> str:
    if chain_id is not None:
        if len(chain_id) != 1:
            raise ValueError(f"{path}: --chain must be one character, got '{chain_id}'")
        if chain_id not in chains:
            available = ",".join(repr(c) for c in sorted(chains))
            raise ValueError(f"{path}: chain '{chain_id}' not found. Available chains: {available}")
        return chain_id

    if len(chains) == 1:
        return next(iter(chains))

    sorted_by_len = sorted(chains.items(), key=lambda item: len(item[1]), reverse=True)
    best_len = len(sorted_by_len[0][1])
    tied = [cid for cid, residues in sorted_by_len if len(residues) == best_len]
    if len(tied) > 1:
        tied_text = ",".join(repr(c) for c in tied)
        raise ValueError(f"{path}: ambiguous chain (multiple top-length protein chains: {tied_text}); use --query*-chain")
    return sorted_by_len[0][0]


def write_chain_only_pdb(query: ResolvedQuery, out_dir: Path) -> Path | None:
    if query.pdb_path is None or query.chain is None:
        return query.pdb_path

    out_path = out_dir / f"{query.label}.surface_input_chain_{query.chain.strip() or 'blank'}.pdb"
    wrote_any = False
    with query.pdb_path.open() as inp, out_path.open("w") as out:
        for raw in inp:
            if raw.startswith(("ATOM", "HETATM")):
                if raw[21:22] != query.chain:
                    continue
                resname = raw[17:20].strip().upper()
                if resname not in AA3_TO_1:
                    continue
                out.write(raw)
                wrote_any = True
            elif wrote_any and raw.startswith("TER"):
                out.write(raw)
        if wrote_any:
            out.write("END\n")
    if not wrote_any:
        try:
            out_path.unlink()
        except FileNotFoundError:
            pass
        return None
    return out_path


def resolve_query(
    label: str,
    fasta_path: Path | None,
    pdb_path: Path | None,
    chain: str | None,
    resolved_dir: Path,
    warnings: list[str],
) -> ResolvedQuery:
    fasta_header: str | None = None
    fasta_seq: str | None = None
    pdb_header: str | None = None
    pdb_seq: str | None = None
    pdb_residue_ids: list[tuple[str, str]] | None = None
    pdb_residue_labels: list[str] | None = None
    pdb_residue_coords: np.ndarray | None = None
    chosen_chain: str | None = None

    if fasta_path is not None:
        fasta_header, fasta_seq = parse_single_fasta(fasta_path)
    if pdb_path is not None:
        pdb_header, pdb_seq, pdb_residue_ids, pdb_residue_labels, pdb_residue_coords = parse_pdb_sequence(pdb_path, chain)
        chosen_chain = pdb_header.rsplit("_", 1)[-1]

    if fasta_seq is None and pdb_seq is None:
        raise ValueError(f"{label}: no input sequence could be resolved")

    if fasta_seq is not None and pdb_seq is not None:
        if fasta_seq != pdb_seq:
            raise ValueError(
                f"{label}: FASTA and PDB sequences differ in length/content; "
                "provide matching inputs or only one source."
            )
        sequence = fasta_seq
        header = fasta_header or pdb_header or label
        source_type = "fasta+pdb"
    elif fasta_seq is not None:
        sequence = fasta_seq
        header = fasta_header or label
        source_type = "fasta"
    else:
        sequence = pdb_seq or ""
        header = pdb_header or label
        source_type = "pdb"

    if source_type == "pdb":
        warnings.append(f"{label}: sequence extracted from PDB chain '{chosen_chain}'")

    fasta_out = resolved_dir / f"{label}.resolved.fasta"
    write_single_fasta(fasta_out, header, sequence)

    return ResolvedQuery(
        label=label,
        source_type=source_type,
        sequence=sequence,
        fasta_path=fasta_out,
        pdb_path=pdb_path,
        chain=chosen_chain or chain,
        header=header,
        pdb_residue_ids=pdb_residue_ids,
        pdb_residue_labels=pdb_residue_labels,
        pdb_residue_coords=pdb_residue_coords,
    )


def write_single_fasta(path: Path, header: str, sequence: str) -> None:
    with path.open("w") as handle:
        handle.write(f">{header}\n")
        handle.write(f"{sequence}\n")


def run_ifrag(
    project_root: Path,
    query1_fasta: Path,
    query2_fasta: Path,
    out_dir: Path,
    args: argparse.Namespace,
) -> None:
    cmd = [
        sys.executable,
        str(project_root / "ifrags.py"),
        "--query1",
        str(query1_fasta),
        "--query2",
        str(query2_fasta),
        "--template-dataset",
        args.ifrag_template_dataset,
        "--out-dir",
        str(out_dir),
        "--threads",
        str(args.threads),
        "--evalue",
        str(args.ifrag_evalue),
        "--redundancy-mode",
        args.ifrag_redundancy_mode,
        "--cdhit-bin",
        args.ifrag_cdhit_bin,
        "--identity-threshold",
        str(args.ifrag_identity_threshold),
        "--max-target-seqs",
        str(args.ifrag_max_target_seqs),
        "--min-pident",
        str(args.ifrag_min_pident),
        "--min-aln-len",
        str(args.ifrag_min_aln_len),
        "--min-cov1",
        str(args.ifrag_min_cov1),
        "--max-cov1",
        str(args.ifrag_max_cov1),
        "--min-cov2",
        str(args.ifrag_min_cov2),
        "--max-cov2",
        str(args.ifrag_max_cov2),
    ]
    if args.ifrag_pairs is not None:
        cmd.extend(["--pairs", str(args.ifrag_pairs)])
    if args.ifrag_blast_db is not None:
        cmd.extend(["--blast-db", str(args.ifrag_blast_db)])
    if args.ifrag_template_fasta is not None:
        cmd.extend(["--template-fasta", str(args.ifrag_template_fasta)])
    for token in args.ifrag_pair_method_substring:
        cmd.extend(["--pair-method-substring", token])
    if args.ifrag_top_k is not None:
        cmd.extend(["--top-k", str(args.ifrag_top_k)])
    if not args.no_heatmap:
        cmd.append("--heatmap")
    result = run_command(cmd, cwd=project_root)
    (out_dir / "combine_ifrag.stdout.log").write_text(result.stdout or "")
    (out_dir / "combine_ifrag.stderr.log").write_text(result.stderr or "")
    if result.returncode != 0:
        msg = result.stderr.strip() or result.stdout.strip() or "unknown error"
        raise RuntimeError(f"iFrag failed: {msg}")


def run_conservation(
    project_root: Path,
    query1_fasta: Path,
    query2_fasta: Path,
    out_dir: Path,
    args: argparse.Namespace,
    query1_search_tsv: Path,
    query2_search_tsv: Path,
    interaction_mode: str,
) -> None:
    radi_pair_defaults = radi_pair_dataset_defaults(args.radi_pair_dataset)
    radi_pairs = args.radi_pairs or radi_pair_defaults["pairs"]
    radi_pairs_meta = args.radi_pairs_meta or radi_pair_defaults["pairs_meta"]
    cmd = [
        sys.executable,
        str(project_root / "conservation.py"),
        "--query1",
        str(query1_fasta),
        "--query2",
        str(query2_fasta),
        "--query1-search-tsv",
        str(query1_search_tsv),
        "--query2-search-tsv",
        str(query2_search_tsv),
        "--pairs",
        str(radi_pairs),
        "--pairs-meta",
        str(radi_pairs_meta),
        "--interaction-mode",
        interaction_mode,
        "--shared-search-mode",
        args.homolog_search_mode,
        "--out-dir",
        str(out_dir),
        "--sequence-fasta",
        str(args.radi_sequence_fasta),
        "--famsa-bin",
        args.radi_famsa_bin,
        "--threads",
        str(args.threads),
    ]
    if args.no_heatmap:
        cmd.append("--no-heatmap")
    if args.verbose:
        cmd.append("--verbose")
    result = run_command(cmd, cwd=project_root)
    (out_dir / "combine_conservation.stdout.log").write_text(result.stdout or "")
    (out_dir / "combine_conservation.stderr.log").write_text(result.stderr or "")
    if result.returncode != 0:
        msg = result.stderr.strip() or result.stdout.strip() or "unknown error"
        raise RuntimeError(f"conservation failed: {msg}")


def run_radi_prepare(
    project_root: Path,
    query1_fasta: Path,
    query2_fasta: Path,
    out_dir: Path,
    args: argparse.Namespace,
    query1_search_tsv: Path,
    query2_search_tsv: Path,
    prepared_dir: Path | None = None,
) -> None:
    radi_pair_defaults = radi_pair_dataset_defaults(args.radi_pair_dataset)
    radi_pairs = args.radi_pairs or radi_pair_defaults["pairs"]
    radi_pairs_meta = args.radi_pairs_meta or radi_pair_defaults["pairs_meta"]
    cmd = [
        sys.executable,
        str(project_root / "radi_prepare.py"),
        "--query1",
        str(query1_fasta),
        "--query2",
        str(query2_fasta),
        "--query1-search-tsv",
        str(query1_search_tsv),
        "--query2-search-tsv",
        str(query2_search_tsv),
        "--pairs",
        str(radi_pairs),
        "--pairs-meta",
        str(radi_pairs_meta),
        "--shared-search-mode",
        args.homolog_search_mode,
        "--sequence-fasta",
        str(args.radi_sequence_fasta),
        "--out-dir",
        str(out_dir),
        "--famsa-bin",
        args.radi_famsa_bin,
        "--threads",
        str(args.threads),
    ]
    if prepared_dir is not None:
        cmd.extend(["--prepared-dir", str(prepared_dir)])
    if args.verbose:
        cmd.append("--verbose")
    result = run_command(cmd, cwd=project_root)
    (out_dir / "combine_radi_prepare.stdout.log").write_text(result.stdout or "")
    (out_dir / "combine_radi_prepare.stderr.log").write_text(result.stderr or "")
    if result.returncode != 0:
        msg = result.stderr.strip() or result.stdout.strip() or "unknown error"
        raise RuntimeError(f"radi_prepare failed: {msg}")


def run_homolog_search(
    project_root: Path,
    query1_fasta: Path,
    query2_fasta: Path,
    out_dir: Path,
    args: argparse.Namespace,
) -> tuple[Path, Path]:
    cmd = [
        sys.executable,
        str(project_root / "homolog_search.py"),
        "--query1",
        str(query1_fasta),
        "--query2",
        str(query2_fasta),
        "--out-dir",
        str(out_dir),
        "--template-dataset",
        args.radi_pair_dataset,
        "--template-mmseqs-db",
        str(args.radi_template_mmseqs_db),
        "--template-proteins",
        str(args.radi_template_proteins),
        "--search-mode",
        args.homolog_search_mode,
        "--mmseqs-bin",
        args.radi_mmseqs_bin,
        "--threads",
        str(args.threads),
        "--stage1-iterations",
        str(args.radi_stage1_iterations),
        "--max-hits",
        str(args.radi_max_hits),
        "--mmseqs-sensitivity",
        str(args.radi_mmseqs_sensitivity),
    ]
    if args.radi_evalue is not None:
        cmd.extend(["--evalue", str(args.radi_evalue)])
    if args.verbose:
        cmd.append("--verbose")
    result = run_command(cmd, cwd=project_root)
    (out_dir / "combine_homolog_search.stdout.log").write_text(result.stdout or "")
    (out_dir / "combine_homolog_search.stderr.log").write_text(result.stderr or "")
    if result.returncode != 0:
        msg = result.stderr.strip() or result.stdout.strip() or "unknown error"
        raise RuntimeError(f"homolog_search failed: {msg}")
    return (
        out_dir / "query1_mmseqs.tsv",
        out_dir / "query2_mmseqs.tsv",
    )


def run_radi(
    project_root: Path,
    prepare_out_dir: Path,
    out_dir: Path,
    query1: ResolvedQuery,
    query2: ResolvedQuery,
    args: argparse.Namespace,
) -> None:
    cmd = [
        sys.executable,
        str(project_root / "radi.py"),
        "--paired-msa",
        str(prepare_out_dir / "paired_msa.txt"),
        "--paired-ssa",
        str(prepare_out_dir / "paired_ssa.txt"),
        "--out-dir",
        str(out_dir),
        "--radi-bin",
        args.radi_bin,
        "--ra",
        str(args.radi_ra),
        "--max-radi-pairs",
        str(args.radi_top_pairs_consensus),
        "--query1-label",
        query1.header,
        "--query2-label",
        query2.header,
    ]
    if args.no_heatmap:
        cmd.append("--no-heatmap")
    if args.verbose:
        cmd.append("--verbose")
    result = run_command(cmd, cwd=project_root)
    (out_dir / "combine_radi.stdout.log").write_text(result.stdout or "")
    (out_dir / "combine_radi.stderr.log").write_text(result.stderr or "")
    if result.returncode != 0:
        msg = result.stderr.strip() or result.stdout.strip() or "unknown error"
        raise RuntimeError(f"radi failed: {msg}")


def run_blastpdb(
    project_root: Path,
    query1_fasta: Path,
    query2_fasta: Path,
    out_dir: Path,
    args: argparse.Namespace,
    interaction_mode: str,
) -> None:
    cmd = [
        sys.executable,
        str(project_root / "blastpdb.py"),
        "--query1",
        str(query1_fasta),
        "--query2",
        str(query2_fasta),
        "--interaction-mode",
        interaction_mode,
        "--out-dir",
        str(out_dir),
        "--cache-dir",
        str(args.blastpdb_cache_dir),
        "--blast-bin",
        args.blastpdb_blast_bin,
        "--makeblastdb-bin",
        args.blastpdb_makeblastdb_bin,
        "--threads",
        str(args.threads),
        "--top-assemblies",
        str(args.blastpdb_top_assemblies),
        "--sequence-search-identity-cutoff",
        str(args.blastpdb_sequence_search_identity_cutoff),
        "--sequence-search-evalue-cutoff",
        str(args.blastpdb_sequence_search_evalue_cutoff),
        "--local-blast-evalue",
        str(args.blastpdb_local_blast_evalue),
        "--local-blast-max-target-seqs",
        str(args.blastpdb_local_blast_max_target_seqs),
        "--cbeta-threshold",
        str(args.blastpdb_cbeta_threshold),
        "--min-template-contacts",
        str(args.blastpdb_min_template_contacts),
    ]
    if args.no_heatmap:
        cmd.append("--no-heatmap")
    if args.verbose:
        cmd.append("--verbose")
    result = run_command(cmd, cwd=project_root)
    (out_dir / "combine_blastpdb.stdout.log").write_text(result.stdout or "")
    (out_dir / "combine_blastpdb.stderr.log").write_text(result.stderr or "")
    if result.returncode != 0:
        msg = result.stderr.strip() or result.stdout.strip() or "unknown error"
        raise RuntimeError(f"blastPDB failed: {msg}")


def load_matrix(path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Matrix file not found: {path}")
    try:
        arr = np.loadtxt(path, delimiter="\t", dtype=float)
    except ValueError as exc:
        raise ValueError(f"Could not parse matrix TSV: {path}") from exc

    if np.isscalar(arr):
        arr = np.array([[float(arr)]], dtype=float)
    elif arr.ndim == 1:
        rows, cols = expected_shape
        if rows == 1:
            arr = arr.reshape(1, -1)
        elif cols == 1:
            arr = arr.reshape(-1, 1)
        else:
            raise ValueError(f"{path}: ambiguous 1D matrix with expected shape {expected_shape}")
    elif arr.ndim != 2:
        raise ValueError(f"{path}: expected 2D matrix")

    if arr.shape != expected_shape:
        raise ValueError(f"{path}: shape {arr.shape} does not match expected {expected_shape}")
    return arr


def select_template_branch_path(ifrag_out: Path, conservation_out: Path, combine_mode: str) -> Path:
    if combine_mode == "ifrag_radi":
        return ifrag_out / "ifrag_matrix.tsv"
    if combine_mode == "conservation_radi":
        return conservation_out / "conservation_matrix.tsv"
    raise ValueError(f"Unsupported single-branch --combine-mode: {combine_mode}")


def load_ifrag_branches(
    ifrag_out: Path,
    conservation_out: Path,
    expected_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    ifrag_matrix = load_matrix(ifrag_out / "ifrag_matrix.tsv", expected_shape)
    conservation_matrix = load_matrix(conservation_out / "conservation_matrix.tsv", expected_shape)
    return ifrag_matrix, conservation_matrix


def load_optional_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def load_conservation_profile_scores(path: Path, expected_length: int) -> np.ndarray | None:
    if not path.exists():
        return None
    values = np.zeros(expected_length, dtype=float)
    seen = 0
    with path.open(encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            idx = int(row["residue_index"]) - 1
            if idx < 0 or idx >= expected_length:
                raise ValueError(f"{path}: residue_index {idx + 1} is outside expected length {expected_length}")
            values[idx] = float(row["profile_score"])
            seen += 1
    if seen != expected_length:
        raise ValueError(f"{path}: expected {expected_length} profile rows, found {seen}")
    return values


def parse_freesasa_rsa(path: Path) -> dict[tuple[str, str], float]:
    rsa_map: dict[tuple[str, str], float] = {}
    with path.open() as handle:
        for raw in handle:
            line = raw.strip()
            if not line.startswith("RES"):
                continue
            fields = line.split()
            if len(fields) < 6:
                continue
            chain = fields[2]
            residue_id = fields[3]
            try:
                rsa_pct = float(fields[5])
            except ValueError:
                continue
            rsa_map[(chain, residue_id)] = rsa_pct
    return rsa_map


def compute_surface_mask(
    query: ResolvedQuery,
    out_dir: Path,
    rsa_threshold: float,
    warnings: list[str],
) -> np.ndarray | None:
    if query.pdb_path is None or query.pdb_residue_ids is None:
        return None

    freesasa_bin = shutil.which("freesasa")
    if freesasa_bin is None:
        warnings.append(f"{query.label}: freesasa executable not found; skipping SASA filtering.")
        return None

    surface_input = write_chain_only_pdb(query, out_dir)
    if surface_input is None:
        warnings.append(f"{query.label}: could not isolate the selected protein chain for SASA; skipping SASA filtering.")
        return None

    rsa_path = out_dir / f"{query.label}.surface.rsa"
    # Use an absolute output path because freesasa runs with cwd=out_dir.
    result = run_command([freesasa_bin, "--format=rsa", str(surface_input.resolve()), "-o", str(rsa_path.resolve())], cwd=out_dir)
    if result.returncode != 0 or not rsa_path.exists():
        message = result.stderr.strip() or result.stdout.strip() or "unknown freesasa error"
        warnings.append(f"{query.label}: freesasa failed; skipping SASA filtering ({message})")
        return None

    rsa_map = parse_freesasa_rsa(rsa_path)
    if not rsa_map:
        warnings.append(f"{query.label}: freesasa produced no parseable RSA residues; skipping SASA filtering.")
        return None

    mask = np.zeros(len(query.pdb_residue_ids), dtype=bool)
    for idx, residue_key in enumerate(query.pdb_residue_ids):
        rsa_pct = rsa_map.get(residue_key, 0.0)
        mask[idx] = rsa_pct >= rsa_threshold
    return mask


def normalize_nonzero_by_percentile(matrix: np.ndarray) -> np.ndarray:
    out = np.zeros_like(matrix, dtype=float)
    mask = matrix > 0.0
    values = matrix[mask]
    if values.size == 0:
        return out
    sorted_values = np.sort(values)
    ranks = np.searchsorted(sorted_values, values, side="right")
    out[mask] = ranks.astype(float) / float(values.size)
    return out


def top_nonzero_max_normalized_values(matrix: np.ndarray, top_n: int) -> tuple[np.ndarray, int]:
    out = np.zeros_like(matrix, dtype=float)
    nonzero_idx = np.argwhere(matrix > 0.0)
    if nonzero_idx.size == 0:
        return out, 0

    values = matrix[nonzero_idx[:, 0], nonzero_idx[:, 1]]
    keep = min(top_n, values.size)
    order = np.argsort(values)[::-1][:keep]
    kept_values = values[order]
    max_value = float(np.max(kept_values))
    if max_value <= 0.0:
        return out, 0

    for idx, di_value in zip(order, kept_values):
        r, c = nonzero_idx[idx]
        out[r, c] = float(di_value) / max_value
    return out, keep


def write_matrix_tsv(path: Path, matrix: np.ndarray) -> None:
    np.savetxt(path, matrix, delimiter="\t", fmt="%.10g")


def write_top_pairs(
    path: Path,
    matrix: np.ndarray,
    seq1: str,
    seq2: str,
    limit: int | None = None,
    row_indices: np.ndarray | None = None,
    col_indices: np.ndarray | None = None,
) -> int:
    mask = matrix > 0.0
    if row_indices is not None:
        row_mask = np.zeros(matrix.shape[0], dtype=bool)
        row_mask[row_indices] = True
        mask &= row_mask[:, None]
    if col_indices is not None:
        col_mask = np.zeros(matrix.shape[1], dtype=bool)
        col_mask[col_indices] = True
        mask &= col_mask[None, :]
    nz = np.argwhere(mask)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["rank", "res1", "aa1", "res2", "aa2", "score"])
        if nz.size == 0:
            return 0
        values = matrix[nz[:, 0], nz[:, 1]]
        order = np.argsort(values)[::-1]
        if limit is not None:
            order = order[:limit]
        for rank, idx in enumerate(order, start=1):
            i0, j0 = nz[idx]
            writer.writerow([rank, i0 + 1, seq1[i0], j0 + 1, seq2[j0], f"{values[idx]:.10g}"])
    return int(order.shape[0])


def weighted_top_k_sum(vec: np.ndarray, k: int) -> float:
    if vec.size == 0:
        return 0.0
    keep = min(k, vec.size)
    part = np.partition(vec, vec.size - keep)
    top = np.sort(part[-keep:])[::-1]
    if top.size == 0:
        return 0.0
    if top.size == 1:
        return float(top[0])
    return float(top[0] + 0.5 * np.sum(top[1:]))


def top_k_specificity(vec: np.ndarray, k: int) -> float:
    positive = vec[vec > 0.0]
    if positive.size == 0:
        return 0.0
    keep = min(k, positive.size)
    top = np.sort(positive)[::-1][:keep]
    denom = float(np.sum(top))
    if denom <= 0.0:
        return 0.0
    return float(top[0]) / denom


def compute_residue_scores(consensus: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    q1_scores = np.array([weighted_top_k_sum(consensus[i, :], top_k) for i in range(consensus.shape[0])], dtype=float)
    q2_scores = np.array([weighted_top_k_sum(consensus[:, j], top_k) for j in range(consensus.shape[1])], dtype=float)
    return q1_scores, q2_scores


def compute_specificity_scores(matrix: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    q1_scores = np.array([top_k_specificity(matrix[i, :], top_k) for i in range(matrix.shape[0])], dtype=float)
    q2_scores = np.array([top_k_specificity(matrix[:, j], top_k) for j in range(matrix.shape[1])], dtype=float)
    return q1_scores, q2_scores


def normalize_positive_vector(scores: np.ndarray) -> np.ndarray:
    out = np.zeros_like(scores, dtype=float)
    mask = scores > 0.0
    if not np.any(mask):
        return out
    values = scores[mask]
    sorted_values = np.sort(values)
    ranks = np.searchsorted(sorted_values, values, side="right")
    out[mask] = ranks.astype(float) / float(values.size)
    return out


def select_nonredundant_seed_indices(
    scores: np.ndarray,
    top_n: int,
    min_gap: int,
    tie_break_scores: np.ndarray | None = None,
) -> np.ndarray:
    if top_n <= 0 or scores.size == 0:
        return np.array([], dtype=int)

    positive = np.flatnonzero(scores > 0.0)
    if positive.size == 0:
        return np.array([], dtype=int)

    local_support = compute_sequence_window_support(scores, window_radius=max(1, min_gap))
    if tie_break_scores is not None and tie_break_scores.shape == scores.shape:
        secondary = np.maximum(tie_break_scores, 0.0)
    else:
        secondary = np.zeros_like(scores, dtype=float)

    ordered = positive[
        np.lexsort(
            (
                positive,
                -secondary[positive],
                -local_support[positive],
                -scores[positive],
            )
        )
    ]

    selected: list[int] = []
    for idx in ordered:
        idx_i = int(idx)
        if any(abs(idx_i - chosen) <= min_gap for chosen in selected):
            continue
        selected.append(idx_i)
        if len(selected) >= top_n:
            break
    return np.array(selected, dtype=int)


def build_seed_region_signal(
    scores: np.ndarray,
    top_n_seeds: int,
    window_radius: int,
    tie_break_scores: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    signal = np.zeros_like(scores, dtype=float)
    if scores.size == 0 or top_n_seeds <= 0:
        return signal, np.array([], dtype=int)

    seeds = select_nonredundant_seed_indices(
        scores,
        top_n=top_n_seeds,
        min_gap=max(1, window_radius),
        tie_break_scores=tie_break_scores,
    )
    if seeds.size == 0:
        return signal, seeds

    denom = float(window_radius + 1)
    for seed in seeds:
        base = float(scores[seed])
        if base <= 0.0:
            continue
        lo = max(0, seed - window_radius)
        hi = min(scores.size, seed + window_radius + 1)
        for idx in range(lo, hi):
            dist = abs(idx - int(seed))
            weight = (window_radius + 1 - dist) / denom
            signal[idx] += base * weight
    return signal, seeds


def fraction_nonzero(matrix: np.ndarray) -> float:
    if matrix.size == 0:
        return 0.0
    return float(np.count_nonzero(matrix > 0.0)) / float(matrix.size)


def compute_ifrag_reliability(matrix: np.ndarray) -> float:
    if matrix.size == 0 or not np.any(matrix > 0.0):
        return 0.0
    density = fraction_nonzero(matrix)
    row_coverage = float(np.count_nonzero(np.any(matrix > 0.0, axis=1))) / float(matrix.shape[0])
    col_coverage = float(np.count_nonzero(np.any(matrix > 0.0, axis=0))) / float(matrix.shape[1])
    breadth = 0.5 * (row_coverage + col_coverage)

    density_term = max(0.0, min(1.0, 1.0 - density))
    breadth_term = max(0.0, min(1.0, (1.0 - breadth) / 0.5))
    reliability = 0.75 * density_term + 0.25 * breadth_term
    return max(0.05, min(1.0, reliability))


def compute_radi_bonus_weight(gated_radi_pairs: int, radi_top_pairs: int) -> float:
    if gated_radi_pairs <= 0 or radi_top_pairs <= 0:
        return 0.0
    coverage = min(1.0, float(gated_radi_pairs) / float(max(1, radi_top_pairs)))
    return 0.35 * coverage


def combine_seed_region_patch(
    conservation_region: np.ndarray,
    ifrag_region: np.ndarray,
    ifrag_reliability: float,
    use_conservation: bool,
    use_ifrag: bool,
) -> np.ndarray:
    ifrag_scaled = ifrag_reliability * ifrag_region if use_ifrag else np.zeros_like(conservation_region, dtype=float)
    conservation_scaled = conservation_region if use_conservation else np.zeros_like(conservation_region, dtype=float)

    if use_conservation and use_ifrag:
        overlap = np.minimum(conservation_scaled, ifrag_scaled)
        patch_raw = conservation_scaled + ifrag_scaled + 0.5 * overlap
        return normalize_positive_vector(patch_raw)
    if use_conservation:
        return normalize_positive_vector(conservation_scaled)
    if use_ifrag:
        return normalize_positive_vector(ifrag_scaled)
    return np.zeros_like(conservation_region, dtype=float)


def compute_patch_guided_component(component: np.ndarray, patch_score: np.ndarray, base_floor: float) -> np.ndarray:
    if component.size == 0:
        return np.zeros_like(component, dtype=float)
    guided = component * (base_floor + (1.0 - base_floor) * patch_score)
    return normalize_positive_vector(guided)


def build_residue_priority_matrix(q1_scores: np.ndarray, q2_scores: np.ndarray) -> np.ndarray:
    q1_norm = normalize_positive_vector(q1_scores)
    q2_norm = normalize_positive_vector(q2_scores)
    return np.outer(q1_norm, q2_norm)


def weighted_average_components(components: list[tuple[np.ndarray, float]]) -> np.ndarray:
    if not components:
        raise ValueError("weighted_average_components requires at least one component")
    shape = components[0][0].shape
    total = np.zeros(shape, dtype=float)
    weight_sum = 0.0
    for values, weight in components:
        if values.shape != shape:
            raise ValueError("all component arrays must have the same shape")
        if weight <= 0.0:
            continue
        total += weight * values
        weight_sum += weight
    if weight_sum <= 0.0:
        return np.zeros(shape, dtype=float)
    return total / weight_sum


def select_top_positive_indices(scores: np.ndarray, count: int) -> np.ndarray:
    nonzero = np.flatnonzero(scores > 0.0)
    if nonzero.size == 0:
        return np.array([], dtype=int)
    order = nonzero[np.argsort(-scores[nonzero], kind="stable")]
    return order[: min(count, order.size)]


def binary_mask_from_indices(length: int, indices: np.ndarray) -> np.ndarray:
    mask = np.zeros(length, dtype=bool)
    if indices.size > 0:
        mask[indices] = True
    return mask


def residue_metadata(query: ResolvedQuery, idx: int) -> tuple[str, str, str, str]:
    chain = ""
    residue_id = ""
    residue_name = ""
    residue_label = ""
    if query.pdb_residue_ids is not None and idx < len(query.pdb_residue_ids):
        chain, residue_id = query.pdb_residue_ids[idx]
    if query.pdb_residue_labels is not None and idx < len(query.pdb_residue_labels):
        residue_label = query.pdb_residue_labels[idx]
        parts = residue_label.split(".")
        if len(parts) >= 3:
            residue_name = parts[1]
    return chain, residue_id, residue_name, residue_label


def write_residue_scores(path: Path, query: ResolvedQuery, scores: np.ndarray) -> None:
    order = np.argsort(-scores, kind="stable")
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "rank",
                "residue_index",
                "aa",
                "pdb_chain",
                "pdb_residue_id",
                "pdb_resname",
                "pdb_residue_label",
                "residue_score",
            ]
        )
        for rank, idx in enumerate(order, start=1):
            chain, residue_id, residue_name, residue_label = residue_metadata(query, idx)
            writer.writerow(
                [
                    rank,
                    idx + 1,
                    query.sequence[idx],
                    chain,
                    residue_id,
                    residue_name,
                    residue_label,
                    f"{scores[idx]:.10g}",
                ]
            )


def write_branch_score_table(
    path: Path,
    query: ResolvedQuery,
    final_scores: np.ndarray,
    patch_scores: np.ndarray,
    ifrag_strength: np.ndarray,
    ifrag_specificity: np.ndarray,
    ifrag_component: np.ndarray,
    conservation_strength: np.ndarray,
    conservation_component: np.ndarray,
    radi_anchor: np.ndarray,
    radi_component: np.ndarray,
    blastpdb_anchor: np.ndarray,
    blastpdb_component: np.ndarray,
) -> None:
    order = np.argsort(-final_scores, kind="stable")
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "rank",
                "residue_index",
                "aa",
                "pdb_chain",
                "pdb_residue_id",
                "pdb_resname",
                "pdb_residue_label",
                "final_score",
                "patch_score",
                "ifrag_strength",
                "ifrag_specificity",
                "ifrag_component",
                "conservation_strength",
                "conservation_component",
                "radi_anchor",
                "radi_component",
                "blastpdb_anchor",
                "blastpdb_component",
            ]
        )
        for rank, idx in enumerate(order, start=1):
            chain, residue_id, residue_name, residue_label = residue_metadata(query, idx)
            writer.writerow(
                [
                    rank,
                    idx + 1,
                    query.sequence[idx],
                    chain,
                    residue_id,
                    residue_name,
                    residue_label,
                    f"{final_scores[idx]:.10g}",
                    f"{patch_scores[idx]:.10g}",
                    f"{ifrag_strength[idx]:.10g}",
                    f"{ifrag_specificity[idx]:.10g}",
                    f"{ifrag_component[idx]:.10g}",
                    f"{conservation_strength[idx]:.10g}",
                    f"{conservation_component[idx]:.10g}",
                    f"{radi_anchor[idx]:.10g}",
                    f"{radi_component[idx]:.10g}",
                    f"{blastpdb_anchor[idx]:.10g}",
                    f"{blastpdb_component[idx]:.10g}",
                ]
            )


def write_structure_feature_table(
    path: Path,
    query: ResolvedQuery,
    final_scores: np.ndarray,
    support_scores: np.ndarray,
    confidence_component: np.ndarray,
    local_mass_component: np.ndarray,
    shape_component: np.ndarray,
    hydrophobic_patch_component: np.ndarray,
) -> None:
    order = np.argsort(-final_scores, kind="stable")
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "rank",
                "residue_index",
                "aa",
                "pdb_chain",
                "pdb_residue_id",
                "pdb_resname",
                "pdb_residue_label",
                "final_score",
                "selection_support",
                "confidence_component",
                "local_mass_component",
                "shape_component",
                "hydrophobic_patch_component",
            ]
        )
        for rank, idx in enumerate(order, start=1):
            chain, residue_id, residue_name, residue_label = residue_metadata(query, idx)
            writer.writerow(
                [
                    rank,
                    idx + 1,
                    query.sequence[idx],
                    chain,
                    residue_id,
                    residue_name,
                    residue_label,
                    f"{final_scores[idx]:.10g}",
                    f"{support_scores[idx]:.10g}",
                    f"{confidence_component[idx]:.10g}",
                    f"{local_mass_component[idx]:.10g}",
                    f"{shape_component[idx]:.10g}",
                    f"{hydrophobic_patch_component[idx]:.10g}",
                ]
            )


def expand_passive_indices(
    scores: np.ndarray,
    active_indices: np.ndarray,
    passive_count: int,
    coords: np.ndarray | None = None,
    eligible_mask: np.ndarray | None = None,
) -> np.ndarray:
    if passive_count <= 0 or active_indices.size == 0:
        return np.array([], dtype=int)

    candidate_indices = np.arange(scores.size, dtype=int)
    if eligible_mask is not None:
        candidate_indices = candidate_indices[eligible_mask[candidate_indices]]
    if candidate_indices.size == 0:
        return np.array([], dtype=int)

    is_active = np.zeros(scores.size, dtype=bool)
    is_active[active_indices] = True
    candidate_indices = candidate_indices[~is_active[candidate_indices]]
    if candidate_indices.size == 0:
        return np.array([], dtype=int)

    candidate_scores = scores[candidate_indices]
    if coords is not None and coords.shape[0] == scores.size:
        active_coords = coords[active_indices]
        candidate_coords = coords[candidate_indices]
        min_dist = np.min(
            np.linalg.norm(candidate_coords[:, None, :] - active_coords[None, :, :], axis=2),
            axis=1,
        )
    else:
        min_dist = np.min(np.abs(candidate_indices[:, None] - active_indices[None, :]), axis=1).astype(float)

    order = np.lexsort((candidate_indices, -candidate_scores, min_dist))
    keep = min(passive_count, candidate_indices.size)
    return candidate_indices[order[:keep]]


def compute_sequence_window_support(
    scores: np.ndarray,
    window_radius: int = 4,
    eligible_mask: np.ndarray | None = None,
) -> np.ndarray:
    if scores.size == 0:
        return np.array([], dtype=float)
    if window_radius <= 0:
        return scores.astype(float, copy=True)

    working = scores.astype(float, copy=True)
    if eligible_mask is not None:
        working[~eligible_mask] = 0.0

    support = np.zeros_like(working)
    for idx in range(working.size):
        lo = max(0, idx - window_radius)
        hi = min(working.size, idx + window_radius + 1)
        support[idx] = float(np.sum(working[lo:hi]))
    return support


def compute_spatial_support(
    scores: np.ndarray,
    coords: np.ndarray | None,
    radius: float,
    eligible_mask: np.ndarray | None = None,
) -> np.ndarray:
    support = np.zeros_like(scores, dtype=float)
    if coords is None or coords.shape[0] != scores.size or radius <= 0.0 or scores.size == 0:
        return support

    candidate_indices = np.flatnonzero(scores > 0.0)
    if eligible_mask is not None:
        candidate_indices = candidate_indices[eligible_mask[candidate_indices]]
    if candidate_indices.size == 0:
        return support

    candidate_coords = coords[candidate_indices]
    candidate_scores = np.maximum(scores[candidate_indices], 0.0)
    distances = np.linalg.norm(candidate_coords[:, None, :] - candidate_coords[None, :, :], axis=2)
    local_mass = (distances <= radius).astype(float) @ candidate_scores
    support[candidate_indices] = local_mass
    return support


def select_spatial_seed_indices(
    scores: np.ndarray,
    coords: np.ndarray | None,
    top_n: int,
    min_distance: float,
    eligible_mask: np.ndarray | None = None,
    tie_break_scores: np.ndarray | None = None,
) -> np.ndarray:
    if coords is None or coords.shape[0] != scores.size:
        return select_nonredundant_seed_indices(
            scores,
            top_n=top_n,
            min_gap=max(1, int(round(min_distance / 3.0))),
            tie_break_scores=tie_break_scores,
        )
    if top_n <= 0 or scores.size == 0:
        return np.array([], dtype=int)

    positive = np.flatnonzero(scores > 0.0)
    if eligible_mask is not None:
        positive = positive[eligible_mask[positive]]
    if positive.size == 0:
        return np.array([], dtype=int)

    spatial_support = compute_spatial_support(scores, coords, radius=min_distance, eligible_mask=eligible_mask)
    if tie_break_scores is not None and tie_break_scores.shape == scores.shape:
        secondary = np.maximum(tie_break_scores, 0.0)
    else:
        secondary = np.zeros_like(scores, dtype=float)

    ordered = positive[
        np.lexsort(
            (
                positive,
                -secondary[positive],
                -spatial_support[positive],
                -scores[positive],
            )
        )
    ]

    selected: list[int] = []
    for idx in ordered:
        idx_i = int(idx)
        if any(float(np.linalg.norm(coords[idx_i] - coords[chosen])) <= min_distance for chosen in selected):
            continue
        selected.append(idx_i)
        if len(selected) >= top_n:
            break
    return np.array(selected, dtype=int)


def select_clustered_active_passive_indices(
    scores: np.ndarray,
    active_count: int,
    passive_count: int,
    coords: np.ndarray | None,
    eligible_mask: np.ndarray | None = None,
    tie_break_scores: np.ndarray | None = None,
    seed_count: int | None = None,
    seed_min_distance: float = 10.0,
    cluster_radius: float = 12.0,
    passive_shell_radius: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if coords is None or coords.shape[0] != scores.size or scores.size == 0:
        return None

    candidate_indices = np.flatnonzero(scores > 0.0)
    if eligible_mask is not None:
        candidate_indices = candidate_indices[eligible_mask[candidate_indices]]
    if candidate_indices.size == 0:
        return None

    if tie_break_scores is not None and tie_break_scores.shape == scores.shape:
        secondary = np.maximum(tie_break_scores, 0.0)
    else:
        secondary = np.zeros_like(scores, dtype=float)

    spatial_support = compute_spatial_support(scores, coords, radius=cluster_radius, eligible_mask=eligible_mask)
    n_seeds = seed_count or max(2, min(4, active_count // 2 + 1))
    seeds = select_spatial_seed_indices(
        scores,
        coords,
        top_n=n_seeds,
        min_distance=seed_min_distance,
        eligible_mask=eligible_mask,
        tie_break_scores=secondary + 0.25 * spatial_support,
    )
    if seeds.size == 0:
        return None

    candidate_coords = coords[candidate_indices]
    best_cluster: np.ndarray | None = None
    best_cluster_score = -1.0
    best_seed = int(seeds[0])

    for seed in seeds:
        distances = np.linalg.norm(candidate_coords - coords[int(seed)], axis=1)
        cluster_members = candidate_indices[distances <= cluster_radius]
        if cluster_members.size == 0:
            cluster_members = np.array([int(seed)], dtype=int)
        cluster_score = weighted_top_k_sum(
            scores[cluster_members],
            max(active_count + passive_count, 4),
        ) + 0.5 * weighted_top_k_sum(
            secondary[cluster_members],
            max(active_count, 3),
        )
        if cluster_score > best_cluster_score:
            best_cluster_score = cluster_score
            best_cluster = cluster_members
            best_seed = int(seed)

    if best_cluster is None or best_cluster.size == 0:
        return None

    cluster_order = best_cluster[
        np.lexsort(
            (
                best_cluster,
                -spatial_support[best_cluster],
                -secondary[best_cluster],
                -scores[best_cluster],
            )
        )
    ]
    active = cluster_order[: min(active_count, cluster_order.size)]
    if active.size == 0:
        return None

    cluster_mask = np.zeros(scores.size, dtype=bool)
    cluster_mask[best_cluster] = True
    cluster_mask[best_seed] = True

    active_coords = coords[active]
    min_dist_to_active = np.min(
        np.linalg.norm(coords[:, None, :] - active_coords[None, :, :], axis=2),
        axis=1,
    )
    shell_mask = cluster_mask | (min_dist_to_active <= passive_shell_radius)
    if eligible_mask is not None:
        shell_mask &= eligible_mask

    passive = expand_passive_indices(
        scores,
        active,
        passive_count,
        coords=coords,
        eligible_mask=shell_mask,
    )
    selected = np.unique(np.sort(np.concatenate([active, passive])))
    return active, passive, selected


def select_active_passive_indices(
    scores: np.ndarray,
    active_count: int,
    passive_count: int,
    coords: np.ndarray | None = None,
    eligible_mask: np.ndarray | None = None,
    tie_break_scores: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if eligible_mask is None:
        nonzero = np.flatnonzero(scores > 0.0)
    else:
        nonzero = np.flatnonzero((scores > 0.0) & eligible_mask)
    if nonzero.size == 0:
        empty = np.array([], dtype=int)
        return empty, empty, empty
    clustered = select_clustered_active_passive_indices(
        scores,
        active_count,
        passive_count,
        coords=coords,
        eligible_mask=eligible_mask,
        tie_break_scores=tie_break_scores,
    )
    if clustered is not None:
        return clustered
    local_support = compute_sequence_window_support(scores, window_radius=4, eligible_mask=eligible_mask)
    if tie_break_scores is not None and tie_break_scores.shape == scores.shape:
        secondary = np.maximum(tie_break_scores, 0.0)
    else:
        secondary = np.zeros_like(scores, dtype=float)
    # When final scores flatten into a broad plateau, prefer residues that sit in
    # the strongest local band rather than the left-most tied index.
    order = nonzero[
        np.lexsort(
            (
                nonzero,
                -secondary[nonzero],
                -local_support[nonzero],
                -scores[nonzero],
            )
        )
    ]
    active = order[: min(active_count, order.size)]
    passive = expand_passive_indices(scores, active, passive_count, coords=coords, eligible_mask=eligible_mask)
    selected = np.unique(np.sort(np.concatenate([active, passive])))
    return active, passive, selected


def compute_direct_support_signal(
    ifrag_component: np.ndarray,
    conservation_component: np.ndarray,
    radi_component: np.ndarray,
    blastpdb_component: np.ndarray,
    ifrag_weight: float,
    radi_weight: float,
    blastpdb_weight: float,
) -> np.ndarray:
    support = np.maximum(conservation_component, 0.0).astype(float, copy=True)
    if ifrag_weight > 0.0:
        support += ifrag_weight * np.maximum(ifrag_component, 0.0)
    if radi_weight > 0.0:
        support += radi_weight * np.maximum(radi_component, 0.0)
    if blastpdb_weight > 0.0:
        support += blastpdb_weight * np.maximum(blastpdb_component, 0.0)
    return support


def order_candidate_indices(
    indices: np.ndarray,
    scores: np.ndarray,
    support_scores: np.ndarray,
    tie_break_scores: np.ndarray | None = None,
) -> np.ndarray:
    if indices.size == 0:
        return indices
    secondary = np.maximum(support_scores, 0.0)
    if tie_break_scores is not None and tie_break_scores.shape == scores.shape:
        secondary = secondary + np.maximum(tie_break_scores, 0.0)
    return indices[
        np.lexsort(
            (
                indices,
                -secondary[indices],
                -support_scores[indices],
                -scores[indices],
            )
        )
    ]


def select_best_support_cluster(
    scores: np.ndarray,
    support_scores: np.ndarray,
    coords: np.ndarray | None = None,
    eligible_mask: np.ndarray | None = None,
    tie_break_scores: np.ndarray | None = None,
    seed_min_distance: float = 10.0,
    cluster_radius: float = 12.0,
    requested_active_count: int = 4,
) -> np.ndarray:
    candidate_mask = (scores > 0.0) & (support_scores > 0.0)
    if eligible_mask is not None:
        candidate_mask &= eligible_mask
    candidate_indices = np.flatnonzero(candidate_mask)
    if candidate_indices.size == 0:
        return np.array([], dtype=int)

    top_n_seeds = max(2, min(6, max(1, requested_active_count)))
    secondary = np.maximum(support_scores, 0.0)
    if tie_break_scores is not None and tie_break_scores.shape == scores.shape:
        secondary = secondary + np.maximum(tie_break_scores, 0.0)

    if coords is not None and coords.shape[0] == scores.size:
        spatial_support = compute_spatial_support(
            support_scores,
            coords,
            radius=cluster_radius,
            eligible_mask=candidate_mask,
        )
        seeds = select_spatial_seed_indices(
            support_scores,
            coords,
            top_n=top_n_seeds,
            min_distance=seed_min_distance,
            eligible_mask=candidate_mask,
            tie_break_scores=secondary + 0.25 * spatial_support,
        )
        if seeds.size == 0:
            return np.array([], dtype=int)
        candidate_coords = coords[candidate_indices]
        best_cluster = np.array([], dtype=int)
        best_score = -1.0
        for seed in seeds:
            distances = np.linalg.norm(candidate_coords - coords[int(seed)], axis=1)
            cluster_members = candidate_indices[distances <= cluster_radius]
            if cluster_members.size == 0:
                cluster_members = np.array([int(seed)], dtype=int)
            cluster_score = (
                float(np.sum(support_scores[cluster_members]))
                + 0.75 * weighted_top_k_sum(scores[cluster_members], min(8, cluster_members.size))
                + 0.25 * weighted_top_k_sum(secondary[cluster_members], min(8, cluster_members.size))
            )
            if cluster_score > best_score:
                best_score = cluster_score
                best_cluster = cluster_members
        return order_candidate_indices(best_cluster, scores, support_scores, tie_break_scores=tie_break_scores)

    window_radius = max(3, int(round(cluster_radius / 3.0)))
    seeds = select_nonredundant_seed_indices(
        support_scores * candidate_mask.astype(float),
        top_n=top_n_seeds,
        min_gap=max(2, int(round(seed_min_distance / 3.0))),
        tie_break_scores=secondary,
    )
    if seeds.size == 0:
        return np.array([], dtype=int)

    best_cluster = np.array([], dtype=int)
    best_score = -1.0
    for seed in seeds:
        lo = max(0, int(seed) - window_radius)
        hi = min(scores.size, int(seed) + window_radius + 1)
        cluster_members = np.arange(lo, hi, dtype=int)
        cluster_members = cluster_members[candidate_mask[cluster_members]]
        if cluster_members.size == 0:
            cluster_members = np.array([int(seed)], dtype=int)
        cluster_score = (
            float(np.sum(support_scores[cluster_members]))
            + 0.75 * weighted_top_k_sum(scores[cluster_members], min(8, cluster_members.size))
            + 0.25 * weighted_top_k_sum(secondary[cluster_members], min(8, cluster_members.size))
        )
        if cluster_score > best_score:
            best_score = cluster_score
            best_cluster = cluster_members
    return order_candidate_indices(best_cluster, scores, support_scores, tie_break_scores=tie_break_scores)


def select_adaptive_docking_indices(
    scores: np.ndarray,
    support_scores: np.ndarray,
    coords: np.ndarray | None = None,
    eligible_mask: np.ndarray | None = None,
    tie_break_scores: np.ndarray | None = None,
    requested_active_count: int = 4,
    requested_passive_count: int | None = None,
    active_score_fraction: float = 0.75,
    passive_score_fraction: float = 0.35,
    passive_shell_radius: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if scores.size == 0:
        empty = np.array([], dtype=int)
        return empty, empty, empty

    best_cluster = select_best_support_cluster(
        scores,
        support_scores,
        coords=coords,
        eligible_mask=eligible_mask,
        tie_break_scores=tie_break_scores,
        requested_active_count=requested_active_count,
    )
    if best_cluster.size == 0:
        empty = np.array([], dtype=int)
        return empty, empty, empty

    candidate_mask = (scores > 0.0) & (support_scores > 0.0)
    if eligible_mask is not None:
        candidate_mask &= eligible_mask

    total_support = float(np.sum(support_scores[candidate_mask]))
    cluster_support = float(np.sum(support_scores[best_cluster]))
    cluster_fraction = (cluster_support / total_support) if total_support > 0.0 else 0.0
    max_cluster_score = float(np.max(scores[best_cluster]))
    if cluster_fraction < 0.10 and max_cluster_score < 0.75:
        empty = np.array([], dtype=int)
        return empty, empty, empty

    active_threshold = active_score_fraction * max_cluster_score
    ordered_cluster = order_candidate_indices(best_cluster, scores, support_scores, tie_break_scores=tie_break_scores)
    active = ordered_cluster[scores[ordered_cluster] >= active_threshold]
    if requested_active_count > 0 and active.size > requested_active_count:
        active = active[:requested_active_count]
    if active.size == 0:
        empty = np.array([], dtype=int)
        return empty, empty, empty

    if coords is not None and coords.shape[0] == scores.size:
        active_coords = coords[active]
        min_dist_to_active = np.min(
            np.linalg.norm(coords[:, None, :] - active_coords[None, :, :], axis=2),
            axis=1,
        )
        shell_mask = candidate_mask & (min_dist_to_active <= passive_shell_radius)
    else:
        shell_mask = np.zeros(scores.size, dtype=bool)
        window_radius = max(3, int(round(passive_shell_radius / 3.0)))
        for idx in active:
            lo = max(0, int(idx) - window_radius)
            hi = min(scores.size, int(idx) + window_radius + 1)
            shell_mask[lo:hi] = True
        shell_mask &= candidate_mask

    passive_threshold = passive_score_fraction * max_cluster_score
    passive_mask = shell_mask & ~binary_mask_from_indices(scores.size, active)
    passive = np.flatnonzero(passive_mask & (scores >= passive_threshold))
    passive = order_candidate_indices(passive, scores, support_scores, tie_break_scores=tie_break_scores)
    if requested_passive_count is not None and requested_passive_count >= 0 and passive.size > requested_passive_count:
        passive = passive[:requested_passive_count]

    selected = np.unique(np.sort(np.concatenate([active, passive])))
    return active, passive, selected


def write_docking_residue_table(
    path: Path,
    query: ResolvedQuery,
    scores: np.ndarray,
    active_indices: np.ndarray,
    passive_indices: np.ndarray,
    ifrag_component: np.ndarray | None = None,
    conservation_component: np.ndarray | None = None,
    radi_component: np.ndarray | None = None,
    blastpdb_component: np.ndarray | None = None,
) -> int:
    global_order = np.argsort(-scores, kind="stable")
    global_rank = {int(idx): rank for rank, idx in enumerate(global_order, start=1)}
    active_set = [int(idx) for idx in active_indices]
    passive_set = [int(idx) for idx in passive_indices]
    rows: list[tuple[str, int, int]] = []
    rows.extend(("active", rank, idx) for rank, idx in enumerate(active_set, start=1))
    rows.extend(("passive", rank, idx) for rank, idx in enumerate(passive_set, start=1))

    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "role",
                "role_rank",
                "global_rank",
                "residue_index",
                "aa",
                "pdb_chain",
                "pdb_residue_id",
                "pdb_resname",
                "pdb_residue_label",
                "residue_score",
                "ifrag_component",
                "conservation_component",
                "radi_component",
                "blastpdb_component",
            ]
        )
        for role, role_rank, idx in rows:
            chain, residue_id, residue_name, residue_label = residue_metadata(query, idx)
            writer.writerow(
                [
                    role,
                    role_rank,
                    global_rank[idx],
                    idx + 1,
                    query.sequence[idx],
                    chain,
                    residue_id,
                    residue_name,
                    residue_label,
                    f"{scores[idx]:.10g}",
                    "" if ifrag_component is None else f"{ifrag_component[idx]:.10g}",
                    "" if conservation_component is None else f"{conservation_component[idx]:.10g}",
                    "" if radi_component is None else f"{radi_component[idx]:.10g}",
                    "" if blastpdb_component is None else f"{blastpdb_component[idx]:.10g}",
                ]
            )
    return len(rows)


def format_lightdock_residue_id(query: ResolvedQuery, idx: int) -> str | None:
    chain, residue_id, residue_name, residue_label = residue_metadata(query, idx)
    if residue_label:
        return residue_label
    if chain and residue_name and residue_id:
        return f"{chain}.{residue_name}.{residue_id}"
    return None


def build_lightdock_restraint_lines(
    query: ResolvedQuery,
    active_indices: np.ndarray,
    passive_indices: np.ndarray,
    molecule_code: str,
    include_passive: bool,
) -> list[str]:
    lines: list[str] = []
    seen: set[str] = set()
    for idx in active_indices:
        residue_id = format_lightdock_residue_id(query, int(idx))
        if not residue_id or residue_id in seen:
            continue
        seen.add(residue_id)
        lines.append(f"{molecule_code} {residue_id}")
    if include_passive:
        for idx in passive_indices:
            residue_id = format_lightdock_residue_id(query, int(idx))
            if not residue_id or residue_id in seen:
                continue
            seen.add(residue_id)
            lines.append(f"{molecule_code} {residue_id} P")
    return lines


def write_lightdock_restraints(
    path: Path,
    receptor_query: ResolvedQuery,
    receptor_active: np.ndarray,
    receptor_passive: np.ndarray,
    ligand_query: ResolvedQuery,
    ligand_active: np.ndarray,
    ligand_passive: np.ndarray,
    include_passive: bool,
) -> int:
    lines = build_lightdock_restraint_lines(
        receptor_query,
        receptor_active,
        receptor_passive,
        molecule_code="R",
        include_passive=include_passive,
    )
    lines.extend(
        build_lightdock_restraint_lines(
            ligand_query,
            ligand_active,
            ligand_passive,
            molecule_code="L",
            include_passive=include_passive,
        )
    )
    path.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")
    return len(lines)


def save_residue_track_plot(path: Path, scores: np.ndarray, title: str, ylabel: str) -> None:
    try:
        import matplotlib
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required for residue score plots (or use --no-heatmap)") from exc
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(1, scores.size + 1, dtype=int)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(x, scores, color="navy", linewidth=1.2)
    ax.fill_between(x, scores, 0.0, color="skyblue", alpha=0.35)
    ax.set_xlabel("Residue index")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(1, scores.size)
    ax.set_ylim(bottom=0.0)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_scored_pdb(query: ResolvedQuery, scores: np.ndarray, out_path: Path) -> bool:
    if query.pdb_path is None or query.pdb_residue_ids is None:
        return False

    max_score = float(np.max(scores)) if scores.size else 0.0
    if max_score > 0.0:
        normalized = scores / max_score
    else:
        normalized = np.zeros_like(scores, dtype=float)
    score_by_residue = {
        residue_key: float(normalized[idx] * 100.0)
        for idx, residue_key in enumerate(query.pdb_residue_ids)
    }

    with query.pdb_path.open() as inp, out_path.open("w") as out:
        for raw in inp:
            if raw.startswith(("ATOM", "HETATM")):
                resname = raw[17:20].strip().upper()
                if resname not in AA3_TO_1:
                    out.write(raw)
                    continue
                chain = raw[21:22]
                residue_id = f"{raw[22:26].strip()}{raw[26:27].strip()}"
                value = score_by_residue.get((chain, residue_id))
                if value is not None:
                    raw = f"{raw[:60]}{value:6.2f}{raw[66:]}"
            out.write(raw)
    return True


def save_heatmap(path: Path, matrix: np.ndarray, title: str, label: str) -> None:
    try:
        import matplotlib
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required for heatmap generation (or use --no-heatmap)") from exc
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, origin="upper", aspect="auto", cmap="viridis")
    ax.set_xlabel("Query2 residue")
    ax.set_ylabel("Query1 residue")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=label)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_overlay_heatmap(
    path: Path,
    base_matrix: np.ndarray,
    overlay_matrix: np.ndarray,
    top_pairs: int,
    base_title: str,
    base_label: str,
    overlay_label: str,
    overlay_cmap: str = "autumn",
) -> int:
    try:
        import matplotlib
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required for heatmap generation (or use --no-heatmap)") from exc
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mask = overlay_matrix > 0.0
    nz = np.argwhere(mask)
    if nz.size == 0:
        pairs_written = 0
        values = np.array([], dtype=float)
        order = np.array([], dtype=int)
    else:
        values = overlay_matrix[nz[:, 0], nz[:, 1]]
        order = np.argsort(values)[::-1][:top_pairs]
        pairs_written = int(order.shape[0])

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(base_matrix, origin="upper", aspect="auto", cmap="viridis")
    overlay_im = None
    if pairs_written > 0:
        sel = nz[order]
        sel_values = values[order]
        vmin = float(np.min(sel_values))
        vmax = float(np.max(sel_values))
        if vmax > vmin:
            sizes = 20.0 + 60.0 * ((sel_values - vmin) / (vmax - vmin))
        else:
            sizes = np.full(sel_values.shape, 40.0, dtype=float)
        overlay_im = ax.scatter(
            sel[:, 1],
            sel[:, 0],
            s=sizes,
            c=sel_values,
            cmap=overlay_cmap,
            vmin=vmin,
            vmax=vmax if vmax > vmin else vmin + 1.0,
            alpha=0.9,
            edgecolors="white",
            linewidths=0.35,
            label=f"Top {pairs_written} {overlay_label}",
        )
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    ax.set_xlabel("Query2 residue")
    ax.set_ylabel("Query1 residue")
    ax.set_title(base_title)
    fig.colorbar(im, ax=ax, label=base_label, fraction=0.046, pad=0.04)
    if overlay_im is not None:
        fig.colorbar(overlay_im, ax=ax, label=overlay_label, fraction=0.046, pad=0.10)
    fig.tight_layout(rect=(0.0, 0.0, 0.86, 1.0))
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return pairs_written


def save_full_heatmap_overlay(
    path: Path,
    base_matrix: np.ndarray,
    overlay_matrix: np.ndarray,
    base_title: str,
    base_label: str,
    overlay_label: str,
    base_cmap: str = "viridis",
    overlay_cmap: str = "cool",
) -> int:
    try:
        import matplotlib
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required for heatmap generation (or use --no-heatmap)") from exc
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    base_im = ax.imshow(base_matrix, origin="upper", aspect="auto", cmap=base_cmap)

    overlay_mask = overlay_matrix > 0.0
    overlay_nonzero = int(np.count_nonzero(overlay_mask))
    overlay_im = None
    if overlay_nonzero > 0:
        overlay_values = overlay_matrix[overlay_mask]
        vmin = float(np.min(overlay_values))
        vmax = float(np.max(overlay_values))
        if vmax > vmin:
            norm_alpha = (overlay_matrix - vmin) / (vmax - vmin)
            norm_alpha = np.clip(norm_alpha, 0.0, 1.0)
        else:
            norm_alpha = np.where(overlay_mask, 1.0, 0.0)
        alpha = np.where(overlay_mask, 0.15 + 0.70 * norm_alpha, 0.0)
        masked_overlay = np.ma.masked_where(~overlay_mask, overlay_matrix)
        overlay_im = ax.imshow(
            masked_overlay,
            origin="upper",
            aspect="auto",
            cmap=overlay_cmap,
            alpha=alpha,
            vmin=vmin,
            vmax=vmax if vmax > vmin else vmin + 1.0,
        )

    ax.set_xlabel("Query2 residue")
    ax.set_ylabel("Query1 residue")
    ax.set_title(base_title)
    fig.colorbar(base_im, ax=ax, label=base_label, fraction=0.046, pad=0.04)
    if overlay_im is not None:
        fig.colorbar(overlay_im, ax=ax, label=overlay_label, fraction=0.046, pad=0.10)
    fig.tight_layout(rect=(0.0, 0.0, 0.86, 1.0))
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return overlay_nonzero


def build_residue_first_scores(
    ifrag_matrix: np.ndarray,
    conservation_matrix: np.ndarray,
    conservation_profile_q1: np.ndarray,
    conservation_profile_q2: np.ndarray,
    radi_matrix: np.ndarray,
    blastpdb_matrix: np.ndarray,
    top_k: int,
    radi_top_pairs: int,
    patch_residues_per_chain: int,
    use_ifrag: bool,
    use_conservation: bool,
    use_radi: bool,
    use_blastpdb: bool,
    low_radi_confidence: bool,
    low_blastpdb_confidence: bool,
) -> dict[str, object]:
    specificity_top_k = max(5, top_k)
    seed_window_radius = 4
    seed_count = max(2, min(4, max(1, patch_residues_per_chain // 4)))

    ifrag_norm = normalize_nonzero_by_percentile(ifrag_matrix) if use_ifrag else np.zeros_like(ifrag_matrix, dtype=float)
    conservation_norm = (
        normalize_nonzero_by_percentile(conservation_matrix) if use_conservation else np.zeros_like(conservation_matrix, dtype=float)
    )
    radi_norm_top, gated_radi_pairs = (
        top_nonzero_max_normalized_values(radi_matrix, radi_top_pairs) if use_radi else (np.zeros_like(radi_matrix), 0)
    )
    blastpdb_norm_top, gated_blastpdb_pairs = (
        top_nonzero_max_normalized_values(blastpdb_matrix, radi_top_pairs)
        if use_blastpdb
        else (np.zeros_like(blastpdb_matrix), 0)
    )

    if use_ifrag:
        ifrag_q1_strength, ifrag_q2_strength = compute_residue_scores(ifrag_norm, top_k)
        ifrag_q1_specificity, ifrag_q2_specificity = compute_specificity_scores(ifrag_norm, specificity_top_k)
        ifrag_q1_component = normalize_positive_vector(ifrag_q1_strength)
        ifrag_q2_component = normalize_positive_vector(ifrag_q2_strength)
        ifrag_reliability = compute_ifrag_reliability(ifrag_matrix)
    else:
        ifrag_q1_strength = np.zeros(ifrag_matrix.shape[0], dtype=float)
        ifrag_q2_strength = np.zeros(ifrag_matrix.shape[1], dtype=float)
        ifrag_q1_specificity = np.zeros(ifrag_matrix.shape[0], dtype=float)
        ifrag_q2_specificity = np.zeros(ifrag_matrix.shape[1], dtype=float)
        ifrag_q1_component = np.zeros(ifrag_matrix.shape[0], dtype=float)
        ifrag_q2_component = np.zeros(ifrag_matrix.shape[1], dtype=float)
        ifrag_reliability = 0.0

    if use_conservation:
        conservation_q1_strength = conservation_profile_q1.astype(float, copy=True)
        conservation_q2_strength = conservation_profile_q2.astype(float, copy=True)
        conservation_q1_component = normalize_positive_vector(conservation_q1_strength)
        conservation_q2_component = normalize_positive_vector(conservation_q2_strength)
    else:
        conservation_q1_strength = np.zeros(conservation_matrix.shape[0], dtype=float)
        conservation_q2_strength = np.zeros(conservation_matrix.shape[1], dtype=float)
        conservation_q1_component = np.zeros(conservation_matrix.shape[0], dtype=float)
        conservation_q2_component = np.zeros(conservation_matrix.shape[1], dtype=float)

    if use_radi and not low_radi_confidence and gated_radi_pairs > 0:
        radi_q1_anchor, radi_q2_anchor = compute_residue_scores(radi_norm_top, top_k)
        radi_q1_component = normalize_positive_vector(radi_q1_anchor)
        radi_q2_component = normalize_positive_vector(radi_q2_anchor)
        radi_weight = compute_radi_bonus_weight(gated_radi_pairs, radi_top_pairs)
    else:
        radi_q1_anchor = np.zeros(radi_matrix.shape[0], dtype=float)
        radi_q2_anchor = np.zeros(radi_matrix.shape[1], dtype=float)
        radi_q1_component = np.zeros(radi_matrix.shape[0], dtype=float)
        radi_q2_component = np.zeros(radi_matrix.shape[1], dtype=float)
        radi_weight = 0.0

    if use_blastpdb and not low_blastpdb_confidence and gated_blastpdb_pairs > 0:
        blastpdb_q1_anchor, blastpdb_q2_anchor = compute_residue_scores(blastpdb_norm_top, top_k)
        blastpdb_q1_component = normalize_positive_vector(blastpdb_q1_anchor)
        blastpdb_q2_component = normalize_positive_vector(blastpdb_q2_anchor)
        blastpdb_weight = compute_radi_bonus_weight(gated_blastpdb_pairs, radi_top_pairs)
    else:
        blastpdb_q1_anchor = np.zeros(blastpdb_matrix.shape[0], dtype=float)
        blastpdb_q2_anchor = np.zeros(blastpdb_matrix.shape[1], dtype=float)
        blastpdb_q1_component = np.zeros(blastpdb_matrix.shape[0], dtype=float)
        blastpdb_q2_component = np.zeros(blastpdb_matrix.shape[1], dtype=float)
        blastpdb_weight = 0.0

    q1_ifrag_seed_region, q1_ifrag_seed_indices = build_seed_region_signal(
        ifrag_q1_component,
        top_n_seeds=seed_count,
        window_radius=seed_window_radius,
        tie_break_scores=conservation_q1_component + 0.5 * (radi_q1_component + blastpdb_q1_component),
    )
    q2_ifrag_seed_region, q2_ifrag_seed_indices = build_seed_region_signal(
        ifrag_q2_component,
        top_n_seeds=seed_count,
        window_radius=seed_window_radius,
        tie_break_scores=conservation_q2_component + 0.5 * (radi_q2_component + blastpdb_q2_component),
    )
    q1_conservation_seed_region, q1_conservation_seed_indices = build_seed_region_signal(
        conservation_q1_component,
        top_n_seeds=seed_count,
        window_radius=seed_window_radius,
        tie_break_scores=ifrag_q1_component + 0.5 * (radi_q1_component + blastpdb_q1_component),
    )
    q2_conservation_seed_region, q2_conservation_seed_indices = build_seed_region_signal(
        conservation_q2_component,
        top_n_seeds=seed_count,
        window_radius=seed_window_radius,
        tie_break_scores=ifrag_q2_component + 0.5 * (radi_q2_component + blastpdb_q2_component),
    )

    q1_patch_score = combine_seed_region_patch(
        conservation_region=q1_conservation_seed_region,
        ifrag_region=q1_ifrag_seed_region,
        ifrag_reliability=ifrag_reliability,
        use_conservation=use_conservation,
        use_ifrag=use_ifrag,
    )
    q2_patch_score = combine_seed_region_patch(
        conservation_region=q2_conservation_seed_region,
        ifrag_region=q2_ifrag_seed_region,
        ifrag_reliability=ifrag_reliability,
        use_conservation=use_conservation,
        use_ifrag=use_ifrag,
    )

    q1_ifrag_guided = normalize_positive_vector(q1_ifrag_seed_region)
    q2_ifrag_guided = normalize_positive_vector(q2_ifrag_seed_region)
    q1_radi_bonus = compute_patch_guided_component(radi_q1_component, q1_patch_score, base_floor=0.25)
    q2_radi_bonus = compute_patch_guided_component(radi_q2_component, q2_patch_score, base_floor=0.25)
    q1_blastpdb_bonus = compute_patch_guided_component(blastpdb_q1_component, q1_patch_score, base_floor=0.25)
    q2_blastpdb_bonus = compute_patch_guided_component(blastpdb_q2_component, q2_patch_score, base_floor=0.25)

    q1_hotspot_raw = q1_patch_score.copy()
    q2_hotspot_raw = q2_patch_score.copy()
    if radi_weight > 0.0:
        q1_hotspot_raw += radi_weight * q1_radi_bonus
        q2_hotspot_raw += radi_weight * q2_radi_bonus
    if blastpdb_weight > 0.0:
        q1_hotspot_raw += blastpdb_weight * q1_blastpdb_bonus
        q2_hotspot_raw += blastpdb_weight * q2_blastpdb_bonus

    q1_template_prior = q1_patch_score.copy()
    q2_template_prior = q2_patch_score.copy()
    q1_final_scores = normalize_positive_vector(q1_hotspot_raw)
    q2_final_scores = normalize_positive_vector(q2_hotspot_raw)

    q1_patch = select_top_positive_indices(q1_template_prior, patch_residues_per_chain)
    q2_patch = select_top_positive_indices(q2_template_prior, patch_residues_per_chain)
    if q1_patch.size == 0:
        q1_patch = select_top_positive_indices(q1_final_scores, patch_residues_per_chain)
    if q2_patch.size == 0:
        q2_patch = select_top_positive_indices(q2_final_scores, patch_residues_per_chain)

    q1_patch_mask = binary_mask_from_indices(q1_template_prior.size, q1_patch)
    q2_patch_mask = binary_mask_from_indices(q2_template_prior.size, q2_patch)
    patch_mask = np.outer(q1_patch_mask, q2_patch_mask).astype(float)

    template_support_matrix = build_residue_priority_matrix(q1_template_prior, q2_template_prior) * patch_mask
    residue_priority_matrix = build_residue_priority_matrix(q1_final_scores, q2_final_scores)
    anchor_matrix = np.zeros_like(radi_matrix, dtype=float)
    if radi_weight > 0.0:
        anchor_matrix = np.maximum(anchor_matrix, radi_norm_top)
    if blastpdb_weight > 0.0:
        anchor_matrix = np.maximum(anchor_matrix, blastpdb_norm_top)
    anchor_pairs_in_patch = int(np.count_nonzero(anchor_matrix * patch_mask > 0.0))

    return {
        "template_support_matrix": template_support_matrix,
        "residue_priority_matrix": residue_priority_matrix,
        "anchor_matrix": anchor_matrix,
        "q1_scores": q1_final_scores,
        "q2_scores": q2_final_scores,
        "q1_template_prior": q1_template_prior,
        "q2_template_prior": q2_template_prior,
        "q1_patch_indices": q1_patch,
        "q2_patch_indices": q2_patch,
        "gated_radi_pairs": gated_radi_pairs,
        "gated_blastpdb_pairs": gated_blastpdb_pairs,
        "anchor_pairs_in_patch": anchor_pairs_in_patch,
        "fallback_to_patch_only": radi_weight == 0.0 and blastpdb_weight == 0.0,
        "ifrag_norm": ifrag_norm,
        "conservation_norm": conservation_norm,
        "ifrag_q1_strength": ifrag_q1_strength,
        "ifrag_q2_strength": ifrag_q2_strength,
        "ifrag_q1_specificity": ifrag_q1_specificity,
        "ifrag_q2_specificity": ifrag_q2_specificity,
        "ifrag_q1_component": ifrag_q1_component,
        "ifrag_q2_component": ifrag_q2_component,
        "conservation_q1_strength": conservation_q1_strength,
        "conservation_q2_strength": conservation_q2_strength,
        "conservation_q1_component": conservation_q1_component,
        "conservation_q2_component": conservation_q2_component,
        "radi_q1_anchor": radi_q1_anchor,
        "radi_q2_anchor": radi_q2_anchor,
        "radi_q1_component": radi_q1_component,
        "radi_q2_component": radi_q2_component,
        "blastpdb_q1_anchor": blastpdb_q1_anchor,
        "blastpdb_q2_anchor": blastpdb_q2_anchor,
        "blastpdb_q1_component": blastpdb_q1_component,
        "blastpdb_q2_component": blastpdb_q2_component,
        "q1_ifrag_seed_indices": q1_ifrag_seed_indices,
        "q2_ifrag_seed_indices": q2_ifrag_seed_indices,
        "q1_conservation_seed_indices": q1_conservation_seed_indices,
        "q2_conservation_seed_indices": q2_conservation_seed_indices,
        "q1_patch_score": q1_patch_score,
        "q2_patch_score": q2_patch_score,
        "ifrag_q1_guided": q1_ifrag_guided,
        "ifrag_q2_guided": q2_ifrag_guided,
        "radi_q1_bonus": q1_radi_bonus,
        "radi_q2_bonus": q2_radi_bonus,
        "blastpdb_q1_bonus": q1_blastpdb_bonus,
        "blastpdb_q2_bonus": q2_blastpdb_bonus,
        "ifrag_weight": ifrag_reliability,
        "conservation_weight": 1.0 if use_conservation else 0.0,
        "radi_weight": radi_weight,
        "blastpdb_weight": blastpdb_weight,
        "ifrag_density": fraction_nonzero(ifrag_matrix) if use_ifrag else 0.0,
    }


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    resolved_dir = out_dir / "resolved_inputs"
    resolved_dir.mkdir(parents=True, exist_ok=True)

    query1 = resolve_query(
        label="query1",
        fasta_path=args.query1_fasta,
        pdb_path=args.query1_pdb,
        chain=args.query1_chain,
        resolved_dir=resolved_dir,
        warnings=warnings,
    )
    query2 = resolve_query(
        label="query2",
        fasta_path=args.query2_fasta,
        pdb_path=args.query2_pdb,
        chain=args.query2_chain,
        resolved_dir=resolved_dir,
        warnings=warnings,
    )
    interaction_mode_used = resolve_interaction_mode(args.interaction_mode, query1.sequence, query2.sequence)
    if interaction_mode_used == "homomer":
        warnings.append("interaction mode resolved to homomer; residue evidence will be symmetrized across both protomers.")

    needs_ifrag = args.combine_mode in {"ifrag_radi", "ifrag_conservation", "ifrag_conservation_radi", "ifrag_blastpdb"}
    needs_conservation = args.combine_mode in {"conservation_radi", "ifrag_conservation", "ifrag_conservation_radi"}
    needs_radi = args.combine_mode in {"ifrag_radi", "conservation_radi", "ifrag_conservation_radi"}
    needs_blastpdb = bool(args.use_blastpdb)
    needs_homolog_search = needs_conservation or needs_radi
    radi_status = "not_requested_by_mode"
    blastpdb_status = "not_requested"
    if not needs_radi:
        warnings.append(
            f"raDI branch not requested because combine mode is '{args.combine_mode}'. "
            "Final residue scores were computed without raDI anchors."
        )

    ifrag_out = out_dir / "ifrag"
    homolog_search_out = out_dir / "homolog_search"
    conservation_out = out_dir / "conservation"
    radi_prepare_out = out_dir / "radi_prepare"
    radi_out = out_dir / "radi"
    blastpdb_out = out_dir / "blastpdb"
    if needs_ifrag:
        ifrag_out.mkdir(parents=True, exist_ok=True)
    if needs_homolog_search:
        homolog_search_out.mkdir(parents=True, exist_ok=True)
    if needs_conservation:
        conservation_out.mkdir(parents=True, exist_ok=True)
    if needs_radi:
        radi_prepare_out.mkdir(parents=True, exist_ok=True)
        radi_out.mkdir(parents=True, exist_ok=True)
    if needs_blastpdb:
        blastpdb_out.mkdir(parents=True, exist_ok=True)

    if needs_ifrag:
        run_ifrag(project_root, query1.fasta_path, query2.fasta_path, ifrag_out, args)
    if needs_blastpdb:
        try:
            run_blastpdb(
                project_root,
                query1.fasta_path,
                query2.fasta_path,
                blastpdb_out,
                args,
                interaction_mode_used,
            )
        except Exception as exc:
            blastpdb_status = "execution_failed"
            warnings.append(
                f"blastPDB failed; continuing without structural anchors ({exc})."
            )
        else:
            blastpdb_status = "executed"
    query1_search_tsv: Path | None = None
    query2_search_tsv: Path | None = None
    if needs_homolog_search:
        query1_search_tsv, query2_search_tsv = run_homolog_search(
            project_root,
            query1.fasta_path,
            query2.fasta_path,
            homolog_search_out,
            args,
        )
    if needs_conservation:
        if query1_search_tsv is None or query2_search_tsv is None:
            raise RuntimeError("conservation requested but shared homolog-search outputs were not created")
        run_conservation(
            project_root,
            query1.fasta_path,
            query2.fasta_path,
            conservation_out,
            args,
            query1_search_tsv,
            query2_search_tsv,
            interaction_mode_used,
        )
    if needs_radi:
        if query1_search_tsv is None or query2_search_tsv is None:
            raise RuntimeError("raDI requested but shared homolog-search outputs were not created")
        try:
            run_radi_prepare(
                project_root,
                query1.fasta_path,
                query2.fasta_path,
                radi_prepare_out,
                args,
                query1_search_tsv,
                query2_search_tsv,
                conservation_out if needs_conservation else None,
            )
        except Exception as exc:
            radi_status = "prepare_failed"
            warnings.append(
                f"raDI prepare failed; continuing without raDI anchors ({exc})."
            )
        else:
            radi_prepare_summary = load_optional_json(radi_prepare_out / "radi_prepare_summary.json")
            radi_rows_available = int(radi_prepare_summary.get("paired_rows_used", 0) or 0)
            if radi_rows_available < 2:
                radi_status = "skipped_insufficient_paired_rows"
                warnings.append(
                    "raDI paired MSA has fewer than 2 homolog rows; skipping raDI execution and continuing with template-only residue scoring."
                )
            else:
                try:
                    run_radi(project_root, radi_prepare_out, radi_out, query1, query2, args)
                except Exception as exc:
                    radi_status = "execution_failed"
                    warnings.append(
                        f"raDI execution failed; continuing without raDI anchors ({exc})."
                    )
                else:
                    radi_status = "executed"

    expected_shape = (len(query1.sequence), len(query2.sequence))
    if needs_ifrag:
        ifrag_matrix = load_matrix(ifrag_out / "ifrag_matrix.tsv", expected_shape)
    else:
        ifrag_matrix = np.zeros(expected_shape, dtype=float)
    if needs_conservation:
        conservation_matrix = load_matrix(conservation_out / "conservation_matrix.tsv", expected_shape)
    else:
        conservation_matrix = np.zeros(expected_shape, dtype=float)
    radi_matrix_path = radi_out / "paired_interchain_matrix.tsv"
    radi_matrix = (
        load_matrix(radi_matrix_path, expected_shape)
        if needs_radi and radi_matrix_path.exists()
        else np.zeros(expected_shape, dtype=float)
    )
    blastpdb_matrix_path = blastpdb_out / "blastpdb_matrix.tsv"
    blastpdb_matrix = (
        load_matrix(blastpdb_matrix_path, expected_shape)
        if needs_blastpdb and blastpdb_matrix_path.exists()
        else np.zeros(expected_shape, dtype=float)
    )
    homolog_search_summary = load_optional_json(homolog_search_out / "homolog_search_summary.json") if needs_homolog_search else {}
    conservation_summary = load_optional_json(conservation_out / "conservation_summary.json") if needs_conservation else {}
    radi_prepare_summary = load_optional_json(radi_prepare_out / "radi_prepare_summary.json") if needs_radi else {}
    radi_summary = load_optional_json(radi_out / "radi_summary.json") if needs_radi else {}
    blastpdb_summary = load_optional_json(blastpdb_out / "blastpdb_summary.json") if needs_blastpdb else {}
    conservation_profile_q1 = (
        load_conservation_profile_scores(conservation_out / "query1_conservation_profile.tsv", expected_shape[0])
        if needs_conservation
        else None
    )
    conservation_profile_q2 = (
        load_conservation_profile_scores(conservation_out / "query2_conservation_profile.tsv", expected_shape[1])
        if needs_conservation
        else None
    )
    if needs_conservation and conservation_profile_q1 is None:
        conservation_profile_q1 = np.max(normalize_nonzero_by_percentile(conservation_matrix), axis=1)
    if needs_conservation and conservation_profile_q2 is None:
        conservation_profile_q2 = np.max(normalize_nonzero_by_percentile(conservation_matrix), axis=0)
    if conservation_profile_q1 is None:
        conservation_profile_q1 = np.zeros(expected_shape[0], dtype=float)
    if conservation_profile_q2 is None:
        conservation_profile_q2 = np.zeros(expected_shape[1], dtype=float)
    radi_paired_rows_used = int(radi_prepare_summary.get("paired_rows_used", 0) or 0) if needs_radi else 0
    radi_weak_flag = bool(radi_prepare_summary.get("weak_msa_warning", False)) if needs_radi else False
    low_radi_confidence_reasons: list[str] = []
    if needs_radi and radi_weak_flag:
        low_radi_confidence_reasons.append("radi_prepare weak_msa_warning")
    if (
        needs_radi
        and
        args.radi_min_trusted_paired_rows > 0
        and radi_paired_rows_used < args.radi_min_trusted_paired_rows
    ):
        low_radi_confidence_reasons.append(
            f"paired_rows_used<{args.radi_min_trusted_paired_rows}"
        )
    low_radi_confidence = bool(low_radi_confidence_reasons)
    effective_radi_matrix = radi_matrix
    blastpdb_retained_templates = int(blastpdb_summary.get("retained_templates", 0) or 0) if needs_blastpdb else 0
    low_blastpdb_confidence_reasons: list[str] = []
    if (
        needs_blastpdb
        and args.blastpdb_min_trusted_templates > 0
        and blastpdb_retained_templates < args.blastpdb_min_trusted_templates
    ):
        low_blastpdb_confidence_reasons.append(
            f"retained_templates<{args.blastpdb_min_trusted_templates}"
        )
    low_blastpdb_confidence = bool(low_blastpdb_confidence_reasons)
    effective_blastpdb_matrix = blastpdb_matrix

    if needs_radi and low_radi_confidence:
        if radi_status == "executed":
            radi_status = "ignored_low_confidence"
        effective_radi_matrix = np.zeros_like(radi_matrix)
        warnings.append(
            "raDI paired MSA is below the configured confidence threshold; ignoring raDI during residue scoring."
        )
    elif needs_radi and radi_status == "executed" and np.count_nonzero(radi_matrix > 0.0) == 0:
        radi_status = "executed_no_anchor_pairs"
        warnings.append("raDI executed but produced no retained inter-chain anchor pairs; final residue scores do not include a raDI bonus.")
    elif needs_radi and radi_status == "executed":
        radi_status = "used"

    if needs_blastpdb and low_blastpdb_confidence:
        if blastpdb_status == "executed":
            blastpdb_status = "ignored_low_confidence"
        effective_blastpdb_matrix = np.zeros_like(blastpdb_matrix)
        warnings.append(
            "blastPDB retained fewer structural templates than the configured confidence threshold; ignoring structural anchors during residue scoring."
        )
    elif needs_blastpdb and blastpdb_status == "executed" and np.count_nonzero(blastpdb_matrix > 0.0) == 0:
        blastpdb_status = "executed_no_anchor_pairs"
        warnings.append("blastPDB executed but produced no retained structural anchor pairs.")
    elif needs_blastpdb and blastpdb_status == "executed":
        blastpdb_status = "used"

    selected_matrix_tsv: Path | None
    if args.combine_mode == "ifrag_radi":
        selected_matrix_tsv = ifrag_out / "ifrag_matrix.tsv"
    elif args.combine_mode == "conservation_radi":
        selected_matrix_tsv = conservation_out / "conservation_matrix.tsv"
    elif args.combine_mode == "ifrag_conservation_radi":
        selected_matrix_tsv = conservation_out / "conservation_matrix.tsv"
    elif args.combine_mode == "ifrag_blastpdb":
        selected_matrix_tsv = ifrag_out / "ifrag_matrix.tsv"
    else:
        selected_matrix_tsv = None

    residue_scores = build_residue_first_scores(
        ifrag_matrix=ifrag_matrix,
        conservation_matrix=conservation_matrix,
        conservation_profile_q1=conservation_profile_q1,
        conservation_profile_q2=conservation_profile_q2,
        radi_matrix=effective_radi_matrix,
        blastpdb_matrix=effective_blastpdb_matrix,
        top_k=args.top_k,
        radi_top_pairs=args.radi_top_pairs_consensus,
        patch_residues_per_chain=args.patch_residues_per_chain,
        use_ifrag=needs_ifrag,
        use_conservation=needs_conservation,
        use_radi=needs_radi,
        use_blastpdb=needs_blastpdb,
        low_radi_confidence=low_radi_confidence,
        low_blastpdb_confidence=low_blastpdb_confidence,
    )

    consensus = residue_scores["residue_priority_matrix"]
    template_support_matrix = residue_scores["template_support_matrix"]
    anchor_matrix = residue_scores["anchor_matrix"]
    residue_priority_matrix = residue_scores["residue_priority_matrix"]
    s1 = residue_scores["q1_scores"]
    s2 = residue_scores["q2_scores"]
    q1_patch_indices = residue_scores["q1_patch_indices"]
    q2_patch_indices = residue_scores["q2_patch_indices"]
    gated_radi_pairs = int(residue_scores["gated_radi_pairs"])
    gated_blastpdb_pairs = int(residue_scores["gated_blastpdb_pairs"])
    anchor_pairs_in_patch = int(residue_scores["anchor_pairs_in_patch"])
    fallback_to_patch_only = bool(residue_scores["fallback_to_patch_only"])
    ifrag_norm = residue_scores["ifrag_norm"]
    conservation_norm = residue_scores["conservation_norm"]
    ifrag_q1_strength = residue_scores["ifrag_q1_strength"]
    ifrag_q2_strength = residue_scores["ifrag_q2_strength"]
    ifrag_q1_specificity = residue_scores["ifrag_q1_specificity"]
    ifrag_q2_specificity = residue_scores["ifrag_q2_specificity"]
    ifrag_q1_component = residue_scores["ifrag_q1_component"]
    ifrag_q2_component = residue_scores["ifrag_q2_component"]
    conservation_q1_strength = residue_scores["conservation_q1_strength"]
    conservation_q2_strength = residue_scores["conservation_q2_strength"]
    conservation_q1_component = residue_scores["conservation_q1_component"]
    conservation_q2_component = residue_scores["conservation_q2_component"]
    radi_q1_anchor = residue_scores["radi_q1_anchor"]
    radi_q2_anchor = residue_scores["radi_q2_anchor"]
    radi_q1_component = residue_scores["radi_q1_component"]
    radi_q2_component = residue_scores["radi_q2_component"]
    blastpdb_q1_anchor = residue_scores["blastpdb_q1_anchor"]
    blastpdb_q2_anchor = residue_scores["blastpdb_q2_anchor"]
    blastpdb_q1_component = residue_scores["blastpdb_q1_component"]
    blastpdb_q2_component = residue_scores["blastpdb_q2_component"]
    q1_template_prior = residue_scores["q1_template_prior"]
    q2_template_prior = residue_scores["q2_template_prior"]
    q1_patch_score = residue_scores["q1_patch_score"]
    q2_patch_score = residue_scores["q2_patch_score"]
    q1_ifrag_guided = residue_scores["ifrag_q1_guided"]
    q2_ifrag_guided = residue_scores["ifrag_q2_guided"]
    q1_radi_bonus = residue_scores["radi_q1_bonus"]
    q2_radi_bonus = residue_scores["radi_q2_bonus"]
    q1_blastpdb_bonus = residue_scores["blastpdb_q1_bonus"]
    q2_blastpdb_bonus = residue_scores["blastpdb_q2_bonus"]
    ifrag_weight = float(residue_scores["ifrag_weight"])
    conservation_weight = float(residue_scores["conservation_weight"])
    radi_weight = float(residue_scores["radi_weight"])
    blastpdb_weight = float(residue_scores["blastpdb_weight"])
    ifrag_density = float(residue_scores["ifrag_density"])

    if needs_radi and gated_radi_pairs == 0:
        warnings.append("raDI produced no retained inter-chain pairs after gating; residue scoring used only template-derived evidence.")
    if needs_blastpdb and gated_blastpdb_pairs == 0 and blastpdb_status in {"executed", "used"}:
        warnings.append("blastPDB produced no retained structural pairs after gating; residue scoring used only the other evidence branches.")

    q1_surface_mask = compute_surface_mask(query1, out_dir, args.surface_rsa_threshold, warnings)
    q2_surface_mask = compute_surface_mask(query2, out_dir, args.surface_rsa_threshold, warnings)
    row_surface_mask = np.ones(expected_shape[0], dtype=bool) if q1_surface_mask is None else q1_surface_mask.copy()
    col_surface_mask = np.ones(expected_shape[1], dtype=bool) if q2_surface_mask is None else q2_surface_mask.copy()
    if interaction_mode_used == "homomer":
        shared_surface_mask = np.logical_or(row_surface_mask, col_surface_mask)
        row_surface_mask = shared_surface_mask.copy()
        col_surface_mask = shared_surface_mask.copy()
    surface_prior_applied = q1_surface_mask is not None or q2_surface_mask is not None
    q1_surface_residues_kept: int | None = None
    q2_surface_residues_kept: int | None = None
    if q1_surface_mask is not None:
        q1_surface_residues_kept = int(np.count_nonzero(q1_surface_mask))
    if q2_surface_mask is not None:
        q2_surface_residues_kept = int(np.count_nonzero(q2_surface_mask))

    if surface_prior_applied:
        for arr in (
            s1,
            q1_patch_score,
            q1_template_prior,
            q1_ifrag_guided,
            q1_radi_bonus,
            q1_blastpdb_bonus,
            ifrag_q1_strength,
            ifrag_q1_specificity,
            ifrag_q1_component,
            conservation_q1_strength,
            conservation_q1_component,
            radi_q1_anchor,
            radi_q1_component,
            blastpdb_q1_anchor,
            blastpdb_q1_component,
        ):
            arr[~row_surface_mask] = 0.0
        for arr in (
            s2,
            q2_patch_score,
            q2_template_prior,
            q2_ifrag_guided,
            q2_radi_bonus,
            q2_blastpdb_bonus,
            ifrag_q2_strength,
            ifrag_q2_specificity,
            ifrag_q2_component,
            conservation_q2_strength,
            conservation_q2_component,
            radi_q2_anchor,
            radi_q2_component,
            blastpdb_q2_anchor,
            blastpdb_q2_component,
        ):
            arr[~col_surface_mask] = 0.0

        q1_patch_indices = select_top_positive_indices(q1_template_prior, args.patch_residues_per_chain)
        q2_patch_indices = select_top_positive_indices(q2_template_prior, args.patch_residues_per_chain)
        if q1_patch_indices.size == 0:
            q1_patch_indices = select_top_positive_indices(s1, args.patch_residues_per_chain)
        if q2_patch_indices.size == 0:
            q2_patch_indices = select_top_positive_indices(s2, args.patch_residues_per_chain)

        q1_patch_mask = binary_mask_from_indices(q1_template_prior.size, q1_patch_indices)
        q2_patch_mask = binary_mask_from_indices(q2_template_prior.size, q2_patch_indices)
        patch_mask = np.outer(q1_patch_mask, q2_patch_mask).astype(float)
        pair_surface_mask = np.outer(row_surface_mask, col_surface_mask).astype(float)

        template_support_matrix = build_residue_priority_matrix(q1_template_prior, q2_template_prior) * patch_mask
        anchor_matrix = anchor_matrix * pair_surface_mask
        residue_priority_matrix = build_residue_priority_matrix(s1, s2)
        consensus = residue_priority_matrix.copy()
        anchor_pairs_in_patch = int(np.count_nonzero(anchor_matrix * patch_mask > 0.0))
        warnings.append("Applied SASA surface prior during residue scoring and docking residue selection.")
    elif query1.pdb_path is None and query2.pdb_path is None and args.combine_mode == "ifrag_conservation_radi" and low_radi_confidence:
        warnings.append("Low-confidence-raDI fallback used without SASA filtering because no query PDB inputs were provided.")

    if interaction_mode_used == "homomer":
        if expected_shape[0] != expected_shape[1]:
            raise RuntimeError("homomer mode currently requires identical query lengths after query resolution")

        def symmetrize_pair(v1: np.ndarray, v2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            shared = 0.5 * (v1 + v2)
            return shared.copy(), shared.copy()

        s1, s2 = symmetrize_pair(s1, s2)
        q1_template_prior, q2_template_prior = symmetrize_pair(q1_template_prior, q2_template_prior)
        q1_patch_score, q2_patch_score = symmetrize_pair(q1_patch_score, q2_patch_score)
        q1_ifrag_guided, q2_ifrag_guided = symmetrize_pair(q1_ifrag_guided, q2_ifrag_guided)
        q1_radi_bonus, q2_radi_bonus = symmetrize_pair(q1_radi_bonus, q2_radi_bonus)
        q1_blastpdb_bonus, q2_blastpdb_bonus = symmetrize_pair(q1_blastpdb_bonus, q2_blastpdb_bonus)
        ifrag_q1_strength, ifrag_q2_strength = symmetrize_pair(ifrag_q1_strength, ifrag_q2_strength)
        ifrag_q1_specificity, ifrag_q2_specificity = symmetrize_pair(ifrag_q1_specificity, ifrag_q2_specificity)
        ifrag_q1_component, ifrag_q2_component = symmetrize_pair(ifrag_q1_component, ifrag_q2_component)
        conservation_q1_strength, conservation_q2_strength = symmetrize_pair(conservation_q1_strength, conservation_q2_strength)
        conservation_q1_component, conservation_q2_component = symmetrize_pair(conservation_q1_component, conservation_q2_component)
        radi_q1_anchor, radi_q2_anchor = symmetrize_pair(radi_q1_anchor, radi_q2_anchor)
        radi_q1_component, radi_q2_component = symmetrize_pair(radi_q1_component, radi_q2_component)
        blastpdb_q1_anchor, blastpdb_q2_anchor = symmetrize_pair(blastpdb_q1_anchor, blastpdb_q2_anchor)
        blastpdb_q1_component, blastpdb_q2_component = symmetrize_pair(blastpdb_q1_component, blastpdb_q2_component)

        shared_patch_indices = select_top_positive_indices(q1_template_prior, args.patch_residues_per_chain)
        if shared_patch_indices.size == 0:
            shared_patch_indices = select_top_positive_indices(s1, args.patch_residues_per_chain)
        q1_patch_indices = shared_patch_indices.copy()
        q2_patch_indices = shared_patch_indices.copy()
        shared_patch_mask = binary_mask_from_indices(q1_template_prior.size, shared_patch_indices)
        patch_mask = np.outer(shared_patch_mask, shared_patch_mask).astype(float)

        template_support_matrix = build_residue_priority_matrix(q1_template_prior, q2_template_prior) * patch_mask
        residue_priority_matrix = build_residue_priority_matrix(s1, s2)
        consensus = residue_priority_matrix.copy()
        if anchor_matrix.shape[0] == anchor_matrix.shape[1]:
            anchor_matrix = np.maximum(anchor_matrix, anchor_matrix.T)
        anchor_pairs_in_patch = int(np.count_nonzero(anchor_matrix * patch_mask > 0.0))

    q1_selection_support = compute_direct_support_signal(
        ifrag_q1_component,
        conservation_q1_component,
        radi_q1_component,
        blastpdb_q1_component,
        ifrag_weight=ifrag_weight,
        radi_weight=radi_weight,
        blastpdb_weight=blastpdb_weight,
    )
    q2_selection_support = compute_direct_support_signal(
        ifrag_q2_component,
        conservation_q2_component,
        radi_q2_component,
        blastpdb_q2_component,
        ifrag_weight=ifrag_weight,
        radi_weight=radi_weight,
        blastpdb_weight=blastpdb_weight,
    )
    q1_struct_conf = np.ones_like(s1, dtype=float)
    q2_struct_conf = np.ones_like(s2, dtype=float)
    q1_struct_local_mass = np.zeros_like(s1, dtype=float)
    q2_struct_local_mass = np.zeros_like(s2, dtype=float)
    q1_struct_shape = np.zeros_like(s1, dtype=float)
    q2_struct_shape = np.zeros_like(s2, dtype=float)
    q1_struct_hydrophobic = np.zeros_like(s1, dtype=float)
    q2_struct_hydrophobic = np.zeros_like(s2, dtype=float)
    q1_struct_conf_source = "off"
    q2_struct_conf_source = "off"
    q1_struct_conf_detected = False
    q2_struct_conf_detected = False
    structaware_applied = False
    q1_confidence_mode_used = resolve_structure_confidence_mode(
        args.structaware_confidence_mode,
        args.query1_structure_source,
    )
    q2_confidence_mode_used = resolve_structure_confidence_mode(
        args.structaware_confidence_mode,
        args.query2_structure_source,
    )
    q1_surface_weights = row_surface_mask.astype(float, copy=True)
    q2_surface_weights = col_surface_mask.astype(float, copy=True)
    if args.structaware_mode == "rerank":
        q1_struct = rerank_with_structure_features(
            sequence=query1.sequence,
            pdb_path=query1.pdb_path,
            chain_id=query1.chain,
            coords=query1.pdb_residue_coords,
            base_scores=s1,
            support_scores=q1_selection_support,
            eligible_mask=row_surface_mask,
            surface_weights=q1_surface_weights,
            confidence_mode=q1_confidence_mode_used,
            hydrophobic_weight=args.structaware_hydrophobic_weight,
        )
        q2_struct = rerank_with_structure_features(
            sequence=query2.sequence,
            pdb_path=query2.pdb_path,
            chain_id=query2.chain,
            coords=query2.pdb_residue_coords,
            base_scores=s2,
            support_scores=q2_selection_support,
            eligible_mask=col_surface_mask,
            surface_weights=q2_surface_weights,
            confidence_mode=q2_confidence_mode_used,
            hydrophobic_weight=args.structaware_hydrophobic_weight,
        )
        q1_can_rerank = (
            query1.pdb_residue_coords is not None
            and query1.pdb_residue_coords.shape[0] == s1.size
        )
        q2_can_rerank = (
            query2.pdb_residue_coords is not None
            and query2.pdb_residue_coords.shape[0] == s2.size
        )
        if q1_can_rerank or q2_can_rerank:
            structaware_applied = True
            s1 = q1_struct.final_scores
            s2 = q2_struct.final_scores
            q1_selection_support = q1_struct.support_scores
            q2_selection_support = q2_struct.support_scores
            q1_struct_conf = q1_struct.confidence_component
            q2_struct_conf = q2_struct.confidence_component
            q1_struct_local_mass = q1_struct.local_mass_component
            q2_struct_local_mass = q2_struct.local_mass_component
            q1_struct_shape = q1_struct.shape_component
            q2_struct_shape = q2_struct.shape_component
            q1_struct_hydrophobic = q1_struct.hydrophobic_patch_component
            q2_struct_hydrophobic = q2_struct.hydrophobic_patch_component
            q1_struct_conf_source = q1_struct.confidence_source
            q2_struct_conf_source = q2_struct.confidence_source
            q1_struct_conf_detected = q1_struct.confidence_detected
            q2_struct_conf_detected = q2_struct.confidence_detected
            if interaction_mode_used == "homomer" and s1.size == s2.size:
                def symmetrize_struct_pair(v1: np.ndarray, v2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                    shared = 0.5 * (v1 + v2)
                    return shared.copy(), shared.copy()

                s1, s2 = symmetrize_struct_pair(s1, s2)
                q1_selection_support, q2_selection_support = symmetrize_struct_pair(q1_selection_support, q2_selection_support)
                q1_struct_conf, q2_struct_conf = symmetrize_struct_pair(q1_struct_conf, q2_struct_conf)
                q1_struct_local_mass, q2_struct_local_mass = symmetrize_struct_pair(q1_struct_local_mass, q2_struct_local_mass)
                q1_struct_shape, q2_struct_shape = symmetrize_struct_pair(q1_struct_shape, q2_struct_shape)
                q1_struct_hydrophobic, q2_struct_hydrophobic = symmetrize_struct_pair(q1_struct_hydrophobic, q2_struct_hydrophobic)
            residue_priority_matrix = build_residue_priority_matrix(s1, s2)
            warnings.append("Applied structure-aware reranking on top of the classical residue score.")

    consensus = residue_priority_matrix.copy()
    consensus_tsv = out_dir / "consensus_pair_matrix.tsv"
    consensus_npy = out_dir / "consensus_pair_matrix.npy"
    consensus_top_pairs = out_dir / "consensus_top_pairs.tsv"
    template_support_tsv = out_dir / "template_support_matrix.tsv"
    template_support_npy = out_dir / "template_support_matrix.npy"
    anchor_pair_tsv = out_dir / "anchor_pair_matrix.tsv"
    anchor_pair_npy = out_dir / "anchor_pair_matrix.npy"
    residue_priority_tsv = out_dir / "residue_priority_matrix.tsv"
    residue_priority_npy = out_dir / "residue_priority_matrix.npy"
    template_support_heatmap = out_dir / "template_support_heatmap.png"
    anchor_overlay_heatmap = out_dir / "template_support_with_anchor_overlay.png"
    ifrag_blastpdb_overlay_heatmap = out_dir / "ifrag_with_blastpdb_overlay.png"
    final_score_heatmap = out_dir / "final_score_heatmap.png"
    legacy_consensus_heatmap = out_dir / "consensus_heatmap.png"
    legacy_top_residue_heatmap = out_dir / "consensus_top_residues_heatmap.png"
    docking_top_pairs = out_dir / "docking_candidate_pairs.tsv"
    docking_top_pairs_loose = out_dir / "docking_candidate_pairs.loose.tsv"
    q1_res_scores = out_dir / "query1_residue_scores.tsv"
    q2_res_scores = out_dir / "query2_residue_scores.tsv"
    q1_branch_scores = out_dir / "query1_branch_scores.tsv"
    q2_branch_scores = out_dir / "query2_branch_scores.tsv"
    q1_struct_features = out_dir / "query1_structure_features.tsv"
    q2_struct_features = out_dir / "query2_structure_features.tsv"
    q1_docking_residues = out_dir / "query1_docking_residues.tsv"
    q2_docking_residues = out_dir / "query2_docking_residues.tsv"
    q1_docking_residues_strict = out_dir / "query1_docking_residues.strict.tsv"
    q2_docking_residues_strict = out_dir / "query2_docking_residues.strict.tsv"
    q1_docking_residues_loose = out_dir / "query1_docking_residues.loose.tsv"
    q2_docking_residues_loose = out_dir / "query2_docking_residues.loose.tsv"
    lightdock_strict = out_dir / "lightdock_restraints.strict.list"
    lightdock_strict_active = out_dir / "lightdock_restraints.strict_active.list"
    lightdock_loose = out_dir / "lightdock_restraints.loose.list"
    lightdock_query1_only = out_dir / "lightdock_restraints.query1_only.strict.list"
    lightdock_query2_only = out_dir / "lightdock_restraints.query2_only.strict.list"
    q1_residue_plot = out_dir / "query1_residue_scores.png"
    q2_residue_plot = out_dir / "query2_residue_scores.png"
    q1_scored_pdb = out_dir / "query1_residue_scores_colored.pdb"
    q2_scored_pdb = out_dir / "query2_residue_scores_colored.pdb"
    summary_path = out_dir / "consensus_summary.json"

    for legacy_path in (legacy_consensus_heatmap, legacy_top_residue_heatmap):
        if legacy_path.exists():
            legacy_path.unlink()
    if anchor_overlay_heatmap.exists() and not (needs_radi or needs_blastpdb):
        anchor_overlay_heatmap.unlink()

    write_matrix_tsv(consensus_tsv, consensus)
    np.save(consensus_npy, consensus)
    write_matrix_tsv(template_support_tsv, template_support_matrix)
    np.save(template_support_npy, template_support_matrix)
    write_matrix_tsv(anchor_pair_tsv, anchor_matrix)
    np.save(anchor_pair_npy, anchor_matrix)
    write_matrix_tsv(residue_priority_tsv, residue_priority_matrix)
    np.save(residue_priority_npy, residue_priority_matrix)
    top_pairs_written = write_top_pairs(consensus_top_pairs, consensus, query1.sequence, query2.sequence)
    write_residue_scores(q1_res_scores, query1, s1)
    write_residue_scores(q2_res_scores, query2, s2)
    write_branch_score_table(
        q1_branch_scores,
        query1,
        s1,
        q1_patch_score,
        ifrag_q1_strength,
        ifrag_q1_specificity,
        ifrag_q1_component,
        conservation_q1_strength,
        conservation_q1_component,
        radi_q1_anchor,
        radi_q1_component,
        blastpdb_q1_anchor,
        blastpdb_q1_component,
    )
    write_branch_score_table(
        q2_branch_scores,
        query2,
        s2,
        q2_patch_score,
        ifrag_q2_strength,
        ifrag_q2_specificity,
        ifrag_q2_component,
        conservation_q2_strength,
        conservation_q2_component,
        radi_q2_anchor,
        radi_q2_component,
        blastpdb_q2_anchor,
        blastpdb_q2_component,
    )
    if structaware_applied:
        write_structure_feature_table(
            q1_struct_features,
            query1,
            s1,
            q1_selection_support,
            q1_struct_conf,
            q1_struct_local_mass,
            q1_struct_shape,
            q1_struct_hydrophobic,
        )
        write_structure_feature_table(
            q2_struct_features,
            query2,
            s2,
            q2_selection_support,
            q2_struct_conf,
            q2_struct_local_mass,
            q2_struct_shape,
            q2_struct_hydrophobic,
        )
    q1_selection_tiebreak = (
        q1_selection_support
        + conservation_q1_component
        + 0.5 * (radi_q1_component + blastpdb_q1_component)
        + 0.25 * q1_patch_score
    )
    q2_selection_tiebreak = (
        q2_selection_support
        + conservation_q2_component
        + 0.5 * (radi_q2_component + blastpdb_q2_component)
        + 0.25 * q2_patch_score
    )
    if interaction_mode_used == "homomer":
        shared_coords = query1.pdb_residue_coords if query1.pdb_residue_coords is not None else query2.pdb_residue_coords
        q1_active_strict, q1_passive_strict, q1_selected_strict = select_adaptive_docking_indices(
            s1,
            q1_selection_support,
            coords=shared_coords,
            eligible_mask=row_surface_mask,
            tie_break_scores=q1_selection_tiebreak,
            requested_active_count=args.strict_active_residues_per_chain,
            requested_passive_count=args.strict_passive_residues_per_chain,
            active_score_fraction=0.75,
            passive_score_fraction=0.35,
        )
        q1_active_loose, q1_passive_loose, q1_selected_loose = select_adaptive_docking_indices(
            s1,
            q1_selection_support,
            coords=shared_coords,
            eligible_mask=row_surface_mask,
            tie_break_scores=q1_selection_tiebreak,
            requested_active_count=args.active_residues_per_chain,
            requested_passive_count=args.passive_residues_per_chain,
            active_score_fraction=0.45,
            passive_score_fraction=0.15,
        )
        q2_active_strict = q1_active_strict.copy()
        q2_passive_strict = q1_passive_strict.copy()
        q2_selected_strict = q1_selected_strict.copy()
        q2_active_loose = q1_active_loose.copy()
        q2_passive_loose = q1_passive_loose.copy()
        q2_selected_loose = q1_selected_loose.copy()
    else:
        q1_active_strict, q1_passive_strict, q1_selected_strict = select_adaptive_docking_indices(
            s1,
            q1_selection_support,
            coords=query1.pdb_residue_coords,
            eligible_mask=row_surface_mask,
            tie_break_scores=q1_selection_tiebreak,
            requested_active_count=args.strict_active_residues_per_chain,
            requested_passive_count=args.strict_passive_residues_per_chain,
            active_score_fraction=0.75,
            passive_score_fraction=0.35,
        )
        q2_active_strict, q2_passive_strict, q2_selected_strict = select_adaptive_docking_indices(
            s2,
            q2_selection_support,
            coords=query2.pdb_residue_coords,
            eligible_mask=col_surface_mask,
            tie_break_scores=q2_selection_tiebreak,
            requested_active_count=args.strict_active_residues_per_chain,
            requested_passive_count=args.strict_passive_residues_per_chain,
            active_score_fraction=0.75,
            passive_score_fraction=0.35,
        )
        q1_active_loose, q1_passive_loose, q1_selected_loose = select_adaptive_docking_indices(
            s1,
            q1_selection_support,
            coords=query1.pdb_residue_coords,
            eligible_mask=row_surface_mask,
            tie_break_scores=q1_selection_tiebreak,
            requested_active_count=args.active_residues_per_chain,
            requested_passive_count=args.passive_residues_per_chain,
            active_score_fraction=0.45,
            passive_score_fraction=0.15,
        )
        q2_active_loose, q2_passive_loose, q2_selected_loose = select_adaptive_docking_indices(
            s2,
            q2_selection_support,
            coords=query2.pdb_residue_coords,
            eligible_mask=col_surface_mask,
            tie_break_scores=q2_selection_tiebreak,
            requested_active_count=args.active_residues_per_chain,
            requested_passive_count=args.passive_residues_per_chain,
            active_score_fraction=0.45,
            passive_score_fraction=0.15,
        )
    if q1_selected_strict.size == 0:
        warnings.append("query1: no strict docking residues passed the direct-support selection thresholds.")
    if q2_selected_strict.size == 0:
        warnings.append("query2: no strict docking residues passed the direct-support selection thresholds.")
    q1_docking_written = write_docking_residue_table(
        q1_docking_residues,
        query1,
        s1,
        q1_active_strict,
        q1_passive_strict,
        ifrag_component=ifrag_q1_component,
        conservation_component=conservation_q1_component,
        radi_component=radi_q1_component,
        blastpdb_component=blastpdb_q1_component,
    )
    write_docking_residue_table(
        q1_docking_residues_strict,
        query1,
        s1,
        q1_active_strict,
        q1_passive_strict,
        ifrag_component=ifrag_q1_component,
        conservation_component=conservation_q1_component,
        radi_component=radi_q1_component,
        blastpdb_component=blastpdb_q1_component,
    )
    q1_docking_loose_written = write_docking_residue_table(
        q1_docking_residues_loose,
        query1,
        s1,
        q1_active_loose,
        q1_passive_loose,
        ifrag_component=ifrag_q1_component,
        conservation_component=conservation_q1_component,
        radi_component=radi_q1_component,
        blastpdb_component=blastpdb_q1_component,
    )
    q2_docking_written = write_docking_residue_table(
        q2_docking_residues,
        query2,
        s2,
        q2_active_strict,
        q2_passive_strict,
        ifrag_component=ifrag_q2_component,
        conservation_component=conservation_q2_component,
        radi_component=radi_q2_component,
        blastpdb_component=blastpdb_q2_component,
    )
    write_docking_residue_table(
        q2_docking_residues_strict,
        query2,
        s2,
        q2_active_strict,
        q2_passive_strict,
        ifrag_component=ifrag_q2_component,
        conservation_component=conservation_q2_component,
        radi_component=radi_q2_component,
        blastpdb_component=blastpdb_q2_component,
    )
    q2_docking_loose_written = write_docking_residue_table(
        q2_docking_residues_loose,
        query2,
        s2,
        q2_active_loose,
        q2_passive_loose,
        ifrag_component=ifrag_q2_component,
        conservation_component=conservation_q2_component,
        radi_component=radi_q2_component,
        blastpdb_component=blastpdb_q2_component,
    )
    lightdock_strict_written = write_lightdock_restraints(
        lightdock_strict,
        receptor_query=query1,
        receptor_active=q1_active_strict,
        receptor_passive=q1_passive_strict,
        ligand_query=query2,
        ligand_active=q2_active_strict,
        ligand_passive=q2_passive_strict,
        include_passive=True,
    )
    lightdock_strict_active_written = write_lightdock_restraints(
        lightdock_strict_active,
        receptor_query=query1,
        receptor_active=q1_active_strict,
        receptor_passive=np.array([], dtype=int),
        ligand_query=query2,
        ligand_active=q2_active_strict,
        ligand_passive=np.array([], dtype=int),
        include_passive=False,
    )
    lightdock_loose_written = write_lightdock_restraints(
        lightdock_loose,
        receptor_query=query1,
        receptor_active=q1_active_loose,
        receptor_passive=q1_passive_loose,
        ligand_query=query2,
        ligand_active=q2_active_loose,
        ligand_passive=q2_passive_loose,
        include_passive=True,
    )
    lightdock_query1_only_written = write_lightdock_restraints(
        lightdock_query1_only,
        receptor_query=query1,
        receptor_active=q1_active_strict,
        receptor_passive=q1_passive_strict,
        ligand_query=query2,
        ligand_active=np.array([], dtype=int),
        ligand_passive=np.array([], dtype=int),
        include_passive=True,
    )
    lightdock_query2_only_written = write_lightdock_restraints(
        lightdock_query2_only,
        receptor_query=query1,
        receptor_active=np.array([], dtype=int),
        receptor_passive=np.array([], dtype=int),
        ligand_query=query2,
        ligand_active=q2_active_strict,
        ligand_passive=q2_passive_strict,
        include_passive=True,
    )
    docking_pairs_written = write_top_pairs(
        docking_top_pairs,
        consensus,
        query1.sequence,
        query2.sequence,
        row_indices=q1_selected_strict,
        col_indices=q2_selected_strict,
    )
    docking_pairs_loose_written = write_top_pairs(
        docking_top_pairs_loose,
        consensus,
        query1.sequence,
        query2.sequence,
        row_indices=q1_selected_loose,
        col_indices=q2_selected_loose,
    )
    q1_scored_pdb_written = write_scored_pdb(query1, s1, q1_scored_pdb)
    q2_scored_pdb_written = write_scored_pdb(query2, s2, q2_scored_pdb)

    overlay_written = False
    ifrag_blastpdb_overlay_written = False
    template_support_heatmap_written = False
    final_score_heatmap_written = False
    overlay_pairs_written = 0
    ifrag_blastpdb_overlay_pairs_written = 0
    residue_plots_written = False
    overlay_anchor_matrix = anchor_matrix
    if not args.no_heatmap:
        overlay_top_pairs = args.overlay_top_pairs or args.radi_top_pairs_consensus
        save_heatmap(
            template_support_heatmap,
            template_support_matrix,
            "iFrag + conservation support heatmap",
            "Template support",
        )
        template_support_heatmap_written = True
        if needs_conservation and (needs_radi or needs_blastpdb):
            overlay_pairs_written = save_overlay_heatmap(
                anchor_overlay_heatmap,
                template_support_matrix,
                overlay_anchor_matrix,
                overlay_top_pairs,
                "Anchor evidence over iFrag + conservation support",
                "Template support",
                "Anchor strength",
            )
            overlay_written = True
        if needs_ifrag and needs_blastpdb:
            ifrag_blastpdb_overlay_pairs_written = save_full_heatmap_overlay(
                ifrag_blastpdb_overlay_heatmap,
                ifrag_matrix,
                blastpdb_matrix,
                "Full blastPDB heatmap over raw iFrag matrix",
                "iFrag support",
                "blastPDB support",
                base_cmap="viridis",
                overlay_cmap="cool",
            )
            ifrag_blastpdb_overlay_written = True
        save_heatmap(final_score_heatmap, consensus, "Final residue-prior heatmap", "Residue prior")
        save_residue_track_plot(q1_residue_plot, s1, "Query1 residue scores", "Residue score")
        save_residue_track_plot(q2_residue_plot, s2, "Query2 residue scores", "Residue score")
        residue_plots_written = True
        final_score_heatmap_written = True

    if args.combine_mode == "ifrag_radi":
        ifrag_role = "main template-derived residue signal in ifrag_radi mode"
        conservation_role = "not used"
        radi_gating = "top_nonzero_max_normalized_values"
        radi_normalization = "retain_top_n_then_divide_by_top_di"
        consensus_rule = "final heatmap = outer(final query1 residue scores, final query2 residue scores)"
    elif args.combine_mode == "ifrag_blastpdb":
        ifrag_role = "main template-derived residue signal in ifrag_blastpdb mode"
        conservation_role = "not used"
        radi_gating = "not used"
        radi_normalization = "not used"
        consensus_rule = "final heatmap = outer(final query1 residue scores, final query2 residue scores)"
    elif args.combine_mode == "conservation_radi":
        ifrag_role = "not used"
        conservation_role = "main template-derived residue signal in conservation_radi mode"
        radi_gating = "top_nonzero_max_normalized_values"
        radi_normalization = "retain_top_n_then_divide_by_top_di"
        consensus_rule = "final heatmap = outer(final query1 residue scores, final query2 residue scores)"
    elif args.combine_mode == "ifrag_conservation":
        ifrag_role = "template-derived residue ranking signal and patch contributor in ifrag_conservation mode"
        conservation_role = "per-residue conservation profile used as a broad patch contributor in ifrag_conservation mode"
        radi_gating = "not used"
        radi_normalization = "not used"
        consensus_rule = "diagnostic residue-priority heatmap = outer(final query1 residue scores, final query2 residue scores)"
    else:
        ifrag_role = "template-derived residue ranking signal and patch contributor in ifrag_conservation_radi mode"
        conservation_role = "per-residue conservation profile used as a broad patch contributor in ifrag_conservation_radi mode"
        radi_gating = "top_nonzero_max_normalized_values"
        radi_normalization = "retain_top_n_then_divide_by_top_di"
        consensus_rule = "diagnostic residue-priority heatmap = outer(final query1 residue scores, final query2 residue scores); raDI and/or blastPDB are shown separately as sparse anchors"

    blastpdb_role = (
        "optional structural-template anchor evidence from experimental PDB biological assemblies"
        if needs_blastpdb
        else "not used"
    )
    anchor_bonus_terms: list[str] = []
    if needs_radi:
        anchor_bonus_terms.append("radi_bonus_weight*patch_guided_radi_bonus")
    if needs_blastpdb:
        anchor_bonus_terms.append("blastpdb_bonus_weight*patch_guided_blastpdb_bonus")
    if anchor_bonus_terms:
        residue_score_rule = (
            "final hotspot score = normalize_positive(seed-region patch score + "
            + " + ".join(anchor_bonus_terms)
            + ")"
        )
    else:
        residue_score_rule = "final hotspot score = normalize_positive(seed-region patch score)"

    warnings_path = out_dir / "run_warnings.txt"
    warnings_path.write_text("\n".join(warnings) + ("\n" if warnings else ""), encoding="utf-8")

    summary = {
        "inputs": {
            "query1_fasta": str(args.query1_fasta) if args.query1_fasta else None,
            "query2_fasta": str(args.query2_fasta) if args.query2_fasta else None,
            "query1_pdb": str(args.query1_pdb) if args.query1_pdb else None,
            "query2_pdb": str(args.query2_pdb) if args.query2_pdb else None,
            "query1_chain": args.query1_chain,
            "query2_chain": args.query2_chain,
            "interaction_mode_requested": args.interaction_mode,
            "interaction_mode_used": interaction_mode_used,
        },
        "input_types": {
            "query1": query1.source_type,
            "query2": query2.source_type,
        },
        "resolved_sequences": {
            "query1_header": query1.header,
            "query2_header": query2.header,
            "query1_length": len(query1.sequence),
            "query2_length": len(query2.sequence),
            "query1_fasta": str(query1.fasta_path),
            "query2_fasta": str(query2.fasta_path),
        },
        "runs": {
            "ifrag_out_dir": str(ifrag_out) if needs_ifrag else None,
            "homolog_search_out_dir": str(homolog_search_out) if needs_homolog_search else None,
            "conservation_out_dir": str(conservation_out) if needs_conservation else None,
            "radi_prepare_out_dir": str(radi_prepare_out) if needs_radi else None,
            "radi_out_dir": str(radi_out) if needs_radi else None,
            "blastpdb_out_dir": str(blastpdb_out) if needs_blastpdb else None,
        },
        "ifrag_parameters": {
            "combine_mode": args.combine_mode,
            "interaction_mode": interaction_mode_used,
            "selected_matrix_tsv": str(selected_matrix_tsv) if selected_matrix_tsv is not None else None,
            "ifrag_matrix_tsv": str(ifrag_out / "ifrag_matrix.tsv") if needs_ifrag else None,
            "blast_db": str(args.ifrag_blast_db) if needs_ifrag else None,
            "evalue": args.ifrag_evalue if needs_ifrag else None,
            "max_target_seqs": args.ifrag_max_target_seqs if needs_ifrag else None,
            "min_pident": args.ifrag_min_pident if needs_ifrag else None,
            "min_aln_len": args.ifrag_min_aln_len if needs_ifrag else None,
            "min_cov1": args.ifrag_min_cov1 if needs_ifrag else None,
            "max_cov1": args.ifrag_max_cov1 if needs_ifrag else None,
            "min_cov2": args.ifrag_min_cov2 if needs_ifrag else None,
            "max_cov2": args.ifrag_max_cov2 if needs_ifrag else None,
            "top_k": args.ifrag_top_k if needs_ifrag else None,
        },
        "radi_parameters": {
            "radi_enabled": needs_radi,
            "radi_status": radi_status,
            "search_backend": f"shared_template_mmseqs_{args.homolog_search_mode}",
            "homolog_search_mode": args.homolog_search_mode,
            "pair_dataset": args.radi_pair_dataset,
            "pairs_file": str(args.radi_pairs),
            "pairs_meta_file": str(args.radi_pairs_meta),
            "pairing_mode_requested": "interaction_only" if needs_radi else None,
            "pairing_mode_used": radi_prepare_summary.get("pairing_mode_used") if needs_radi else None,
            "ra": args.radi_ra if needs_radi else None,
            "template_mmseqs_db": str(args.radi_template_mmseqs_db),
            "template_sequence_fasta": str(args.radi_sequence_fasta),
            "template_proteins_tsv": str(args.radi_template_proteins),
            "mmseqs_bin": args.radi_mmseqs_bin,
            "mmseqs_sensitivity": args.radi_mmseqs_sensitivity,
            "row_builder": "shared_resolved_homolog_hits",
            "pairing_builder": "radi_prepare.py" if needs_radi else None,
            "stage1_iterations": args.radi_stage1_iterations,
            "evalue": args.radi_evalue,
            "max_hits": args.radi_max_hits,
            "min_trusted_paired_rows_for_combine": args.radi_min_trusted_paired_rows if needs_radi else None,
            "conservation_matrix_tsv": str(conservation_out / "conservation_matrix.tsv") if needs_conservation else None,
            "radi_prepare_summary_json": str(radi_prepare_out / "radi_prepare_summary.json") if needs_radi else None,
        },
        "blastpdb_parameters": {
            "blastpdb_enabled": needs_blastpdb,
            "blastpdb_status": blastpdb_status,
            "cache_dir": str(args.blastpdb_cache_dir) if needs_blastpdb else None,
            "blast_bin": args.blastpdb_blast_bin if needs_blastpdb else None,
            "makeblastdb_bin": args.blastpdb_makeblastdb_bin if needs_blastpdb else None,
            "top_assemblies": args.blastpdb_top_assemblies if needs_blastpdb else None,
            "sequence_search_identity_cutoff": args.blastpdb_sequence_search_identity_cutoff if needs_blastpdb else None,
            "sequence_search_evalue_cutoff": args.blastpdb_sequence_search_evalue_cutoff if needs_blastpdb else None,
            "local_blast_evalue": args.blastpdb_local_blast_evalue if needs_blastpdb else None,
            "local_blast_max_target_seqs": args.blastpdb_local_blast_max_target_seqs if needs_blastpdb else None,
            "cbeta_threshold": args.blastpdb_cbeta_threshold if needs_blastpdb else None,
            "min_template_contacts": args.blastpdb_min_template_contacts if needs_blastpdb else None,
            "min_trusted_templates_for_combine": args.blastpdb_min_trusted_templates if needs_blastpdb else None,
            "blastpdb_summary_json": str(blastpdb_out / "blastpdb_summary.json") if needs_blastpdb else None,
        },
        "matrix_shapes": {
            "ifrag": list(ifrag_matrix.shape),
            "conservation": list(conservation_matrix.shape),
            "radi": list(radi_matrix.shape),
            "blastpdb": list(blastpdb_matrix.shape),
            "template_support": list(template_support_matrix.shape),
            "anchor_pairs": list(anchor_matrix.shape),
            "consensus": list(consensus.shape),
        },
        "nonzero_counts": {
            "ifrag": int(np.count_nonzero(ifrag_matrix > 0.0)),
            "conservation": int(np.count_nonzero(conservation_matrix > 0.0)),
            "radi": int(np.count_nonzero(radi_matrix > 0.0)),
            "blastpdb": int(np.count_nonzero(blastpdb_matrix > 0.0)),
            "template_support": int(np.count_nonzero(template_support_matrix > 0.0)),
            "anchor_pairs": int(np.count_nonzero(anchor_matrix > 0.0)),
            "consensus": int(np.count_nonzero(consensus > 0.0)),
        },
        "scoring": {
            "combine_mode": args.combine_mode,
            "interaction_mode": interaction_mode_used,
            "structaware_mode": args.structaware_mode,
            "structaware_applied": structaware_applied,
            "structaware_confidence_mode": args.structaware_confidence_mode,
            "structaware_hydrophobic_weight": args.structaware_hydrophobic_weight,
            "query1_structure_source": args.query1_structure_source,
            "query2_structure_source": args.query2_structure_source,
            "query1_confidence_mode_used": q1_confidence_mode_used,
            "query2_confidence_mode_used": q2_confidence_mode_used,
            "query1_confidence_source": q1_struct_conf_source,
            "query2_confidence_source": q2_struct_conf_source,
            "query1_confidence_detected": q1_struct_conf_detected,
            "query2_confidence_detected": q2_struct_conf_detected,
            "ifrag_role": ifrag_role,
            "conservation_role": conservation_role,
            "blastpdb_role": blastpdb_role,
            "template_normalization": "nonzero_percentile_rank",
            "radi_gating": radi_gating,
            "radi_normalization": radi_normalization,
            "consensus_rule": consensus_rule,
            "ifrag_weight": ifrag_weight,
            "conservation_weight": conservation_weight,
            "radi_weight": radi_weight,
            "blastpdb_weight": blastpdb_weight,
            "ifrag_density": ifrag_density,
            "radi_top_pairs_consensus": args.radi_top_pairs_consensus,
            "radi_paired_rows_used": radi_paired_rows_used,
            "radi_low_confidence": low_radi_confidence,
            "radi_low_confidence_reasons": low_radi_confidence_reasons,
            "gated_radi_pairs_selected": gated_radi_pairs,
            "blastpdb_retained_templates": blastpdb_retained_templates,
            "blastpdb_low_confidence": low_blastpdb_confidence,
            "blastpdb_low_confidence_reasons": low_blastpdb_confidence_reasons,
            "gated_blastpdb_pairs_selected": gated_blastpdb_pairs,
            "anchor_pairs_inside_patch": anchor_pairs_in_patch,
            "fallback_to_patch_only": fallback_to_patch_only,
            "surface_rsa_threshold": args.surface_rsa_threshold,
            "surface_filter_applied": surface_prior_applied,
            "surface_prior_applied": surface_prior_applied,
            "q1_surface_residues_kept": q1_surface_residues_kept,
            "q2_surface_residues_kept": q2_surface_residues_kept,
            "residue_top_k": args.top_k,
            "strict_active_residues_per_chain": args.strict_active_residues_per_chain,
            "strict_passive_residues_per_chain": args.strict_passive_residues_per_chain,
            "active_residues_per_chain": args.active_residues_per_chain,
            "passive_residues_per_chain": args.passive_residues_per_chain,
            "patch_residues_per_chain": args.patch_residues_per_chain,
            "q1_patch_residues_selected": int(q1_patch_indices.size),
            "q2_patch_residues_selected": int(q2_patch_indices.size),
            "overlay_top_pairs": args.overlay_top_pairs or args.radi_top_pairs_consensus,
            "residue_score_rule": residue_score_rule,
            "patch_score_rule": "build local seed regions from top ifrag and conservation residues, combine them with an overlap bonus, and normalize only after reliability-scaled merging",
            "docking_selector_rule": "choose a compact surface-supported cluster per chain, require direct branch support for active residues, use passive residues only as the shell around those actives, and allow zero residues on weak chains",
            "ifrag_component_rule": "normalize_positive(ifrag_strength); ifrag_specificity is kept as a diagnostic only",
            "conservation_component_rule": "normalize_positive(conservation_profile_score)",
            "radi_component_rule": "normalize_positive(weighted_top_k_sum(gated_radi_pairs_by_residue))",
            "blastpdb_component_rule": "normalize_positive(weighted_top_k_sum(gated_blastpdb_pairs_by_residue))",
            "ifrag_guided_rule": "normalize_positive(local ifrag seed-region support); used diagnostically rather than added a second time to the final hotspot score",
            "radi_bonus_rule": "normalize_positive(radi_component * (0.25 + 0.75*patch_score))",
            "blastpdb_bonus_rule": "normalize_positive(blastpdb_component * (0.25 + 0.75*patch_score))",
            "residue_priority_matrix_rule": "outer(final query1 residue scores, final query2 residue scores)",
            "structaware_rule": "classical residue scores are built first; structure-aware reranking only reorders surface-eligible residues using local score mass, patch shape, optional hydrophobic patchiness, and optional pLDDT-style confidence while preserving classical branch-derived support gating",
        },
        "output_semantics": {
            "consensus_pair_matrix": "diagnostic outer-product projection of final residue scores; not direct pair evidence",
            "residue_priority_matrix": "same diagnostic outer-product projection as consensus_pair_matrix",
            "template_support_matrix": "diagnostic outer-product support view of the template-derived residue prior inside the selected patch",
            "anchor_pair_matrix": "sparse retained anchor pairs after gating from raDI and/or blastPDB",
            "blastpdb_matrix": "normalized structural-template support transferred from experimental PDB biological assemblies before anchor gating in the combined scorer",
            "docking_candidate_pairs": "diagnostic top cells from the residue-priority projection restricted to the recommended strict docking set",
            "docking_candidate_pairs_loose": "diagnostic top cells from the residue-priority projection restricted to the broader loose docking set",
            "query1_docking_residues.tsv": "primary recommended strict docking residue set for query1",
            "query2_docking_residues.tsv": "primary recommended strict docking residue set for query2",
            "query1_docking_residues.loose.tsv": "broader alternative docking residue set for query1",
            "query2_docking_residues.loose.tsv": "broader alternative docking residue set for query2",
            "query1_structure_features.tsv": "structure-aware reranking components for query1 when structure-aware scoring is applied",
            "query2_structure_features.tsv": "structure-aware reranking components for query2 when structure-aware scoring is applied",
            "lightdock_restraints.strict_active.list": "recommended first LightDock run: strict residues, active restraints only",
            "lightdock_restraints.strict.list": "recommended second LightDock run: strict residues with passive shell retained",
            "lightdock_restraints.loose.list": "broader third LightDock run: loose residues with passive shell retained",
            "lightdock_restraints.query1_only.strict.list": "one-sided LightDock restraints using only query1 strict residues",
            "lightdock_restraints.query2_only.strict.list": "one-sided LightDock restraints using only query2 strict residues",
            "primary_product": "per-chain residue scores, strict/loose docking residue tables, and colored PDBs",
        },
        "docking_selection": {
            "primary_recommended_set": "strict",
            "selection_strategy": "choose a compact surface-supported hotspot cluster from direct-support residues on each chain independently; write a stricter set and a broader loose companion set without forcing equal residue counts across chains",
            "strict_set_role": "first docking run / highest-confidence restraints from direct-support residues only",
            "loose_set_role": "second docking run / broader shell around the same scoring landscape without zero-support fillers",
            "lightdock_receptor_assignment": "query1 is written as receptor (R) and query2 as ligand (L) in the generated LightDock restraint files",
            "lightdock_primary_restraint_file": "lightdock_restraints.strict_active.list",
            "lightdock_secondary_restraint_file": "lightdock_restraints.strict.list",
            "lightdock_tertiary_restraint_file": "lightdock_restraints.loose.list",
            "lightdock_one_sided_files": [
                "lightdock_restraints.query1_only.strict.list",
                "lightdock_restraints.query2_only.strict.list"
            ],
            "lightdock_note": "Receptor-only restraints are the cleanest one-sided LightDock mode because receptor restraints affect swarm placement. Query2-only restraints are still written for benchmarking, but if query2 is the only trusted chain you may also want a swapped receptor/ligand docking run.",
            "structure_aware_selection": bool(query1.pdb_residue_coords is not None or query2.pdb_residue_coords is not None),
            "surface_filter_used": surface_prior_applied,
            "query1_strict_selected": int(q1_selected_strict.size),
            "query2_strict_selected": int(q2_selected_strict.size),
            "query1_loose_selected": int(q1_selected_loose.size),
            "query2_loose_selected": int(q2_selected_loose.size),
            "lightdock_strict_lines": lightdock_strict_written,
            "lightdock_strict_active_lines": lightdock_strict_active_written,
            "lightdock_loose_lines": lightdock_loose_written,
            "lightdock_query1_only_lines": lightdock_query1_only_written,
            "lightdock_query2_only_lines": lightdock_query2_only_written,
        },
        "stage_summaries": {
            "homolog_search": homolog_search_summary,
            "conservation": conservation_summary,
            "radi_prepare": radi_prepare_summary,
            "radi": radi_summary,
            "blastpdb": blastpdb_summary,
        },
        "outputs": {
            "consensus_pair_matrix.tsv": str(consensus_tsv),
            "consensus_pair_matrix.npy": str(consensus_npy),
            "template_support_matrix.tsv": str(template_support_tsv),
            "template_support_matrix.npy": str(template_support_npy),
            "anchor_pair_matrix.tsv": str(anchor_pair_tsv),
            "anchor_pair_matrix.npy": str(anchor_pair_npy),
            "residue_priority_matrix.tsv": str(residue_priority_tsv),
            "residue_priority_matrix.npy": str(residue_priority_npy),
            "consensus_top_pairs.tsv": str(consensus_top_pairs),
            "docking_candidate_pairs.tsv": str(docking_top_pairs),
            "docking_candidate_pairs.loose.tsv": str(docking_top_pairs_loose),
            "query1_residue_scores.tsv": str(q1_res_scores),
            "query2_residue_scores.tsv": str(q2_res_scores),
            "query1_branch_scores.tsv": str(q1_branch_scores),
            "query2_branch_scores.tsv": str(q2_branch_scores),
            "query1_structure_features.tsv": str(q1_struct_features) if structaware_applied else None,
            "query2_structure_features.tsv": str(q2_struct_features) if structaware_applied else None,
            "query1_docking_residues.tsv": str(q1_docking_residues),
            "query2_docking_residues.tsv": str(q2_docking_residues),
            "query1_docking_residues.strict.tsv": str(q1_docking_residues_strict),
            "query2_docking_residues.strict.tsv": str(q2_docking_residues_strict),
            "query1_docking_residues.loose.tsv": str(q1_docking_residues_loose),
            "query2_docking_residues.loose.tsv": str(q2_docking_residues_loose),
            "lightdock_restraints.strict_active.list": str(lightdock_strict_active),
            "lightdock_restraints.strict.list": str(lightdock_strict),
            "lightdock_restraints.loose.list": str(lightdock_loose),
            "lightdock_restraints.query1_only.strict.list": str(lightdock_query1_only),
            "lightdock_restraints.query2_only.strict.list": str(lightdock_query2_only),
            "query1_residue_scores.png": str(q1_residue_plot) if residue_plots_written else None,
            "query2_residue_scores.png": str(q2_residue_plot) if residue_plots_written else None,
            "query1_residue_scores_colored.pdb": str(q1_scored_pdb) if q1_scored_pdb_written else None,
            "query2_residue_scores_colored.pdb": str(q2_scored_pdb) if q2_scored_pdb_written else None,
            "homolog_search_summary.json": str(homolog_search_out / "homolog_search_summary.json") if needs_homolog_search else None,
            "blastpdb_matrix.tsv": str(blastpdb_out / "blastpdb_matrix.tsv") if needs_blastpdb else None,
            "blastpdb_matrix.npy": str(blastpdb_out / "blastpdb_matrix.npy") if needs_blastpdb else None,
            "blastpdb_top_pairs.tsv": str(blastpdb_out / "blastpdb_top_pairs.tsv") if needs_blastpdb else None,
            "blastpdb_template_matches.tsv": str(blastpdb_out / "blastpdb_template_matches.tsv") if needs_blastpdb else None,
            "blastpdb_summary.json": str(blastpdb_out / "blastpdb_summary.json") if needs_blastpdb else None,
            "template_support_heatmap.png": str(template_support_heatmap) if template_support_heatmap_written else None,
            "template_support_with_anchor_overlay.png": str(anchor_overlay_heatmap) if overlay_written else None,
            "ifrag_with_blastpdb_overlay.png": str(ifrag_blastpdb_overlay_heatmap) if ifrag_blastpdb_overlay_written else None,
            "final_score_heatmap.png": str(final_score_heatmap) if final_score_heatmap_written else None,
            "consensus_summary.json": str(summary_path),
            "run_warnings.txt": str(warnings_path),
        },
        "rows_written": {
            "consensus_top_pairs": top_pairs_written,
            "docking_candidate_pairs": docking_pairs_written,
            "docking_candidate_pairs_loose": docking_pairs_loose_written,
            "query1_residue_scores": int(len(query1.sequence)),
            "query2_residue_scores": int(len(query2.sequence)),
            "query1_branch_scores": int(len(query1.sequence)),
            "query2_branch_scores": int(len(query2.sequence)),
            "query1_structure_features": int(len(query1.sequence)) if structaware_applied else 0,
            "query2_structure_features": int(len(query2.sequence)) if structaware_applied else 0,
            "query1_docking_residues": q1_docking_written,
            "query2_docking_residues": q2_docking_written,
            "query1_docking_residues_loose": q1_docking_loose_written,
            "query2_docking_residues_loose": q2_docking_loose_written,
            "lightdock_restraints_strict_active": lightdock_strict_active_written,
            "lightdock_restraints_strict": lightdock_strict_written,
            "lightdock_restraints_loose": lightdock_loose_written,
            "lightdock_restraints_query1_only": lightdock_query1_only_written,
            "lightdock_restraints_query2_only": lightdock_query2_only_written,
            "anchor_overlay_pairs": overlay_pairs_written,
            "ifrag_blastpdb_overlay_pairs": ifrag_blastpdb_overlay_pairs_written,
        },
        "warnings": warnings,
    }
    summary_path.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
