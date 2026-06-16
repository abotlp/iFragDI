#!/usr/bin/env python3
"""
Hybrid cached blastPDB branch for structural contact transfer.

Version 1 scope:
- experimental PDB biological assemblies only
- remote RCSB sequence search for candidate assembly discovery
- local cached assembly download and contact extraction
- local BLAST mapping against chains parsed from the assembly
- transfer template contacts onto query residue pairs

Biologically, this is a modern hybrid adaptation of thesis blastPDB:
- it is not a prebuilt local PDBContact/SwissprotContacts replica
- but it preserves the runtime idea of sequence alignment + contact transfer
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import shlex
import subprocess
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


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
    "MSE": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}

RCSB_SEQUENCE_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_ENTRY_METADATA_URL = "https://data.rcsb.org/rest/v1/core/entry/{entry_id}"
RCSB_ASSEMBLY_MMCIF_URL = "https://files.rcsb.org/download/{entry_id_lower}-assembly{assembly_id}.cif.gz"
BLAST_OUTFMT = "6 qseqid sseqid pident evalue bitscore length qstart qend sstart send qseq sseq"


@dataclass(frozen=True)
class AssemblySearchHit:
    identifier: str
    entry_id: str
    assembly_id: str
    score: float
    rank: int


@dataclass(frozen=True)
class EntryMetadata:
    entry_id: str
    title: str
    methodology: str | None
    experimental_method: str | None
    resolution: float | None


@dataclass(frozen=True)
class ChainTemplate:
    chain_id: str
    sequence: str
    residue_labels: Tuple[str, ...]
    contact_positions: Tuple[int, ...]
    contact_coords: np.ndarray


@dataclass(frozen=True)
class ContactPair:
    left_chain: str
    right_chain: str
    contacts: Tuple[Tuple[int, int, float], ...]


@dataclass(frozen=True)
class AssemblyContactData:
    entry_id: str
    assembly_id: str
    chains: Dict[str, ChainTemplate]
    contact_pairs: Tuple[ContactPair, ...]


@dataclass(frozen=True)
class BlastHit:
    subject_id: str
    pident: float
    evalue: float
    bitscore: float
    alnlen: int
    qstart: int
    qend: int
    sstart: int
    send: int
    qseq: str
    sseq: str
    query_coverage: float


@dataclass(frozen=True)
class RetainedAssemblyCandidate:
    identifier: str
    entry_id: str
    assembly_id: str
    title: str
    experimental_method: str | None
    resolution: float | None
    contacting_chain_pairs_used: Tuple[str, ...]
    supported_cells: frozenset[Tuple[int, int]]
    transferred_contacts_total: int
    combined_bitscore_best: float
    combined_evalue_best: float
    q1_best_chain: str | None
    q2_best_chain: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hybrid cached blastPDB branch using experimental biological assemblies from the PDB."
    )
    parser.add_argument("--query1", required=True, type=Path, help="Single-sequence FASTA for protein 1.")
    parser.add_argument("--query2", required=True, type=Path, help="Single-sequence FASTA for protein 2.")
    parser.add_argument(
        "--interaction-mode",
        choices=("heteromer", "homomer", "auto"),
        default="heteromer",
        help="Interpret the query pair as heteromer, homomer, or auto-resolve from identical sequences.",
    )
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory.")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/cache/blastpdb"),
        help="Shared cache for search results, entry metadata, assemblies, and extracted contacts.",
    )
    parser.add_argument("--blast-bin", default="blastp", help="blastp executable used for local chain mapping.")
    parser.add_argument(
        "--makeblastdb-bin",
        default="makeblastdb",
        help="makeblastdb executable used to build temporary local chain databases.",
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--top-assemblies",
        type=int,
        default=25,
        help="Maximum number of shared candidate assemblies kept after remote discovery.",
    )
    parser.add_argument(
        "--sequence-search-identity-cutoff",
        type=float,
        default=0.30,
        help="RCSB sequence-search identity cutoff for candidate discovery.",
    )
    parser.add_argument(
        "--sequence-search-evalue-cutoff",
        type=float,
        default=1.0,
        help="RCSB sequence-search E-value cutoff for candidate discovery.",
    )
    parser.add_argument(
        "--local-blast-evalue",
        type=float,
        default=0.01,
        help="Local BLAST E-value used to map the query onto assembly chains.",
    )
    parser.add_argument(
        "--local-blast-max-target-seqs",
        type=int,
        default=1000,
        help="Maximum local BLAST chain hits retained per query/assembly.",
    )
    parser.add_argument(
        "--cbeta-threshold",
        type=float,
        default=12.0,
        help="C-beta contact threshold in angstroms. Gly uses CA as a fallback.",
    )
    parser.add_argument(
        "--min-template-contacts",
        type=int,
        default=5,
        help="Minimum structural contacts required to keep a chain pair as a template.",
    )
    parser.add_argument("--no-heatmap", action="store_true", help="Skip heatmap PNG output.")
    parser.add_argument("--verbose", action="store_true", help="Print progress information.")
    args = parser.parse_args()

    if args.threads <= 0:
        raise SystemExit("--threads must be > 0")
    if args.top_assemblies <= 0:
        raise SystemExit("--top-assemblies must be > 0")
    if not (0.0 < args.sequence_search_identity_cutoff <= 1.0):
        raise SystemExit("--sequence-search-identity-cutoff must be in (0, 1].")
    if args.sequence_search_evalue_cutoff <= 0.0:
        raise SystemExit("--sequence-search-evalue-cutoff must be > 0.")
    if args.local_blast_evalue <= 0.0:
        raise SystemExit("--local-blast-evalue must be > 0.")
    if args.local_blast_max_target_seqs <= 0:
        raise SystemExit("--local-blast-max-target-seqs must be > 0")
    if args.cbeta_threshold <= 0.0:
        raise SystemExit("--cbeta-threshold must be > 0.")
    if args.min_template_contacts <= 0:
        raise SystemExit("--min-template-contacts must be > 0")
    return args


def read_single_fasta(path: Path) -> tuple[str, str]:
    header = None
    seq_parts: List[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    raise SystemExit(f"{path}: expected a single FASTA record")
                header = line[1:].strip() or path.stem
            else:
                seq_parts.append(line)
    if header is None:
        raise SystemExit(f"{path}: FASTA header not found")
    sequence = "".join(seq_parts).replace(" ", "").replace("\t", "").upper()
    if not sequence:
        raise SystemExit(f"{path}: FASTA sequence is empty")
    return header, sequence


def resolve_interaction_mode(requested_mode: str, q1_seq: str, q2_seq: str) -> str:
    if requested_mode == "auto":
        return "homomer" if q1_seq == q2_seq else "heteromer"
    if requested_mode == "homomer" and q1_seq != q2_seq:
        raise ValueError("homomer mode currently requires identical query sequences")
    return requested_mode


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def load_json(path: Path) -> object:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: Path, payload: object) -> None:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def request_json(url: str, payload: object | None = None, timeout: int = 60) -> object:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method="POST" if data is not None else "GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def download_file(url: str, destination: Path, timeout: int = 120) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".tmp")
    request = urllib.request.Request(url, headers={"Accept": "*/*"})
    with urllib.request.urlopen(request, timeout=timeout) as response, tmp_path.open("wb") as handle:
        handle.write(response.read())
    tmp_path.replace(destination)


def search_assemblies_rcsb(
    sequence: str,
    top_hits: int,
    identity_cutoff: float,
    evalue_cutoff: float,
    cache_dir: Path,
) -> List[AssemblySearchHit]:
    cache_key = sha1_text(f"{sequence}|{top_hits}|{identity_cutoff}|{evalue_cutoff}")
    cache_path = ensure_dir(cache_dir / "search") / f"{cache_key}.json"
    if cache_path.exists():
        raw = load_json(cache_path)
    else:
        payload = {
            "query": {
                "type": "terminal",
                "service": "sequence",
                "parameters": {
                    "value": sequence,
                    "sequence_type": "protein",
                    "identity_cutoff": identity_cutoff,
                    "evalue_cutoff": evalue_cutoff,
                },
            },
            "return_type": "assembly",
            "request_options": {
                "results_verbosity": "minimal",
                "paginate": {"start": 0, "rows": top_hits},
            },
        }
        try:
            raw = request_json(RCSB_SEQUENCE_SEARCH_URL, payload=payload)
        except urllib.error.HTTPError:
            query_json = urllib.parse.quote(json.dumps(payload, separators=(",", ":")))
            raw = request_json(f"{RCSB_SEQUENCE_SEARCH_URL}?json={query_json}")
        dump_json(cache_path, raw)

    result_set = raw.get("result_set") if isinstance(raw, dict) else None
    if not result_set:
        return []
    hits: List[AssemblySearchHit] = []
    for rank, row in enumerate(result_set, start=1):
        identifier = str(row.get("identifier") or "").strip()
        if "-" not in identifier:
            continue
        entry_id, assembly_id = identifier.split("-", 1)
        try:
            score = float(row.get("score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        hits.append(
            AssemblySearchHit(
                identifier=identifier,
                entry_id=entry_id.upper(),
                assembly_id=assembly_id,
                score=score,
                rank=rank,
            )
        )
    return hits


def fetch_entry_metadata(entry_id: str, cache_dir: Path) -> EntryMetadata:
    cache_path = ensure_dir(cache_dir / "entries") / f"{entry_id.upper()}.json"
    if cache_path.exists():
        raw = load_json(cache_path)
    else:
        raw = request_json(RCSB_ENTRY_METADATA_URL.format(entry_id=entry_id.upper()))
        dump_json(cache_path, raw)
    info = raw.get("rcsb_entry_info", {}) if isinstance(raw, dict) else {}
    accession = raw.get("rcsb_accession_info", {}) if isinstance(raw, dict) else {}
    methodology = info.get("structure_determination_methodology")
    experimental_method = info.get("experimental_method")
    resolution_values = info.get("resolution_combined") or []
    resolution = None
    if isinstance(resolution_values, list) and resolution_values:
        try:
            resolution = float(resolution_values[0])
        except (TypeError, ValueError):
            resolution = None
    title = ""
    if isinstance(raw, dict):
        title = str(raw.get("struct", {}).get("title") or accession.get("initial_release_date") or "")
    return EntryMetadata(
        entry_id=entry_id.upper(),
        title=title,
        methodology=str(methodology).strip().lower() if methodology is not None else None,
        experimental_method=str(experimental_method).strip() if experimental_method is not None else None,
        resolution=resolution,
    )


def is_experimental_entry(metadata: EntryMetadata) -> bool:
    return metadata.methodology == "experimental"


def download_assembly_mmcif(entry_id: str, assembly_id: str, cache_dir: Path) -> Path:
    destination = ensure_dir(cache_dir / "assemblies") / f"{entry_id.upper()}-assembly{assembly_id}.cif.gz"
    if destination.exists():
        return destination
    url = RCSB_ASSEMBLY_MMCIF_URL.format(entry_id_lower=entry_id.lower(), assembly_id=assembly_id)
    download_file(url, destination)
    return destination


def tokenize_mmcif_line(line: str) -> List[str]:
    stripped = line.strip()
    if not stripped or stripped == "#":
        return []
    try:
        return shlex.split(stripped, posix=True)
    except ValueError:
        return stripped.split()


def parse_assembly_chains_from_mmcif(path: Path) -> Dict[str, ChainTemplate]:
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        lines = handle.read().splitlines()

    atom_columns: List[str] = []
    atom_rows: List[List[str]] = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped != "loop_":
            i += 1
            continue
        i += 1
        columns: List[str] = []
        while i < len(lines) and lines[i].lstrip().startswith("_"):
            columns.append(lines[i].strip().split()[0])
            i += 1
        if not columns:
            continue
        is_atom_site = columns[0].startswith("_atom_site.")
        buffer: List[str] = []
        while i < len(lines):
            stripped = lines[i].strip()
            if not stripped:
                i += 1
                continue
            if stripped == "#" or stripped == "loop_" or stripped.startswith("data_") or stripped.startswith("_"):
                break
            buffer.extend(tokenize_mmcif_line(lines[i]))
            while len(buffer) >= len(columns):
                row = buffer[: len(columns)]
                buffer = buffer[len(columns) :]
                if is_atom_site:
                    atom_columns = columns
                    atom_rows.append(row)
            i += 1

    if not atom_columns or not atom_rows:
        raise RuntimeError(f"{path}: atom_site loop not found in assembly mmCIF")

    index = {name: idx for idx, name in enumerate(atom_columns)}
    required = [
        "_atom_site.group_PDB",
        "_atom_site.label_asym_id",
        "_atom_site.label_comp_id",
        "_atom_site.label_atom_id",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
    ]
    for column in required:
        if column not in index:
            raise RuntimeError(f"{path}: required mmCIF atom_site column missing: {column}")

    model_idx = index.get("_atom_site.pdbx_PDB_model_num")
    auth_seq_idx = index.get("_atom_site.auth_seq_id")
    label_seq_idx = index.get("_atom_site.label_seq_id")
    ins_idx = index.get("_atom_site.pdbx_PDB_ins_code")

    residue_order: Dict[str, List[Tuple[str, str]]] = {}
    residue_data: Dict[Tuple[str, str, str], Dict[str, object]] = {}

    for row in atom_rows:
        if row[index["_atom_site.group_PDB"]] != "ATOM":
            continue
        if model_idx is not None and row[model_idx] not in {"1", ".", "?"}:
            continue
        chain_id = row[index["_atom_site.label_asym_id"]].strip()
        comp_id = row[index["_atom_site.label_comp_id"]].strip().upper()
        atom_name = row[index["_atom_site.label_atom_id"]].strip().upper()
        if comp_id not in AA3_TO_1:
            continue
        seq_id = ""
        if label_seq_idx is not None:
            seq_id = row[label_seq_idx].strip()
        if not seq_id or seq_id in {".", "?"}:
            if auth_seq_idx is None:
                continue
            seq_id = row[auth_seq_idx].strip()
        if not seq_id or seq_id in {".", "?"}:
            continue
        ins_code = ""
        if ins_idx is not None:
            ins_code = row[ins_idx].strip()
            if ins_code in {".", "?"}:
                ins_code = ""
        residue_key = (chain_id, seq_id, ins_code)
        if residue_key not in residue_data:
            residue_data[residue_key] = {
                "aa": AA3_TO_1[comp_id],
                "residue_label": f"{seq_id}{ins_code}",
                "ca": None,
                "cb": None,
            }
            residue_order.setdefault(chain_id, []).append((seq_id, ins_code))
        try:
            coord = np.array(
                [
                    float(row[index["_atom_site.Cartn_x"]]),
                    float(row[index["_atom_site.Cartn_y"]]),
                    float(row[index["_atom_site.Cartn_z"]]),
                ],
                dtype=float,
            )
        except (TypeError, ValueError):
            continue
        if atom_name == "CA":
            residue_data[residue_key]["ca"] = coord
        elif atom_name == "CB":
            residue_data[residue_key]["cb"] = coord

    chains: Dict[str, ChainTemplate] = {}
    for chain_id, ordered_residues in residue_order.items():
        sequence_chars: List[str] = []
        residue_labels: List[str] = []
        contact_positions: List[int] = []
        contact_coords: List[np.ndarray] = []
        for seq_position, (seq_id, ins_code) in enumerate(ordered_residues, start=1):
            residue_key = (chain_id, seq_id, ins_code)
            info = residue_data[residue_key]
            aa = str(info["aa"])
            sequence_chars.append(aa)
            residue_labels.append(str(info["residue_label"]))
            ca_coord = info["ca"]
            cb_coord = info["cb"]
            contact_coord = cb_coord if cb_coord is not None else ca_coord
            if contact_coord is not None:
                contact_positions.append(seq_position)
                contact_coords.append(contact_coord)
        if not sequence_chars:
            continue
        coord_array = np.vstack(contact_coords) if contact_coords else np.zeros((0, 3), dtype=float)
        chains[chain_id] = ChainTemplate(
            chain_id=chain_id,
            sequence="".join(sequence_chars),
            residue_labels=tuple(residue_labels),
            contact_positions=tuple(contact_positions),
            contact_coords=coord_array,
        )
    return chains


def compute_contact_pairs(
    chains: Dict[str, ChainTemplate],
    cbeta_threshold: float,
    min_template_contacts: int,
) -> Tuple[ContactPair, ...]:
    threshold_sq = cbeta_threshold * cbeta_threshold
    contact_pairs: List[ContactPair] = []
    chain_ids = sorted(chains)
    for left_index, left_chain in enumerate(chain_ids):
        left = chains[left_chain]
        if left.contact_coords.size == 0:
            continue
        left_positions = np.array(left.contact_positions, dtype=int)
        for right_chain in chain_ids[left_index + 1 :]:
            right = chains[right_chain]
            if right.contact_coords.size == 0:
                continue
            right_positions = np.array(right.contact_positions, dtype=int)
            deltas = left.contact_coords[:, None, :] - right.contact_coords[None, :, :]
            distances_sq = np.sum(deltas * deltas, axis=2)
            match_indices = np.argwhere(distances_sq <= threshold_sq)
            if match_indices.shape[0] < min_template_contacts:
                continue
            contacts: List[Tuple[int, int, float]] = []
            for left_pos_idx, right_pos_idx in match_indices:
                distance = math.sqrt(float(distances_sq[left_pos_idx, right_pos_idx]))
                contacts.append(
                    (
                        int(left_positions[left_pos_idx]),
                        int(right_positions[right_pos_idx]),
                        distance,
                    )
                )
            contact_pairs.append(
                ContactPair(
                    left_chain=left_chain,
                    right_chain=right_chain,
                    contacts=tuple(contacts),
                )
            )
    return tuple(contact_pairs)


def load_or_build_contact_cache(
    entry_id: str,
    assembly_id: str,
    assembly_path: Path,
    cache_dir: Path,
    cbeta_threshold: float,
    min_template_contacts: int,
) -> AssemblyContactData:
    cache_path = ensure_dir(cache_dir / "contacts") / f"{entry_id.upper()}-assembly{assembly_id}.json.gz"
    if cache_path.exists():
        raw = load_json(cache_path)
        chains = {
            chain_id: ChainTemplate(
                chain_id=chain_id,
                sequence=str(chain_info["sequence"]),
                residue_labels=tuple(str(token) for token in chain_info["residue_labels"]),
                contact_positions=tuple(int(value) for value in chain_info["contact_positions"]),
                contact_coords=np.array(chain_info["contact_coords"], dtype=float),
            )
            for chain_id, chain_info in raw["chains"].items()
        }
        contact_pairs = tuple(
            ContactPair(
                left_chain=str(row["left_chain"]),
                right_chain=str(row["right_chain"]),
                contacts=tuple((int(a), int(b), float(c)) for a, b, c in row["contacts"]),
            )
            for row in raw["contact_pairs"]
        )
        return AssemblyContactData(
            entry_id=str(raw["entry_id"]),
            assembly_id=str(raw["assembly_id"]),
            chains=chains,
            contact_pairs=contact_pairs,
        )

    chains = parse_assembly_chains_from_mmcif(assembly_path)
    contact_pairs = compute_contact_pairs(chains, cbeta_threshold=cbeta_threshold, min_template_contacts=min_template_contacts)
    raw = {
        "entry_id": entry_id.upper(),
        "assembly_id": str(assembly_id),
        "chains": {
            chain_id: {
                "sequence": chain.sequence,
                "residue_labels": list(chain.residue_labels),
                "contact_positions": list(chain.contact_positions),
                "contact_coords": chain.contact_coords.tolist(),
            }
            for chain_id, chain in chains.items()
        },
        "contact_pairs": [
            {
                "left_chain": pair.left_chain,
                "right_chain": pair.right_chain,
                "contacts": [[a, b, dist] for a, b, dist in pair.contacts],
            }
            for pair in contact_pairs
        ],
    }
    dump_json(cache_path, raw)
    return AssemblyContactData(
        entry_id=entry_id.upper(),
        assembly_id=str(assembly_id),
        chains=chains,
        contact_pairs=contact_pairs,
    )


def write_chain_fasta(path: Path, chains: Dict[str, ChainTemplate]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for chain_id in sorted(chains):
            sequence = chains[chain_id].sequence
            handle.write(f">{chain_id}\n")
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start : start + 80] + "\n")


def run_makeblastdb(makeblastdb_bin: str, fasta_path: Path, db_prefix: Path) -> None:
    result = subprocess.run(
        [
            makeblastdb_bin,
            "-in",
            str(fasta_path),
            "-dbtype",
            "prot",
            "-out",
            str(db_prefix),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "makeblastdb failed.\n"
            f"Command: {makeblastdb_bin} -in {fasta_path} -dbtype prot -out {db_prefix}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )


def run_blastp(
    blast_bin: str,
    query_fasta: Path,
    db_prefix: Path,
    out_tsv: Path,
    threads: int,
    evalue: float,
    max_target_seqs: int,
) -> None:
    cmd = [
        blast_bin,
        "-query",
        str(query_fasta),
        "-db",
        str(db_prefix),
        "-outfmt",
        BLAST_OUTFMT,
        "-evalue",
        str(evalue),
        "-max_target_seqs",
        str(max_target_seqs),
        "-num_threads",
        str(threads),
        "-out",
        str(out_tsv),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            "BLAST search failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )


def parse_chain_blast_hits(tsv_path: Path, query_len: int) -> Dict[str, BlastHit]:
    best_by_subject: Dict[str, BlastHit] = {}
    with tsv_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n\r")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 12:
                continue
            try:
                subject_id = parts[1].strip()
                pident = float(parts[2])
                evalue = float(parts[3])
                bitscore = float(parts[4])
                alnlen = int(parts[5])
                qstart = int(parts[6])
                qend = int(parts[7])
                sstart = int(parts[8])
                send = int(parts[9])
            except ValueError:
                continue
            qseq = parts[10]
            sseq = parts[11]
            mapping = subject_to_query_position_map(qseq, sseq, qstart, qend, sstart, send)
            if not mapping:
                continue
            query_coverage = len({qpos for qpos in mapping.values()}) / float(query_len)
            hit = BlastHit(
                subject_id=subject_id,
                pident=pident,
                evalue=evalue,
                bitscore=bitscore,
                alnlen=alnlen,
                qstart=qstart,
                qend=qend,
                sstart=sstart,
                send=send,
                qseq=qseq,
                sseq=sseq,
                query_coverage=query_coverage,
            )
            current = best_by_subject.get(subject_id)
            if current is None or (hit.evalue, -hit.bitscore, -hit.query_coverage, subject_id) < (
                current.evalue,
                -current.bitscore,
                -current.query_coverage,
                current.subject_id,
            ):
                best_by_subject[subject_id] = hit
    return best_by_subject


def subject_to_query_position_map(
    qseq: str,
    sseq: str,
    qstart: int,
    qend: int,
    sstart: int,
    send: int,
) -> Dict[int, int]:
    if len(qseq) != len(sseq):
        return {}
    qpos = qstart
    spos = sstart
    qstep = 1 if qend >= qstart else -1
    sstep = 1 if send >= sstart else -1
    mapping: Dict[int, int] = {}
    for qchar, schar in zip(qseq, sseq):
        current_q = qpos if qchar != "-" else None
        current_s = spos if schar != "-" else None
        if qchar != "-":
            qpos += qstep
        if schar != "-":
            spos += sstep
        if current_q is None or current_s is None:
            continue
        mapping[current_s] = current_q
    return mapping


def run_local_chain_blast(
    blast_bin: str,
    makeblastdb_bin: str,
    query_fasta: Path,
    query_len: int,
    chains: Dict[str, ChainTemplate],
    threads: int,
    evalue: float,
    max_target_seqs: int,
) -> Dict[str, BlastHit]:
    if not chains:
        return {}
    with tempfile.TemporaryDirectory(prefix="blastpdb_chain_db_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        fasta_path = tmp_dir / "assembly_chains.fa"
        db_prefix = tmp_dir / "assembly_chains_db"
        out_tsv = tmp_dir / "blast.tsv"
        write_chain_fasta(fasta_path, chains)
        run_makeblastdb(makeblastdb_bin, fasta_path, db_prefix)
        run_blastp(
            blast_bin=blast_bin,
            query_fasta=query_fasta,
            db_prefix=db_prefix,
            out_tsv=out_tsv,
            threads=threads,
            evalue=evalue,
            max_target_seqs=max_target_seqs,
        )
        return parse_chain_blast_hits(out_tsv, query_len=query_len)


def transfer_contacts(
    contacts: Iterable[Tuple[int, int, float]],
    left_map: Dict[int, int],
    right_map: Dict[int, int],
    supported_cells: set[Tuple[int, int]],
) -> int:
    transferred = 0
    for left_pos, right_pos, _distance in contacts:
        q1_pos = left_map.get(left_pos)
        q2_pos = right_map.get(right_pos)
        if q1_pos is None or q2_pos is None:
            continue
        supported_cells.add((q1_pos, q2_pos))
        transferred += 1
    return transferred


def build_assembly_candidate(
    assembly_hit: AssemblySearchHit,
    entry_metadata: EntryMetadata,
    contact_data: AssemblyContactData,
    q1_hits: Dict[str, BlastHit],
    q2_hits: Dict[str, BlastHit],
    interaction_mode: str,
) -> RetainedAssemblyCandidate | None:
    if not contact_data.contact_pairs:
        return None
    supported_cells: set[Tuple[int, int]] = set()
    transferred_contacts_total = 0
    used_chain_pairs: List[str] = []
    best_combined_bitscore = float("-inf")
    best_combined_evalue = float("inf")
    best_q1_chain: str | None = None
    best_q2_chain: str | None = None

    for pair in contact_data.contact_pairs:
        left_chain = pair.left_chain
        right_chain = pair.right_chain
        left_hit_q1 = q1_hits.get(left_chain)
        right_hit_q2 = q2_hits.get(right_chain)
        if left_hit_q1 is not None and right_hit_q2 is not None:
            left_map = subject_to_query_position_map(
                left_hit_q1.qseq,
                left_hit_q1.sseq,
                left_hit_q1.qstart,
                left_hit_q1.qend,
                left_hit_q1.sstart,
                left_hit_q1.send,
            )
            right_map = subject_to_query_position_map(
                right_hit_q2.qseq,
                right_hit_q2.sseq,
                right_hit_q2.qstart,
                right_hit_q2.qend,
                right_hit_q2.sstart,
                right_hit_q2.send,
            )
            transferred = transfer_contacts(pair.contacts, left_map, right_map, supported_cells)
            if transferred > 0:
                used_chain_pairs.append(f"{left_chain}:{right_chain}")
                combined_bitscore = left_hit_q1.bitscore + right_hit_q2.bitscore
                combined_evalue = left_hit_q1.evalue * right_hit_q2.evalue
                if (combined_evalue, -combined_bitscore) < (best_combined_evalue, -best_combined_bitscore):
                    best_combined_bitscore = combined_bitscore
                    best_combined_evalue = combined_evalue
                    best_q1_chain = left_chain
                    best_q2_chain = right_chain
                transferred_contacts_total += transferred

        swapped_left_hit_q1 = q1_hits.get(right_chain)
        swapped_right_hit_q2 = q2_hits.get(left_chain)
        if swapped_left_hit_q1 is not None and swapped_right_hit_q2 is not None:
            if interaction_mode == "heteromer" and left_chain == right_chain:
                continue
            left_map = subject_to_query_position_map(
                swapped_left_hit_q1.qseq,
                swapped_left_hit_q1.sseq,
                swapped_left_hit_q1.qstart,
                swapped_left_hit_q1.qend,
                swapped_left_hit_q1.sstart,
                swapped_left_hit_q1.send,
            )
            right_map = subject_to_query_position_map(
                swapped_right_hit_q2.qseq,
                swapped_right_hit_q2.sseq,
                swapped_right_hit_q2.qstart,
                swapped_right_hit_q2.qend,
                swapped_right_hit_q2.sstart,
                swapped_right_hit_q2.send,
            )
            transferred = transfer_contacts(pair.contacts, left_map, right_map, supported_cells)
            if transferred > 0:
                used_chain_pairs.append(f"{right_chain}:{left_chain}")
                combined_bitscore = swapped_left_hit_q1.bitscore + swapped_right_hit_q2.bitscore
                combined_evalue = swapped_left_hit_q1.evalue * swapped_right_hit_q2.evalue
                if (combined_evalue, -combined_bitscore) < (best_combined_evalue, -best_combined_bitscore):
                    best_combined_bitscore = combined_bitscore
                    best_combined_evalue = combined_evalue
                    best_q1_chain = right_chain
                    best_q2_chain = left_chain
                transferred_contacts_total += transferred

    if not supported_cells:
        return None

    return RetainedAssemblyCandidate(
        identifier=assembly_hit.identifier,
        entry_id=assembly_hit.entry_id,
        assembly_id=assembly_hit.assembly_id,
        title=entry_metadata.title,
        experimental_method=entry_metadata.experimental_method,
        resolution=entry_metadata.resolution,
        contacting_chain_pairs_used=tuple(sorted(set(used_chain_pairs))),
        supported_cells=frozenset(supported_cells),
        transferred_contacts_total=transferred_contacts_total,
        combined_bitscore_best=best_combined_bitscore if best_combined_bitscore != float("-inf") else 0.0,
        combined_evalue_best=best_combined_evalue if best_combined_evalue != float("inf") else 1.0,
        q1_best_chain=best_q1_chain,
        q2_best_chain=best_q2_chain,
    )


def score_candidates(
    candidates: List[RetainedAssemblyCandidate],
    q1_len: int,
    q2_len: int,
) -> np.ndarray:
    matrix = np.zeros((q1_len, q2_len), dtype=float)
    if not candidates:
        return matrix
    for candidate in candidates:
        for q1_pos, q2_pos in candidate.supported_cells:
            if 1 <= q1_pos <= q1_len and 1 <= q2_pos <= q2_len:
                matrix[q1_pos - 1, q2_pos - 1] += 1.0
    matrix /= float(len(candidates))
    return matrix


def write_matrix_tsv(path: Path, matrix: np.ndarray) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        for row in matrix:
            writer.writerow(f"{float(value):.10g}" for value in row)


def write_template_matches(path: Path, candidates: List[RetainedAssemblyCandidate]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "template_id",
                "entry_id",
                "assembly_id",
                "q1_best_chain",
                "q2_best_chain",
                "combined_bitscore_best",
                "combined_evalue_best",
                "transferred_contacts_total",
                "supported_cell_count",
                "contacting_chain_pairs_used",
                "experimental_method",
                "resolution",
                "title",
            ]
        )
        for candidate in candidates:
            writer.writerow(
                [
                    candidate.identifier,
                    candidate.entry_id,
                    candidate.assembly_id,
                    candidate.q1_best_chain or "",
                    candidate.q2_best_chain or "",
                    f"{candidate.combined_bitscore_best:.10g}",
                    f"{candidate.combined_evalue_best:.10g}",
                    candidate.transferred_contacts_total,
                    len(candidate.supported_cells),
                    ";".join(candidate.contacting_chain_pairs_used),
                    candidate.experimental_method or "",
                    "" if candidate.resolution is None else f"{candidate.resolution:.10g}",
                    candidate.title,
                ]
            )


def write_top_pairs(path: Path, matrix: np.ndarray, top_n: int = 200) -> None:
    nonzero = np.argwhere(matrix > 0.0)
    rows: List[Tuple[int, int, float]] = []
    for i, j in nonzero:
        rows.append((int(i) + 1, int(j) + 1, float(matrix[i, j])))
    rows.sort(key=lambda item: (-item[2], item[0], item[1]))
    rows = rows[:top_n]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["query1_residue_index", "query2_residue_index", "score"])
        for i, j, score in rows:
            writer.writerow([i, j, f"{score:.10g}"])


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


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = ensure_dir(args.cache_dir)

    q1_header, q1_seq = read_single_fasta(args.query1)
    q2_header, q2_seq = read_single_fasta(args.query2)
    interaction_mode = resolve_interaction_mode(args.interaction_mode, q1_seq, q2_seq)

    if args.verbose:
        print(f"[INFO] Query1: {q1_header} (length {len(q1_seq)})")
        print(f"[INFO] Query2: {q2_header} (length {len(q2_seq)})")
        print(f"[INFO] Interaction mode: {interaction_mode}")

    q1_search_hits = search_assemblies_rcsb(
        q1_seq,
        top_hits=args.top_assemblies,
        identity_cutoff=args.sequence_search_identity_cutoff,
        evalue_cutoff=args.sequence_search_evalue_cutoff,
        cache_dir=cache_dir,
    )
    q2_search_hits = search_assemblies_rcsb(
        q2_seq,
        top_hits=args.top_assemblies,
        identity_cutoff=args.sequence_search_identity_cutoff,
        evalue_cutoff=args.sequence_search_evalue_cutoff,
        cache_dir=cache_dir,
    )

    q1_by_id = {hit.identifier: hit for hit in q1_search_hits}
    q2_by_id = {hit.identifier: hit for hit in q2_search_hits}
    common_ids = sorted(
        set(q1_by_id) & set(q2_by_id),
        key=lambda identifier: (
            q1_by_id[identifier].rank + q2_by_id[identifier].rank,
            -(q1_by_id[identifier].score + q2_by_id[identifier].score),
            identifier,
        ),
    )[: args.top_assemblies]

    warnings: List[str] = []
    retained: List[RetainedAssemblyCandidate] = []
    considered_identifiers: List[str] = []
    skipped_nonexperimental = 0
    skipped_empty_contacts = 0
    skipped_no_transfer = 0

    for identifier in common_ids:
        q1_hit = q1_by_id[identifier]
        considered_identifiers.append(identifier)
        entry_metadata = fetch_entry_metadata(q1_hit.entry_id, cache_dir)
        if not is_experimental_entry(entry_metadata):
            skipped_nonexperimental += 1
            continue
        assembly_path = download_assembly_mmcif(q1_hit.entry_id, q1_hit.assembly_id, cache_dir)
        contact_data = load_or_build_contact_cache(
            q1_hit.entry_id,
            q1_hit.assembly_id,
            assembly_path,
            cache_dir,
            cbeta_threshold=args.cbeta_threshold,
            min_template_contacts=args.min_template_contacts,
        )
        if not contact_data.contact_pairs:
            skipped_empty_contacts += 1
            continue
        q1_chain_hits = run_local_chain_blast(
            blast_bin=args.blast_bin,
            makeblastdb_bin=args.makeblastdb_bin,
            query_fasta=args.query1,
            query_len=len(q1_seq),
            chains=contact_data.chains,
            threads=args.threads,
            evalue=args.local_blast_evalue,
            max_target_seqs=args.local_blast_max_target_seqs,
        )
        q2_chain_hits = run_local_chain_blast(
            blast_bin=args.blast_bin,
            makeblastdb_bin=args.makeblastdb_bin,
            query_fasta=args.query2,
            query_len=len(q2_seq),
            chains=contact_data.chains,
            threads=args.threads,
            evalue=args.local_blast_evalue,
            max_target_seqs=args.local_blast_max_target_seqs,
        )
        candidate = build_assembly_candidate(
            assembly_hit=q1_hit,
            entry_metadata=entry_metadata,
            contact_data=contact_data,
            q1_hits=q1_chain_hits,
            q2_hits=q2_chain_hits,
            interaction_mode=interaction_mode,
        )
        if candidate is None:
            skipped_no_transfer += 1
            continue
        retained.append(candidate)
        if args.verbose:
            print(
                "[INFO] Retained blastPDB template "
                f"{candidate.identifier} chains={','.join(candidate.contacting_chain_pairs_used)} "
                f"cells={len(candidate.supported_cells)} contacts={candidate.transferred_contacts_total}"
            )

    retained.sort(
        key=lambda candidate: (
            candidate.combined_evalue_best,
            -candidate.combined_bitscore_best,
            -len(candidate.supported_cells),
            candidate.identifier,
        )
    )

    matrix = score_candidates(retained, q1_len=len(q1_seq), q2_len=len(q2_seq))
    write_matrix_tsv(out_dir / "blastpdb_matrix.tsv", matrix)
    np.save(out_dir / "blastpdb_matrix.npy", matrix)
    write_top_pairs(out_dir / "blastpdb_top_pairs.tsv", matrix)
    write_template_matches(out_dir / "blastpdb_template_matches.tsv", retained)
    if not args.no_heatmap:
        save_heatmap(
            out_dir / "blastpdb_heatmap.png",
            matrix,
            title="blastPDB structural-template support",
            label="Normalized template support",
        )

    if not retained:
        warnings.append("No structural templates survived contact transfer.")

    summary = {
        "query1_header": q1_header,
        "query2_header": q2_header,
        "query1_length": len(q1_seq),
        "query2_length": len(q2_seq),
        "interaction_mode_used": interaction_mode,
        "search_backend": "rcsb_sequence_search_api",
        "candidate_source": "experimental_pdb_biological_assemblies",
        "top_assemblies_requested": args.top_assemblies,
        "sequence_search_identity_cutoff": args.sequence_search_identity_cutoff,
        "sequence_search_evalue_cutoff": args.sequence_search_evalue_cutoff,
        "local_blast_evalue": args.local_blast_evalue,
        "local_blast_max_target_seqs": args.local_blast_max_target_seqs,
        "cbeta_threshold": args.cbeta_threshold,
        "min_template_contacts": args.min_template_contacts,
        "query1_remote_assembly_hits": len(q1_search_hits),
        "query2_remote_assembly_hits": len(q2_search_hits),
        "shared_candidate_assemblies": len(common_ids),
        "considered_assemblies": considered_identifiers,
        "skipped_nonexperimental_assemblies": skipped_nonexperimental,
        "skipped_empty_contact_assemblies": skipped_empty_contacts,
        "skipped_no_transfer_assemblies": skipped_no_transfer,
        "retained_templates": len(retained),
        "nonzero_cells": int(np.count_nonzero(matrix > 0.0)),
        "matrix_shape": list(matrix.shape),
        "cache_dir": str(cache_dir),
        "warnings": warnings,
        "outputs": {
            "blastpdb_matrix_tsv": str(out_dir / "blastpdb_matrix.tsv"),
            "blastpdb_matrix_npy": str(out_dir / "blastpdb_matrix.npy"),
            "blastpdb_top_pairs_tsv": str(out_dir / "blastpdb_top_pairs.tsv"),
            "blastpdb_template_matches_tsv": str(out_dir / "blastpdb_template_matches.tsv"),
            "blastpdb_summary_json": str(out_dir / "blastpdb_summary.json"),
        },
    }
    (out_dir / "blastpdb_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.verbose:
        print(f"[INFO] Remote assembly hits: q1={len(q1_search_hits)} q2={len(q2_search_hits)}")
        print(f"[INFO] Shared candidate assemblies: {len(common_ids)}")
        print(f"[INFO] Retained structural templates: {len(retained)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
