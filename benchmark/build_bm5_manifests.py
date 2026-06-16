#!/usr/bin/env python3
"""Build BM5.5 control manifests for iFragDI benchmark planning.

This script intentionally does not run iFragDI, inspect template databases, or
train an ML model. It only parses the official BM5.5 table, maps rows to local
structure files, decomposes chain-pair tasks, and labels native heavy-atom
contacts in the bound structures.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
from xml.etree import ElementTree as ET


CONTACT_CUTOFFS = (3.9, 5.0, 8.0)
CONTACT_LABELS = {3.9: "3p9A", 5.0: "5A", 8.0: "8A"}

RUN_STATUS = "not_run"
LABEL_STATUS = "manifest_only"
LEAKAGE_STATUS = "not_checked"
TEMPLATE_DATASET_PLAN = "intact_biogrid"
USE_BLASTPDB = "false"

DIFFICULTY_ROWS = {
    "Rigid-body (162)": "rigid",
    "Medium Difficulty (60)": "medium",
    "Difficult (35)": "difficult",
}
DIFFICULTY_ORDER = {"rigid": 0, "medium": 1, "difficult": 2}

SYNTHETIC_LOCAL_ID_CANDIDATES = {
    "1QFW": ["1QFW", "9QFW"],
    "1OYV": ["1OYV", "BOYV"],
    "3AAD": ["3AAD", "BAAD"],
    "3P57": ["3P57", "BP57", "CP57"],
}

MAPPING_MIN_IDENTITY = 0.90
MAPPING_MIN_COVERAGE_BOUND = 0.80
MAPPING_MIN_COVERAGE_UNBOUND = 0.80
MAPPING_MIN_BEST_IDENTITY_DELTA = 0.05

AA_CODES = {
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
    "SEC": "U",
    "PYL": "O",
}
AA_RESNAMES = set(AA_CODES)

ENTITY_COLUMNS = [
    "entity_id",
    "benchmark_row_index",
    "table_complex_id",
    "local_file_id",
    "in_bm55_table",
    "difficulty",
    "category_code",
    "protein1_name",
    "protein2_name",
    "irmsd_A",
    "dasa_A2",
    "bm_version_introduced",
    "receptor_unbound_pdb",
    "ligand_unbound_pdb",
    "receptor_bound_pdb",
    "ligand_bound_pdb",
    "has_all_required_files",
    "missing_files",
    "receptor_unbound_chain_ids",
    "ligand_unbound_chain_ids",
    "receptor_bound_chain_ids",
    "ligand_bound_chain_ids",
    "receptor_unbound_chain_count",
    "ligand_unbound_chain_count",
    "receptor_bound_chain_count",
    "ligand_bound_chain_count",
    "receptor_chain_mapping_status",
    "ligand_chain_mapping_status",
    "receptor_chain_mapping_details",
    "ligand_chain_mapping_details",
    "is_single_chain_receptor",
    "is_single_chain_ligand",
    "is_single_chain_pair",
    "is_multichain_entity",
    "chainpair_decomposition_needed",
    "n_chainpair_tasks",
    "receptor_total_length_unbound",
    "ligand_total_length_unbound",
    "total_entity_length",
    "size_bin",
    "native_contacting_entity_3p9A",
    "native_contacting_entity_5A",
    "native_contacting_entity_8A",
    "entity_interface_residue_count_receptor_5A",
    "entity_interface_residue_count_ligand_5A",
    "entity_total_interface_residue_count_5A",
    "entity_min_heavy_atom_distance_A",
    "entity_heavy_atom_contact_count_5A",
    "run_status",
    "label_status",
    "leakage_status",
    "template_dataset_plan",
    "use_blastpdb",
    "notes",
]

CHAINPAIR_COLUMNS = [
    "chainpair_id",
    "entity_id",
    "benchmark_row_index",
    "table_complex_id",
    "local_file_id",
    "difficulty",
    "category_code",
    "receptor_chain",
    "ligand_chain",
    "query1_role",
    "query2_role",
    "receptor_unbound_pdb",
    "ligand_unbound_pdb",
    "receptor_bound_pdb",
    "ligand_bound_pdb",
    "query1_chain",
    "query2_chain",
    "query1_length_unbound",
    "query2_length_unbound",
    "query1_length_bound",
    "query2_length_bound",
    "query1_chain_mapping_identity",
    "query2_chain_mapping_identity",
    "query1_chain_mapping_coverage_bound",
    "query2_chain_mapping_coverage_bound",
    "query1_chain_mapping_coverage_unbound",
    "query2_chain_mapping_coverage_unbound",
    "query1_chain_mapping_status",
    "query2_chain_mapping_status",
    "chainpair_runnable",
    "chainpair_exclusion_reason",
    "total_chainpair_length",
    "length_ratio",
    "size_bin",
    "length_balance_bin",
    "native_contacting_chainpair_3p9A",
    "native_contacting_chainpair_5A",
    "native_contacting_chainpair_8A",
    "query1_interface_residue_count_3p9A",
    "query2_interface_residue_count_3p9A",
    "query1_interface_residue_count_5A",
    "query2_interface_residue_count_5A",
    "query1_interface_residue_count_8A",
    "query2_interface_residue_count_8A",
    "total_interface_residue_count_5A",
    "interface_fraction_query1_5A",
    "interface_fraction_query2_5A",
    "min_heavy_atom_distance_A",
    "heavy_atom_contact_count_3p9A",
    "heavy_atom_contact_count_5A",
    "heavy_atom_contact_count_8A",
    "direct_single_chain_case",
    "decomposed_multichain_case",
    "noncontacting_chainpair_control",
    "recommended_for_first_singlechain_pilot",
    "recommended_for_first_multichain_pilot",
    "run_status",
    "label_status",
    "leakage_status",
    "template_dataset_plan",
    "use_blastpdb",
    "planned_output_dir",
    "notes",
]


@dataclass(frozen=True)
class TableRow:
    row_index: int
    table_complex_id: str
    base_id: str
    table_receptor_chains: Tuple[str, ...]
    table_ligand_chains: Tuple[str, ...]
    difficulty: str
    category_code: str
    protein1_name: str
    protein2_name: str
    irmsd_A: str
    dasa_A2: str
    bm_version_introduced: str


@dataclass(frozen=True)
class Atom:
    chain: str
    residue_key: Tuple[str, str, str, str]
    x: float
    y: float
    z: float


@dataclass
class StructureInfo:
    path: Path
    chain_ids: List[str]
    residues_by_chain: Dict[str, Set[Tuple[str, str, str, str]]]
    atoms_by_chain: Dict[str, List[Atom]]
    sequence_by_chain: Dict[str, str]

    @property
    def chain_count(self) -> int:
        return len(self.chain_ids)

    def length(self, chain: str) -> int:
        return len(self.residues_by_chain.get(chain, set()))

    def total_length(self) -> int:
        return sum(len(self.residues_by_chain.get(chain, set())) for chain in self.chain_ids)

    def sequence(self, chain: str) -> str:
        return self.sequence_by_chain.get(chain, "")


@dataclass(frozen=True)
class ChainMapping:
    bound_chain: str
    query_chain: Optional[str]
    sequence_identity: Optional[float]
    coverage_bound: Optional[float]
    coverage_unbound: Optional[float]
    mapping_method: str
    mapping_status: str
    reason: str
    second_best_identity: Optional[float] = None
    fallback_query_chain: Optional[str] = None


@dataclass
class ContactStats:
    min_distance: Optional[float]
    contact_counts: Dict[float, int]
    receptor_interface: Dict[float, Set[Tuple[str, str, str, str]]]
    ligand_interface: Dict[float, Set[Tuple[str, str, str, str]]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build BM5.5 entity, chain-pair, pilot, and summary manifests."
    )
    parser.add_argument("--benchmark-root", required=True, type=Path)
    parser.add_argument("--table", required=True, type=Path)
    parser.add_argument("--entity-out", required=True, type=Path)
    parser.add_argument("--chainpair-out", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--pilot-single-out", required=True, type=Path)
    parser.add_argument("--pilot-multichain-out", required=True, type=Path)
    return parser.parse_args()


def cell_column_index(cell_ref: str) -> int:
    match = re.match(r"([A-Z]+)", cell_ref)
    if not match:
        return 0
    value = 0
    for char in match.group(1):
        value = value * 26 + ord(char) - ord("A") + 1
    return value - 1


def read_xlsx_rows(path: Path) -> List[Tuple[int, List[str]]]:
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as archive:
        shared_strings: List[str] = []
        try:
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall("a:si", ns):
                shared_strings.append("".join(t.text or "" for t in item.findall(".//a:t", ns)))
        except KeyError:
            pass

        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        first_sheet_name = workbook.find(".//a:sheets/a:sheet", ns).attrib["name"]
        rels_root = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        rel_ns = {"r": "http://schemas.openxmlformats.org/package/2006/relationships"}
        sheet_rid = workbook.find(".//a:sheets/a:sheet", ns).attrib[
            "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
        ]
        sheet_target = None
        for rel in rels_root.findall("r:Relationship", rel_ns):
            if rel.attrib["Id"] == sheet_rid:
                sheet_target = rel.attrib["Target"]
                break
        if not sheet_target:
            raise ValueError(f"Could not locate worksheet target for {first_sheet_name}")
        sheet_path = "xl/" + sheet_target.lstrip("/")
        sheet = ET.fromstring(archive.read(sheet_path))

    rows: List[Tuple[int, List[str]]] = []
    for row_node in sheet.findall(".//a:sheetData/a:row", ns):
        values: Dict[int, str] = {}
        for cell in row_node.findall("a:c", ns):
            value_node = cell.find("a:v", ns)
            inline_node = cell.find("a:is", ns)
            value = ""
            if value_node is not None:
                value = value_node.text or ""
                if cell.attrib.get("t") == "s":
                    value = shared_strings[int(value)]
            elif inline_node is not None:
                value = "".join(t.text or "" for t in inline_node.findall(".//a:t", ns))
            values[cell_column_index(cell.attrib.get("r", "A1"))] = str(value).strip()
        if values:
            rows.append(
                (
                    int(row_node.attrib.get("r", len(rows) + 1)),
                    [values.get(i, "") for i in range(max(values) + 1)],
                )
            )
    return rows


def parse_table_complex_id(value: str) -> Tuple[str, Tuple[str, ...], Tuple[str, ...]]:
    cleaned = value.replace("*", "").strip()
    if "_" not in cleaned or ":" not in cleaned:
        raise ValueError(f"Unrecognized BM5 complex id: {value!r}")
    base_id, chain_part = cleaned.split("_", 1)
    receptor, ligand = chain_part.split(":", 1)
    return base_id.upper(), tuple(receptor.strip()), tuple(ligand.strip())


def parse_bm5_table(path: Path) -> List[TableRow]:
    rows = read_xlsx_rows(path)
    parsed: List[TableRow] = []
    difficulty = ""
    for excel_row, row in rows:
        if not row:
            continue
        first = row[0].strip()
        if first in DIFFICULTY_ROWS:
            difficulty = DIFFICULTY_ROWS[first]
            continue
        if not re.match(r"^[A-Za-z0-9]{4}_.+:.+", first):
            continue
        if len(row) < 9:
            raise ValueError(f"BM5 table row {excel_row} has fewer columns than expected: {row}")
        base_id, receptor_chains, ligand_chains = parse_table_complex_id(first)
        parsed.append(
            TableRow(
                row_index=excel_row,
                table_complex_id=first.replace("*", "").strip(),
                base_id=base_id,
                table_receptor_chains=receptor_chains,
                table_ligand_chains=ligand_chains,
                difficulty=difficulty,
                category_code=row[1].strip(),
                protein1_name=row[3].strip(),
                protein2_name=row[5].strip(),
                irmsd_A=row[6].strip(),
                dasa_A2=row[7].strip(),
                bm_version_introduced=row[8].strip(),
            )
        )
    return parsed


def bool_s(value: bool) -> str:
    return "true" if value else "false"


def path_s(path: Optional[Path]) -> str:
    return "" if path is None else str(path)


def list_s(values: Sequence[str]) -> str:
    return ",".join(values)


def na_if_none(value: Optional[object]) -> str:
    if value is None:
        return "NA"
    return str(value)


def format_float(value: Optional[float], digits: int = 3) -> str:
    if value is None or math.isnan(value):
        return "NA"
    return f"{value:.{digits}f}"


def size_bin(total_length: Optional[int]) -> str:
    if total_length is None:
        return "unknown"
    if total_length < 300:
        return "small"
    if total_length <= 700:
        return "medium"
    return "large"


def length_balance_bin(length_ratio: Optional[float]) -> str:
    if length_ratio is None:
        return "unknown"
    if length_ratio <= 1.5:
        return "balanced"
    if length_ratio <= 3.0:
        return "moderately_imbalanced"
    return "highly_imbalanced"


def infer_element(line: str) -> str:
    element = line[76:78].strip() if len(line) >= 78 else ""
    if element:
        return element.upper()
    atom_name = line[12:16].strip()
    atom_name = atom_name.lstrip("0123456789")
    return atom_name[:1].upper()


def is_hydrogen(line: str) -> bool:
    return infer_element(line) in {"H", "D"}


def parse_structure(path: Path) -> StructureInfo:
    chain_ids: List[str] = []
    seen_chains: Set[str] = set()
    seen_residues: Set[Tuple[str, str, str, str]] = set()
    residues_by_chain: Dict[str, Set[Tuple[str, str, str, str]]] = defaultdict(set)
    atoms_by_chain: Dict[str, List[Atom]] = defaultdict(list)
    sequence_lists: Dict[str, List[str]] = defaultdict(list)

    with path.open(errors="ignore") as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            altloc = line[16].strip()
            if altloc not in {"", "A"}:
                continue
            resname = line[17:20].strip().upper()
            if resname not in AA_RESNAMES:
                continue
            chain = line[21].strip() or "_"
            if chain not in seen_chains:
                seen_chains.add(chain)
                chain_ids.append(chain)
            resseq = line[22:26].strip()
            icode = line[26].strip()
            residue_key = (chain, resseq, icode, resname)
            residues_by_chain[chain].add(residue_key)
            if residue_key not in seen_residues:
                seen_residues.add(residue_key)
                sequence_lists[chain].append(AA_CODES.get(resname, "X"))
            if is_hydrogen(line):
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            atoms_by_chain[chain].append(Atom(chain, residue_key, x, y, z))

    return StructureInfo(
        path=path,
        chain_ids=chain_ids,
        residues_by_chain=dict(residues_by_chain),
        atoms_by_chain=dict(atoms_by_chain),
        sequence_by_chain={chain: "".join(sequence_lists.get(chain, [])) for chain in chain_ids},
    )


def get_structure_paths(benchmark_root: Path, local_id: str) -> Dict[str, Path]:
    structures = benchmark_root / "structures"
    return {
        "r_u": structures / f"{local_id}_r_u.pdb",
        "l_u": structures / f"{local_id}_l_u.pdb",
        "r_b": structures / f"{local_id}_r_b.pdb",
        "l_b": structures / f"{local_id}_l_b.pdb",
    }


def required_files_exist(paths: Dict[str, Path]) -> Tuple[bool, List[str]]:
    missing = [str(path) for path in paths.values() if not path.exists()]
    return len(missing) == 0, missing


def available_complete_local_ids(benchmark_root: Path) -> Set[str]:
    structures = benchmark_root / "structures"
    ids_by_suffix: Dict[str, Set[str]] = {"r_u": set(), "l_u": set(), "r_b": set(), "l_b": set()}
    for path in structures.glob("*.pdb"):
        if path.name.startswith("._"):
            continue
        match = re.match(r"^(.+)_([rl]_[ub])\.pdb$", path.name)
        if match:
            ids_by_suffix[match.group(2)].add(match.group(1).upper())
    complete = set.intersection(*(ids_by_suffix[suffix] for suffix in ids_by_suffix))
    return complete


def candidate_local_ids(row: TableRow, complete_ids: Set[str]) -> List[str]:
    candidates = SYNTHETIC_LOCAL_ID_CANDIDATES.get(row.base_id, [row.base_id])
    result = [candidate for candidate in candidates if candidate in complete_ids]
    if row.base_id in complete_ids and row.base_id not in result:
        result.insert(0, row.base_id)
    return result


def candidate_matches_bound_chains(
    row: TableRow,
    local_id: str,
    structure_cache: Dict[Path, StructureInfo],
    benchmark_root: Path,
) -> bool:
    paths = get_structure_paths(benchmark_root, local_id)
    ok, _missing = required_files_exist(paths)
    if not ok:
        return False
    r_b = load_structure(paths["r_b"], structure_cache)
    l_b = load_structure(paths["l_b"], structure_cache)
    return tuple(r_b.chain_ids) == row.table_receptor_chains and tuple(l_b.chain_ids) == row.table_ligand_chains


def choose_local_id(
    row: TableRow,
    complete_ids: Set[str],
    structure_cache: Dict[Path, StructureInfo],
    benchmark_root: Path,
    warnings: List[str],
    ambiguous: List[Dict[str, object]],
    synthetic_used: List[Dict[str, object]],
) -> Optional[str]:
    candidates = candidate_local_ids(row, complete_ids)
    if not candidates:
        ambiguous.append(
            {
                "benchmark_row_index": row.row_index,
                "table_complex_id": row.table_complex_id,
                "base_id": row.base_id,
                "reason": "no_complete_local_files_for_base_or_synthetic_candidates",
                "candidates": SYNTHETIC_LOCAL_ID_CANDIDATES.get(row.base_id, [row.base_id]),
            }
        )
        return None

    if len(candidates) == 1 and candidates[0] == row.base_id:
        return row.base_id

    matching = [
        candidate
        for candidate in candidates
        if candidate_matches_bound_chains(row, candidate, structure_cache, benchmark_root)
    ]
    if len(matching) == 1:
        local_id = matching[0]
    elif len(matching) > 1:
        local_id = matching[0]
        ambiguous.append(
            {
                "benchmark_row_index": row.row_index,
                "table_complex_id": row.table_complex_id,
                "base_id": row.base_id,
                "reason": "multiple_candidates_match_bound_chains_first_used",
                "candidates": matching,
            }
        )
    elif row.base_id in candidates:
        local_id = row.base_id
        if len(candidates) > 1:
            ambiguous.append(
                {
                    "benchmark_row_index": row.row_index,
                    "table_complex_id": row.table_complex_id,
                    "base_id": row.base_id,
                    "reason": "no_synthetic_candidate_matched_bound_chains_base_id_used",
                    "candidates": candidates,
                }
            )
    else:
        local_id = candidates[0]
        ambiguous.append(
            {
                "benchmark_row_index": row.row_index,
                "table_complex_id": row.table_complex_id,
                "base_id": row.base_id,
                "reason": "no_candidate_matched_bound_chains_first_candidate_used",
                "candidates": candidates,
            }
        )

    if local_id != row.base_id:
        synthetic_used.append(
            {
                "benchmark_row_index": row.row_index,
                "table_complex_id": row.table_complex_id,
                "source_base_id": row.base_id,
                "local_file_id": local_id,
                "reason": "README synthetic/local ID mapped by bound-chain content",
            }
        )
    if local_id not in complete_ids:
        warnings.append(f"{row.table_complex_id}: selected local id {local_id} is not complete")
    return local_id


def load_structure(path: Path, cache: Dict[Path, StructureInfo]) -> StructureInfo:
    if path not in cache:
        cache[path] = parse_structure(path)
    return cache[path]


def build_grid(atoms: Sequence[Atom], cell_size: float) -> Dict[Tuple[int, int, int], List[Atom]]:
    grid: Dict[Tuple[int, int, int], List[Atom]] = defaultdict(list)
    for atom in atoms:
        key = (
            math.floor(atom.x / cell_size),
            math.floor(atom.y / cell_size),
            math.floor(atom.z / cell_size),
        )
        grid[key].append(atom)
    return grid


def neighbor_keys(key: Tuple[int, int, int]) -> Iterable[Tuple[int, int, int]]:
    x, y, z = key
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                yield (x + dx, y + dy, z + dz)


def point_to_cell_min_dist_sq(atom: Atom, key: Tuple[int, int, int], cell_size: float) -> float:
    total = 0.0
    for coord, cell_index in zip((atom.x, atom.y, atom.z), key):
        low = cell_index * cell_size
        high = low + cell_size
        if coord < low:
            delta = low - coord
        elif coord > high:
            delta = coord - high
        else:
            delta = 0.0
        total += delta * delta
    return total


def compute_contact_stats(receptor_atoms: Sequence[Atom], ligand_atoms: Sequence[Atom]) -> ContactStats:
    counts = {cutoff: 0 for cutoff in CONTACT_CUTOFFS}
    receptor_interface = {cutoff: set() for cutoff in CONTACT_CUTOFFS}
    ligand_interface = {cutoff: set() for cutoff in CONTACT_CUTOFFS}
    if not receptor_atoms or not ligand_atoms:
        return ContactStats(None, counts, receptor_interface, ligand_interface)

    max_cutoff = max(CONTACT_CUTOFFS)
    max_cutoff_sq = max_cutoff * max_cutoff
    cutoff_squares = {cutoff: cutoff * cutoff for cutoff in CONTACT_CUTOFFS}
    grid = build_grid(ligand_atoms, max_cutoff)
    occupied_keys = list(grid)

    first_rec = receptor_atoms[0]
    first_lig = ligand_atoms[0]
    min_distance_sq: float = (
        (first_rec.x - first_lig.x) ** 2
        + (first_rec.y - first_lig.y) ** 2
        + (first_rec.z - first_lig.z) ** 2
    )

    for rec in receptor_atoms:
        key = (
            math.floor(rec.x / max_cutoff),
            math.floor(rec.y / max_cutoff),
            math.floor(rec.z / max_cutoff),
        )

        # Contact counts only need neighboring max-cutoff grid cells.
        for neighbor in neighbor_keys(key):
            for lig in grid.get(neighbor, []):
                dx = rec.x - lig.x
                dy = rec.y - lig.y
                dz = rec.z - lig.z
                dist_sq = dx * dx + dy * dy + dz * dz
                if dist_sq < min_distance_sq:
                    min_distance_sq = dist_sq
                if dist_sq > max_cutoff_sq:
                    continue
                for cutoff in CONTACT_CUTOFFS:
                    if dist_sq <= cutoff_squares[cutoff]:
                        counts[cutoff] += 1
                        receptor_interface[cutoff].add(rec.residue_key)
                        ligand_interface[cutoff].add(lig.residue_key)

        # Exact global minimum: check every ligand cell whose bounding cube could
        # improve the current best distance. This is independent of contact cutoffs.
        for cell_key in occupied_keys:
            if point_to_cell_min_dist_sq(rec, cell_key, max_cutoff) > min_distance_sq:
                continue
            for lig in grid[cell_key]:
                dx = rec.x - lig.x
                dy = rec.y - lig.y
                dz = rec.z - lig.z
                dist_sq = dx * dx + dy * dy + dz * dz
                if dist_sq < min_distance_sq:
                    min_distance_sq = dist_sq

    return ContactStats(math.sqrt(min_distance_sq), counts, receptor_interface, ligand_interface)


def sequence_match_stats(bound_sequence: str, unbound_sequence: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if not bound_sequence or not unbound_sequence:
        return None, None, None
    matcher = SequenceMatcher(None, bound_sequence, unbound_sequence, autojunk=False)
    matches = sum(block.size for block in matcher.get_matching_blocks())
    identity = matches / max(len(bound_sequence), len(unbound_sequence))
    coverage_bound = matches / len(bound_sequence)
    coverage_unbound = matches / len(unbound_sequence)
    return identity, coverage_bound, coverage_unbound


def format_mapping_value(value: Optional[float]) -> str:
    return format_float(value, digits=4)


def fallback_chain_by_order(
    bound_chain: str,
    bound_chains: Sequence[str],
    unbound_chains: Sequence[str],
) -> Optional[str]:
    if bound_chain in unbound_chains:
        return bound_chain
    if len(bound_chains) == len(unbound_chains):
        return unbound_chains[bound_chains.index(bound_chain)]
    if len(unbound_chains) == 1:
        return unbound_chains[0]
    return None


def mapping_to_dict(mapping: ChainMapping) -> Dict[str, object]:
    return {
        "bound_chain": mapping.bound_chain,
        "query_chain": mapping.query_chain or "",
        "sequence_identity": None if mapping.sequence_identity is None else round(mapping.sequence_identity, 4),
        "alignment_coverage_bound": None if mapping.coverage_bound is None else round(mapping.coverage_bound, 4),
        "alignment_coverage_unbound": None if mapping.coverage_unbound is None else round(mapping.coverage_unbound, 4),
        "mapping_method": mapping.mapping_method,
        "mapping_status": mapping.mapping_status,
        "second_best_identity": None if mapping.second_best_identity is None else round(mapping.second_best_identity, 4),
        "fallback_query_chain": mapping.fallback_query_chain or "",
        "reason": mapping.reason,
    }


def mapping_status_for_side(mappings: Dict[str, ChainMapping]) -> str:
    if not mappings:
        return "not_evaluated"
    if all(mapping.mapping_status == "high_confidence" for mapping in mappings.values()):
        return "high_confidence"
    return "ambiguous_or_low_confidence"


def map_bound_to_unbound_chains(
    bound_info: StructureInfo,
    unbound_info: StructureInfo,
    notes: List[str],
    side_label: str,
) -> Dict[str, ChainMapping]:
    mapping: Dict[str, ChainMapping] = {}
    for bound_chain in bound_info.chain_ids:
        bound_sequence = bound_info.sequence(bound_chain)
        candidates = []
        for query_chain in unbound_info.chain_ids:
            identity, cov_bound, cov_unbound = sequence_match_stats(
                bound_sequence,
                unbound_info.sequence(query_chain),
            )
            candidates.append((query_chain, identity, cov_bound, cov_unbound))
        candidates.sort(
            key=lambda item: (
                -1.0 if item[1] is None else item[1],
                -1.0 if item[2] is None else item[2],
                -1.0 if item[3] is None else item[3],
            ),
            reverse=True,
        )
        fallback = fallback_chain_by_order(bound_chain, bound_info.chain_ids, unbound_info.chain_ids)
        if not candidates:
            mapping[bound_chain] = ChainMapping(
                bound_chain=bound_chain,
                query_chain=None,
                sequence_identity=None,
                coverage_bound=None,
                coverage_unbound=None,
                mapping_method="sequence_match",
                mapping_status="ambiguous_or_low_confidence",
                reason="no_unbound_candidate_chains",
                fallback_query_chain=fallback,
            )
            notes.append(f"{side_label} chain {bound_chain} has no unbound chain candidates")
            continue

        best_chain, best_identity, best_cov_bound, best_cov_unbound = candidates[0]
        second_identity = candidates[1][1] if len(candidates) > 1 else None
        clearly_best = second_identity is None or (
            best_identity is not None and second_identity is not None and
            best_identity - second_identity >= MAPPING_MIN_BEST_IDENTITY_DELTA
        )
        high_confidence = (
            best_identity is not None
            and best_cov_bound is not None
            and best_cov_unbound is not None
            and best_identity >= MAPPING_MIN_IDENTITY
            and best_cov_bound >= MAPPING_MIN_COVERAGE_BOUND
            and best_cov_unbound >= MAPPING_MIN_COVERAGE_UNBOUND
            and clearly_best
        )
        if high_confidence:
            mapping[bound_chain] = ChainMapping(
                bound_chain=bound_chain,
                query_chain=best_chain,
                sequence_identity=best_identity,
                coverage_bound=best_cov_bound,
                coverage_unbound=best_cov_unbound,
                mapping_method="sequence_match",
                mapping_status="high_confidence",
                reason="accepted_sequence_match",
                second_best_identity=second_identity,
                fallback_query_chain=fallback,
            )
        else:
            reason_parts = []
            if best_identity is None:
                reason_parts.append("missing_sequence")
            elif best_identity < MAPPING_MIN_IDENTITY:
                reason_parts.append("identity_below_threshold")
            if best_cov_bound is None or best_cov_bound < MAPPING_MIN_COVERAGE_BOUND:
                reason_parts.append("bound_coverage_below_threshold")
            if best_cov_unbound is None or best_cov_unbound < MAPPING_MIN_COVERAGE_UNBOUND:
                reason_parts.append("unbound_coverage_below_threshold")
            if not clearly_best:
                reason_parts.append("best_hit_not_clearly_better_than_second")
            reason = ",".join(reason_parts) or "ambiguous_or_low_confidence"
            mapping[bound_chain] = ChainMapping(
                bound_chain=bound_chain,
                query_chain=None,
                sequence_identity=best_identity,
                coverage_bound=best_cov_bound,
                coverage_unbound=best_cov_unbound,
                mapping_method="sequence_match",
                mapping_status="ambiguous_or_low_confidence",
                reason=reason,
                second_best_identity=second_identity,
                fallback_query_chain=fallback,
            )
            fallback_text = f"; unsafe fallback would be {fallback}" if fallback else ""
            notes.append(f"{side_label} chain {bound_chain} sequence mapping rejected: {reason}{fallback_text}")
    return mapping


def ratio_larger_over_smaller(length_a: Optional[int], length_b: Optional[int]) -> Optional[float]:
    if length_a is None or length_b is None or min(length_a, length_b) <= 0:
        return None
    return max(length_a, length_b) / min(length_a, length_b)


def fraction(numerator: int, denominator: Optional[int]) -> Optional[float]:
    if denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def interface_size_bin(row: Dict[str, str]) -> str:
    try:
        total = int(row["total_interface_residue_count_5A"])
    except ValueError:
        return "unknown"
    if total < 20:
        return "small_interface"
    if total <= 60:
        return "medium_interface"
    return "large_interface"


def entity_chain_count_key(entity: Dict[str, str]) -> str:
    return (
        f"ru{entity['receptor_unbound_chain_count']}_lu{entity['ligand_unbound_chain_count']}_"
        f"rb{entity['receptor_bound_chain_count']}_lb{entity['ligand_bound_chain_count']}"
    )


def pick_diverse(
    candidates: Sequence[Dict[str, str]],
    n: int,
    novelty_keys: Sequence[str],
    required_key: Optional[str] = None,
    entity_lookup: Optional[Dict[str, Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    selected: List[Dict[str, str]] = []
    remaining = list(candidates)
    seen_values: Dict[str, Set[str]] = {key: set() for key in novelty_keys}
    required_counts: Counter[str] = Counter()

    def sort_key(row: Dict[str, str]) -> Tuple[int, int, str]:
        difficulty_rank = DIFFICULTY_ORDER.get(row.get("difficulty", ""), 99)
        try:
            interface_count = int(row.get("total_interface_residue_count_5A", "0"))
        except ValueError:
            interface_count = 0
        return (difficulty_rank, interface_count, row.get("chainpair_id", ""))

    remaining.sort(key=sort_key)
    while remaining and len(selected) < n:
        best_index = 0
        best_score: Optional[Tuple[int, int, int, str]] = None
        for idx, row in enumerate(remaining):
            score = 0
            if required_key:
                value = row.get(required_key, "")
                if required_counts[value] == 0:
                    score += 100
                score -= required_counts[value] * 3
            for key in novelty_keys:
                if key == "interface_size_bin":
                    value = interface_size_bin(row)
                elif key == "parent_chain_count_key" and entity_lookup is not None:
                    value = entity_chain_count_key(entity_lookup[row["entity_id"]])
                else:
                    value = row.get(key, "")
                if value not in seen_values[key]:
                    score += 10
            try:
                interface_count = int(row.get("total_interface_residue_count_5A", "0"))
            except ValueError:
                interface_count = 0
            tie = (
                score,
                -abs(interface_count - 40),
                -DIFFICULTY_ORDER.get(row.get("difficulty", ""), 99),
                row.get("chainpair_id", ""),
            )
            if best_score is None or tie > best_score:
                best_index = idx
                best_score = tie
        row = remaining.pop(best_index)
        selected.append(row)
        if required_key:
            required_counts[row.get(required_key, "")] += 1
        for key in novelty_keys:
            if key == "interface_size_bin":
                value = interface_size_bin(row)
            elif key == "parent_chain_count_key" and entity_lookup is not None:
                value = entity_chain_count_key(entity_lookup[row["entity_id"]])
            else:
                value = row.get(key, "")
            seen_values[key].add(value)
    return selected


def int_field(row: Dict[str, str], column: str) -> Optional[int]:
    value = row.get(column, "NA")
    if value in {"", "NA"}:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def float_field(row: Dict[str, str], column: str) -> Optional[float]:
    value = row.get(column, "NA")
    if value in {"", "NA"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def has_non_na_query_lengths(row: Dict[str, str]) -> bool:
    return int_field(row, "query1_length_unbound") is not None and int_field(row, "query2_length_unbound") is not None


def chainpair_is_safe(row: Dict[str, str]) -> bool:
    return (
        row.get("chainpair_runnable") == "true"
        and row.get("query1_chain_mapping_status") == "high_confidence"
        and row.get("query2_chain_mapping_status") == "high_confidence"
        and has_non_na_query_lengths(row)
    )


def strict_multichain_positive(row: Dict[str, str]) -> bool:
    return (
        row.get("native_contacting_chainpair_5A") == "true"
        and (int_field(row, "total_interface_residue_count_5A") or 0) >= 10
        and (int_field(row, "heavy_atom_contact_count_5A") or 0) >= 20
    )


def strict_multichain_negative(row: Dict[str, str]) -> bool:
    min_distance = float_field(row, "min_heavy_atom_distance_A")
    return (
        row.get("native_contacting_chainpair_5A") == "false"
        and int_field(row, "heavy_atom_contact_count_8A") == 0
        and min_distance is not None
        and min_distance > 8.0
    )


def choose_pilots(
    chainpairs: List[Dict[str, str]],
    entities: List[Dict[str, str]],
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[str]]:
    warnings: List[str] = []
    entity_lookup = {entity["entity_id"]: entity for entity in entities}

    direct_candidates = [
        row
        for row in chainpairs
        if row["direct_single_chain_case"] == "true"
        and chainpair_is_safe(row)
        and row["native_contacting_chainpair_5A"] == "true"
    ]
    single_pilot = pick_diverse(
        direct_candidates,
        min(20, len(direct_candidates)),
        ["category_code", "size_bin", "interface_size_bin", "length_balance_bin"],
        required_key="difficulty",
    )
    if len(single_pilot) < 20:
        warnings.append(f"Single-chain pilot has only {len(single_pilot)} runnable native-contacting rows available")
    single_difficulties = {row["difficulty"] for row in single_pilot}
    missing_difficulties = {"rigid", "medium", "difficult"} - single_difficulties
    if missing_difficulties:
        warnings.append(
            "Single-chain pilot could not include difficulties: " + ",".join(sorted(missing_difficulties))
        )

    multichain_candidates = [
        row
        for row in chainpairs
        if row["decomposed_multichain_case"] == "true" and chainpair_is_safe(row)
    ]
    strict_positive = [row for row in multichain_candidates if strict_multichain_positive(row)]
    strict_negative = [row for row in multichain_candidates if strict_multichain_negative(row)]
    first_contacting = pick_diverse(
        strict_positive,
        min(10, len(strict_positive)),
        ["category_code", "size_bin", "interface_size_bin", "parent_chain_count_key"],
        required_key="difficulty",
        entity_lookup=entity_lookup,
    )
    first_noncontacting = pick_diverse(
        strict_negative,
        min(10, len(strict_negative)),
        ["category_code", "size_bin", "interface_size_bin", "parent_chain_count_key"],
        required_key="difficulty",
        entity_lookup=entity_lookup,
    )
    if len(first_contacting) < 10:
        warnings.append(
            f"Multi-chain pilot strict positive criteria yielded {len(first_contacting)} rows; filling only from safe runnable rows"
        )
    if len(first_noncontacting) < 10:
        warnings.append(
            f"Multi-chain pilot strict negative criteria yielded {len(first_noncontacting)} rows; filling only from safe runnable rows"
        )

    multichain_pilot = first_contacting + first_noncontacting
    selected_ids = {row["chainpair_id"] for row in multichain_pilot}

    if len(first_contacting) < 10:
        positive_fill = [
            row for row in multichain_candidates
            if row["chainpair_id"] not in selected_ids and row["native_contacting_chainpair_5A"] == "true"
        ]
        added = pick_diverse(
            positive_fill,
            min(10 - len(first_contacting), len(positive_fill)),
            ["category_code", "size_bin", "interface_size_bin", "parent_chain_count_key"],
            required_key="difficulty",
            entity_lookup=entity_lookup,
        )
        multichain_pilot.extend(added)
        selected_ids.update(row["chainpair_id"] for row in added)

    current_negative_count = sum(1 for row in multichain_pilot if row["native_contacting_chainpair_5A"] == "false")
    if current_negative_count < 10:
        negative_fill = [
            row for row in multichain_candidates
            if row["chainpair_id"] not in selected_ids and row["native_contacting_chainpair_5A"] == "false"
        ]
        added = pick_diverse(
            negative_fill,
            min(10 - current_negative_count, len(negative_fill)),
            ["category_code", "size_bin", "interface_size_bin", "parent_chain_count_key"],
            required_key="difficulty",
            entity_lookup=entity_lookup,
        )
        multichain_pilot.extend(added)
        selected_ids.update(row["chainpair_id"] for row in added)

    if len(multichain_pilot) < 20:
        fill = [row for row in multichain_candidates if row["chainpair_id"] not in selected_ids]
        multichain_pilot.extend(
            pick_diverse(
                fill,
                min(20 - len(multichain_pilot), len(fill)),
                ["category_code", "size_bin", "interface_size_bin", "parent_chain_count_key"],
                required_key="difficulty",
                entity_lookup=entity_lookup,
            )
        )
    if len(multichain_pilot) < 20:
        warnings.append(f"Multi-chain pilot has only {len(multichain_pilot)} safe runnable rows available")

    single_ids = {row["chainpair_id"] for row in single_pilot}
    multi_ids = {row["chainpair_id"] for row in multichain_pilot}
    for row in chainpairs:
        row["recommended_for_first_singlechain_pilot"] = bool_s(row["chainpair_id"] in single_ids)
        row["recommended_for_first_multichain_pilot"] = bool_s(row["chainpair_id"] in multi_ids)

    return single_pilot, multichain_pilot, warnings


def write_tsv(path: Path, rows: Sequence[Dict[str, str]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def count_bool(rows: Sequence[Dict[str, str]], column: str) -> Dict[str, int]:
    counts = Counter(row.get(column, "NA") for row in rows)
    return {key: counts.get(key, 0) for key in ["true", "false", "NA"] if counts.get(key, 0) > 0}


def counter_dict(rows: Sequence[Dict[str, str]], column: str) -> Dict[str, int]:
    return dict(sorted(Counter(row.get(column, "") for row in rows).items()))


def value_range(rows: Sequence[Dict[str, str]], column: str, as_float: bool = False) -> Dict[str, Optional[float]]:
    values: List[float] = []
    for row in rows:
        value = row.get(column, "NA")
        if value in {"", "NA"}:
            continue
        try:
            values.append(float(value) if as_float else int(value))
        except ValueError:
            continue
    if not values:
        return {"min": None, "max": None, "count": 0}
    return {"min": min(values), "max": max(values), "count": len(values)}


def pilot_summary(rows: Sequence[Dict[str, str]]) -> Dict[str, object]:
    return {
        "row_count": len(rows),
        "counts_by_difficulty": counter_dict(rows, "difficulty"),
        "counts_by_category_code": counter_dict(rows, "category_code"),
        "counts_by_size_bin": counter_dict(rows, "size_bin"),
        "counts_by_length_balance_bin": counter_dict(rows, "length_balance_bin"),
        "native_contacting_chainpair_counts_5A": count_bool(rows, "native_contacting_chainpair_5A"),
        "chainpair_runnable_counts": count_bool(rows, "chainpair_runnable"),
        "mapping_status_counts_query1": counter_dict(rows, "query1_chain_mapping_status"),
        "mapping_status_counts_query2": counter_dict(rows, "query2_chain_mapping_status"),
        "total_interface_residue_count_range_5A": value_range(rows, "total_interface_residue_count_5A"),
    }


def nonrunnable_reason_counts(rows: Sequence[Dict[str, str]]) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        if row.get("chainpair_runnable") == "true":
            continue
        reason_text = row.get("chainpair_exclusion_reason", "")
        if not reason_text:
            counts["unspecified"] += 1
            continue
        for reason in reason_text.split(";"):
            if reason:
                counts[reason] += 1
    return dict(sorted(counts.items()))


def sequence_mapping_status_counts(rows: Sequence[Dict[str, str]]) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts[f"query1:{row.get('query1_chain_mapping_status', 'missing')}"] += 1
        counts[f"query2:{row.get('query2_chain_mapping_status', 'missing')}"] += 1
    return dict(sorted(counts.items()))


def pilot_qc_warnings(rows: Sequence[Dict[str, str]], pilot_name: str) -> List[str]:
    warnings: List[str] = []
    if any(row.get("chainpair_runnable") != "true" for row in rows):
        warnings.append(f"{pilot_name} pilot contains non-runnable chain pairs")
    if any(
        row.get("query1_chain_mapping_status") != "high_confidence"
        or row.get("query2_chain_mapping_status") != "high_confidence"
        for row in rows
    ):
        warnings.append(f"{pilot_name} pilot contains ambiguous/low-confidence sequence mappings")
    if any(not has_non_na_query_lengths(row) for row in rows):
        warnings.append(f"{pilot_name} pilot contains NA query lengths")
    if any(
        row.get("native_contacting_chainpair_5A") == "false" and row.get("min_heavy_atom_distance_A") == "NA"
        for row in rows
    ):
        warnings.append(f"{pilot_name} pilot contains noncontacting controls without min distance")
    return warnings


def first_lines(path: Path, n: int) -> str:
    lines = path.read_text().splitlines()
    return "\n".join(lines[: n + 1])


def write_manifest_report(
    report_path: Path,
    entity_path: Path,
    chainpair_path: Path,
    summary_path: Path,
    pilot_single_path: Path,
    pilot_multichain_path: Path,
) -> None:
    chainpairs = list(csv.DictReader(chainpair_path.open(), delimiter="\t"))
    out: List[str] = []
    out.append("# BM5.5 manifest build report")
    out.append("")
    out.append("Generated by `benchmark/build_bm5_manifests.py`. iFragDI was not run; ML was not started.")
    out.append("")
    out.append("## First 10 rows: bm5_entity_manifest.tsv")
    out.append("```tsv")
    out.append(first_lines(entity_path, 10))
    out.append("```")
    out.append("")
    out.append("## First 10 rows: bm5_chainpair_manifest.tsv")
    out.append("```tsv")
    out.append(first_lines(chainpair_path, 10))
    out.append("```")
    out.append("")
    out.append("## bm5_manifest_summary.json")
    out.append("```json")
    out.append(summary_path.read_text().rstrip())
    out.append("```")
    out.append("")
    out.append("## Direct single-chain chain pairs by difficulty")
    out.append("```json")
    out.append(json.dumps(dict(Counter(row["difficulty"] for row in chainpairs if row["direct_single_chain_case"] == "true")), indent=2, sort_keys=True))
    out.append("```")
    out.append("")
    out.append("## Decomposed multi-chain chain pairs by difficulty")
    out.append("```json")
    out.append(json.dumps(dict(Counter(row["difficulty"] for row in chainpairs if row["decomposed_multichain_case"] == "true")), indent=2, sort_keys=True))
    out.append("```")
    out.append("")
    out.append("## Native-contacting vs noncontacting chain pairs")
    out.append("```json")
    out.append(json.dumps({label: dict(Counter(row[f"native_contacting_chainpair_{label}"] for row in chainpairs)) for label in ["3p9A", "5A", "8A"]}, indent=2, sort_keys=True))
    out.append("```")
    out.append("")
    out.append("## Full single-chain pilot table")
    out.append("```tsv")
    out.append(pilot_single_path.read_text().rstrip())
    out.append("```")
    out.append("")
    out.append("## Full multi-chain pilot table")
    out.append("```tsv")
    out.append(pilot_multichain_path.read_text().rstrip())
    out.append("```")
    out.append("")
    report_path.write_text("\n".join(out) + "\n")


def main() -> None:
    args = parse_args()
    table_rows = parse_bm5_table(args.table)
    complete_ids = available_complete_local_ids(args.benchmark_root)
    structure_cache: Dict[Path, StructureInfo] = {}
    warnings: List[str] = []
    ambiguous: List[Dict[str, object]] = []
    synthetic_used: List[Dict[str, object]] = []

    sidecar_count = sum(1 for path in (args.benchmark_root / "structures").glob("._*.pdb"))
    if sidecar_count:
        warnings.append(f"Ignored {sidecar_count} macOS sidecar PDB files matching ._*.pdb")

    entities: List[Dict[str, str]] = []
    chainpairs: List[Dict[str, str]] = []
    used_local_ids: Set[str] = set()
    bound_chain_mismatch_rows: List[str] = []

    for entity_number, row in enumerate(table_rows, start=1):
        entity_id = f"BM5E{entity_number:04d}"
        local_id = choose_local_id(
            row,
            complete_ids,
            structure_cache,
            args.benchmark_root,
            warnings,
            ambiguous,
            synthetic_used,
        )
        paths = get_structure_paths(args.benchmark_root, local_id) if local_id else {}
        has_all_files, missing_files = required_files_exist(paths) if paths else (False, [])
        if local_id and has_all_files:
            used_local_ids.add(local_id)

        notes: List[str] = []
        if not has_all_files:
            notes.append("missing required BM5 structure files")

        r_u = l_u = r_b = l_b = None
        if has_all_files:
            r_u = load_structure(paths["r_u"], structure_cache)
            l_u = load_structure(paths["l_u"], structure_cache)
            r_b = load_structure(paths["r_b"], structure_cache)
            l_b = load_structure(paths["l_b"], structure_cache)
            if tuple(r_b.chain_ids) != row.table_receptor_chains or tuple(l_b.chain_ids) != row.table_ligand_chains:
                notes.append("bound chain IDs do not exactly match table_complex_id annotation")
                bound_chain_mismatch_rows.append(row.table_complex_id)

        receptor_unbound_chains = r_u.chain_ids if r_u else []
        ligand_unbound_chains = l_u.chain_ids if l_u else []
        receptor_bound_chains = r_b.chain_ids if r_b else []
        ligand_bound_chains = l_b.chain_ids if l_b else []

        is_single_chain_receptor = (
            len(receptor_unbound_chains) == 1 and len(receptor_bound_chains) == 1
        )
        is_single_chain_ligand = len(ligand_unbound_chains) == 1 and len(ligand_bound_chains) == 1
        is_single_chain_pair = is_single_chain_receptor and is_single_chain_ligand
        is_multichain_entity = any(
            count > 1
            for count in [
                len(receptor_unbound_chains),
                len(ligand_unbound_chains),
                len(receptor_bound_chains),
                len(ligand_bound_chains),
            ]
        )
        n_chainpair_tasks = len(receptor_bound_chains) * len(ligand_bound_chains) if has_all_files else 0

        receptor_total_unbound = r_u.total_length() if r_u else None
        ligand_total_unbound = l_u.total_length() if l_u else None
        total_entity_length = (
            receptor_total_unbound + ligand_total_unbound
            if receptor_total_unbound is not None and ligand_total_unbound is not None
            else None
        )

        receptor_interface_entity_5A: Set[Tuple[str, str, str, str]] = set()
        ligand_interface_entity_5A: Set[Tuple[str, str, str, str]] = set()
        entity_contact_counts = {cutoff: 0 for cutoff in CONTACT_CUTOFFS}
        entity_min_distance: Optional[float] = None

        receptor_query_map: Dict[str, ChainMapping] = {}
        ligand_query_map: Dict[str, ChainMapping] = {}
        receptor_mapping_status = "not_evaluated"
        ligand_mapping_status = "not_evaluated"
        receptor_mapping_details = "[]"
        ligand_mapping_details = "[]"
        if has_all_files and r_u and l_u and r_b and l_b:
            receptor_query_map = map_bound_to_unbound_chains(r_b, r_u, notes, "receptor")
            ligand_query_map = map_bound_to_unbound_chains(l_b, l_u, notes, "ligand")
            receptor_mapping_status = mapping_status_for_side(receptor_query_map)
            ligand_mapping_status = mapping_status_for_side(ligand_query_map)
            receptor_mapping_details = json.dumps(
                [mapping_to_dict(receptor_query_map[chain]) for chain in receptor_bound_chains],
                sort_keys=True,
                separators=(",", ":"),
            )
            ligand_mapping_details = json.dumps(
                [mapping_to_dict(ligand_query_map[chain]) for chain in ligand_bound_chains],
                sort_keys=True,
                separators=(",", ":"),
            )

            for receptor_chain in receptor_bound_chains:
                for ligand_chain in ligand_bound_chains:
                    contact_stats = compute_contact_stats(
                        r_b.atoms_by_chain.get(receptor_chain, []),
                        l_b.atoms_by_chain.get(ligand_chain, []),
                    )
                    if contact_stats.min_distance is not None and (
                        entity_min_distance is None or contact_stats.min_distance < entity_min_distance
                    ):
                        entity_min_distance = contact_stats.min_distance
                    for cutoff in CONTACT_CUTOFFS:
                        entity_contact_counts[cutoff] += contact_stats.contact_counts[cutoff]
                    receptor_interface_entity_5A.update(contact_stats.receptor_interface[5.0])
                    ligand_interface_entity_5A.update(contact_stats.ligand_interface[5.0])

                    query1_mapping = receptor_query_map.get(receptor_chain)
                    query2_mapping = ligand_query_map.get(ligand_chain)
                    query1_chain = query1_mapping.query_chain if query1_mapping else None
                    query2_chain = query2_mapping.query_chain if query2_mapping else None
                    query1_length_unbound = r_u.length(query1_chain) if query1_chain else None
                    query2_length_unbound = l_u.length(query2_chain) if query2_chain else None
                    query1_length_bound = r_b.length(receptor_chain)
                    query2_length_bound = l_b.length(ligand_chain)
                    total_chainpair_length = (
                        query1_length_unbound + query2_length_unbound
                        if query1_length_unbound is not None and query2_length_unbound is not None
                        else None
                    )
                    length_ratio = ratio_larger_over_smaller(query1_length_unbound, query2_length_unbound)
                    chainpair_notes: List[str] = []
                    exclusion_reasons: List[str] = []
                    if query1_mapping is None:
                        exclusion_reasons.append("query1_mapping_not_evaluated")
                    elif query1_mapping.mapping_status != "high_confidence":
                        exclusion_reasons.append(f"query1_mapping_{query1_mapping.mapping_status}:{query1_mapping.reason}")
                    if query2_mapping is None:
                        exclusion_reasons.append("query2_mapping_not_evaluated")
                    elif query2_mapping.mapping_status != "high_confidence":
                        exclusion_reasons.append(f"query2_mapping_{query2_mapping.mapping_status}:{query2_mapping.reason}")
                    if not query1_chain:
                        exclusion_reasons.append("query1_chain_empty")
                    if not query2_chain:
                        exclusion_reasons.append("query2_chain_empty")
                    if query1_length_unbound is None:
                        exclusion_reasons.append("query1_length_unbound_NA")
                    if query2_length_unbound is None:
                        exclusion_reasons.append("query2_length_unbound_NA")
                    chainpair_runnable = len(exclusion_reasons) == 0
                    if exclusion_reasons:
                        chainpair_notes.append("non-runnable: " + ";".join(exclusion_reasons))

                    chainpair_index = len(chainpairs) + 1
                    chainpair_id = f"BM5CP{chainpair_index:05d}"
                    interface_q1_5 = len(contact_stats.receptor_interface[5.0])
                    interface_q2_5 = len(contact_stats.ligand_interface[5.0])
                    chainpair = {
                        "chainpair_id": chainpair_id,
                        "entity_id": entity_id,
                        "benchmark_row_index": str(row.row_index),
                        "table_complex_id": row.table_complex_id,
                        "local_file_id": local_id or "",
                        "difficulty": row.difficulty,
                        "category_code": row.category_code,
                        "receptor_chain": receptor_chain,
                        "ligand_chain": ligand_chain,
                        "query1_role": "receptor",
                        "query2_role": "ligand",
                        "receptor_unbound_pdb": path_s(paths["r_u"]),
                        "ligand_unbound_pdb": path_s(paths["l_u"]),
                        "receptor_bound_pdb": path_s(paths["r_b"]),
                        "ligand_bound_pdb": path_s(paths["l_b"]),
                        "query1_chain": query1_chain or "",
                        "query2_chain": query2_chain or "",
                        "query1_length_unbound": na_if_none(query1_length_unbound),
                        "query2_length_unbound": na_if_none(query2_length_unbound),
                        "query1_length_bound": str(query1_length_bound),
                        "query2_length_bound": str(query2_length_bound),
                        "query1_chain_mapping_identity": format_mapping_value(query1_mapping.sequence_identity) if query1_mapping else "NA",
                        "query2_chain_mapping_identity": format_mapping_value(query2_mapping.sequence_identity) if query2_mapping else "NA",
                        "query1_chain_mapping_coverage_bound": format_mapping_value(query1_mapping.coverage_bound) if query1_mapping else "NA",
                        "query2_chain_mapping_coverage_bound": format_mapping_value(query2_mapping.coverage_bound) if query2_mapping else "NA",
                        "query1_chain_mapping_coverage_unbound": format_mapping_value(query1_mapping.coverage_unbound) if query1_mapping else "NA",
                        "query2_chain_mapping_coverage_unbound": format_mapping_value(query2_mapping.coverage_unbound) if query2_mapping else "NA",
                        "query1_chain_mapping_status": query1_mapping.mapping_status if query1_mapping else "not_evaluated",
                        "query2_chain_mapping_status": query2_mapping.mapping_status if query2_mapping else "not_evaluated",
                        "chainpair_runnable": bool_s(chainpair_runnable),
                        "chainpair_exclusion_reason": ";".join(exclusion_reasons),
                        "total_chainpair_length": na_if_none(total_chainpair_length),
                        "length_ratio": format_float(length_ratio),
                        "size_bin": size_bin(total_chainpair_length),
                        "length_balance_bin": length_balance_bin(length_ratio),
                        "native_contacting_chainpair_3p9A": bool_s(contact_stats.contact_counts[3.9] > 0),
                        "native_contacting_chainpair_5A": bool_s(contact_stats.contact_counts[5.0] > 0),
                        "native_contacting_chainpair_8A": bool_s(contact_stats.contact_counts[8.0] > 0),
                        "query1_interface_residue_count_3p9A": str(len(contact_stats.receptor_interface[3.9])),
                        "query2_interface_residue_count_3p9A": str(len(contact_stats.ligand_interface[3.9])),
                        "query1_interface_residue_count_5A": str(interface_q1_5),
                        "query2_interface_residue_count_5A": str(interface_q2_5),
                        "query1_interface_residue_count_8A": str(len(contact_stats.receptor_interface[8.0])),
                        "query2_interface_residue_count_8A": str(len(contact_stats.ligand_interface[8.0])),
                        "total_interface_residue_count_5A": str(interface_q1_5 + interface_q2_5),
                        "interface_fraction_query1_5A": format_float(
                            fraction(interface_q1_5, query1_length_bound), digits=4
                        ),
                        "interface_fraction_query2_5A": format_float(
                            fraction(interface_q2_5, query2_length_bound), digits=4
                        ),
                        "min_heavy_atom_distance_A": format_float(contact_stats.min_distance),
                        "heavy_atom_contact_count_3p9A": str(contact_stats.contact_counts[3.9]),
                        "heavy_atom_contact_count_5A": str(contact_stats.contact_counts[5.0]),
                        "heavy_atom_contact_count_8A": str(contact_stats.contact_counts[8.0]),
                        "direct_single_chain_case": bool_s(is_single_chain_pair),
                        "decomposed_multichain_case": bool_s(not is_single_chain_pair),
                        "noncontacting_chainpair_control": bool_s(contact_stats.contact_counts[5.0] == 0),
                        "recommended_for_first_singlechain_pilot": "false",
                        "recommended_for_first_multichain_pilot": "false",
                        "run_status": RUN_STATUS,
                        "label_status": LABEL_STATUS,
                        "leakage_status": LEAKAGE_STATUS,
                        "template_dataset_plan": TEMPLATE_DATASET_PLAN,
                        "use_blastpdb": USE_BLASTPDB,
                        "planned_output_dir": f"benchmark/bm5_ifragdi_runs/{chainpair_id}",
                        "notes": "; ".join(chainpair_notes),
                    }
                    chainpairs.append(chainpair)

        entity_row = {
            "entity_id": entity_id,
            "benchmark_row_index": str(row.row_index),
            "table_complex_id": row.table_complex_id,
            "local_file_id": local_id or "",
            "in_bm55_table": "true",
            "difficulty": row.difficulty,
            "category_code": row.category_code,
            "protein1_name": row.protein1_name,
            "protein2_name": row.protein2_name,
            "irmsd_A": row.irmsd_A,
            "dasa_A2": row.dasa_A2,
            "bm_version_introduced": row.bm_version_introduced,
            "receptor_unbound_pdb": path_s(paths.get("r_u")) if paths else "",
            "ligand_unbound_pdb": path_s(paths.get("l_u")) if paths else "",
            "receptor_bound_pdb": path_s(paths.get("r_b")) if paths else "",
            "ligand_bound_pdb": path_s(paths.get("l_b")) if paths else "",
            "has_all_required_files": bool_s(has_all_files),
            "missing_files": ",".join(missing_files),
            "receptor_unbound_chain_ids": list_s(receptor_unbound_chains),
            "ligand_unbound_chain_ids": list_s(ligand_unbound_chains),
            "receptor_bound_chain_ids": list_s(receptor_bound_chains),
            "ligand_bound_chain_ids": list_s(ligand_bound_chains),
            "receptor_unbound_chain_count": str(len(receptor_unbound_chains)),
            "ligand_unbound_chain_count": str(len(ligand_unbound_chains)),
            "receptor_bound_chain_count": str(len(receptor_bound_chains)),
            "ligand_bound_chain_count": str(len(ligand_bound_chains)),
            "receptor_chain_mapping_status": receptor_mapping_status,
            "ligand_chain_mapping_status": ligand_mapping_status,
            "receptor_chain_mapping_details": receptor_mapping_details,
            "ligand_chain_mapping_details": ligand_mapping_details,
            "is_single_chain_receptor": bool_s(is_single_chain_receptor),
            "is_single_chain_ligand": bool_s(is_single_chain_ligand),
            "is_single_chain_pair": bool_s(is_single_chain_pair),
            "is_multichain_entity": bool_s(is_multichain_entity),
            "chainpair_decomposition_needed": bool_s(has_all_files and not is_single_chain_pair),
            "n_chainpair_tasks": str(n_chainpair_tasks),
            "receptor_total_length_unbound": na_if_none(receptor_total_unbound),
            "ligand_total_length_unbound": na_if_none(ligand_total_unbound),
            "total_entity_length": na_if_none(total_entity_length),
            "size_bin": size_bin(total_entity_length),
            "native_contacting_entity_3p9A": (
                bool_s(entity_contact_counts[3.9] > 0) if has_all_files else "NA"
            ),
            "native_contacting_entity_5A": (
                bool_s(entity_contact_counts[5.0] > 0) if has_all_files else "NA"
            ),
            "native_contacting_entity_8A": (
                bool_s(entity_contact_counts[8.0] > 0) if has_all_files else "NA"
            ),
            "entity_interface_residue_count_receptor_5A": (
                str(len(receptor_interface_entity_5A)) if has_all_files else "NA"
            ),
            "entity_interface_residue_count_ligand_5A": (
                str(len(ligand_interface_entity_5A)) if has_all_files else "NA"
            ),
            "entity_total_interface_residue_count_5A": (
                str(len(receptor_interface_entity_5A) + len(ligand_interface_entity_5A))
                if has_all_files
                else "NA"
            ),
            "entity_min_heavy_atom_distance_A": format_float(entity_min_distance) if has_all_files else "NA",
            "entity_heavy_atom_contact_count_5A": (
                str(entity_contact_counts[5.0]) if has_all_files else "NA"
            ),
            "run_status": RUN_STATUS,
            "label_status": LABEL_STATUS,
            "leakage_status": LEAKAGE_STATUS,
            "template_dataset_plan": TEMPLATE_DATASET_PLAN,
            "use_blastpdb": USE_BLASTPDB,
            "notes": "; ".join(dict.fromkeys(notes)),
        }
        entities.append(entity_row)

    single_pilot, multichain_pilot, pilot_warnings = choose_pilots(chainpairs, entities)
    warnings.extend(pilot_warnings)
    warnings.extend(pilot_qc_warnings(single_pilot, "Single-chain"))
    warnings.extend(pilot_qc_warnings(multichain_pilot, "Multi-chain"))

    extra_local_ids = sorted(complete_ids - used_local_ids)
    if bound_chain_mismatch_rows:
        warnings.append(
            f"{len(bound_chain_mismatch_rows)} official rows have local bound chain IDs that differ from the Excel table annotation; see entity notes"
        )

    summary = {
        "total_rows_in_excel_table": len(table_rows),
        "total_entities_mapped_to_local_files": sum(1 for row in entities if row["has_all_required_files"] == "true"),
        "total_entities_missing_files": sum(1 for row in entities if row["has_all_required_files"] == "false"),
        "extra_local_ids_not_in_excel_table": extra_local_ids,
        "counts_by_difficulty": counter_dict(entities, "difficulty"),
        "counts_by_category_code": counter_dict(entities, "category_code"),
        "single_chain_entity_count": sum(1 for row in entities if row["is_single_chain_pair"] == "true"),
        "multichain_entity_count": sum(1 for row in entities if row["is_multichain_entity"] == "true"),
        "total_chainpair_tasks": len(chainpairs),
        "chainpair_runnable_count": sum(1 for row in chainpairs if row["chainpair_runnable"] == "true"),
        "chainpair_nonrunnable_count": sum(1 for row in chainpairs if row["chainpair_runnable"] != "true"),
        "nonrunnable_reasons": nonrunnable_reason_counts(chainpairs),
        "sequence_mapping_status_counts": sequence_mapping_status_counts(chainpairs),
        "singlechain_pilot_runnable_count": sum(1 for row in single_pilot if row["chainpair_runnable"] == "true"),
        "multichain_pilot_runnable_count": sum(1 for row in multichain_pilot if row["chainpair_runnable"] == "true"),
        "ambiguous_mapping_chainpair_count": sum(
            1 for row in chainpairs
            if row["query1_chain_mapping_status"] != "high_confidence"
            or row["query2_chain_mapping_status"] != "high_confidence"
        ),
        "na_query_length_chainpair_count": sum(1 for row in chainpairs if not has_non_na_query_lengths(row)),
        "noncontacting_controls_with_min_distance_count": sum(
            1 for row in chainpairs
            if row["native_contacting_chainpair_5A"] == "false" and row["min_heavy_atom_distance_A"] != "NA"
        ),
        "noncontacting_controls_missing_min_distance_count": sum(
            1 for row in chainpairs
            if row["native_contacting_chainpair_5A"] == "false" and row["min_heavy_atom_distance_A"] == "NA"
        ),
        "direct_single_chain_chainpair_count": sum(
            1 for row in chainpairs if row["direct_single_chain_case"] == "true"
        ),
        "decomposed_multichain_chainpair_count": sum(
            1 for row in chainpairs if row["decomposed_multichain_case"] == "true"
        ),
        "native_contacting_chainpair_counts_3p9A": count_bool(chainpairs, "native_contacting_chainpair_3p9A"),
        "native_contacting_chainpair_counts_5A": count_bool(chainpairs, "native_contacting_chainpair_5A"),
        "native_contacting_chainpair_counts_8A": count_bool(chainpairs, "native_contacting_chainpair_8A"),
        "noncontacting_chainpair_control_count_5A": sum(
            1 for row in chainpairs if row["noncontacting_chainpair_control"] == "true"
        ),
        "counts_by_size_bin_entity": counter_dict(entities, "size_bin"),
        "counts_by_size_bin_chainpair": counter_dict(chainpairs, "size_bin"),
        "counts_by_length_balance_bin_chainpair": counter_dict(chainpairs, "length_balance_bin"),
        "entity_length_ranges": {
            "total_entity_length": value_range(entities, "total_entity_length"),
            "receptor_total_length_unbound": value_range(entities, "receptor_total_length_unbound"),
            "ligand_total_length_unbound": value_range(entities, "ligand_total_length_unbound"),
        },
        "chainpair_length_ranges": {
            "total_chainpair_length": value_range(chainpairs, "total_chainpair_length"),
            "query1_length_unbound": value_range(chainpairs, "query1_length_unbound"),
            "query2_length_unbound": value_range(chainpairs, "query2_length_unbound"),
            "length_ratio": value_range(chainpairs, "length_ratio", as_float=True),
        },
        "interface_residue_count_ranges_5A": {
            "entity_total_interface_residue_count_5A": value_range(
                entities, "entity_total_interface_residue_count_5A"
            ),
            "chainpair_total_interface_residue_count_5A": value_range(
                chainpairs, "total_interface_residue_count_5A"
            ),
            "chainpair_query1_interface_residue_count_5A": value_range(
                chainpairs, "query1_interface_residue_count_5A"
            ),
            "chainpair_query2_interface_residue_count_5A": value_range(
                chainpairs, "query2_interface_residue_count_5A"
            ),
        },
        "heavy_atom_contact_count_ranges_5A": {
            "entity_heavy_atom_contact_count_5A": value_range(
                entities, "entity_heavy_atom_contact_count_5A"
            ),
            "chainpair_heavy_atom_contact_count_5A": value_range(
                chainpairs, "heavy_atom_contact_count_5A"
            ),
        },
        "pilot_singlechain_summary": pilot_summary(single_pilot),
        "pilot_multichain_summary": pilot_summary(multichain_pilot),
        "missing_or_ambiguous_local_ids": ambiguous,
        "synthetic_id_mapping_used": synthetic_used,
        "warnings": warnings,
    }

    write_tsv(args.entity_out, entities, ENTITY_COLUMNS)
    write_tsv(args.chainpair_out, chainpairs, CHAINPAIR_COLUMNS)
    write_tsv(args.pilot_single_out, single_pilot, CHAINPAIR_COLUMNS)
    write_tsv(args.pilot_multichain_out, multichain_pilot, CHAINPAIR_COLUMNS)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    write_manifest_report(
        args.summary.parent / "bm5_manifest_report.md",
        args.entity_out,
        args.chainpair_out,
        args.summary,
        args.pilot_single_out,
        args.pilot_multichain_out,
    )


if __name__ == "__main__":
    main()
