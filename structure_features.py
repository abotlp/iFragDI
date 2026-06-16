#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "MSE": "M",
}


AA_HYDROPHOBICITY = {
    "A": 1.8,
    "R": -4.5,
    "N": -3.5,
    "D": -3.5,
    "C": 2.5,
    "Q": -3.5,
    "E": -3.5,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "L": 3.8,
    "K": -3.9,
    "M": 1.9,
    "F": 2.8,
    "P": -1.6,
    "S": -0.8,
    "T": -0.7,
    "W": -0.9,
    "Y": -1.3,
    "V": 4.2,
}


@dataclass(frozen=True)
class StructureRerankResult:
    final_scores: np.ndarray
    support_scores: np.ndarray
    confidence_component: np.ndarray
    local_mass_component: np.ndarray
    shape_component: np.ndarray
    hydrophobic_patch_component: np.ndarray
    confidence_source: str
    confidence_detected: bool


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


def _parse_residue_bfactors(path: Path, chain_id: str | None) -> np.ndarray | None:
    if chain_id is None or not path.exists():
        return None

    residues: list[dict[str, object]] = []
    residue_lookup: dict[tuple[str, str, str], int] = {}
    seen_atoms: set[tuple[str, str, str, str]] = set()

    with path.open() as handle:
        for raw in handle:
            if not raw.startswith(("ATOM", "HETATM")):
                continue
            if raw[21:22] != chain_id:
                continue
            altloc = raw[16:17]
            if altloc not in (" ", "A"):
                continue
            resname = raw[17:20].strip().upper()
            aa = AA3_TO_1.get(resname)
            if aa is None:
                continue
            atom_name = raw[12:16].strip()
            resseq = raw[22:26].strip()
            icode = raw[26:27].strip()
            atom_key = (resseq, icode, resname, atom_name)
            if atom_key in seen_atoms:
                continue
            seen_atoms.add(atom_key)
            try:
                bfactor = float(raw[60:66])
            except ValueError:
                continue
            resid = (resseq, icode, resname)
            idx = residue_lookup.get(resid)
            if idx is None:
                idx = len(residues)
                residue_lookup[resid] = idx
                residues.append(
                    {
                        "aa": aa,
                        "bfactors": [bfactor],
                        "ca_bfactor": bfactor if atom_name == "CA" else None,
                    }
                )
            else:
                entry = residues[idx]
                entry["bfactors"].append(bfactor)
                if atom_name == "CA":
                    entry["ca_bfactor"] = bfactor

    if not residues:
        return None

    values = np.zeros(len(residues), dtype=float)
    for i, entry in enumerate(residues):
        ca_bfactor = entry["ca_bfactor"]
        if ca_bfactor is not None:
            values[i] = float(ca_bfactor)
        else:
            values[i] = float(np.mean(np.asarray(entry["bfactors"], dtype=float)))
    return values


def infer_confidence_weights(
    pdb_path: Path | None,
    chain_id: str | None,
    expected_length: int,
    mode: str = "auto",
) -> tuple[np.ndarray, bool, str]:
    default = np.ones(expected_length, dtype=float)
    if pdb_path is None or expected_length <= 0 or mode == "off":
        return default, False, "off"

    b_factors = _parse_residue_bfactors(pdb_path, chain_id)
    if b_factors is None or b_factors.size != expected_length:
        return default, False, "unavailable"

    finite = np.isfinite(b_factors)
    if not np.all(finite):
        return default, False, "invalid_bfactor"

    if mode == "plddt_bfactor":
        confidence = np.clip((b_factors - 50.0) / 40.0, 0.0, 1.0)
        return confidence, True, "plddt_bfactor_forced"

    # Conservative auto-detection: treat B-factors as pLDDT only when they look
    # like AlphaFold-style confidence values rather than experimental B-factors.
    valid_range = np.all((b_factors >= 0.0) & (b_factors <= 100.0))
    median = float(np.median(b_factors))
    q75 = float(np.percentile(b_factors, 75))
    q25 = float(np.percentile(b_factors, 25))
    detected = valid_range and median >= 70.0 and q75 >= 82.5 and q25 >= 55.0
    if not detected:
        return default, False, "auto_not_detected"

    confidence = np.clip((b_factors - 50.0) / 40.0, 0.0, 1.0)
    return confidence, True, "plddt_bfactor_auto"


def _eligible_indices(length: int, eligible_mask: np.ndarray | None = None) -> np.ndarray:
    if eligible_mask is None:
        return np.arange(length, dtype=int)
    return np.flatnonzero(eligible_mask)


def compute_local_score_mass(
    scores: np.ndarray,
    coords: np.ndarray | None,
    radius: float,
    eligible_mask: np.ndarray | None = None,
) -> np.ndarray:
    out = np.zeros_like(scores, dtype=float)
    if coords is None or coords.shape[0] != scores.size or radius <= 0.0 or scores.size == 0:
        return out

    indices = _eligible_indices(scores.size, eligible_mask)
    if indices.size == 0:
        return out

    values = np.maximum(scores[indices], 0.0)
    distances = np.linalg.norm(coords[indices][:, None, :] - coords[indices][None, :, :], axis=2)
    local_mass = (distances <= radius).astype(float) @ values
    out[indices] = local_mass
    return out


def compute_surface_patch_density(
    coords: np.ndarray | None,
    length: int,
    radius: float,
    eligible_mask: np.ndarray | None = None,
) -> np.ndarray:
    out = np.zeros(length, dtype=float)
    if coords is None or coords.shape[0] != length or radius <= 0.0 or length == 0:
        return out

    indices = _eligible_indices(length, eligible_mask)
    if indices.size == 0:
        return out

    distances = np.linalg.norm(coords[indices][:, None, :] - coords[indices][None, :, :], axis=2)
    counts = np.sum(distances <= radius, axis=1).astype(float) - 1.0
    counts[counts < 0.0] = 0.0
    out[indices] = counts
    return out


def compute_hydrophobic_patch(
    sequence: str,
    coords: np.ndarray | None,
    radius: float,
    eligible_mask: np.ndarray | None = None,
) -> np.ndarray:
    length = len(sequence)
    out = np.zeros(length, dtype=float)
    if coords is None or coords.shape[0] != length or radius <= 0.0 or length == 0:
        return out

    indices = _eligible_indices(length, eligible_mask)
    if indices.size == 0:
        return out

    hydrophobicity = np.array([max(AA_HYDROPHOBICITY.get(aa, 0.0), 0.0) for aa in sequence], dtype=float)
    values = hydrophobicity[indices]
    distances = np.linalg.norm(coords[indices][:, None, :] - coords[indices][None, :, :], axis=2)
    mask = distances <= radius
    denom = np.sum(mask, axis=1).astype(float)
    denom[denom == 0.0] = 1.0
    out[indices] = (mask.astype(float) @ values) / denom
    return out


def rerank_with_structure_features(
    sequence: str,
    pdb_path: Path | None,
    chain_id: str | None,
    coords: np.ndarray | None,
    base_scores: np.ndarray,
    support_scores: np.ndarray,
    eligible_mask: np.ndarray | None = None,
    surface_weights: np.ndarray | None = None,
    confidence_mode: str = "auto",
    hydrophobic_weight: float = 0.02,
) -> StructureRerankResult:
    # Structural features are allowed to rerank residues, but they should not
    # manufacture interface support on their own. The biological support score
    # is passed through unchanged so downstream eligibility stays tied to the
    # iFrag/conservation/raDI branches.
    if coords is None or coords.shape[0] != base_scores.size or base_scores.size == 0:
        empty = np.zeros_like(base_scores, dtype=float)
        return StructureRerankResult(
            final_scores=base_scores.astype(float, copy=True),
            support_scores=support_scores.astype(float, copy=True),
            confidence_component=np.ones_like(base_scores, dtype=float),
            local_mass_component=empty,
            shape_component=empty,
            hydrophobic_patch_component=empty,
            confidence_source="off",
            confidence_detected=False,
        )

    confidence_weights, detected, source = infer_confidence_weights(
        pdb_path,
        chain_id,
        expected_length=base_scores.size,
        mode=confidence_mode,
    )

    local_mass = normalize_positive_vector(
        compute_local_score_mass(base_scores, coords, radius=10.0, eligible_mask=eligible_mask)
    )
    shape_density = normalize_positive_vector(
        compute_surface_patch_density(coords, base_scores.size, radius=10.0, eligible_mask=eligible_mask)
    )
    hydrophobic_patch = normalize_positive_vector(
        compute_hydrophobic_patch(sequence, coords, radius=8.0, eligible_mask=eligible_mask)
    )

    confidence_multiplier = 0.6 + 0.4 * confidence_weights
    rerank_raw = np.maximum(base_scores, 0.0) * confidence_multiplier
    rerank_raw += 0.35 * local_mass
    rerank_raw += 0.15 * shape_density
    if hydrophobic_weight > 0.0:
        rerank_raw += float(hydrophobic_weight) * hydrophobic_patch
    if surface_weights is not None and surface_weights.shape == base_scores.shape:
        rerank_raw = rerank_raw * np.clip(surface_weights.astype(float), 0.0, 1.0)

    support_out = support_scores.astype(float, copy=True)

    if eligible_mask is not None:
        rerank_raw = rerank_raw.astype(float, copy=True)
        rerank_raw[~eligible_mask] = 0.0

    final_scores = normalize_positive_vector(rerank_raw)
    return StructureRerankResult(
        final_scores=final_scores,
        support_scores=support_out,
        confidence_component=confidence_weights,
        local_mass_component=local_mass,
        shape_component=shape_density,
        hydrophobic_patch_component=hydrophobic_patch,
        confidence_source=source,
        confidence_detected=detected,
    )
