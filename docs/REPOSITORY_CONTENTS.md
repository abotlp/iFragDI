# Repository Contents

This document lists the main code and orchestration files in the current working snapshot.

## Core Pipeline

- `combine_ifrag_radi.py`: top-level combiner for iFrag, conservation, raDI, optional blastPDB anchors, residue scoring, and LightDock restraint output.
- `conservation.py`: builds the interaction-supported conservation prior. Current snapshot includes unverified `--allow-no-evidence` handling for empty conservation evidence.
- `radi_prepare.py`: builds paired alignments and support files for the raDI branch.
- `radi.py`: runs and parses raDI outputs into inter-chain score matrices.
- `ifrags.py`: classical iFrag template detection and matrix generation.
- `homolog_search.py`: shared homolog-search driver used upstream of conservation and raDI.
- `template_mmseqs.py`: MMseqs template-search helpers, resolved-hit parsing, and template resource defaults.
- `blastpdb.py`: optional experimental-PDB structural anchor discovery and transfer branch.
- `structure_features.py`: structure-derived surface and residue feature utilities.
- `run_lightdock_from_ifragdi.py`: helper for launching LightDock from iFragDI-generated restraint files.

## Benchmark Orchestration

- `benchmark/build_bm5_manifests.py`: builds BM5 chain-pair manifests and related benchmark planning tables.
- `benchmark/run_ifragdi_smoke_from_manifest.py`: creates or executes selected iFragDI smoke-test commands from a manifest.
- `benchmark/sbatch_ifragdi_smoke_2jobs.sh`: Slurm wrapper for the first two-job smoke test.
- `benchmark/sbatch_ifragdi_BM5CP00019_retry.sh`: Slurm wrapper for retrying only `BM5CP00019`.

## Tests

- `tests/test_conservation_allow_no_evidence.py`: lightweight local test for the no-evidence conservation branch.

## Requested File Presence

All files requested for this inventory are present in this working snapshot.
