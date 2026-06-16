# iFragDI Pipeline Guide

This file explains the project end to end:

1. what the pipeline is trying to do
2. how `iFrag`, `conservation`, and `raDI` are combined
3. what changes when PDB structures are available
4. how docking restraints are prepared
5. which commands to run

This replaces the older split notes about `iFrag` and docking selection.

Important implementation note:

- the current stable/default runner is [`combine_ifrag_radi.py`](/home/patricia/cluster_shiva/iFragDI/combine_ifrag_radi.py)
- the current experimental structure-aware runner is [`combine_ifrag_radi_structaware.py`](/home/patricia/cluster_shiva/iFragDI/combine_ifrag_radi_structaware.py)

The BM5 benchmark helper currently calls the stable runner, not the experimental one.

Important data/runtime note:

- `iFrag` now supports two UniProt-based template universes:
  - curated core: `intact_biogrid`
  - STRING-expanded: `intact_biogrid_string`
- the stable runner now defaults to the STRING-expanded dataset:
  - `intact_biogrid_string`
- `iFrag` searches the PPI-derived template BLAST DB built from the selected template dataset
- `conservation.py` and `radi_prepare.py` now share one template-backed MMseqs search by default:
  - MMseqs search DB: `data/db/mmseqs_templates_<dataset>/templates_db*`
  - sequence recovery FASTA: `data/interaction_templates/<dataset>/templates.fasta`
  - taxids: `data/interaction_templates/<dataset>/proteins.final.tsv`
- the default shared homolog-search mode is:
  - `template_iterative`

## 0. Current Status

What is already in good shape:

- `ifrags.py` is the most paper-faithful branch and now supports both datasets:
  - `intact_biogrid`
  - `intact_biogrid_string`
- the stable/default combined runner is still:
  - [`combine_ifrag_radi.py`](/home/patricia/cluster_shiva/iFragDI/combine_ifrag_radi.py)
- the experimental structure-aware runner is still:
  - [`combine_ifrag_radi_structaware.py`](/home/patricia/cluster_shiva/iFragDI/combine_ifrag_radi_structaware.py)

What to benchmark first:

- benchmark the stable/default runner first
- treat the structure-aware runner as a later comparison, not the benchmark baseline

What biological level each branch really predicts:

- `iFrag`: template-derived interface-region evidence
- `conservation`: broad per-chain patch prior
- `raDI`: sparse inter-chain anchor pairs when the paired MSA is good enough
- `blastPDB`: sparse structural-template anchor pairs from experimental PDB biological assemblies

Most important interpretation rule:

- the final combined 2D matrix is a **diagnostic residue-priority matrix**
- it is **not** a literal predicted contact map unless you are looking specifically at the retained `raDI` and/or `blastPDB` anchor pairs

## 0.1 Pipeline Schema

```text
query1/query2 FASTA or PDB
        |
        +--------------------+-----------------------------+
        |                    |                             |
        v                    v                             v
     iFrag              homolog search                 blastPDB
        |                    |                             |
        |            +-------+-------+                     |
        |            |               |                     |
        |            v               v                     |
        |       conservation      radi_prepare             |
        |            |               |                     |
        |            |               v                     |
        |            |             paired MSA              |
        |            |               |                     |
        |            |               v                     |
        |            +------------> raDI                   |
        |                                                  |
        +------------------------+-------------------------+
                                 |
                                 v
                       residue-first combination
                                 |
                                 v
                      per-chain residue scores
                                 |
                                 v
                 surface filtering + compact cluster selection
                                 |
                                 v
                   strict / loose docking residue sets
                                 |
                                 v
                       LightDock restraint exports
```

## 1. Project Scope

The practical goal of this project is:

- input: two proteins
- output: which residues are most likely to belong to the interface
- when structures are available: convert those residues into docking restraints
- final benchmark: test whether guided docking is better than blind docking

So the method is not mainly a contact-map predictor.

It is:

- a residue-level interface predictor
- plus a docking-guidance pipeline

The most important use case is:

- two monomer structures in PDB format
- predict interface residues
- run docking with those restraints

FASTA-only mode still exists, but it is secondary.

## 2. Input Modes

### A. FASTA-only mode

Use this when the user gives:

- `query1.fa`
- `query2.fa`

Main output:

- per-chain residue scores
- a diagnostic 2D heatmap
- optional sparse `raDI` anchor pairs

Important:

- the final 2D heatmap is **not** a literal contact map
- it is a diagnostic projection of per-residue scores

### B. PDB mode

Use this when the user gives:

- `query1.pdb`
- `query2.pdb`

Main output:

- per-chain residue scores
- surface-filtered interface candidates
- strict docking residues
- loose docking residues
- LightDock-ready restraint files

This is the main mode for the future webserver.

## 3. The Four Evidence Branches

### A. `iFrag`

Biological role:

- template-derived interaction-region prior

Classical idea:

- if query1 matches a fragment from template protein A
- and query2 matches a fragment from template protein B
- and A/B are known to interact
- then the aligned query regions are plausible interaction regions

Important:

- `iFrag` does **not** require template complex structures
- `iFrag` does **not** say that residue `i` contacts residue `j` structurally
- it says the query fragments resemble fragments from a known interacting template pair

Current data inputs:

- template dataset selector:
  - `--template-dataset intact_biogrid`
  - `--template-dataset intact_biogrid_string`
- default curated pair/provenance file:
  - `data/datasets/intact_biogrid/intact_biogrid.final.tsv`
- default curated sequence search DB:
  - `data/db/blast_templates_intact_biogrid/templates_db*`
- default curated template FASTA:
  - `data/interaction_templates/intact_biogrid/templates.fasta`
- STRING-expanded alternatives:
  - `data/datasets/intact_biogrid_string/intact_biogrid_string.final.tsv`
  - `data/db/blast_templates_intact_biogrid_string/templates_db*`
  - `data/interaction_templates/intact_biogrid_string/templates.fasta`

### B. `conservation`

Biological role:

- broad interface-patch prior

Idea:

- positions that stay conserved across interacting homologs are more likely to be interface-relevant

Important:

- in this pipeline, conservation is used mainly as a **per-chain residue signal**
- it is not treated as a direct residue-pair predictor

Current data inputs:

- homolog-side pair universe:
  - selected at runtime with `--radi-pair-dataset`
  - `intact_biogrid`
  - `intact_biogrid_string`
- shared template-backed homolog search:
  - `data/db/mmseqs_templates_<dataset>/templates_db*`
- sequence recovery FASTA:
  - `data/interaction_templates/<dataset>/templates.fasta`
- taxids:
  - `data/interaction_templates/<dataset>/proteins.final.tsv`

Conservation stays per-chain:

- it uses the shared resolved homolog hits
- builds one per-chain MSA for query1 and one for query2
- computes per-residue conservation/alignment coverage
- contributes a broad patch prior rather than explicit residue-pair anchors

### C. `raDI`

Biological role:

- sparse coevolutionary anchor signal

Idea:

- if a paired homolog MSA can be built from interaction-supported template hits
- then `raDI` can highlight a small number of residue-pair anchors

Important:

- `raDI` is the most specific branch when it works
- but it often fails because homolog-pair depth is too low
- therefore `raDI` is used as a **bonus anchor branch**, not the sole predictor

Current data inputs:

- homolog-side pair universe:
  - selected at runtime with `--radi-pair-dataset`
  - `intact_biogrid`
  - `intact_biogrid_string`
- shared template-backed homolog search:
  - `data/db/mmseqs_templates_<dataset>/templates_db*`
- sequence recovery FASTA:
  - `data/interaction_templates/<dataset>/templates.fasta`
- taxids:
  - `data/interaction_templates/<dataset>/proteins.final.tsv`

raDI stays paired:

- it reuses the same resolved homolog pool as conservation
- builds paired interolog rows from interaction-supported template hits
- reuses the prepared per-chain MSAs when conservation ran first
- then concatenates the paired trimmed rows into the final raDI input alignment

### D. `blastPDB`

Biological role:

- sparse structural-template anchor signal

Idea:

- if both query proteins match proteins that occur together in an experimental PDB biological assembly
- and that assembly contains residue-level inter-chain contacts
- then those contacts can be transferred to query residue pairs through the alignments

Important:

- `blastPDB` is a structural anchor branch, not a broad patch prior
- the current implementation is a hybrid cached runtime branch:
  - remote candidate discovery against experimental PDB sequence resources
  - local assembly download/cache
  - local contact extraction
  - local alignment-based contact transfer

Current data inputs:

- query sequences resolved by the stable runner
- experimental PDB biological assemblies discovered at runtime
- local cache under:
  - `data/cache/blastpdb`

## 4. Current Runtime Truth

Right now the runtime is:

- `iFrag` can choose between two base-accession PPI universes:
  - `intact_biogrid`
  - `intact_biogrid_string`
- `conservation.py` and `radi_prepare.py` can choose between the same two homolog-side pair universes with `--radi-pair-dataset`
- `iFrag` has a template sequence FASTA/BLAST DB per dataset
- `conservation.py` and `radi_prepare.py` now share one template-backed MMseqs homolog-search DB per dataset
- `blastPDB` is separate from those local datasets and uses experimental PDB biological assemblies as structural templates

So the current codebase now has a consistent two-dataset model for all three branches:
- `intact_biogrid`
- `intact_biogrid_string`

Important separation of roles:

- template universe for `iFrag`:
  - interacting template pairs + template FASTA/BLAST DB
- shared homolog search for `conservation.py` / `radi_prepare.py`:
  - MMseqs directly against the selected template universe
  - sequence recovery from the same dataset `templates.fasta`
  - taxids from `proteins.final.tsv`
- structural-template universe for `blastPDB`:
  - experimental PDB biological assemblies discovered and cached at runtime

## 5. How Classical `iFrag` Is Scored

The standalone `iFrag` matrix from [`ifrags.py`](/home/patricia/cluster_shiva/iFragDI/ifrags.py) is the paper-faithful part.

Evidence hierarchy:

1. one BLAST HSP = one local fragment hit
2. one left-HSP + one right-HSP = one fragment-pair support event
3. all fragment pairs from one retained template interaction are merged
4. each retained template interaction contributes at most one vote to each supported cell

So the classical `iFrag` matrix is:

`ifrag(i,j) = number_of_retained_template_interactions_supporting_(i,j) / N`

where:

- `N` = number of retained nonredundant template interactions

This is the score written by:

- [`ifrag_matrix.tsv`](/home/patricia/cluster_shiva/iFragDI/ifrags.py#L700)

This matrix is broad and blocky by design. That is expected classical behavior.

## 6. How The Stable Combined Predictor Works

The main combine step in [`combine_ifrag_radi.py`](/home/patricia/cluster_shiva/iFragDI/combine_ifrag_radi.py#L2132) turns the active evidence branches into **per-residue interface scores**.

### Step 1. Normalize branch evidence

- `iFrag` matrix: nonzero percentile normalization
- `conservation` matrix/profile: nonzero percentile normalization for the matrix; per-chain profile used directly
- `raDI` matrix: keep top `N` nonzero pairs, divide by the top DI score
- `blastPDB` matrix: keep top `N` nonzero pairs, divide by the top transferred structural score

This makes the branches comparable without pretending they use the same raw scale.

### Step 2. Convert pairwise evidence into residue-level evidence

For `iFrag`, `raDI`, and `blastPDB`, residue strength is computed from the strongest row/column cells:

`top1 + 0.5*(top2 + ... + topk)`

with `k = --top-k`, default `3`.

Interpretation:

- one very strong supported cell matters most
- several extra strong cells help, but with lower weight

For `conservation`, the per-chain profile already exists, so it is used directly.

### Step 3. Build local seed regions

For each chain:

- choose top nonredundant residues
- grow a short local band around each seed
- weight nearby positions with triangular decay

This expresses the assumption that interfaces form local patches rather than isolated single residues.

### Step 4. Build the template patch

The patch score is built from:

- conservation seed-region support
- reliability-scaled `iFrag` seed-region support
- an overlap bonus

Schematic rule:

`patch = normalize_positive(conservation_region + ifrag_weight*ifrag_region + 0.5*overlap)`

`iFrag` reliability is important:

- diffuse/blocky `iFrag` maps are downweighted
- compact `iFrag` maps contribute more

This reliability term is a docking-oriented heuristic:

- it makes sense if the goal is to localize a compact interface patch
- it should not be interpreted as a universal biological truth about template evidence

When conservation is disabled, the patch falls back to the `iFrag`-guided template prior.

### Step 5. Add `raDI` and/or `blastPDB` as anchor bonuses

If `raDI` is available and trusted:

- it does not redefine the whole patch
- it adds a sparse bonus inside or near the patch

If `blastPDB` is available and trusted:

- it also does not redefine the whole patch
- it adds a sparse structural-template bonus inside or near the patch

Schematic rule:

`radi_bonus = normalize_positive(radi_component * (0.25 + 0.75*patch))`
`blastpdb_bonus = normalize_positive(blastpdb_component * (0.25 + 0.75*patch))`

Final residue score:

- without `raDI` and without `blastPDB`:
  - `final_score = normalize_positive(patch)`
- with `raDI` only:
  - `final_score = normalize_positive(patch + radi_weight*radi_bonus)`
- with `blastPDB` only:
  - `final_score = normalize_positive(patch + blastpdb_weight*blastpdb_bonus)`
- with both:
  - `final_score = normalize_positive(patch + radi_weight*radi_bonus + blastpdb_weight*blastpdb_bonus)`

Current `raDI` trust logic in the stable runner:

- `radi_prepare.py` can build a paired MSA in:
  - `interaction_only`
  - `species_besthit`
  - `auto`
- the stable combine runner only uses `raDI` in scoring if:
  - `radi_prepare.py` succeeded
  - the paired MSA produced at least 2 homolog rows
  - the paired-row depth is not below `--radi-min-trusted-paired-rows`

So biologically:

- `raDI` is treated as a sparse anchor branch
- low-depth `raDI` is explicitly ignored rather than silently blended into the final score

So in `ifrag_blastpdb` mode, the final score is **not** "raw iFrag matrix + raw blastPDB matrix".

It is:

1. build the template patch prior mainly from `iFrag`
2. convert retained blastPDB anchors into a per-residue anchor component
3. add a patch-guided blastPDB bonus
4. normalize the final per-chain residue scores

### Step 6. Build the final diagnostic heatmap

The final 2D heatmap is:

`outer(query1_final_scores, query2_final_scores)`

This is why it should be interpreted as a **diagnostic residue-priority heatmap**, not a literal predicted contact map.

Real pairwise evidence is shown separately through the retained `raDI` and/or `blastPDB` anchor pairs.

This distinction matters for benchmarking:

- benchmark residue scores / docking residues as the main product
- benchmark `raDI` top pairs separately if you want to evaluate true pairwise anchors

## 7. What To Present To The User

### If the user gives only FASTAs

Present:

- per-residue interface score for chain 1
- per-residue interface score for chain 2
- optional top residue tables
- optional diagnostic pair-priority heatmap
- optional `raDI` top anchor pairs

Best wording:

- "interface-likelihood residues"
- "predicted interface patch"

Avoid saying:

- "predicted contact map"

### If the user gives PDBs

Present:

- colored structures
- residue score tables
- strict docking residues
- loose docking residues
- LightDock restraint files

This is the main intended product.

## 8. What Changes When Structures Are Available

There are two relevant code paths.

### A. Stable/default PDB path

[`combine_ifrag_radi.py`](/home/patricia/cluster_shiva/iFragDI/combine_ifrag_radi.py) is the current stable runner used by the benchmark helper.

Its structural role is deliberately modest:

- compute SASA / surface eligibility
- suppress clearly buried residues
- choose a compact supported cluster
- write strict/loose docking sets and LightDock restraints

It does **not** currently use:

- explicit per-chain structure provenance
- pLDDT-aware confidence reranking
- soft RSA weighting
- the newer monomer-structure reranker in [`structure_features.py`](/home/patricia/cluster_shiva/iFragDI/structure_features.py)

So the stable runner is:

- residue-first biological scoring
- plus hard surface filtering
- plus compact cluster-based docking selection

### B. Experimental structure-aware path

[`combine_ifrag_radi_structaware.py`](/home/patricia/cluster_shiva/iFragDI/combine_ifrag_radi_structaware.py) is the newer experimental runner.

This path keeps the same biological core, but adds:

- hard surface eligibility plus soft RSA weighting
- explicit per-chain structure source:
  - `experimental`
  - `alphafold_like`
  - `auto`
- optional pLDDT-from-B-factor confidence handling for AlphaFold-like inputs
- monomer-structure reranking through [`structure_features.py`](/home/patricia/cluster_shiva/iFragDI/structure_features.py)
- anchor-aware strict residue selection

Important:

- in the experimental runner, structure is used to **rerank** biologically supported residues
- it is **not** supposed to create biological support on its own

### C. Stable runner behavior in PDB mode

When PDB structures exist and you run the stable/default path, the scoring branch is the same at first, and then a simpler docking-oriented structural filter is applied.

#### SASA / surface filtering

The code computes exposed residues and suppresses buried ones during final scoring and docking selection.

This means:

- buried residues are not allowed to dominate the docking set
- the predicted patch is restricted to the molecular surface

#### Cluster-based selection

The pipeline does **not** simply take the top residues globally.

Instead, it:

1. builds a direct-support signal from `conservation + iFrag + raDI`
2. finds the best compact cluster on each chain
3. selects residues from that cluster only

This avoids:

- scattered restraints
- broad noisy sets
- large percentile-based patches

#### One-sided selection is allowed

If one chain looks weak:

- it can end up with zero docking residues

This is intentional. Docking does not need equally strong restraints on both chains.

## 9. Docking Residue Selection

The docking selector works per chain.

### Direct-support rule

Active residues must come from residues with real branch support:

- conservation support
- `iFrag` support
- `raDI` support when available

Passive residues are only the shell around active residues.

The selector does **not** fill quotas with zero-support residues.

More explicitly, the stable runner does this per chain:

1. build a direct-support vector from:
   - conservation component
   - reliability-weighted `iFrag` component
   - `raDI` component when trusted
2. choose the best compact supported cluster, using geometry when PDB coordinates exist
3. reject the cluster entirely if it is too weak:
   - cluster support fraction `< 0.10` and max cluster residue score `< 0.75`
4. choose active residues from the cluster by score fraction:
   - strict: `>= 0.75 * max_cluster_score`
   - loose: `>= 0.45 * max_cluster_score`
5. choose passive residues only from the shell around the actives:
   - strict: `>= 0.35 * max_cluster_score`
   - loose: `>= 0.15 * max_cluster_score`
6. cap the final residue counts with the CLI limits

Important:

- the selector is cluster-first and threshold-first
- the residue-count arguments are maximum caps, not quota-filling guarantees
- one weak chain is allowed to end up with zero selected residues

### Strict set

Purpose:

- first docking run
- highest-confidence restraints

Defaults:

- `--strict-active-residues-per-chain 4`
- `--strict-passive-residues-per-chain 4`

Outputs:

- `query1_docking_residues.tsv`
- `query2_docking_residues.tsv`
- `query1_docking_residues.strict.tsv`
- `query2_docking_residues.strict.tsv`

### Loose set

Purpose:

- second docking run
- broader shell around the same hotspot

Defaults:

- `--active-residues-per-chain 8`
- `--passive-residues-per-chain 8`

These are now enforced as maximum counts after thresholding and shell selection.

Outputs:

- `query1_docking_residues.loose.tsv`
- `query2_docking_residues.loose.tsv`

## 10. LightDock Restraint Files

The pipeline writes LightDock-ready restraint files.

Recommended order:

1. `lightdock_restraints.strict_active.list`
2. `lightdock_restraints.strict.list`
3. `lightdock_restraints.loose.list`

One-sided files:

- `lightdock_restraints.query1_only.strict.list`
- `lightdock_restraints.query2_only.strict.list`

Current encoding:

- `query1 = receptor (R)`
- `query2 = ligand (L)`

Important practical note:

- receptor-only restraints are the cleanest one-sided LightDock mode
- if only query2 is trusted, it can still be worth benchmarking a swapped receptor/ligand run

Current recommendation:

- treat `query1_docking_residues.strict.tsv` / `query2_docking_residues.strict.tsv` as the primary high-confidence export
- treat the loose sets as a second-run, higher-recall docking configuration
- treat `docking_candidate_pairs.tsv` as diagnostic pair priorities, not hard residue-pair restraints

## 11. Main Output Files

Most important files from the stable runner [`combine_ifrag_radi.py`](/home/patricia/cluster_shiva/iFragDI/combine_ifrag_radi.py):

- `query1_residue_scores.tsv`
- `query2_residue_scores.tsv`
- `query1_branch_scores.tsv`
- `query2_branch_scores.tsv`
- `query1_docking_residues.tsv`
- `query2_docking_residues.tsv`
- `query1_docking_residues.loose.tsv`
- `query2_docking_residues.loose.tsv`
- `final_score_heatmap.png`
- `template_support_matrix.tsv`
- `anchor_pair_matrix.tsv`
- `template_support_with_anchor_overlay.png`
- `ifrag_with_blastpdb_overlay.png`
- `consensus_summary.json`

How to interpret them:

- `query*_residue_scores.tsv`:
  final per-residue ranking
- `query*_branch_scores.tsv`:
  where the residue score came from
- `query*_docking_residues*.tsv`:
  residues intended for docking
- `final_score_heatmap.png`:
  diagnostic residue-priority projection
- `template_support_matrix.tsv`:
  diagnostic outer-product view of the template-derived patch prior inside the selected patch
- `anchor_pair_matrix.tsv`:
  sparse retained anchor pairs from `raDI` and/or `blastPDB`
- `template_support_with_anchor_overlay.png`:
  diagnostic visualization of template patch + retained anchor pairs
- `ifrag_with_blastpdb_overlay.png`:
  diagnostic pre-combination view of the full raw `iFrag` matrix with the full `blastPDB` heatmap overlaid using a second colormap

The experimental runner writes the same main product types, but may also include extra structure-aware components in the branch score tables and summary.

## 12. Benchmark-Side Sanity Check

For BM5 cases, before docking, check whether the predicted patch is near the native interface:

```bash
python3 bm5_interface_proximity.py \
  --combine-out-dir benchmark/pilot5/1S1Q_ifrag_conservation_radi_template \
  --bound-query1-pdb benchmark/benchmark5.5/structures/1S1Q_r_b.pdb \
  --bound-query2-pdb benchmark/benchmark5.5/structures/1S1Q_l_b.pdb \
  --top-n 30
```

This writes:

- `native_interface_proximity.summary.json`
- `native_interface_proximity.query1.tsv`
- `native_interface_proximity.query2.tsv`
- `native_interface_proximity.png` when plotting libraries are available

Use this to check whether the selected patch is:

- near the real interface
- borderline
- or likely too noisy for docking

## 13. Example Commands

These examples assume the standard local dataset layout. The combined runner now defaults to the STRING-expanded template universe plus shared template-backed MMseqs search for conservation and `raDI`, so no extra dataset flags are needed for the normal setup.

The stable/default shared homolog-search mode is:

- `template_iterative`

For a faster smoke test, you can switch the shared search to:

- `--homolog-search-mode template_single_pass`

### A. FASTA-only interface prediction

```bash
python3 combine_ifrag_radi.py \
  --query1-fasta path/to/query1.fa \
  --query2-fasta path/to/query2.fa \
  --combine-mode ifrag_conservation_radi \
  --threads 32 \
  --out-dir out/fasta_prediction
```

### B. PDB-guided interface prediction and docking preparation

```bash
python3 combine_ifrag_radi.py \
  --query1-pdb path/to/query1.pdb \
  --query2-pdb path/to/query2.pdb \
  --combine-mode ifrag_conservation_radi \
  --threads 32 \
  --out-dir out/pdb_prediction
```

### C. BM5 pilot case

```bash
python3 combine_ifrag_radi.py \
  --query1-pdb benchmark/benchmark5.5/structures/1S1Q_r_u.pdb \
  --query2-pdb benchmark/benchmark5.5/structures/1S1Q_l_u.pdb \
  --combine-mode ifrag_conservation_radi \
  --threads 32 \
  --out-dir benchmark/pilot5/1S1Q_ifrag_conservation_radi_template
```

### D. Experimental structure-aware PDB run

```bash
python3 combine_ifrag_radi_structaware.py \
  --query1-pdb path/to/query1.pdb \
  --query2-pdb path/to/query2.pdb \
  --query1-structure-source experimental \
  --query2-structure-source alphafold_like \
  --combine-mode ifrag_conservation_radi \
  --threads 32 \
  --out-dir out/pdb_prediction_structaware
```

### E. Faster smoke test with template single-pass shared search

```bash
python3 combine_ifrag_radi.py \
  --query1-fasta path/to/query1.fa \
  --query2-fasta path/to/query2.fa \
  --combine-mode ifrag_conservation_radi \
  --homolog-search-mode template_single_pass \
  --threads 32 \
  --out-dir out/fasta_prediction_template_single_pass
```

### F. `iFrag + blastPDB` without conservation or `raDI`

```bash
python3 combine_ifrag_radi.py \
  --query1-pdb path/to/query1.pdb \
  --query2-pdb path/to/query2.pdb \
  --combine-mode ifrag_blastpdb \
  --use-blastpdb \
  --threads 32 \
  --out-dir out/pdb_prediction_ifrag_blastpdb
```

## 13. Recommended Interpretation

The clean conceptual split is:

### Prediction layer

- combine `iFrag + conservation + raDI`
- choose one shared PPI universe for a run:
  - `intact_biogrid`
  - `intact_biogrid_string`
- produce final per-residue interface scores

### Docking layer, stable path

- apply structure-aware filtering
- choose compact surface-supported residues
- write strict/loose docking restraints

### Docking layer, experimental path

- keep the same biological support
- apply soft-RSA / compactness / confidence-aware reranking
- choose strict/loose docking restraints from the supported compact cluster

That is the best way to think about the current method.

## 14. Current Template Resources

`iFrag` currently supports two template datasets:

- `intact_biogrid`
  - pairs: `data/datasets/intact_biogrid/intact_biogrid.final.tsv`
  - fasta: `data/interaction_templates/intact_biogrid/templates.fasta`
  - blast db: `data/db/blast_templates_intact_biogrid/templates_db`
- `intact_biogrid_string`
  - pairs: `data/datasets/intact_biogrid_string/intact_biogrid_string.final.tsv`
  - fasta: `data/interaction_templates/intact_biogrid_string/templates.fasta`
  - blast db: `data/db/blast_templates_intact_biogrid_string/templates_db`

The dataset can be selected explicitly:

```bash
--template-dataset intact_biogrid
--template-dataset intact_biogrid_string
```

Optional filtering by detection-method substring is supported:

```bash
--pair-method-substring "two hybrid"
--pair-method-substring "affinity capture"
```

## 15. Exact Build Commands Used

All commands below were run from:

```bash
cd /users/sbi/patricia/iFragDI
```

### IntAct + BioGRID dataset

```bash
sbatch data/scripts_dataset/run_build_intact_biogrid.sbatch
```

### IntAct + BioGRID FASTA + BLAST DB

```bash
sbatch data/scripts_dataset/run_build_intact_biogrid_blastdb.sbatch
```

### IntAct + BioGRID + STRING dataset

Recommended production run used here:

```bash
MIN_STRING_COMBINED_SCORE=700 MIN_STRING_EXPERIMENTAL=0 sbatch data/scripts_dataset/run_build_intact_biogrid_string.sbatch
```

### IntAct + BioGRID + STRING FASTA + BLAST DB

```bash
sbatch data/scripts_dataset/run_build_intact_biogrid_string_blastdb.sbatch
```

## 16. Exact iFrag FASTA Benchmark Commands

Curated `IntAct + BioGRID` benchmark run:

```bash
cd /users/sbi/patricia/iFragDI
module load BLAST/2.12.0-Linux_x86_64
python3 ifrags.py \
  --query1 benchmark/SEQUENCE1_QUERY.fa \
  --query2 benchmark/SEQUENCE2_QUERY.fa \
  --template-dataset intact_biogrid \
  --blast-bin /soft/system/software/BLAST/2.12.0-Linux_x86_64/bin/blastp \
  --out-dir benchmark/ifrag_fasta_intact_biogrid_paper \
  --threads 8 \
  --heatmap
```

STRING-expanded benchmark run:

```bash
cd /users/sbi/patricia/iFragDI
module load BLAST/2.12.0-Linux_x86_64
python3 ifrags.py \
  --query1 benchmark/SEQUENCE1_QUERY.fa \
  --query2 benchmark/SEQUENCE2_QUERY.fa \
  --template-dataset intact_biogrid_string \
  --blast-bin /soft/system/software/BLAST/2.12.0-Linux_x86_64/bin/blastp \
  --out-dir benchmark/ifrag_fasta_intact_biogrid_string_paper \
  --threads 8 \
  --heatmap
```

## 17. Conservation and `raDI` Search Design

The current stable design is:

1. use one shared MMseqs search for query1 and query2
2. search directly against the selected template universe:
   - `intact_biogrid`
   - `intact_biogrid_string`
3. recover full sequences from the same dataset `templates.fasta`
4. recover taxids from the same dataset `proteins.final.tsv`
5. reuse the resolved homolog-hit TSVs for both `conservation.py` and `radi_prepare.py`

Default shared search mode:

- `template_iterative`

This is the paper-like default:

- MMseqs sensitivity `-s 7.5`
- `--max-seq-id 1.0`
- `--num-iterations 4`
- one iterative MMseqs search on the selected template DB
- no explicit MMseqs `-e` cutoff unless you pass `--radi-evalue`

Faster smoke-test mode:

- `template_single_pass`

The active pipeline no longer uses the older UniRef-backed MMseqs path.

### How the shared template MMseqs search works

The template-backed MMseqs path used by `homolog_search.py`, `conservation.py`, and `radi_prepare.py` is simpler than the older UniRef path because the search DB is already in the same accession universe used by the interaction-template datasets.

The shared search does this:

1. run MMseqs for query1 against `data/db/mmseqs_templates_<dataset>/templates_db*`
2. run MMseqs for query2 against the same DB
3. parse the direct accession-level hits
4. attach taxids from `data/interaction_templates/<dataset>/proteins.final.tsv`
5. write resolved TSVs that both downstream branches reuse

The resolved TSV contains accession-level rows such as:

- `accession`
- `sequence_id`
- `taxid`
- `evalue`
- `bitscore`
- `row`

### How the shared resolved hits diverge downstream

`conservation.py`:

- keeps the shared homolog pool
- builds one per-chain MSA per query
- computes per-residue conservation / alignment coverage
- stays per-chain

`radi_prepare.py`:

- starts from the same shared homolog pool
- builds interaction-supported paired rows
- reuses the prepared per-chain MSAs when available
- concatenates the paired trimmed rows into the final `raDI` paired alignment

So the two branches now share:

- the same search backend
- the same resolved homolog-hit TSVs
- and, in the combined runner, the same prepared per-chain MSA files when conservation runs first

But they do **not** share the same final MSA:

- conservation is per-chain
- `raDI` is paired

Small example:

- suppose MMseqs hits template accession `P12345` for query1
- `P12345` is already in the same accession space used by:
  - `template_pairs.final.tsv`
  - `templates.fasta`
  - `proteins.final.tsv`
- if the chosen pair universe contains an interaction involving `P12345`, then `P12345` can directly participate in:
  - conservation-side homolog filtering
  - `raDI`-side interaction-supported pairing
- the full sequence for `P12345` is recovered from the selected dataset `templates.fasta`
- the taxid for `P12345` is recovered from the selected dataset `proteins.final.tsv`

In short:

- MMseqs gives accession-space hits directly
- the pair universe lives in the same accession space
- FAMSA sequence recovery uses the same dataset FASTA
- no UniRef membership or cluster-expansion layer is involved anymore

### New raDI / conservation dataset switch

Standalone tools now support:

```bash
--pair-dataset intact_biogrid
--pair-dataset intact_biogrid_string
```

Combined runners now support:

```bash
--radi-pair-dataset intact_biogrid
--radi-pair-dataset intact_biogrid_string
```
