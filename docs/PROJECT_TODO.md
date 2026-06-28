# iFragDI project TODO roadmap

This document collects current design decisions, benchmark tasks, and future ideas for the iFragDI project. It is intended to keep the project scientifically traceable: every new feature should be linked to a biological question, a benchmark question, and an implementation task.

## Current project framing

iFragDI is an interpretable interface-residue and docking-restraint prioritization pipeline. The goal is not to replace full complex predictors. The goal is to rank residues on each interaction partner so that the top-ranked residues are useful as interface priors or docking restraints.

Core biological signals currently used or planned:

- iFrag/interface-fragment evidence: partner/interface-like residue support from fragment or template-like information.
- Conservation: broad evolutionary importance, useful but not always interface-specific.
- raDI/co-evolution: sparse inter-chain residue-pair evidence from paired MSAs, useful only when the paired MSA is reliable.
- Structure exposure: SASA/RSA, buried/surface flags, DSSP secondary-structure context.
- Patch context: local windows and residue-neighborhood evidence, because interfaces are patches rather than isolated residues.
- Future PLM/MSA features: protein language model support for MSA quality, sequence pairing, and co-evolution filtering.

## Current benchmark status

Completed milestones:

- BM5 Phase 1 feature generation completed.
- Native residue labels generated using 3.9 Å, 5 Å, and 8 Å interface definitions.
- Primary target is `interface_5A`.
- Original residue-score evaluation completed.
- Patch/window feature builder completed.
- Patch/window ML benchmark completed.
- Structure feature builder completed using FreeSASA and DSSP/mkdssp.
- Structure-aware ML benchmark completed.
- Conservative hybrid structure-aware ranking score tested.
- Structure-aware evidence-block ablation script added and submitted on Shiva.

Important current interpretation:

- The original hand-written `final_score` did not outperform conservation under global ranking metrics.
- Patch/window ML improved over the original manual score and conservation.
- Structure-aware ML improved further.
- The current best tested ranking score is the conservative hybrid `min(full_structure_radi, reduced_direct_radi_structure)`.
- This hybrid should be treated as a ranking score for residue selection, not as a calibrated probability.

## Immediate TODO: fixed-budget docking-relevant evaluation

### Biological question

Global ranking metrics such as AUPRC are useful but do not fully match docking-restraint selection. Conservation can score residues across the whole protein, including buried or non-interface conserved residues. For docking, we need to know whether a method can nominate a small interface-sized residue set with many true positives and few false positives.

### Main task

Implement a fixed-budget residue-recovery evaluator.

Suggested script:

`benchmark/evaluate_bm5_phase1_fixed_budget_recovery.py`

### Inputs

Use available prediction/feature tables:

- `benchmark/labels/bm5_phase1_training_table.tsv`
- `benchmark/labels/bm5_phase1_structure_ml_logreg.predictions.tsv`
- `benchmark/labels/bm5_phase1_structure_ablation_logreg.predictions.tsv` when available
- `benchmark/labels/bm5_phase1_patch_structure_features.tsv` for structure/surface flags if needed

### Methods to compare

At minimum:

- `ifrag_component`
- `conservation_component`
- `radi_component`
- `final_score`
- sequence patch ML score
- full structure-aware ML score
- hybrid structure-aware score
- ablation model scores when ablation output is available

### Prediction budgets

Evaluate fixed-count budgets:

- top 5 residues per chain
- top 10 residues per chain
- top 15 residues per chain
- top 20 residues per chain

Evaluate percentage budgets:

- top 3 percent of residues
- top 5 percent of residues
- top 10 percent of residues

Evaluate true-interface-size budget:

- top N residues, where N is the true number of interface residues in that chain

The top-N-true-interface-size metric is not usable in real prediction, because N is unknown in new cases, but it is a clean scientific benchmark.

### Metrics

For every method, budget, subset, and surface mode, report:

- selected residues
- true interface residues
- true positives
- false positives
- false negatives
- precision
- recall
- F1
- mean true positives per chain
- mean false positives per chain
- median true positives per chain
- median false positives per chain

### Surface modes

Evaluate both:

- all residues
- surface-only residues

Surface-only definition should initially use:

- `struct_surface_flag_rsa_ge_0p20 == 1`

or equivalently RSA >= 0.20 when available.

Rationale: conserved buried residues may be biologically important but are usually poor docking restraints.

### Outputs

Suggested outputs:

- `benchmark/labels/bm5_phase1_fixed_budget_recovery.tsv`
- `benchmark/labels/bm5_phase1_fixed_budget_recovery.per_group.tsv`
- `benchmark/labels/bm5_phase1_fixed_budget_recovery.summary.json`

Per-group output should include selected residue indices/numbers when available, to support manual inspection.

### Key comparisons to inspect first

- iFrag vs conservation, top 15, all residues
- iFrag vs conservation, top N true-interface-size, all residues
- iFrag vs conservation, top 5 percent, all residues
- iFrag vs conservation, top 15, surface-only
- iFrag vs conservation, top N true-interface-size, surface-only
- full/hybrid ML score vs iFrag and conservation under the same fixed budgets
- ablation scores under top 15 and top N when ablation predictions are available

## Immediate TODO: interface-size distribution

### Biological question

Which fixed budget is biologically reasonable? If the average or median BM5 interface size is close to 15 residues per chain, top 15 is a good intuitive metric. If the distribution is broader, percentage-based and top-N metrics become more important.

### Task

Compute true interface size distribution per query-side group using `interface_5A`.

Report:

- number of groups
- number of groups with zero interface residues
- mean interface residue count
- median interface residue count
- 25th percentile
- 75th percentile
- min/max
- number of groups with 1-5, 6-10, 11-15, 16-20, 21-30, >30 interface residues

### Output

Suggested output:

- `benchmark/labels/bm5_phase1_interface_size_distribution.tsv`
- `benchmark/labels/bm5_phase1_interface_size_distribution.summary.json`

## Immediate TODO: finish and interpret ablation benchmark

### Biological question

Is the final structure-aware score genuinely integrating iFrag, conservation, raDI, and structure, or is it mostly conservation plus surface exposure?

### Current script

`benchmark/train_bm5_phase1_structure_ablation.py`

### Outputs expected

- `benchmark/labels/bm5_phase1_structure_ablation_logreg.predictions.tsv`
- `benchmark/labels/bm5_phase1_structure_ablation_logreg.metrics.tsv`
- `benchmark/labels/bm5_phase1_structure_ablation_logreg.group_metrics.tsv`
- `benchmark/labels/bm5_phase1_structure_ablation_logreg.best_models.tsv`
- `benchmark/labels/bm5_phase1_structure_ablation_logreg.feature_sets.tsv`
- `benchmark/labels/bm5_phase1_structure_ablation_logreg.summary.json`

### Interpret with both metric types

Use global ranking metrics:

- AUPRC
- ROC-AUC
- top-L/10 recall
- top-L/5 recall

Also use fixed-budget metrics:

- top 15 precision/recall/F1
- top N precision/recall/F1
- top 5 percent precision/recall/F1
- surface-only top 15 precision/recall/F1
- surface-only top N precision/recall/F1

### Expected scientific conclusions to test

- Removing conservation should reduce performance if conservation is central.
- Removing structure should reduce performance if surface exposure is critical.
- Removing iFrag should reduce performance if iFrag contributes partner/interface-fragment specificity.
- Removing all raDI should show whether co-evolution adds signal after iFrag, conservation, and structure.
- Removing only direct raDI while keeping contextual raDI should show whether raDI is safest as a context-supported sparse anchor.

## Near-term TODO: representative case inspection

### Biological question

Do top-ranked residues form coherent, plausible surface patches, or are they scattered false positives?

### Select cases

Inspect:

- strongest improvement cases from old score to structure/hybrid score
- strongest worsening cases
- average/moderate cases
- cases where conservation beats iFrag
- cases where iFrag beats conservation under fixed-budget scoring
- cases where raDI helps
- cases where raDI hurts

### For each case, compare

- true native interface residues
- old `final_score` selected residues
- conservation selected residues
- iFrag selected residues
- sequence ML selected residues
- structure/hybrid selected residues
- surface exposure/RSA of selected residues
- whether selected residues cluster into a patch
- whether false positives are buried, scattered, or biologically plausible alternative surfaces

### Output idea

Create a per-case inspection table and optional PyMOL/ChimeraX selection scripts.

Suggested files:

- `benchmark/labels/bm5_phase1_case_inspection_candidates.tsv`
- `benchmark/case_inspection/README.md`

## Near-term TODO: final score-definition document

### Biological question

What scores should the production pipeline output, and how should users interpret them?

### Draft definitions

Potential outputs:

- `legacy_final_score`: old/manual score retained for diagnostics and backward compatibility.
- `sequence_score`: ML ranking score when no PDB/query structure is available.
- `structure_score`: structure-aware ML ranking score when query structure is available.
- `hybrid_strict_score`: conservative ranking score for strict docking restraints.
- evidence components: conservation, iFrag, raDI, structure, patch context.

### Suggested rule for residue sets

Initial candidate rule to benchmark:

- strict residues: top 5 percent with minimum 8 and maximum 25 residues per chain
- loose residues: top 10 percent with minimum 15 and maximum 50 residues per chain

This should be revised after fixed-budget evaluation.

### Output document

Suggested file:

- `docs/SCORE_DEFINITIONS.md`

## Near-term TODO: integrate ML scores into production pipeline

### Biological question

Can iFragDI produce practical residue scores and docking-restraint files using the learned scoring logic?

### Candidate integration points

- `combine_ifrag_radi.py`
- `radi.py`
- `radi_prepare.py`
- `run_lightdock_from_ifragdi.py`

### Required production behavior

When no PDB is available:

- compute sequence/patch score if required features exist
- output sequence-based strict/loose residues

When PDB is available:

- compute FreeSASA/RSA features
- run DSSP when available, but do not fail if DSSP is unavailable
- compute structure-aware score
- compute hybrid strict score if both model scores are available
- output strict/loose residue sets and restraint files

### Model packaging questions

Need to decide whether to:

- save trained logistic-regression coefficients and scaler parameters as JSON/TSV
- retrain deployable final models on the full primary training set
- keep cross-validation predictions only for benchmark diagnostics

Production must not use CV predictions as if they were deployable model parameters.

## Near-term TODO: docking benchmark

### Biological question

Does better residue ranking improve actual protein-protein docking?

### Compare docking conditions

- unrestrained docking
- conservation-only restraints
- iFrag-only restraints
- old `final_score` restraints
- sequence ML restraints
- structure-aware ML restraints
- hybrid strict restraints

### Docking metrics

Use accepted docking-quality metrics where possible:

- interface RMSD
- ligand RMSD
- fraction of native contacts
- CAPRI quality category
- top-1, top-5, top-10 success
- number of acceptable/medium/high-quality models

### Practical note

Docking benchmark should come after fixed-budget evaluation and case inspection, because residue-set size and strict/loose rules should be selected first.

## Future TODO: PLM/MSA ideas for raDI

### Biological question

Can protein language models make raDI/co-evolution more reliable by improving paired MSA quality, sequence pairing, or co-evolution filtering?

### Rationale

raDI is potentially partner-specific but noisy. It depends on the quality of the paired MSA. PLM/MSA models may help identify when the paired MSA is reliable and whether raDI anchors are supported by independent learned evolutionary context.

### Good first PLM direction

Do not start by adding thousands of raw PLM embedding dimensions to the main ML model. Start with diagnostic, interpretable PLM-derived features around raDI.

### Candidate PLM tasks

1. MSA quality scoring

Ask whether the paired MSA contains coherent evolutionary information before trusting raDI.

Potential features:

- paired MSA row count
- effective sequence count
- gap fraction
- sequence-identity distribution
- PLM masked-token likelihood or pseudo-likelihood summary
- PLM confidence difference between correctly paired and shuffled paired MSAs

2. Sequence-pairing confidence

Use PLM/MSA ideas inspired by DiffPALM-like approaches to assess whether homolog pairs are likely correctly matched.

Potential experiment:

- take paired MSA
- generate shuffled partner pairings
- evaluate whether PLM score distinguishes original pairing from shuffled controls
- use this as a pairing-confidence feature for raDI reliability

3. PLM inter-chain attention support

Use an MSA-based PLM on concatenated paired MSAs to extract inter-chain attention or contact-like support.

Potential residue features:

- max PLM inter-chain attention per residue
- sum/count of PLM-supported inter-chain pairs per residue
- overlap between top raDI pairs and top PLM-attention pairs
- local window count of PLM-supported anchors
- PLM support only for surface-exposed residues

4. raDI reliability filtering

Use PLM-derived features to classify raDI signal quality.

Potential tests:

- Does PLM support distinguish raDI pairs that are native contacts from raDI false positives?
- Do raDI anchors with PLM support recover interface residues better than raDI anchors without PLM support?
- Does PLM support help most in weak-MSA or paralog-rich cases?

5. MSA augmentation, later only

MSA generation/augmentation should be treated as a later and risky idea. It may help shallow MSAs, but artificial homologs could amplify artifacts. Do not start here.

### First PLM prototype plan

Prototype on a small subset before large-scale integration:

- 10 cases where raDI helps
- 10 cases where raDI hurts
- 10 cases with weak MSA warnings
- 10 cases with strong MSA support

For each case:

- collect paired MSA/SSA input from `radi_prepare.py`
- run an MSA-based PLM if feasible
- extract inter-chain attention/contact-like scores
- compare PLM-supported residue pairs to native inter-chain contacts
- compare PLM-supported residue anchors to `interface_5A`
- compare original paired MSA to shuffled-pair controls

### First PLM benchmark outputs

Suggested files:

- `benchmark/labels/bm5_phase1_plm_radi_subset_manifest.tsv`
- `benchmark/labels/bm5_phase1_plm_radi_pair_scores.tsv`
- `benchmark/labels/bm5_phase1_plm_radi_residue_features.tsv`
- `benchmark/labels/bm5_phase1_plm_radi_summary.json`

### Integration criterion

Only integrate PLM features into the main model if they improve fixed-budget residue recovery or raDI-specific diagnostics without harming interpretability.

## Future TODO: protein-language-model features for general interface scoring

### Biological question

Can single-sequence PLMs improve sequence-only interface ranking when no structure is available?

### Candidate features

Use small, reduced PLM features rather than full raw embeddings:

- ESM2/ProtT5 residue embeddings reduced by PCA
- predicted disorder tendency
- predicted surface/binding-region tendency
- PLM conservation-like residue importance
- local window summaries of PLM-derived residue features

### Caution

BM5 Phase 1 is small. High-dimensional PLM features can overfit. Use group-wise validation and external validation before making strong claims.

## Future TODO: external validation

### Biological question

Does the method generalize beyond BM5 Phase 1 design choices?

### Candidate validation sets

- held-out BM5/DB5 cases not used during model development
- nonredundant recent PPI complexes
- CAPRI-like targets when feasible
- DIPS/DIPS-Plus-derived filtered test set

### Required caution

Avoid leakage from homologous complexes, template-derived structures, or duplicated chain pairs. Split by complex/family, not residue.

## Future TODO: optional feature ideas

Potential biologically interpretable additions:

- surface patch contiguity/clustering score
- hydrophobic-patch score
- charged-patch complementarity score
- residue physicochemical class windows
- disorder/flexibility features
- predicted binding-site propensity
- partner-specific negative controls
- shuffled MSA controls for raDI
- phylogenetic/redundancy correction for paired MSA statistics

## Decision log template

When adding a new feature or benchmark, record:

- Date
- Biological question
- Implementation change
- Input files
- Output files
- Metrics used
- Main result
- Whether it changes the production scoring plan
- Whether it needs external validation

## Current next-day work plan

1. Check whether the ablation job finished.
2. If finished, inspect logs and output files.
3. Implement fixed-budget residue recovery evaluation.
4. Generate interface-size distribution summary.
5. Run fixed-budget evaluation on existing structure ML predictions.
6. Compare iFrag vs conservation under top 15, top N, top 5 percent, and surface-only versions.
7. If ablation predictions are available, run fixed-budget evaluation on ablation outputs.
8. Decide which metric panel will be the official benchmark panel.
9. Start PLM/raDI prototype planning from a small subset, not full integration.
10. Keep production pipeline integration paused until fixed-budget and ablation interpretation are complete.
