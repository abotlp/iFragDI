# Idea Improvement Notes

## Why a structure-aware extension is justified

The pipeline is no longer purely sequence-based once monomer structures are available.

That is biologically reasonable because the final goal is not to predict an exact full contact map from sequence alone. The goal is:

- input two proteins, usually as two monomer PDBs
- infer interface-relevant residues on each chain
- convert those residues into docking restraints
- improve guided docking relative to blind docking

For that goal, structural information is useful even before the complex is known. Monomer structure can tell you:

- which residues are surface-exposed
- whether predicted interface signal forms a compact patch or diffuse noise
- whether a residue lies in a hydrophobic or shape-compatible patch
- whether a predicted anchor is geometrically plausible on the exposed surface
- whether the structure itself is trustworthy, for example with pLDDT-like confidence from AlphaFold models

So adding structure-aware logic is not a departure from the pipeline goal. It is a direct way to make the residue-prior and docking-restraint steps more realistic.

## Useful ideas not fully borrowed yet from StructureDCA

### 1. True inter-chain-only raDI inference

This is the strongest idea to borrow.

Your current inter-chain raDI adaptation is conceptually correct:

- build paired rows
- concatenate chain A and chain B
- run raDI on the full concatenated alignment
- keep only the inter-chain block afterward

That works, but it is noisier than the `cMSA-inter` idea in `DCA_struct.pdf`, where only inter-chain couplings are retained.

The stronger version for your pipeline would be:

- define the concatenated paired MSA as you do now
- during DI inference, allow couplings only between chain A positions and chain B positions
- do not fit A-A or B-B couplings at all

Why this is attractive:

- fewer parameters
- less dilution by strong intra-chain coevolution
- better fit to the biological question, which is interface anchoring
- easier interpretation of raDI output

Near-term practical version:

- keep current raDI as baseline
- add an experimental `interchain_only` inference mode later if you ever touch the raDI source itself

### 2. User-defined sparse structural masks for raDI

StructureDCA emphasizes user-defined sparse contact maps.

You usually do not know the true complex contact map, so you cannot use exact inter-chain contacts as a mask. But you do know useful structural eligibility on each monomer.

Good candidate masks for your setting:

- surface-exposed residues on chain A x surface-exposed residues on chain B
- predicted interface patch on chain A x predicted interface patch on chain B
- top iFrag/conservation patch on chain A x top iFrag/conservation patch on chain B
- surface-exposed residues x residues in high-confidence structural regions only

Ways to use these masks:

- post hoc: filter raDI anchors after inference
- intermediate: reweight raDI anchors by mask compatibility
- strongest version: constrain the allowed inter-chain couplings during inference

This is the cleanest way to make raDI more structure-aware without pretending you know the true complex.

### 3. Continuous RSA weighting instead of only a hard cutoff

Right now the structure-aware combiner applies a hard surface filter. Residues below `--surface-rsa-threshold` are zeroed out.

That is simple and useful, but a softer strategy may be better:

- use RSA as a multiplicative weight rather than a binary gate
- strongly penalize buried residues
- keep partially exposed residues with reduced score

Why this matters:

- interfaces are not always maximally exposed in unbound monomers
- induced fit and side-chain rearrangements can expose residues that look only moderately accessible in the starting structure
- a hard cutoff can kill borderline but real interface residues

Practical version:

- keep the hard filter as an optional strict mode
- add a soft RSA weighting mode for scoring and docking residue selection

### 4. Separate structural confidence weighting from surface weighting

StructureDCA[RSA] treats structural context as a weight, not just a geometric filter.

You already partly do this with optional pLDDT-style confidence from B-factors in the structure-aware reranker. It is worth making this more explicit as a first-class score component:

- confidence weight from pLDDT-like B-factors
- surface/exposure weight from RSA
- patch density / local mass weight from geometry

These should remain separate because they mean different things:

- confidence answers: can I trust this local monomer model?
- RSA answers: is this residue even reachable for binding?
- patch geometry answers: does this residue sit in a coherent candidate interface region?

## Additional ideas worth adding

### 5. Precision-first restraint production

For docking, precision matters more than recall.

This is important enough to make explicit in the pipeline design:

- a few good active residues can help docking a lot
- too many false positives can drive docking toward the wrong face

This argues for:

- a stricter active set
- a looser backup set
- explicit branch-confidence weighting when generating restraints
- stronger penalties on diffuse interface predictions

This matches your current philosophy and should stay central to the scoring function.

### 6. Taxon-aware or clade-aware coevolution integration

A useful idea for the homolog side is to avoid one giant heterogeneous paired MSA.

Instead:

- build paired MSAs in narrower taxonomic clades
- run raDI/DI separately within each clade
- integrate the inter-chain signals afterward

Why this is attractive:

- ortholog pairing is usually cleaner inside narrower clades
- paralog confusion is reduced
- coevolution signals that repeat across clades are more trustworthy
- noisy clades can be downweighted rather than merged blindly

This is especially relevant to your inter-chain raDI branch, because pairing quality is the main biological risk.

### 7. Pair-row reweighting instead of hard best-hit pruning

You were right to be cautious about enforcing one pair per species as the default. That can destroy depth.

A better long-term idea is:

- keep multiple compatible pairs when depth is scarce
- but reweight rows so paralog-rich taxa do not dominate

Examples:

- cap the total contribution per species/taxon
- divide row weight by the number of accepted pairs in that taxon
- prefer high-support curated pairs but do not discard all alternatives

This is a softer anti-paralog strategy than hard best-1-per-species.

### 8. Structure-aware anchor gating

Not all raDI anchors should contribute equally once structures are available.

Possible reweighting terms for each retained inter-chain anchor:

- both residues surface-exposed
- both residues inside or near the predicted patch
- local structural confidence on both monomers
- neighborhood support from nearby predicted interface residues

This would keep raDI as a sparse specificity signal while reducing structurally implausible anchors.

### 9. Explicit uncertainty propagation into docking

The combine stage should keep track of uncertainty, not just scores.

Useful uncertainty signals:

- weak paired MSA depth
- diffuse iFrag map
- conservation built from too few interaction-supported homologs
- poor structural confidence in a candidate patch

These could control:

- whether raDI is allowed to influence the final score
- whether only strict or also loose restraints are written
- how many active residues are allowed
- whether one-sided docking should be preferred

### 10. Optional shallow docking feedback loop

If you want one more stage later, a small feedback loop could be useful:

- run a cheap coarse docking stage using the current restraints
- collect recurrent interface patches from top poses
- use those as a reranking signal for residue priorities

This would make the pipeline more partner-specific without replacing the biological core of iFrag or raDI.

## Literature notes on methods related to your pipeline

These methods are not identical to your pipeline, but they are relevant neighbors.

### Structure-aware inter-protein contact prediction

1. DRN-1D2D_Inter, 2023

- Paper: Si and Yan, Briefings in Bioinformatics, 2023
- Link: https://doi.org/10.1093/bib/bbad039
- Why relevant:
  - predicts inter-protein contacts
  - combines protein language models, MSA-derived features, and coevolution
  - explicitly reports that predicted contacts can improve docking
- What to learn:
  - your pipeline is aiming at the same downstream use, but with a more interpretable classical design
  - this is a good benchmark family for comparison

2. DeepInter, 2023

- Paper: Lin et al., Nature Machine Intelligence, 2023
- Link: https://doi.org/10.1038/s42256-023-00741-2
- Why relevant:
  - predicts inter-protein contacts
  - uses monomer structures plus a triangle-aware deep architecture
- What to learn:
  - geometric structure cues can improve partner-specific inter-chain prediction
  - this supports your move toward structure-aware reranking and anchor filtering

3. DeepHomo2.0, 2023

- Paper: Lin et al., Briefings in Bioinformatics, 2023
- Link: https://doi.org/10.1093/bib/bbac499
- Why relevant:
  - homomer-focused inter-protein contact prediction
  - uses deep learning plus MSA/contextual information
- What to learn:
  - useful reference for your homomer mode

### Structure-based interface-site prediction

4. EquiPPIS, 2023

- Paper: Roche et al., PLOS Computational Biology, 2023
- Link: https://doi.org/10.1371/journal.pcbi.1011435
- Why relevant:
  - predicts interface residues from monomer structure
  - works well even on AlphaFold2-predicted models
- What to learn:
  - structure-aware monomer interface priors are worth using at scale
  - this supports your idea that the pipeline should not remain purely sequence-based once structures are available

5. MIPPIS, 2024

- Paper: Wang et al., BMC Bioinformatics, 2024
- Link: https://doi.org/10.1186/s12859-024-05964-7
- Why relevant:
  - recent interface-site predictor using multi-information fusion
- What to learn:
  - the field is still actively improving partner/interface residue prediction from fused inputs

### Restraint-guided structure modeling and docking

6. ColabDock, 2024

- Paper: Feng et al., Nature Machine Intelligence, 2024
- Link: https://doi.org/10.1038/s42256-024-00873-z
- Why relevant:
  - integrates residue and surface restraints into deep-learning-based complex prediction
  - reports gains over standard docking baselines in restraint-assisted settings
- What to learn:
  - your idea of converting interface evidence into restraints is absolutely aligned with where the field is going
  - residue and surface restraints are both useful, not only exact pairwise contacts

7. Scoring docking models utilizing predicted interface residues, 2022

- Paper: Pozzati et al., Proteins, 2022
- Link: https://doi.org/10.1002/prot.26330
- Why relevant:
  - tests how predicted interface residues help docking
- What to learn:
  - precision is a critical property when predictions are used as docking constraints
  - false positives can easily drive docking toward the wrong face

### Paired-MSA and coevolution improvements

8. DeepMSA2, 2024

- Paper: Zheng et al., Nature Methods, 2024
- Link: https://doi.org/10.1038/s41592-023-02130-4
- Why relevant:
  - focuses on better monomer and multimer MSA construction
  - includes explicit multimer MSA pairing and selection
- What to learn:
  - the quality of the paired MSA itself is a major determinant of downstream complex prediction
  - this supports investing effort in your `radi_prepare.py` stage

9. Enhancing coevolutionary signals in protein-protein interaction prediction through clade-wise alignment integration, 2024

- Paper: Fang et al., Scientific Reports, 2024
- Link: https://doi.org/10.1038/s41598-024-55655-9
- Why relevant:
  - runs DCA separately in narrower clades and integrates signals afterward
- What to learn:
  - this is directly relevant to your paired-homolog problem
  - clade-wise integration may be a strong future improvement for raDI

10. DeepSCFold, 2025

- Paper: Hou et al., Nature Communications, 2025
- Link: https://doi.org/10.1038/s41467-025-65090-7
- Why relevant:
  - improves complex prediction through better paired MSA construction
  - uses learned interaction probability and structural similarity to pair homologs across monomer MSAs
- What to learn:
  - a smarter pairing stage can materially improve downstream complex modeling
  - this is highly relevant to future upgrades of your homolog-pair builder

## What seems distinctive about your pipeline

I did not find a recent published method with exactly the same combination of:

- classical iFrag-style interaction-template transfer
- interaction-aware homolog-side conservation
- paired-homolog inter-chain DI
- post-combination conversion into docking restraints

Most recent methods instead do one of these:

- direct deep learning contact prediction
- partner-independent interface-site prediction from monomer structure
- end-to-end complex prediction with restraints

So your pipeline still has a distinctive niche:

- interpretable branch structure
- explicit biological rationale per branch
- direct compatibility with classical docking

## Additional ideas from newly added PPI papers

The following papers do not argue for replacing the core of the pipeline. Instead, they suggest ways to sharpen the current `iFrag + conservation + raDI + docking` design.

### 11. PPI3D

- Paper file: `papers/PPI3D.pdf`
- Title: `PPI3D: a web server for searching, analyzing and modeling protein-protein, protein-peptide and protein-nucleic acid interactions`
- Main idea:
  - structural interaction templates should be clustered and analyzed at the interface level, not only at the protein-pair level
  - homology-based transfer of interface residues remains useful even in the AlphaFold era
- What seems most useful for this pipeline:
  - treat structural templates as interface clusters rather than isolated examples
  - explicitly capture alternative interfaces within the same homologous family
  - retain interface/contact surface area information, not only binary contact presence
- Concrete upgrades suggested:
  - add structural annotation of template pairs whenever a template complex structure is available
  - cluster structurally similar template interfaces to avoid overcounting redundant interaction geometries
  - distinguish consensus template interfaces from alternative/outlier interfaces
  - use template interface surface area or contact-density features to weight iFrag evidence
- Why this matters:
  - this would make the iFrag branch more interface-aware and less dependent on raw template count alone

### 12. PPI-hotspotID

- Paper file: `papers/PPI-hotspot.pdf`
- Title: `PPI-hotspotID for detecting protein-protein interaction hot spots from the free protein structure`
- Main idea:
  - important binding residues can be predicted from the unbound/free structure
  - critical residues are not always identical to all interface residues
  - some true hot spots are not obvious direct contact residues in the complex
- What seems most useful for this pipeline:
  - separate broad interface-patch prediction from hotspot/anchor prediction
  - use free-structure information to prioritize the most important residues for docking
  - combine interface evidence with hotspot evidence rather than assuming they are the same thing
- Concrete upgrades suggested:
  - add a hotspot-like reranking layer on top of the current final residue scores
  - use conservation + residue type + SASA + local energetic stability as a stricter active-residue prior
  - distinguish:
    - broad predicted interface region
    - high-priority likely hot spots for active restraints
  - downweight diffuse residues that look like generic surface patch but not like energetic anchors
- Why this matters:
  - docking benefits more from a few precise active residues than from a wide low-specificity interface patch

### 13. PPI-ID

- Paper file: `papers/PPI-ID.pdf`
- Title: `PPI-ID: Streamlining protein-protein interaction prediction through domain and SLiM mapping`
- Main idea:
  - many PPIs are constrained by known domain-domain or domain-SLiM logic
  - mapping domains and motifs onto sequence/model space can identify plausible interaction regions and reduce search space
- What seems most useful for this pipeline:
  - motif-mediated and IDR-mediated interactions need different expectations than globular domain-domain interfaces
  - domain/motif priors can help interpret or filter predictions
- Concrete upgrades suggested:
  - annotate query proteins with domains and SLiMs before or after scoring
  - flag interactions likely to be motif-mediated
  - if a predicted patch overlaps a known domain/motif interaction region, boost confidence
  - if the predicted patch is incompatible with known domain/motif logic, lower confidence or label the case as biologically suspicious
  - use domain/SLiM information to guide interpretation of weak or conflicting iFrag/raDI cases
- Why this matters:
  - the pipeline would gain a biological interpretation layer, especially useful for regulatory or disorder-containing proteins

## How these three papers fit the current pipeline

These three papers suggest that the final pipeline should distinguish four related but non-identical concepts:

- interaction-supported interface region
- sparse inter-chain anchor residues
- energetic/functional hot spots
- domain/motif compatibility

Your current pipeline already covers the first two reasonably well:

- `iFrag + conservation` gives the broad interface-region logic
- `raDI` gives sparse inter-chain anchors

The new papers mainly strengthen the last two:

- `PPI-hotspotID` adds the idea of hotspot prioritization from monomer structure
- `PPI-ID` adds the idea of domain/SLiM compatibility
- `PPI3D` adds the idea that template interactions should be treated as clustered structural interface families

## Updated practical priority after reading these papers

If the aim is to improve the current pipeline without replacing its core methods, the most sensible order now looks like this:

1. Add hotspot-aware reranking for the final docking residue selection.
2. Add structure-aware reweighting of retained raDI anchors.
3. Add clade-wise or taxon-aware raDI integration for cleaner paired-homolog signal.
4. Add structural clustering/annotation of template interfaces for the iFrag template universe.
5. Add optional domain/SLiM annotation as a biological interpretation and confidence layer.
6. Longer term: explore a true inter-chain-only raDI inference mode.

## Recommended implementation priority

If you want a practical order:

1. Add soft RSA weighting in addition to the current hard surface cutoff.
2. Add structure-aware reweighting of retained raDI anchors.
3. Add taxon/clade-aware reweighting or capped weighting for paired rows.
4. Add a patch-derived inter-chain structural mask for raDI anchor retention.
5. Longer term: explore a true inter-chain-only raDI inference mode.

## Take-home message

Yes, there are useful ideas left to borrow from the structure-aware literature.

The best ones are not about replacing iFrag or replacing raDI with a black-box model. They are about:

- making the paired-MSA/coevolution stage cleaner
- making the structure-aware reranking more continuous and principled
- making the final docking restraints more precision-focused and geometry-aware

That fits your pipeline well.
