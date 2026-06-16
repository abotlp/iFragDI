# iFragDI Scoring Explainer

This note explains the stable scoring path in `combine_ifrag_radi.py` with one small toy example.

It covers:

1. how `iFrag` contributes residue scores
2. how conservation contributes residue scores
3. how `raDI` contributes residue-anchor scores
4. how `ifrag_conservation_radi` combines them
5. how the final docking residue lists are retained

This is meant to be a practical companion to `README_pipeline.md`, not a replacement for it.

## One Toy System

To keep the arithmetic readable, use a toy heteromer with:

- query1 residues: `A1 A2 A3 A4`
- query2 residues: `B1 B2 B3 B4`
- `top_k = 3`

The real code often works on much larger matrices and uses a larger seed window, but the logic is the same.

## Branch Matrices

The combined runner sees one matrix per pairwise branch.

For the toy example, assume the branch-specific preprocessing already happened:

- `iFrag` matrix has already been percentile-normalized across its nonzero cells
- conservation is shown as a normalized matrix view for intuition, even though the stable runner mainly uses per-chain conservation profiles
- `raDI` matrix has already been gated to its top retained pairs and divided by its top DI score

### Toy `iFrag` matrix

| q1 \\ q2 | B1 | B2 | B3 | B4 |
| --- | ---: | ---: | ---: | ---: |
| A1 | 0.0 | 0.0 | 0.8 | 0.0 |
| A2 | 0.0 | 0.6 | 0.7 | 0.0 |
| A3 | 0.0 | 0.0 | 0.0 | 0.0 |
| A4 | 0.1 | 0.2 | 0.0 | 1.0 |

### Toy conservation matrix view

| q1 \\ q2 | B1 | B2 | B3 | B4 |
| --- | ---: | ---: | ---: | ---: |
| A1 | 0.2 | 0.1 | 0.1 | 0.0 |
| A2 | 0.8 | 1.0 | 0.9 | 0.1 |
| A3 | 0.2 | 0.1 | 0.2 | 0.0 |
| A4 | 0.6 | 0.7 | 0.5 | 0.1 |

### Toy `raDI` matrix

| q1 \\ q2 | B1 | B2 | B3 | B4 |
| --- | ---: | ---: | ---: | ---: |
| A1 | 0.0 | 1.0 | 0.0 | 0.0 |
| A2 | 0.0 | 0.0 | 0.0 | 0.0 |
| A3 | 0.6 | 0.0 | 0.8 | 0.0 |
| A4 | 0.0 | 0.0 | 0.0 | 0.4 |

## 1. How `iFrag` Becomes Per-Residue Scores

In the stable combined scorer, the first residue-level step is:

`weighted_top_k_sum(vec, k) = top1 + 0.5 * (top2 + top3 + ...)`

For `top_k = 3`, the runner applies this to every row and every column of the `iFrag` matrix.

### Query1-side `iFrag` strength

- `A1`: top values are `0.8, 0.0, 0.0`
  result: `0.8`
- `A2`: top values are `0.7, 0.6, 0.0`
  result: `0.7 + 0.5*0.6 = 1.0`
- `A3`: all zero
  result: `0.0`
- `A4`: top values are `1.0, 0.2, 0.1`
  result: `1.0 + 0.5*(0.2 + 0.1) = 1.15`

So:

`ifrag_q1_strength = [0.8, 1.0, 0.0, 1.15]`

### Query2-side `iFrag` strength

- `B1`: `0.1`
- `B2`: `0.6 + 0.5*0.2 = 0.7`
- `B3`: `0.8 + 0.5*0.7 = 1.15`
- `B4`: `1.0`

So:

`ifrag_q2_strength = [0.1, 0.7, 1.15, 1.0]`

### Normalized `iFrag` component

The code then rank-normalizes only the positive values.

For query1, the positive values are:

`[0.8, 1.0, 1.15]`

Their percentile-style ranks become:

`ifrag_q1_component = [0.333, 0.667, 0.0, 1.0]`

For query2, the positive values are:

`[0.1, 0.7, 1.0, 1.15]`

Their ranks become:

`ifrag_q2_component = [0.25, 0.5, 1.0, 0.75]`

### `iFrag` specificity

The code also computes an `ifrag_specificity`, but that is mainly diagnostic in the stable runner.

Example:

- for `A4`, top values are `1.0, 0.2, 0.1`
- specificity is `1.0 / (1.0 + 0.2 + 0.1) = 0.769`

This can tell you whether a residue is supported by one dominant cell or by several comparable cells, but the final stable score does not add it directly.

## 2. How Conservation Enters the Combined Score

Biologically, conservation is treated as a broad interface-patch prior, not as a sparse residue-pair anchor set.

The stable combined runner prefers per-chain conservation profile files from `conservation.py`.

If those profiles are missing, it falls back to:

- query1 profile = row-wise maxima of the normalized conservation matrix
- query2 profile = column-wise maxima of the normalized conservation matrix

For the toy example, using the matrix above:

- query1 row maxima: `[0.2, 1.0, 0.2, 0.7]`
- query2 column maxima: `[0.8, 1.0, 0.9, 0.1]`

After the same positive-rank normalization:

- `conservation_q1_component = [0.5, 1.0, 0.5, 0.75]`
- `conservation_q2_component = [0.5, 1.0, 0.75, 0.25]`

The key idea is:

- conservation is broad
- it says which residues look generally interface-like across interacting homologs
- it does not claim a sharp partner-specific contact pattern by itself

## 3. How `raDI` Becomes Per-Residue Anchor Scores

`raDI` starts from a sparse paired-homolog alignment and produces DI-like inter-chain residue pairs.

In the stable combined runner:

1. keep only the top retained nonzero `raDI` pairs
2. divide each kept cell by the best retained DI value
3. compute row-wise and column-wise `weighted_top_k_sum`
4. positive-rank normalize the resulting residue vectors

Using the toy `raDI` matrix:

### Query1-side `raDI` anchor strength

- `A1`: `1.0`
- `A2`: `0.0`
- `A3`: `0.8 + 0.5*0.6 = 1.1`
- `A4`: `0.4`

So:

`radi_q1_anchor = [1.0, 0.0, 1.1, 0.4]`

After positive-rank normalization:

`radi_q1_component = [0.667, 0.0, 1.0, 0.333]`

### Query2-side `raDI` anchor strength

- `B1`: `0.6`
- `B2`: `1.0`
- `B3`: `0.8`
- `B4`: `0.4`

After normalization:

`radi_q2_component = [0.5, 1.0, 0.75, 0.25]`

The important difference from conservation is:

- conservation is broad patch evidence
- `raDI` is sparse anchor evidence

## 4. How `ifrag_conservation_radi` Combines the Branches

The stable mode does not simply add three residue vectors together.

It does this in stages.

### 4.1 Compute `iFrag` reliability

The code downweights `iFrag` when the raw matrix is too dense or too broad.

For the toy `iFrag` matrix:

- nonzero density = `6 / 16 = 0.375`
- row coverage = `3 / 4 = 0.75`
- column coverage = `4 / 4 = 1.0`

This gives an `ifrag_reliability` of about `0.53`.

That value is later used when combining the `iFrag` patch with conservation.

### 4.2 Build local seed regions

The combined scorer does not trust isolated residue spikes.

Instead, it:

1. picks a few nonredundant seed residues from `ifrag_component`
2. spreads each seed into a local window
3. does the same for conservation
4. merges the two local patch signals

To keep the toy arithmetic readable, use:

- `top_n_seeds = 2`
- `window_radius = 1`

The real code usually uses a larger window and up to 4 seeds.

### 4.3 Toy query1 patch construction

From the toy `ifrag_q1_component = [0.333, 0.667, 0.0, 1.0]`

The best nonredundant seeds are:

- `A4`
- `A2`

Their local seed-region signal is:

`q1_ifrag_seed_region = [0.333, 0.667, 0.834, 1.0]`

From the toy `conservation_q1_component = [0.5, 1.0, 0.5, 0.75]`

The best conservation seeds are:

- `A2`
- `A4`

Their local region is:

`q1_conservation_seed_region = [0.5, 1.0, 0.875, 0.75]`

Now scale the `iFrag` region by `ifrag_reliability ~ 0.53`:

`ifrag_scaled = [0.177, 0.354, 0.443, 0.531]`

Also let:

`conservation_scaled = q1_conservation_seed_region = [0.5, 1.0, 0.875, 0.75]`

Then combine with an overlap bonus:

`patch_raw = conservation_scaled + ifrag_scaled + 0.5 * min(conservation_scaled, ifrag_scaled)`

This gives approximately:

`patch_raw = [0.766, 1.531, 1.540, 1.547]`

After positive-rank normalization:

`q1_patch_score = [0.25, 0.5, 0.75, 1.0]`

### 4.4 Toy query2 patch construction

Doing the same on the query2 side gives:

`q2_patch_score = [0.5, 1.0, 0.75, 0.25]`

### 4.5 Patch-guided `raDI` bonus

`raDI` does not replace the patch. It gets filtered through the patch:

`radi_bonus = normalize_positive(radi_component * (0.25 + 0.75 * patch_score))`

For toy query1:

- `radi_q1_component = [0.667, 0.0, 1.0, 0.333]`
- `q1_patch_score = [0.25, 0.5, 0.75, 1.0]`

So the patch-guided raw values are:

- `A1: 0.667 * (0.25 + 0.75*0.25) = 0.292`
- `A2: 0.0`
- `A3: 1.0 * (0.25 + 0.75*0.75) = 0.8125`
- `A4: 0.333 * 1.0 = 0.333`

After normalization:

`radi_q1_bonus = [0.333, 0.0, 1.0, 0.667]`

For toy query2:

`radi_q2_bonus = [0.5, 1.0, 0.75, 0.25]`

### 4.6 Final per-residue hotspot scores

Now the code starts from the patch scores and adds the anchor bonus:

`final_hotspot_raw = patch_score + radi_weight * radi_bonus + blastpdb_weight * blastpdb_bonus`

If the retained `raDI` pairs are at the requested top-N cap, then:

`radi_weight = 0.35`

Using the toy values:

- `q1_final_raw = [0.367, 0.5, 1.1, 1.233]`
- `q2_final_raw = [0.675, 1.35, 1.013, 0.338]`

After positive-rank normalization:

- `q1_final_scores = [0.25, 0.5, 0.75, 1.0]`
- `q2_final_scores = [0.5, 1.0, 0.75, 0.25]`

In this toy case, `raDI` sharpened some preferences but did not completely change the order.

That is common in real runs:

- `iFrag + conservation` define the broad face
- `raDI` sharpens or reranks residues within or near that face

### 4.7 Final 2D heatmap

The final matrix is:

`residue_priority_matrix = outer(q1_final_scores, q2_final_scores)`

For the toy example:

| q1 \\ q2 | B1 | B2 | B3 | B4 |
| --- | ---: | ---: | ---: | ---: |
| A1 | 0.125 | 0.25 | 0.188 | 0.062 |
| A2 | 0.25 | 0.5 | 0.375 | 0.125 |
| A3 | 0.375 | 0.75 | 0.562 | 0.188 |
| A4 | 0.5 | 1.0 | 0.75 | 0.25 |

This is very important:

- the final heatmap is a residue-priority projection
- it is not a direct predicted contact map
- a bright final cell means both residues scored highly on their own chains
- it does not mean the branch evidence ever contained that exact pair

## 5. How the Docking Residue Lists Are Retained

After the final per-residue scores exist, the pipeline still does not simply take the top N residues globally.

It builds a direct-support signal:

`direct_support = conservation_component + ifrag_weight * ifrag_component + radi_weight * radi_component + blastpdb_weight * blastpdb_component`

Then it:

1. finds the best compact supported cluster on each chain
2. rejects the cluster if it is too weak
3. chooses active residues from that cluster by threshold
4. chooses passive residues only from the shell around the actives
5. applies the strict or loose caps

### Strict defaults

- active cap per chain: `4`
- passive cap per chain: `4`

So the usual primary export is at most:

- `8` residues per chain

### Loose defaults

- active cap per chain: `8`
- passive cap per chain: `8`

So the loose export is at most:

- `16` residues per chain

### The caps are not quotas

This is a common source of confusion.

The selector is:

- cluster-first
- threshold-first
- cap-limited afterward

So a chain can end with:

- fewer than the cap
- or even zero residues if the supported cluster is too weak

## 6. What Each Branch Is Really Doing

### `iFrag`

- starts from template-supported residue pairs
- turns them into per-residue strength by top-k row and column scoring
- provides a sharper template-derived residue-ranking signal

### Conservation

- starts from paired homolog evidence but contributes mainly a per-chain profile
- provides a broader interface-patch prior
- is usually the branch that keeps the predicted face coherent

### `raDI`

- starts from a paired homolog alignment and sparse DI-like inter-chain pairs
- provides sparse anchor evidence
- is strongest when its anchors overlap the template/conservation patch

## 7. Important Interpretation Rules

### Rule 1: final heatmap is not raw pair evidence

The final heatmap is an outer product of final query1 and query2 residue scores.

### Rule 2: `raDI` can be active and still have little effect

If its anchors fall outside the selected patch, its residue bonus can be small or even effectively zero in the final docking residues.

### Rule 3: broad ranking and strict docking set are different products

The top residue ranking can still look biologically reasonable even when the strict docking selector chooses a smaller or less ideal cluster.

## 8. Where This Lives in the Code

If you want to follow the implementation directly, the main functions are:

- `normalize_nonzero_by_percentile`
- `top_nonzero_max_normalized_values`
- `weighted_top_k_sum`
- `compute_residue_scores`
- `build_seed_region_signal`
- `combine_seed_region_patch`
- `compute_patch_guided_component`
- `build_residue_first_scores`
- `compute_direct_support_signal`
- `select_adaptive_docking_indices`

All of those live in `combine_ifrag_radi.py`, except the raw `raDI` matrix construction, which is handled in `radi.py`.

## 9. Short Mental Model

If you want one short summary:

- `iFrag` says: these template-supported residues look important
- conservation says: this whole face keeps looking interface-like
- `raDI` says: here are a few sparse anchor residues that may sharpen the face
- final stable score says: keep the face defined by `iFrag + conservation`, then let trusted anchors rerank residues inside or near that face
