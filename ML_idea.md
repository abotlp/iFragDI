Yes — now the picture is much clearer.

What your PI is describing is **supervised learning**:

* you run your pipeline on many benchmark complexes
* you compute the input feature matrices
* and you give the model the **true contact map from the bound complex** as the label

So the answer to your main question is:

## What contact map should be given to the learning model?

**The target should be the native inter-chain contact map from the bound structure, not the contact map you predicted.**

If you train on your own predicted map, the model just learns to reproduce your current heuristic pipeline. That is circular and useless.
In supervised contact prediction, the standard setup is exactly: **features for residue pairs in, contact labels from known structures out**.   ([Springer][1])

---

# The clean formulation

For each benchmark complex:

## Input `X`

Build these from the information you would really have at prediction time:

* `iFrag(i,j)` pair matrix
* `raDI(i,j)` pair matrix, when available
* conservation on chain 1
* conservation on chain 2
* monomer structure features for chain 1
* monomer structure features for chain 2
* masks telling the model what is missing or weak, for example low paired-MSA depth

This fits your current biology well:

* `iFrag` is pairwise template evidence
* `raDI` gives sparse inter-chain anchor pairs from interaction-supported paired interolog rows
* conservation stays per-chain and can be projected into pair space
* the current final 2D matrix is only diagnostic, not the real target.

## Target `Y`

A binary contact map from the **bound complex**:

* `Y(i,j) = 1` if residue `i` from chain 1 and residue `j` from chain 2 are in contact in the bound structure
* `Y(i,j) = 0` otherwise

That is the label.

This is how inter-protein contact predictors are trained in the literature: residue-pair features are used to predict whether those residue pairs form contacts in the known complex structure. ([PMC][2])

---

# So what exactly do you give the model?

## You do **not** give it:

* the predicted contact map as the truth
* the final outer-product matrix as the truth
* a map that “looks biologically reasonable”

Those can be **input features**, but not labels. Your README is explicit that the final 2D heatmap in the current pipeline is only a **diagnostic residue-priority projection**, not a literal contact map.

## You **do** give it:

* your branch-derived feature maps
* plus the **native contact map from the bound structure** as supervision

That is the core idea.

---

# What contact definition should you use?

Pick **one consistent definition** and keep it fixed across training and evaluation.

In the literature, several contact definitions are used for protein contacts, including:

* any two heavy atoms within a cutoff
* Cβ–Cβ distance cutoffs
* other restricted distance rules. ([PMC][2])

Since your current BM5 helper already uses a **heavy-atom cutoff** to define native-interface residues, the simplest and cleanest thing is:

> use the **same bound-structure contact rule** for the pair labels that you will use in evaluation

That avoids confusion.

So yes, if your benchmark helper uses the bound complex with your chosen heavy-atom cutoff, then **that is the right source of truth**.

---

# How to think about one training example

Suppose one benchmark complex has:

* chain 1 length = `L1`
* chain 2 length = `L2`

Then you build:

## Feature tensor

Shape roughly:

* `L1 × L2 × C`

where `C` is the number of channels.

Possible channels:

* `iFrag(i,j)`
* `raDI(i,j)`
* `cons1(i)` broadcast across columns
* `cons2(j)` broadcast across rows
* `cons1(i) * cons2(j)`
* `RSA1(i)` broadcast
* `RSA2(j)` broadcast
* local geometry around `i`
* local geometry around `j`
* branch-availability masks
* MSA-depth / confidence channels

## Label matrix

Shape:

* `L1 × L2`

with:

* `1` = real contact in the bound complex
* `0` = not in contact

That is the supervised dataset.

---

# Why the bound complex is the correct label

Because the model must learn:

> “given these input signals, which residue pairs are truly in contact?”

Only the **bound complex** tells you that.

Your current matrices are evidence:

* template evidence
* coevolution evidence
* conservation evidence
* geometry evidence

They are **inputs**, not ground truth.

---

# What the model would then learn

The model could learn things like:

* when `iFrag` is trustworthy
* when low-depth `raDI` should be ignored
* when conservation helps only one side
* when one chain has a plausible patch but the other is weak
* what geometrically coherent inter-chain contact patterns look like

That is the part your PI probably means by “learn with triangulation like AlphaFold.”

In AlphaFold, pair representations are updated so that pairwise relations are geometrically consistent, and DeepInter applies a triangle-aware mechanism to inter-protein contact prediction. ([Nature][3])

---

# What you should **not** do

Do **not** do this:

1. run your current pipeline
2. get a predicted matrix
3. ask the model whether that predicted matrix “looks biologically reasonable”
4. retrain from that alone

That is not proper supervision.

You need:

* **predicted/evidence matrices as input**
* **true bound contact map as label**

---

# A very simple first plan

If I were you, I would do this first:

## Dataset

For each benchmark case:

* run `iFrag`
* run conservation
* run `raDI`
* store all matrices and per-residue features
* generate the **native inter-chain contact map from the bound structure**

## Model

Start very simple:

* a small 2D CNN over the pair tensor

Not AlphaFold-scale yet.

## Loss

Binary cross-entropy or focal loss on the contact map.

## Output

Predicted contact map `C(i,j)`.

## Then derive

* row pooled score = interface score on chain 1
* column pooled score = interface score on chain 2
* compact patches for docking

That is the right first ML step.

---

# The simplest possible explanation

Think of it like image segmentation:

* your matrices are the **input image channels**
* the bound contact map is the **segmentation mask**
* the model learns to map inputs to the true contacts

That is all.

---

# The one sentence answer

**Yes: you should give the model the true inter-chain contact map from the bound structures as the training label, while the matrices from iFrag, conservation, raDI, and monomer structure features are the input features.**

If you want, next I can write the exact dataset format for one complex, like:

* tensor channels
* label matrix
* loss
* pooling from contact map to residue scores

so you can actually implement it.

[1]: https://link.springer.com/article/10.1186/s12859-019-3051-7?utm_source=chatgpt.com "Predicting protein inter-residue contacts using composite likelihood maximization and deep learning | BMC Bioinformatics | Springer Nature Link"
[2]: https://pmc.ncbi.nlm.nih.gov/articles/PMC8425427/?utm_source=chatgpt.com "Accurate prediction of inter-protein residue–residue contacts for homo-oligomeric protein complexes - PMC"
[3]: https://www.nature.com/articles/s41586-021-03819-2?utm_source=chatgpt.com "Highly accurate protein structure prediction with AlphaFold | Nature"
