# IntAct + BioGRID Dataset Builder

This is a clean-start builder script for the first PPI universe:

- `IntAct + BioGRID`

The script lives in `data/scripts_dataset`.

The outputs are written under `data/datasets`.

It does not reuse the older dataset scripts.

## What it builds

The script reads:

- `data/raw/intact.zip`
- `data/raw/BIOGRID-SYSTEM-5.0.256.tab3.zip`

and writes:

- `intact_biogrid.final.tsv`
  - one unique undirected pair per row
  - columns:
    - `protein_1`
    - `protein_2`
    - `detection_method`
    - `detection_id`
    - `source`
  - methods, method IDs, and sources are aggregated per pair
- `intact_biogrid.evidence.final.tsv`
  - deduplicated 5-column evidence table
  - columns:
    - `protein_1`
    - `protein_2`
    - `detection_method`
    - `detection_id`
    - `source`
- `template_pairs.final.tsv`
  - one unique undirected pair per row
- `template_pairs.meta.final.tsv`
  - aggregated per-pair provenance and species metadata
- `proteins.final.tsv`
  - unique proteins retained in the universe
- `build_summary.tsv`
  - grouped coverage stats, discard counts, and drop reasons

## Biological choices baked in

- all organisms are kept
- both same-species and cross-species interactions are kept
- self-interactions are kept
- the pair universe is undirected
- isoforms are collapsed to UniProt base accessions
- canonical protein IDs are UniProt accessions
- IntAct negatives are removed
- IntAct expanded complex rows are kept by default
- IntAct rows must be protein-protein
- BioGRID rows must have `Experimental System Type = physical`

Important consequence:

- `intact_biogrid.final.tsv` has unique pairs only with aggregated provenance
- `template_pairs.final.tsv` has unique pairs only
- `intact_biogrid.evidence.final.tsv` is also deduplicated, but at the
  `(protein_1, protein_2, detection_method, detection_id, source)` level

## Filtering policy

For IntAct the script keeps broadly usable physical protein-protein evidence:

- protein-protein rows only
- negative rows removed
- obvious genetic / non-physical interaction types removed
- expanded complex rows kept by default
- self-interactions kept
- detection methods preserved in the output

For BioGRID:

- every row with `Experimental System Type = physical` is kept
- self-interactions kept
- detection methods are preserved in the output

Optional conservative switch:

- `--drop-intact-expanded`

Optional runtime progress:

- `--progress-every N`
  - prints a progress line to stderr every `N` parsed rows per source
  - use `0` to disable

## Script

- builder: [build_intact_biogrid_dataset.py](/home/patricia/cluster_shiva/iFragDI/data/scripts_dataset/build_intact_biogrid_dataset.py)

## How to run on the cluster

Start an interactive node:

```bash
srun -p normal --mem=200G --pty bash -l
```

Then run the default clean-start build:

```bash
cd /home/patricia/cluster_shiva/iFragDI
python3 data/scripts_dataset/build_intact_biogrid_dataset.py \
  --intact-zip /home/patricia/cluster_shiva/iFragDI/data/raw/intact.zip \
  --biogrid-zip /home/patricia/cluster_shiva/iFragDI/data/raw/BIOGRID-SYSTEM-5.0.256.tab3.zip \
  --out-dir /home/patricia/cluster_shiva/iFragDI/data/datasets/intact_biogrid
```

To submit the full run with `sbatch` instead of running interactively:

```bash
cd /users/sbi/patricia/iFragDI
sbatch data/scripts_dataset/run_build_intact_biogrid.sbatch
```

If you submit from the mounted path instead:

```bash
cd /home/patricia/cluster_shiva/iFragDI
sbatch data/scripts_dataset/run_build_intact_biogrid.sbatch
```

The sbatch script auto-detects the project root from its own location.

If you want more frequent progress messages while it runs:

```bash
cd /home/patricia/cluster_shiva/iFragDI
python3 data/scripts_dataset/build_intact_biogrid_dataset.py \
  --intact-zip /home/patricia/cluster_shiva/iFragDI/data/raw/intact.zip \
  --biogrid-zip /home/patricia/cluster_shiva/iFragDI/data/raw/BIOGRID-SYSTEM-5.0.256.tab3.zip \
  --out-dir /home/patricia/cluster_shiva/iFragDI/data/datasets/intact_biogrid \
  --progress-every 100000
```

Optional conservative IntAct variant:

```bash
cd /home/patricia/cluster_shiva/iFragDI
python3 data/scripts_dataset/build_intact_biogrid_dataset.py \
  --intact-zip /home/patricia/cluster_shiva/iFragDI/data/raw/intact.zip \
  --biogrid-zip /home/patricia/cluster_shiva/iFragDI/data/raw/BIOGRID-SYSTEM-5.0.256.tab3.zip \
  --out-dir /home/patricia/cluster_shiva/iFragDI/data/datasets/intact_biogrid_noexpanded \
  --drop-intact-expanded
```

## Output meaning

### `intact_biogrid.final.tsv`

This is the merged final table.

It has one unique undirected pair per row, with aggregated:

- detection methods
- detection IDs
- sources

So if the same pair appears in both IntAct and BioGRID, it appears once here
and the `source` field contains both databases.

### `intact_biogrid.evidence.final.tsv`

This is the evidence-level support table.

Each row is a deduplicated support record with:

- canonical protein pair
- detection method
- detection ID
- source database

For IntAct, if one source row contains multiple detection methods, the script
splits them into separate evidence rows so assay filtering stays clean later.

### `template_pairs.final.tsv`

This is the real unique pair universe.

If the same pair appears many times in IntAct and BioGRID, it appears only once
here.

### `template_pairs.meta.final.tsv`

This keeps the evidence you would otherwise lose when collapsing to unique
pairs. It includes:

- aggregated sources
- aggregated detection methods
- aggregated detection IDs
- support count
- taxid sets
- same-species / cross-species flags
- self-interaction flag

### `build_summary.tsv`

This is the coverage and discard report.

It has three columns:

- `section`
- `metric`
- `value`

Sections include:

- `intact`
- `biogrid`
- `merged`

Examples of what it records:

- total parsed rows
- kept source rows
- duplicate source rows
- kept evidence rows
- duplicate evidence rows
- discarded rows total
- discard fractions
- drop reasons such as missing or ambiguous mappings
- merged counts for unique pairs and proteins
- how many pairs are supported by IntAct only, BioGRID only, or both
- how many retained pairs are self-interactions

## Next step later

When you are ready, we can extend the same design to build:

- `IntAct + BioGRID + STRING`

without changing the output philosophy:

- one clean evidence table
- one unique pair universe
- one protein list for later FASTA/template generation

## Exact Commands Used

Run all commands from:

```bash
cd /users/sbi/patricia/iFragDI
```

### Build `intact_biogrid`

```bash
sbatch data/scripts_dataset/run_build_intact_biogrid.sbatch
```

### Build `intact_biogrid` FASTA + BLAST DB

```bash
sbatch data/scripts_dataset/run_build_intact_biogrid_blastdb.sbatch
```

### Build `intact_biogrid_string`

Recommended production run used here:

```bash
MIN_STRING_COMBINED_SCORE=700 MIN_STRING_EXPERIMENTAL=0 sbatch data/scripts_dataset/run_build_intact_biogrid_string.sbatch
```

### Build `intact_biogrid_string` FASTA + BLAST DB

```bash
sbatch data/scripts_dataset/run_build_intact_biogrid_string_blastdb.sbatch
```

## Exact iFrag FASTA Test Commands

Curated dataset:

```bash
cd /users/sbi/patricia/iFragDI
module load BLAST/2.12.0-Linux_x86_64
module load CD-HIT/4.8.1-GCC-13.3.0
python3 ifrags.py \
  --query1 benchmark/SEQUENCE1_QUERY.fa \
  --query2 benchmark/SEQUENCE2_QUERY.fa \
  --template-dataset intact_biogrid \
  --blast-bin /soft/system/software/BLAST/2.12.0-Linux_x86_64/bin/blastp \
  --out-dir benchmark/ifrag_fasta_intact_biogrid_paper \
  --threads 8 \
  --heatmap
```

STRING-expanded dataset:

```bash
cd /users/sbi/patricia/iFragDI
module load BLAST/2.12.0-Linux_x86_64
module load CD-HIT/4.8.1-GCC-13.3.0
python3 ifrags.py \
  --query1 benchmark/SEQUENCE1_QUERY.fa \
  --query2 benchmark/SEQUENCE2_QUERY.fa \
  --template-dataset intact_biogrid_string \
  --blast-bin /soft/system/software/BLAST/2.12.0-Linux_x86_64/bin/blastp \
  --out-dir benchmark/ifrag_fasta_intact_biogrid_string_paper \
  --threads 8 \
  --heatmap
```

## Template MMseqs DBs for the stable conservation / raDI path

The stable/default pipeline now uses only the template-backed MMseqs DB built from the interaction-template FASTA.
This keeps homolog search, sequence recovery, and taxid lookup inside the same dataset universe used by the PPI template tables.

STRING-expanded default build:

```bash
cd /users/sbi/patricia/iFragDI
sbatch data/scripts_dataset/run_build_template_mmseqs_db.sbatch
```

Curated core build:

```bash
cd /users/sbi/patricia/iFragDI
DATASET=intact_biogrid sbatch data/scripts_dataset/run_build_template_mmseqs_db.sbatch
```

The old UniRef MMseqs builders and UniRef membership fallback scripts were removed from the active pipeline because the homolog search now resolves directly inside the selected template/PPI universe.
