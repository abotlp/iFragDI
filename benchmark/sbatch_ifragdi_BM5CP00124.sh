#!/usr/bin/env bash
#SBATCH --job-name=ifragdi_BM5CP00124
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=benchmark/logs/slurm/ifragdi_BM5CP00124_%j.out
#SBATCH --error=benchmark/logs/slurm/ifragdi_BM5CP00124_%j.err

set -euo pipefail

cd /users/sbi/patricia/iFragDI

source /soft/system/software/Miniconda3/202411/etc/profile.d/conda.sh
conda activate /users/sbi/patricia/.conda/envs/ifrag-env

module load MMseqs2/16-747c6-cpu-avx2
module load CD-HIT/4.8.1-GCC-13.3.0

echo "=== Environment ==="
hostname
date
which python3
python3 --version

echo "=== Tool check ==="
for x in mmseqs famsa freesasa blastp makeblastdb cd-hit; do
  command -v "$x" || { echo "Missing $x"; exit 1; }
done
test -x tools/RADI/bin/raDI || { echo "Missing executable tools/RADI/bin/raDI"; exit 1; }

echo "=== Running BM5CP00124 direct single-chain smoke ==="
python3 benchmark/run_ifragdi_smoke_from_manifest.py \
  --manifest benchmark/manifests/bm5_smoke_nonAA_nonHL_12.tsv \
  --only-chainpair-ids BM5CP00124 \
  --execute

echo "=== Done ==="
