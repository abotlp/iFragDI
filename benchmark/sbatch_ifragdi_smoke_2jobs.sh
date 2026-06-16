#!/usr/bin/env bash
#SBATCH --job-name=ifragdi_smoke2
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=benchmark/logs/slurm/ifragdi_smoke2_%j.out
#SBATCH --error=benchmark/logs/slurm/ifragdi_smoke2_%j.err

set -euo pipefail

cd /users/sbi/patricia/iFragDI

source /soft/system/software/Miniconda3/202411/etc/profile.d/conda.sh
conda activate /users/sbi/patricia/.conda/envs/ifrag-env

module load MMseqs2/16-747c6-cpu-avx2
module load CD-HIT/4.8.1-GCC-13.3.0

echo "=== Environment ==="
echo "HOSTNAME=$(hostname)"
echo "CONDA_PREFIX=${CONDA_PREFIX}"
which python
python --version

missing=0
for tool in mmseqs famsa freesasa blastp makeblastdb cd-hit; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "MISSING from PATH: $tool"
    missing=1
  else
    echo "OK: $tool -> $(command -v "$tool")"
  fi
done

if [[ ! -x tools/RADI/bin/raDI ]]; then
  echo "MISSING or not executable: tools/RADI/bin/raDI"
  missing=1
else
  echo "OK: tools/RADI/bin/raDI"
fi

if [[ "$missing" -ne 0 ]]; then
  echo "STOP: environment incomplete."
  exit 1
fi

echo "=== Running first 2-job iFragDI smoke test ==="

python3 benchmark/run_ifragdi_smoke_from_manifest.py \
  --manifest benchmark/manifests/bm5_smoke_nonAA_nonHL_12.tsv \
  --only-chainpair-ids BM5CP00019,BM5CP00237 \
  --execute

echo "=== Done ==="
