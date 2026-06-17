#!/usr/bin/env bash
#SBATCH --job-name=ifragdi_phase1
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=benchmark/logs/slurm/ifragdi_phase1_%A_%a.out
#SBATCH --error=benchmark/logs/slurm/ifragdi_phase1_%A_%a.err

set -euo pipefail

cd /users/sbi/patricia/iFragDI

source /soft/system/software/Miniconda3/202411/etc/profile.d/conda.sh
conda activate /users/sbi/patricia/.conda/envs/ifrag-env

module load MMseqs2/16-747c6-cpu-avx2
module load CD-HIT/4.8.1-GCC-13.3.0

MANIFEST="benchmark/manifests/bm5_phase1_runnable_chainpairs.tsv"

CID="$(
python3 - "$MANIFEST" "$SLURM_ARRAY_TASK_ID" <<'PY'
import csv
import sys

manifest = sys.argv[1]
idx = int(sys.argv[2])

with open(manifest, newline="") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))

if idx < 1 or idx > len(rows):
    raise SystemExit(f"Array task {idx} outside manifest range 1..{len(rows)}")

print(rows[idx - 1]["chainpair_id"])
PY
)"

echo "=== Phase 1 iFragDI task ==="
echo "date: $(date)"
echo "host: $(hostname)"
echo "job_id: ${SLURM_JOB_ID}"
echo "array_task_id: ${SLURM_ARRAY_TASK_ID}"
echo "chainpair_id: ${CID}"
echo "manifest: ${MANIFEST}"
echo "python: $(which python3)"
python3 --version

python3 benchmark/run_ifragdi_smoke_from_manifest.py \
  --manifest "${MANIFEST}" \
  --only-chainpair-ids "${CID}" \
  --execute

echo "=== Done ${CID} ==="
date
