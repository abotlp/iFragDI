#!/bin/bash
#SBATCH -J ifragdi_intact_biogrid
#SBATCH -p normal
#SBATCH --mem=200G

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

INTACT_ZIP="${INTACT_ZIP:-${PROJECT_DIR}/data/raw/intact.zip}"
BIOGRID_ZIP="${BIOGRID_ZIP:-${PROJECT_DIR}/data/raw/BIOGRID-SYSTEM-5.0.256.tab3.zip}"
OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/data/datasets/intact_biogrid}"
PROGRESS_EVERY="${PROGRESS_EVERY:-1000000}"
DROP_INTACT_EXPANDED="${DROP_INTACT_EXPANDED:-0}"

echo "[$(date '+%F %T')] Starting IntAct + BioGRID build"
echo "Project dir: ${PROJECT_DIR}"
echo "IntAct zip: ${INTACT_ZIP}"
echo "BioGRID zip: ${BIOGRID_ZIP}"
echo "Output dir: ${OUT_DIR}"
echo "Progress every: ${PROGRESS_EVERY}"
echo "Drop IntAct expanded: ${DROP_INTACT_EXPANDED}"

cd "${PROJECT_DIR}"

CMD=(
  python3
  "${PROJECT_DIR}/data/scripts_dataset/build_intact_biogrid_dataset.py"
  --intact-zip "${INTACT_ZIP}"
  --biogrid-zip "${BIOGRID_ZIP}"
  --out-dir "${OUT_DIR}"
  --progress-every "${PROGRESS_EVERY}"
)

if [[ "${DROP_INTACT_EXPANDED}" == "1" ]]; then
  CMD+=(--drop-intact-expanded)
fi

printf 'Command:'
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"

echo "[$(date '+%F %T')] Finished IntAct + BioGRID build"
