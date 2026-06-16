#!/bin/bash
#SBATCH -J ifragdi_intact_biogrid_string
#SBATCH -p normal
#SBATCH --mem=200G

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

INTACT_ZIP="${INTACT_ZIP:-${PROJECT_DIR}/data/raw/intact.zip}"
BIOGRID_ZIP="${BIOGRID_ZIP:-${PROJECT_DIR}/data/raw/BIOGRID-SYSTEM-5.0.256.tab3.zip}"
STRING_ALIASES_GZ="${STRING_ALIASES_GZ:-${PROJECT_DIR}/data/raw/protein.aliases.v12.0.txt.gz}"
STRING_LINKS_GZ="${STRING_LINKS_GZ:-${PROJECT_DIR}/data/raw/protein.physical.links.detailed.v12.0.txt.gz}"
OUT_DIR="${OUT_DIR:-${PROJECT_DIR}/data/datasets/intact_biogrid_string}"
PROGRESS_EVERY="${PROGRESS_EVERY:-1000000}"
DROP_INTACT_EXPANDED="${DROP_INTACT_EXPANDED:-0}"
MIN_STRING_COMBINED_SCORE="${MIN_STRING_COMBINED_SCORE:-400}"
MIN_STRING_EXPERIMENTAL="${MIN_STRING_EXPERIMENTAL:-0}"

echo "[$(date '+%F %T')] Starting IntAct + BioGRID + STRING build"
echo "Project dir: ${PROJECT_DIR}"
echo "IntAct zip: ${INTACT_ZIP}"
echo "BioGRID zip: ${BIOGRID_ZIP}"
echo "STRING aliases: ${STRING_ALIASES_GZ}"
echo "STRING links: ${STRING_LINKS_GZ}"
echo "Output dir: ${OUT_DIR}"
echo "Progress every: ${PROGRESS_EVERY}"
echo "Drop IntAct expanded: ${DROP_INTACT_EXPANDED}"
echo "Min STRING combined score: ${MIN_STRING_COMBINED_SCORE}"
echo "Min STRING experimental: ${MIN_STRING_EXPERIMENTAL}"

cd "${PROJECT_DIR}"

CMD=(
  python3
  "${PROJECT_DIR}/data/scripts_dataset/build_intact_biogrid_string_dataset.py"
  --intact-zip "${INTACT_ZIP}"
  --biogrid-zip "${BIOGRID_ZIP}"
  --string-aliases-gz "${STRING_ALIASES_GZ}"
  --string-links-gz "${STRING_LINKS_GZ}"
  --out-dir "${OUT_DIR}"
  --min-string-combined-score "${MIN_STRING_COMBINED_SCORE}"
  --min-string-experimental "${MIN_STRING_EXPERIMENTAL}"
  --progress-every "${PROGRESS_EVERY}"
)

if [[ "${DROP_INTACT_EXPANDED}" == "1" ]]; then
  CMD+=(--drop-intact-expanded)
fi

printf 'Command:'
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"

echo "[$(date '+%F %T')] Finished IntAct + BioGRID + STRING build"
