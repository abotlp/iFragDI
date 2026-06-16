#!/bin/bash
#SBATCH -J ifragdi_tpl_mmseqs
#SBATCH -p normal
#SBATCH --mem=200G
#SBATCH -c 16

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

DATASET="${DATASET:-intact_biogrid_string}"
MMSEQS_MODULE="${MMSEQS_MODULE:-MMseqs2/16-747c6-cpu-avx2}"
MMSEQS_BIN="${MMSEQS_BIN:-mmseqs}"
THREADS="${THREADS:-${SLURM_CPUS_PER_TASK:-16}}"
CREATE_INDEX="${CREATE_INDEX:-1}"
FORCE_REINDEX="${FORCE_REINDEX:-0}"

case "${DATASET}" in
  intact_biogrid|intact_biogrid_string)
    ;;
  *)
    echo "ERROR: DATASET must be 'intact_biogrid' or 'intact_biogrid_string' (got '${DATASET}')." >&2
    exit 1
    ;;
esac

TEMPLATES_FASTA="${TEMPLATES_FASTA:-${PROJECT_DIR}/data/interaction_templates/${DATASET}/templates.fasta}"
DB_PREFIX="${DB_PREFIX:-${PROJECT_DIR}/data/db/mmseqs_templates_${DATASET}/templates_db}"
TMP_ROOT="${TMP_ROOT:-${PROJECT_DIR}/data/db/mmseqs_templates_${DATASET}/tmp}"
INDEX_SENTINEL="${DB_PREFIX}.idx.dbtype"

echo "[$(date '+%F %T')] Starting template MMseqs DB build"
echo "Project dir: ${PROJECT_DIR}"
echo "Dataset: ${DATASET}"
echo "Templates FASTA: ${TEMPLATES_FASTA}"
echo "DB prefix: ${DB_PREFIX}"
echo "Tmp root: ${TMP_ROOT}"
echo "MMseqs module: ${MMSEQS_MODULE}"
echo "MMseqs bin: ${MMSEQS_BIN}"
echo "Threads: ${THREADS}"
echo "Create index: ${CREATE_INDEX}"
echo "Force reindex: ${FORCE_REINDEX}"

cd "${PROJECT_DIR}"

if ! command -v "${MMSEQS_BIN}" >/dev/null 2>&1; then
  if [[ -f /etc/profile.d/modules.sh ]]; then
    source /etc/profile.d/modules.sh
  fi
  if command -v module >/dev/null 2>&1; then
    module load "${MMSEQS_MODULE}" || true
  fi
fi

if ! command -v "${MMSEQS_BIN}" >/dev/null 2>&1; then
  echo "ERROR: mmseqs not found in PATH. Set MMSEQS_BIN or load a valid MMseqs2 module before sbatch." >&2
  exit 1
fi

if [[ ! -f "${TEMPLATES_FASTA}" ]]; then
  echo "ERROR: Missing templates FASTA: ${TEMPLATES_FASTA}" >&2
  exit 1
fi

mkdir -p "$(dirname "${DB_PREFIX}")" "${TMP_ROOT}"

if [[ ! -f "${DB_PREFIX}.dbtype" ]]; then
  CREATEDB_CMD=("${MMSEQS_BIN}" createdb "${TEMPLATES_FASTA}" "${DB_PREFIX}")
  printf 'createdb command:'
  printf ' %q' "${CREATEDB_CMD[@]}"
  printf '\n'
  "${CREATEDB_CMD[@]}"
else
  echo "Reusing existing MMseqs DB: ${DB_PREFIX}"
fi

if [[ "${CREATE_INDEX}" == "1" ]]; then
  if [[ ! -f "${INDEX_SENTINEL}" || "${FORCE_REINDEX}" == "1" ]]; then
    INDEX_CMD=("${MMSEQS_BIN}" createindex "${DB_PREFIX}" "${TMP_ROOT}" --threads "${THREADS}")
    printf 'createindex command:'
    printf ' %q' "${INDEX_CMD[@]}"
    printf '\n'
    "${INDEX_CMD[@]}"
  else
    echo "Reusing existing MMseqs index: ${INDEX_SENTINEL}"
  fi
else
  echo "Skipping createindex (CREATE_INDEX=${CREATE_INDEX})"
fi

echo "[$(date '+%F %T')] Finished template MMseqs DB build"
