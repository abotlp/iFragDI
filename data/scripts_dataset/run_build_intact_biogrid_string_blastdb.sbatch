#!/bin/bash
#SBATCH -J ifragdi_ibs_blastdb
#SBATCH -p normal
#SBATCH --mem=200G

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

DATASET_DIR="${DATASET_DIR:-${PROJECT_DIR}/data/datasets/intact_biogrid_string}"
TEMPLATES_DIR="${TEMPLATES_DIR:-${PROJECT_DIR}/data/interaction_templates/intact_biogrid_string}"
DB_PREFIX="${DB_PREFIX:-${PROJECT_DIR}/data/db/blast_templates_intact_biogrid_string/templates_db}"
UNIPROT_SPROT="${UNIPROT_SPROT:-${PROJECT_DIR}/data/raw/uniprot_sprot.fasta.gz}"
UNIPROT_TREMBL="${UNIPROT_TREMBL:-${PROJECT_DIR}/data/raw/uniprot_trembl.fasta.gz}"
PROGRESS_EVERY="${PROGRESS_EVERY:-1000000}"
MAKEBLASTDB_BIN="${MAKEBLASTDB_BIN:-makeblastdb}"
BLAST_MODULE="${BLAST_MODULE:-BLAST/2.12.0-Linux_x86_64}"

echo "[$(date '+%F %T')] Starting IntAct + BioGRID + STRING template FASTA / BLAST DB build"
echo "Project dir: ${PROJECT_DIR}"
echo "Dataset dir: ${DATASET_DIR}"
echo "Templates dir: ${TEMPLATES_DIR}"
echo "DB prefix: ${DB_PREFIX}"
echo "Swiss-Prot FASTA: ${UNIPROT_SPROT}"
echo "TrEMBL FASTA: ${UNIPROT_TREMBL}"
echo "Progress every: ${PROGRESS_EVERY}"
echo "makeblastdb binary: ${MAKEBLASTDB_BIN}"
echo "Preferred BLAST module: ${BLAST_MODULE}"

cd "${PROJECT_DIR}"

FASTA_CMD=(
  python3
  "${PROJECT_DIR}/data/scripts_dataset/build_intact_biogrid_template_fasta.py"
  --dataset-dir "${DATASET_DIR}"
  --out-dir "${TEMPLATES_DIR}"
  --uniprot-sprot "${UNIPROT_SPROT}"
  --uniprot-trembl "${UNIPROT_TREMBL}"
  --progress-every "${PROGRESS_EVERY}"
)

printf 'FASTA command:'
printf ' %q' "${FASTA_CMD[@]}"
printf '\n'

"${FASTA_CMD[@]}"

if ! command -v "${MAKEBLASTDB_BIN}" >/dev/null 2>&1; then
  if [[ -f /etc/profile.d/modules.sh ]]; then
    source /etc/profile.d/modules.sh
  fi
  if command -v module >/dev/null 2>&1; then
    module load "${BLAST_MODULE}" || module load BLAST/2.11.0-Linux_x86_64 || true
  fi
fi

if ! command -v "${MAKEBLASTDB_BIN}" >/dev/null 2>&1; then
  echo "ERROR: makeblastdb not found in PATH. Set MAKEBLASTDB_BIN to the full path or load a valid BLAST module before sbatch." >&2
  exit 1
fi

mkdir -p "$(dirname "${DB_PREFIX}")"

BLAST_CMD=(
  "${MAKEBLASTDB_BIN}"
  -in "${TEMPLATES_DIR}/templates.fasta"
  -dbtype prot
  -parse_seqids
  -out "${DB_PREFIX}"
)

printf 'BLAST command:'
printf ' %q' "${BLAST_CMD[@]}"
printf '\n'

"${BLAST_CMD[@]}"

echo "[$(date '+%F %T')] Finished IntAct + BioGRID + STRING template FASTA / BLAST DB build"
