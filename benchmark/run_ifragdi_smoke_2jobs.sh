#!/usr/bin/env bash
# Run this on Shiva, not Masada.
# Load the required environment/modules before running.
# This is only the first 2-job smoke test.
# Do not use this for the full benchmark.

set -euo pipefail

PROJECT_ROOT="/users/sbi/patricia/iFragDI"

cd "${PROJECT_ROOT}"

echo "WARNING: exact environment/module commands are not encoded in this smoke wrapper." >&2
echo "WARNING: load the required environment before running; checking expected binaries now." >&2

required_tools=(
  "mmseqs"
  "famsa"
  "freesasa"
  "blastp"
  "makeblastdb"
  "cd-hit"
  "tools/RADI/bin/raDI"
)

for tool in "${required_tools[@]}"; do
  if [[ "${tool}" == */* ]]; then
    if [[ ! -x "${tool}" ]]; then
      echo "WARNING: missing or not executable: ${tool}" >&2
    fi
  else
    if ! command -v "${tool}" >/dev/null 2>&1; then
      echo "WARNING: missing from PATH: ${tool}" >&2
    fi
  fi
done

python3 benchmark/run_ifragdi_smoke_from_manifest.py \
  --manifest benchmark/manifests/bm5_smoke_nonAA_nonHL_12.tsv \
  --only-chainpair-ids BM5CP00019,BM5CP00237 \
  --execute
