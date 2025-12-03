#!/bin/bash
#SBATCH -J vlm-g25
#SBATCH -o /vast/sxr203/newspaper-downloads/dedupe-webp/logs/%A_%a.out
#SBATCH -e /vast/sxr203/newspaper-downloads/dedupe-webp/logs/%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --array=0-199

set -euo pipefail

# Paths
LAYOUTS_PREFIX=${LAYOUTS_PREFIX:-/vast/sxr203/newspaper-downloads/dedupe-webp/layouts_chunk_}
PNG_ROOT=${PNG_ROOT:-/vast/sxr203/newspaper-downloads/dedupe-webp/unique_png}
OUT_DIR=${OUT_DIR:-/vast/sxr203/newspaper-downloads/dedupe-webp/vlm_gemini25}
CROPS_DIR=${CROPS_DIR:-/vast/sxr203/newspaper-downloads/dedupe-webp/vlm_crops}
LOG_DIR=$(dirname $(dirname ${LAYOUTS_PREFIX}))/logs
mkdir -p "${OUT_DIR}" "${CROPS_DIR}" "${LOG_DIR}"

# Pick layouts file for this task
IDX=$(printf "%02d" ${SLURM_ARRAY_TASK_ID})
LAYOUTS_FILE="${LAYOUTS_PREFIX}${IDX}"
if [[ ! -f "${LAYOUTS_FILE}" ]]; then
  echo "Layouts file ${LAYOUTS_FILE} not found" >&2
  exit 1
fi

# Virtualenv (expects repo cloned with submodules, venv precreated)
if [[ -z "${VENV:-}" ]]; then
  echo "Set VENV to your virtualenv path" >&2
  exit 1
fi
source "${VENV}/bin/activate"

# Start gateway
PORT=$((8000 + (RANDOM % 500)))
DEFAULT_PROVIDER=${DEFAULT_PROVIDER:-gemini}
GEMINI_KEY=${GEMINI_KEY:?Need GEMINI_KEY in env}
uvicorn gateway.app:create_app --factory --host 127.0.0.1 --port ${PORT} --log-level error &
GWPID=$!
sleep 5

python scripts/run_vlm_gateway.py \
  --layouts "@${LAYOUTS_FILE}" \
  --png-root "${PNG_ROOT}" \
  --output-dir "${OUT_DIR}" \
  --model gemini:gemini-2.5-flash \
  --gateway-url "http://127.0.0.1:${PORT}" \
  --max-concurrency 4 \
  --timeout 120 \
  --max-retries 1 \
  --save-crops-dir "${CROPS_DIR}/${SLURM_ARRAY_TASK_ID}"

kill ${GWPID} || true
