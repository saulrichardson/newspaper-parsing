#!/bin/bash
# Submit OpenAI Batch jobs for zoning classification (per-box).
#
# This job CALLS the OpenAI API (uploads JSONL + creates a Batch job). It is NOT offline.
#
# Expected request dir contents:
#   - openai_requests_shardNNN.jsonl
#   - mapping_shardNNN.jsonl
#
# Key handling:
#   - Source a secrets file with PROJECT_OPENAI_KEY (preferred) and/or OPENAI_KEY / OPENAI_API_KEY.
#   - See: scripts/push_openai_key_to_greene.sh
#
# IMPORTANT:
#   - Each array task submits exactly ONE shard (SLURM_ARRAY_TASK_ID).
#   - To avoid concurrent writes to the same submitted_batches.jsonl, each task writes its own record:
#       <REQUEST_DIR>/submitted_batches_shardNNN.jsonl
#   - You can later merge records if desired.
#
# Usage (on Greene):
#   REQUEST_DIR=/vast/sxr203/newspaper-downloads/dedupe-webp/<your_request_dir> \
#   sbatch --array=0-75%8 slurm/submit_openai_zoning_boxes_array.sbatch
#
# Optional:
#   DRY_RUN=1 ... (validates + prints; no API calls)

#SBATCH -J submit-openai-zoning
#SBATCH -o /vast/sxr203/newspaper-downloads/dedupe-webp/logs/submit_openai_zoning_%A_%a.out
#SBATCH -e /vast/sxr203/newspaper-downloads/dedupe-webp/logs/submit_openai_zoning_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --array=0-75%8
#SBATCH -p short

set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/scratch/sxr203/newspaper-parsing-export}
VENV=${VENV:-/scratch/sxr203/newspaper-parsing/venv}

REQUEST_DIR=${REQUEST_DIR:?Set REQUEST_DIR to the request shard directory on /vast or /scratch}
SECRETS_FILE=${SECRETS_FILE:-/scratch/sxr203/.secrets/openai.env}

DRY_RUN=${DRY_RUN:-0}

shard="${SLURM_ARRAY_TASK_ID}"
idx=$(printf "%03d" "${shard}")

req_path="${REQUEST_DIR}/openai_requests_shard${idx}.jsonl"
map_path="${REQUEST_DIR}/mapping_shard${idx}.jsonl"
rec_path="${REQUEST_DIR}/submitted_batches_shard${idx}.jsonl"
rec_path="${RECORD_PATH:-${rec_path}}"

if [[ ! -d "${REQUEST_DIR}" ]]; then
  echo "Missing REQUEST_DIR: ${REQUEST_DIR}" >&2
  exit 1
fi
if [[ ! -f "${req_path}" ]]; then
  echo "Missing requests file: ${req_path}" >&2
  exit 1
fi
if [[ ! -f "${map_path}" ]]; then
  echo "Missing mapping file: ${map_path}" >&2
  exit 1
fi

if [[ ! -f "${SECRETS_FILE}" ]]; then
  echo "Missing secrets file: ${SECRETS_FILE}" >&2
  echo "Create/update it via: scripts/push_openai_key_to_greene.sh (run locally)" >&2
  exit 1
fi
source "${SECRETS_FILE}"

source "${VENV}/bin/activate"

cmd=(python "${PROJECT_ROOT}/scripts/submit_batch_shards.py" \
  --request-dir "${REQUEST_DIR}" \
  --providers openai \
  --openai-endpoint /v1/responses \
  --shards "${shard}" \
  --record "${rec_path}")

if [[ "${DRY_RUN}" == "1" ]]; then
  cmd+=(--dry-run)
fi

PYTHONUNBUFFERED=1 "${cmd[@]}"
