#!/bin/bash
# Submit OpenAI Batch jobs for the dedup export request shards (Dell text boxes only).
#
# This job CALLS the OpenAI API (uploads JSONL + creates a Batch job). It is NOT offline.
#
# Prereqs on Greene:
#   - Request files exist under IN_BASE/part_XXXX/OPENAI_SUBDIR:
#       - openai_requests_shardNNN.jsonl
#       - mapping_shardNNN.jsonl
#   - Python venv has `openai` installed (VENV below)
#   - OPENAI_KEY is available via SECRETS_FILE (see scripts/push_openai_key_to_greene.sh)
#
# Output:
#   - Per-part submission record: IN_BASE/part_XXXX/OPENAI_SUBDIR/submitted_batches.jsonl
#   - Per-part done flag: IN_BASE/part_XXXX/OPENAI_SUBDIR/SUBMITTED_OPENAI
#
# Safe to re-run:
#   - submit_batch_shards.py is resumable via submitted_batches.jsonl
#   - done flag prevents redundant work once a part fully succeeded

#SBATCH -J submit-openai-dedup
#SBATCH -o /vast/sxr203/newspaper-downloads/dedupe-webp/logs/submit_openai_dedup_%A_%a.out
#SBATCH -e /vast/sxr203/newspaper-downloads/dedupe-webp/logs/submit_openai_dedup_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --array=0-469%10
#SBATCH -p short

set -euo pipefail

PROJECT_ROOT=/scratch/sxr203/newspaper-parsing-export
VENV=/scratch/sxr203/newspaper-parsing/venv

IN_BASE=${IN_BASE:-/vast/sxr203/newspaper-downloads/dedupe-webp/batch_requests_dedup_jpeg1024_mp1_q80}
OPENAI_SUBDIR=${OPENAI_SUBDIR:-openai_gpt52_reasoning_medium_split}
SECRETS_FILE=${SECRETS_FILE:-/scratch/sxr203/.secrets/openai.env}

# If DRY_RUN=1, validates shards + prints what would be submitted, but does not call the API.
DRY_RUN=${DRY_RUN:-0}

idx=$(printf "%04d" "${SLURM_ARRAY_TASK_ID}")
part_dir="${IN_BASE}/part_${idx}/${OPENAI_SUBDIR}"
done_flag="${part_dir}/SUBMITTED_OPENAI"

if [[ ! -d "${part_dir}" ]]; then
  echo "Missing OpenAI request dir: ${part_dir}" >&2
  exit 1
fi

if [[ "${DRY_RUN}" != "1" ]] && [[ -f "${done_flag}" ]]; then
  echo "Already done: ${done_flag}"
  exit 0
fi

if [[ ! -f "${SECRETS_FILE}" ]]; then
  echo "Missing secrets file: ${SECRETS_FILE}" >&2
  echo "Create it via: scripts/push_openai_key_to_greene.sh (run locally)" >&2
  exit 1
fi

source "${SECRETS_FILE}"
if [[ -z "${OPENAI_KEY:-}" ]]; then
  echo "OPENAI_KEY was not set after sourcing ${SECRETS_FILE}" >&2
  exit 1
fi

source "${VENV}/bin/activate"

cmd=(python "${PROJECT_ROOT}/scripts/submit_batch_shards.py" \
  --request-dir "${part_dir}" \
  --providers openai \
  --openai-endpoint /v1/responses)

if [[ "${DRY_RUN}" == "1" ]]; then
  cmd+=(--dry-run)
fi

PYTHONUNBUFFERED=1 "${cmd[@]}"

if [[ "${DRY_RUN}" != "1" ]]; then
  touch "${done_flag}"
  echo "Wrote DONE flag: ${done_flag}"
fi

