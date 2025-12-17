#!/bin/bash
# Download OpenAI Batch result files for the dedup export shards (Dell text boxes only).
#
# This job CALLS the OpenAI API to download completed batch output/error JSONL.
#
# Output:
#   - <part_dir>/<OPENAI_SUBDIR>/results/openai_results_shardNNN.jsonl
#   - <part_dir>/<OPENAI_SUBDIR>/results/openai_errors_shardNNN.jsonl
#   - <part_dir>/<OPENAI_SUBDIR>/results/downloaded_batches.jsonl
#
# Safe to re-run:
#   - download_openai_batch_results.py supports --skip-existing

#SBATCH -J dl-openai-dedup
#SBATCH -o /vast/sxr203/newspaper-downloads/dedupe-webp/logs/download_openai_dedup_%A_%a.out
#SBATCH -e /vast/sxr203/newspaper-downloads/dedupe-webp/logs/download_openai_dedup_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --array=0-469%15
#SBATCH -p short

set -euo pipefail

PROJECT_ROOT=/scratch/sxr203/newspaper-parsing-export
VENV=/scratch/sxr203/newspaper-parsing/venv

IN_BASE=${IN_BASE:-/vast/sxr203/newspaper-downloads/dedupe-webp/batch_requests_dedup_jpeg1024_mp1_q80}
OPENAI_SUBDIR=${OPENAI_SUBDIR:-openai_gpt52_reasoning_medium_split}
SECRETS_FILE=${SECRETS_FILE:-/scratch/sxr203/.secrets/openai.env}

DRY_RUN=${DRY_RUN:-0}

idx=$(printf "%04d" "${SLURM_ARRAY_TASK_ID}")
request_dir="${IN_BASE}/part_${idx}/${OPENAI_SUBDIR}"
record_path="${request_dir}/submitted_batches.jsonl"
out_dir="${request_dir}/results"
done_flag="${out_dir}/DOWNLOADED_OPENAI"

if [[ ! -d "${request_dir}" ]]; then
  echo "Missing request dir: ${request_dir} (skip)" >&2
  exit 0
fi

if [[ ! -f "${record_path}" ]]; then
  echo "No OpenAI submissions in ${request_dir} (missing ${record_path}); skip."
  exit 0
fi

if [[ ! -f "${SECRETS_FILE}" ]]; then
  echo "Missing secrets file: ${SECRETS_FILE}" >&2
  exit 1
fi

source "${SECRETS_FILE}"
if [[ -z "${OPENAI_KEY:-}" ]] && [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_KEY/OPENAI_API_KEY not set after sourcing ${SECRETS_FILE}" >&2
  exit 1
fi

source "${VENV}/bin/activate"

cmd=(python "${PROJECT_ROOT}/scripts/download_openai_batch_results.py" \
  --request-dir "${request_dir}" \
  --out-dir "${out_dir}" \
  --skip-existing)

if [[ "${DRY_RUN}" == "1" ]]; then
  cmd+=(--dry-run)
fi

PYTHONUNBUFFERED=1 "${cmd[@]}"

if [[ "${DRY_RUN}" != "1" ]]; then
  touch "${done_flag}"
  echo "Wrote DONE flag: ${done_flag}"
fi

