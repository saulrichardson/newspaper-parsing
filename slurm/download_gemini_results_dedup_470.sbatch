#!/bin/bash
# Download Gemini Batch result files for the dedup export shards (Dell text boxes only).
#
# This job CALLS the Gemini API to download completed batch output JSONL.
#
# Output:
#   - <part_dir>/results_gemini/gemini_results_shardNNN.jsonl
#   - <part_dir>/results_gemini/downloaded_batches.jsonl
#
# Safe to re-run:
#   - download_gemini_batch_results.py supports --skip-existing

#SBATCH -J dl-gemini-dedup
#SBATCH -o /vast/sxr203/newspaper-downloads/dedupe-webp/logs/download_gemini_dedup_%A_%a.out
#SBATCH -e /vast/sxr203/newspaper-downloads/dedupe-webp/logs/download_gemini_dedup_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --array=0-469%15
#SBATCH -p short

set -euo pipefail

PROJECT_ROOT=/scratch/sxr203/newspaper-parsing-export
VENV=/scratch/sxr203/newspaper-parsing/venv

IN_BASE=${IN_BASE:-/vast/sxr203/newspaper-downloads/dedupe-webp/batch_requests_dedup_jpeg1024_mp1_q80}
SECRETS_FILE=${SECRETS_FILE:-/scratch/sxr203/.secrets/gemini.env}

DRY_RUN=${DRY_RUN:-0}

idx=$(printf "%04d" "${SLURM_ARRAY_TASK_ID}")
part_dir="${IN_BASE}/part_${idx}"
record_path="${part_dir}/submitted_batches.jsonl"
out_dir="${part_dir}/results_gemini"
done_flag="${out_dir}/DOWNLOADED_GEMINI"

if [[ ! -d "${part_dir}" ]]; then
  echo "Missing part dir: ${part_dir} (skip)" >&2
  exit 0
fi

if [[ ! -f "${record_path}" ]]; then
  echo "No Gemini submissions in ${part_dir} (missing ${record_path}); skip."
  exit 0
fi

if [[ ! -f "${SECRETS_FILE}" ]]; then
  echo "Missing secrets file: ${SECRETS_FILE}" >&2
  exit 1
fi

source "${SECRETS_FILE}"
if [[ -z "${GEMINI_KEY:-}" ]]; then
  echo "GEMINI_KEY not set after sourcing ${SECRETS_FILE}" >&2
  exit 1
fi

source "${VENV}/bin/activate"

cmd=(python "${PROJECT_ROOT}/scripts/download_gemini_batch_results.py" \
  --request-dir "${part_dir}" \
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

