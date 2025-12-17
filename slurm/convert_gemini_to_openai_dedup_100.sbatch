#!/bin/bash
# Convert existing Gemini batch request JSONLs into upload-ready OpenAI Batch JSONLs.
#
# This job is "offline": it only reshapes JSONL files on VAST.
# It does NOT call provider batch APIs.
#
# Expected inputs on Greene (already generated):
#   - IN_BASE/part_XXXX/gemini_requests_shardNNN.jsonl
#   - IN_BASE/part_XXXX/mapping_shardNNN.jsonl
#
# Output:
#   - IN_BASE/part_XXXX/OPENAI_OUT_SUBDIR/ with:
#       - openai_requests_shardMMM.jsonl
#       - mapping_shardMMM.jsonl
#       - split_manifest.jsonl

#SBATCH -J convert-openai-dedup
#SBATCH -o /vast/sxr203/newspaper-downloads/dedupe-webp/logs/convert_openai_dedup_%A_%a.out
#SBATCH -e /vast/sxr203/newspaper-downloads/dedupe-webp/logs/convert_openai_dedup_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --array=0-469%40
#SBATCH -p short

set -euo pipefail

PROJECT_ROOT=/scratch/sxr203/newspaper-parsing-export
VENV=/scratch/sxr203/newspaper-parsing/venv

IN_BASE=${IN_BASE:-/vast/sxr203/newspaper-downloads/dedupe-webp/batch_requests_dedup_jpeg1024_mp1_q80}

OPENAI_MODEL=${OPENAI_MODEL:-gpt-5.2}
OPENAI_REASONING_EFFORT=${OPENAI_REASONING_EFFORT:-medium}
MAX_OPENAI_BYTES=${MAX_OPENAI_BYTES:-180000000}
LINES_PER_SHARD=${LINES_PER_SHARD:-0}

OPENAI_OUT_SUBDIR=${OPENAI_OUT_SUBDIR:-openai_gpt52_reasoning_medium_split}

idx=$(printf "%04d" "${SLURM_ARRAY_TASK_ID}")
part_dir="${IN_BASE}/part_${idx}"
out_dir="${part_dir}/${OPENAI_OUT_SUBDIR}"
done_flag="${out_dir}/DONE"

if [[ ! -d "${part_dir}" ]]; then
  echo "Missing part dir: ${part_dir}" >&2
  exit 1
fi

if [[ -f "${done_flag}" ]]; then
  echo "Already done: ${done_flag}"
  exit 0
fi

mkdir -p "${out_dir}"

source "${VENV}/bin/activate"

PYTHONUNBUFFERED=1 \
python "${PROJECT_ROOT}/scripts/convert_gemini_batch_dir_to_openai.py" \
  --in-dir "${part_dir}" \
  --out-dir "${out_dir}" \
  --openai-model "${OPENAI_MODEL}" \
  --openai-reasoning-effort "${OPENAI_REASONING_EFFORT}" \
  --max-openai-bytes "${MAX_OPENAI_BYTES}" \
  --lines-per-shard "${LINES_PER_SHARD}"

touch "${done_flag}"
echo "Wrote DONE flag: ${done_flag}"

