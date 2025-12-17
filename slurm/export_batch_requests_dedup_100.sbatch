#!/bin/bash
# Export base64-inline batch request JSONLs (Gemini/OpenAI) for the deduped layout set.
#
# This is intentionally "offline": it only generates request JSONL files on VAST.
# It does NOT call provider batch APIs.
#
# Expected inputs on Greene:
#   - Layout chunk manifest: /scratch/sxr203/newspaper-parsing/chunks_dedup_100/manifest.txt
#   - PNGs: /vast/sxr203/newspaper-downloads/dedupe-webp/unique_png/*.png
#
# Output:
#   - Per-task dirs under OUT_BASE/part_XXXX with:
#       - gemini_requests_shardNNN.jsonl (if PROVIDER includes gemini)
#       - openai_requests_shardNNN.jsonl (if PROVIDER includes openai)
#       - mapping_shardNNN.jsonl

#SBATCH -J export-batch-dedup
#SBATCH -o /vast/sxr203/newspaper-downloads/dedupe-webp/logs/export_batch_dedup_%A_%a.out
#SBATCH -e /vast/sxr203/newspaper-downloads/dedupe-webp/logs/export_batch_dedup_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --array=0-469%20
#SBATCH -p short

set -euo pipefail

PROJECT_ROOT=/scratch/sxr203/newspaper-parsing-export
VENV=/scratch/sxr203/newspaper-parsing/venv

CHUNK_MANIFEST=/scratch/sxr203/newspaper-parsing/chunks_dedup_100/manifest.txt
PNG_ROOT=/vast/sxr203/newspaper-downloads/dedupe-webp/unique_png

OUT_BASE=${OUT_BASE:-/vast/sxr203/newspaper-downloads/dedupe-webp/batch_requests_dedup_jpeg1024_mp1_q80}
PROVIDER=${PROVIDER:-both}           # gemini|openai|both
OPENAI_MODEL=${OPENAI_MODEL:-gpt-4o-mini}

IMAGE_FORMAT=${IMAGE_FORMAT:-jpeg}    # png|jpeg
JPEG_QUALITY=${JPEG_QUALITY:-80}
MAX_CROP_DIM=${MAX_CROP_DIM:-1024}
MAX_CROP_MEGAPIXELS=${MAX_CROP_MEGAPIXELS:-1.0}
# NOTE: OpenAI batch file size limits are typically tighter than Gemini. This default is chosen to
# reduce the chance a single shard becomes too large when PROVIDER includes openai.
BOXES_PER_SHARD=${BOXES_PER_SHARD:-2500}

if [[ ! -f "${CHUNK_MANIFEST}" ]]; then
  echo "Missing chunk manifest: ${CHUNK_MANIFEST}" >&2
  exit 1
fi

chunk=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "${CHUNK_MANIFEST}" || true)
if [[ -z "${chunk}" ]]; then
  echo "No chunk entry for task ${SLURM_ARRAY_TASK_ID} in ${CHUNK_MANIFEST}; exiting." >&2
  exit 0
fi
if [[ ! -f "${chunk}" ]]; then
  echo "Chunk file not found: ${chunk}" >&2
  exit 1
fi

idx=$(printf "%04d" "${SLURM_ARRAY_TASK_ID}")
out_dir="${OUT_BASE}/part_${idx}"
done_flag="${out_dir}/DONE_${PROVIDER}"

if [[ -f "${done_flag}" ]]; then
  echo "Already done: ${done_flag}"
  exit 0
fi

mkdir -p "${out_dir}"

source "${VENV}/bin/activate"

PYTHONUNBUFFERED=1 \
python "${PROJECT_ROOT}/scripts/export_batch_requests.py" \
  --layouts "@${chunk}" \
  --png-root "${PNG_ROOT}" \
  --output-dir "${out_dir}" \
  --provider "${PROVIDER}" \
  --openai-model "${OPENAI_MODEL}" \
  --boxes-per-shard "${BOXES_PER_SHARD}" \
  --image-format "${IMAGE_FORMAT}" \
  --jpeg-quality "${JPEG_QUALITY}" \
  --max-crop-dim "${MAX_CROP_DIM}" \
  --max-crop-megapixels "${MAX_CROP_MEGAPIXELS}" \
  --skip-bad-layouts

touch "${done_flag}"
echo "Wrote DONE flag: ${done_flag}"
