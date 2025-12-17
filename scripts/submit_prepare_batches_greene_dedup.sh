#!/bin/bash
#
# Greene helper: (1) ensure dedup Gemini exports are complete, then (2) generate OpenAI Batch shards.
#
# Run on Greene login node:
#   bash /scratch/sxr203/newspaper-parsing-export/scripts/submit_prepare_batches_greene_dedup.sh
#
# You can override defaults via env vars:
#   IN_BASE=... OPENAI_MODEL=... OPENAI_REASONING_EFFORT=... bash ...

set -euo pipefail

PROJECT_ROOT=/scratch/sxr203/newspaper-parsing-export

IN_BASE=${IN_BASE:-/vast/sxr203/newspaper-downloads/dedupe-webp/batch_requests_dedup_jpeg1024_mp1_q80}

export OUT_BASE=${OUT_BASE:-${IN_BASE}}
export PROVIDER=${PROVIDER:-gemini}
export OPENAI_MODEL=${OPENAI_MODEL:-gpt-5.2}
export OPENAI_REASONING_EFFORT=${OPENAI_REASONING_EFFORT:-medium}
export MAX_OPENAI_BYTES=${MAX_OPENAI_BYTES:-180000000}
export LINES_PER_SHARD=${LINES_PER_SHARD:-0}
export OPENAI_OUT_SUBDIR=${OPENAI_OUT_SUBDIR:-openai_gpt52_reasoning_medium_split}

missing=()
for i in $(seq 0 469); do
  idx=$(printf "%04d" "${i}")
  if [[ ! -f "${IN_BASE}/part_${idx}/DONE" ]]; then
    missing+=("${i}")
  fi
done

export_job=""
if [[ ${#missing[@]} -gt 0 ]]; then
  array_spec=$(IFS=,; echo "${missing[*]}")
  echo "Missing DONE for ${#missing[@]} parts; resubmitting export for array=${array_spec}"
  export_job=$(sbatch --parsable --array="${array_spec}" "${PROJECT_ROOT}/slurm/export_batch_requests_dedup_100.sbatch")
  echo "Submitted export job: ${export_job}"
else
  echo "All parts have DONE under ${IN_BASE}"
fi

dep=()
if [[ -n "${export_job}" ]]; then
  dep=(--dependency="afterok:${export_job}")
fi

convert_job=$(sbatch --parsable "${dep[@]}" "${PROJECT_ROOT}/slurm/convert_gemini_to_openai_dedup_100.sbatch")
echo "Submitted convert job: ${convert_job}"

echo ""
echo "Monitor with:"
echo "  squeue -u ${USER}"
echo "Logs:"
echo "  /vast/sxr203/newspaper-downloads/dedupe-webp/logs/ (convert_openai_dedup_* and export_batch_dedup_*)"

