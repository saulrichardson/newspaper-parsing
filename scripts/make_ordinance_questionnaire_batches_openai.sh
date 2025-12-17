#!/usr/bin/env bash
set -euo pipefail

#
# Wrapper: export OpenAI (and optionally Gemini) batch request shards for the ordinance/amendment questionnaire step.
#
# Inputs:
#   - The PRIOR zoning box-classifier run artifacts:
#       * mapping_shard*.jsonl (from the classifier request dir)
#       * openai_results_shard*.jsonl (from the classifier results dir)
#   - Questions workbook (default: ~/Downloads/Questions.xlsx, sheet: "Processed Info")
#
# Usage:
#   ./scripts/make_ordinance_questionnaire_batches_openai.sh \
#     <CLASSIFIER_REQUEST_DIR> <CLASSIFIER_RESULTS_DIR>
#
# Or with env vars:
#   CLS_REQUEST_DIR=... CLS_RESULTS_DIR=... ./scripts/make_ordinance_questionnaire_batches_openai.sh
#

LATEST_CLS_REQUEST_DIR="$(ls -td newspaper-parsing-local/data/batch_requests_zoning_boxes_openai_* 2>/dev/null | head -n 1 || true)"
LATEST_CLS_RESULTS_DIR="$(ls -td newspaper-parsing-local/data/batch_results_zoning_boxes_openai_* 2>/dev/null | head -n 1 || true)"

CLS_REQUEST_DIR="${CLS_REQUEST_DIR:-${LATEST_CLS_REQUEST_DIR}}"
CLS_RESULTS_DIR="${CLS_RESULTS_DIR:-${LATEST_CLS_RESULTS_DIR}}"

if [[ $# -ge 1 ]] && [[ -n "${1:-}" ]]; then
  CLS_REQUEST_DIR="$1"
fi
if [[ $# -ge 2 ]] && [[ -n "${2:-}" ]]; then
  CLS_RESULTS_DIR="$2"
fi

if [[ -z "${CLS_REQUEST_DIR}" ]] || [[ ! -d "${CLS_REQUEST_DIR}" ]]; then
  echo "Missing CLS_REQUEST_DIR (classifier request dir with mapping_shard*.jsonl): ${CLS_REQUEST_DIR}" >&2
  exit 1
fi
if [[ -z "${CLS_RESULTS_DIR}" ]] || [[ ! -d "${CLS_RESULTS_DIR}" ]]; then
  echo "Missing CLS_RESULTS_DIR (classifier results dir with openai_results_shard*.jsonl): ${CLS_RESULTS_DIR}" >&2
  exit 1
fi

QUESTIONS_XLSX="${QUESTIONS_XLSX:-${HOME}/Downloads/Questions.xlsx}"
PROMPT_PATH="${PROMPT_PATH:-prompts/ordinance_box_questionnaire_prompt_text.txt}"

OUT_ROOT="${OUT_ROOT:-newspaper-parsing-local/data}"
TAG="${TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${OUT_ROOT}/batch_requests_ordinance_questionnaire_openai_${TAG}}"

OPENAI_MODEL="${OPENAI_MODEL:-gpt-5-nano}"
OPENAI_REASONING="${OPENAI_REASONING:-medium}"

INCLUDE_LABELS="${INCLUDE_LABELS:-full_ordinance,amendment_substantial,amendment_targeted}"
MIN_CONFIDENCE="${MIN_CONFIDENCE:-0.0}"

SHARDS="${SHARDS:-}"
MAX_REQUESTS="${MAX_REQUESTS:-}"
ALLOW_PARTIAL_RESULTS="${ALLOW_PARTIAL_RESULTS:-0}"

cmd=(python scripts/export_ordinance_questionnaire_batch_requests.py \
  --classification-request-dir "${CLS_REQUEST_DIR}" \
  --classification-results-dir "${CLS_RESULTS_DIR}" \
  --output-dir "${OUT_DIR}" \
  --questions-xlsx "${QUESTIONS_XLSX}" \
  --prompt-path "${PROMPT_PATH}" \
  --provider openai \
  --include-labels "${INCLUDE_LABELS}" \
  --min-confidence "${MIN_CONFIDENCE}" \
  --openai-model "${OPENAI_MODEL}" \
  --openai-reasoning-effort "${OPENAI_REASONING}" \
  --openai-text-format none \
  --requests-per-shard 5000 \
  --max-bytes-per-shard 180000000)

if [[ -n "${SHARDS}" ]]; then
  cmd+=(--shards "${SHARDS}")
fi
if [[ -n "${MAX_REQUESTS}" ]]; then
  cmd+=(--max-requests "${MAX_REQUESTS}")
fi
if [[ "${ALLOW_PARTIAL_RESULTS}" == "1" ]]; then
  cmd+=(--allow-partial-results)
fi

PYTHONUNBUFFERED=1 "${cmd[@]}"

echo ""
echo "Wrote questionnaire batch requests to: ${OUT_DIR}"
echo ""
echo "Next (submit locally):"
echo "  python scripts/submit_batch_shards.py --request-dir \"${OUT_DIR}\" --providers openai --openai-endpoint /v1/responses"
echo ""
echo "Next (submit on Greene via Slurm):"
echo "  REQUEST_DIR=<path_on_greene_to_${OUT_DIR}> sbatch --array=0-<N-1>%8 slurm/submit_openai_zoning_boxes_array.sbatch"
