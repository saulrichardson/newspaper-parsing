#!/usr/bin/env bash
set -euo pipefail

#
# Wrapper: run the ordinance/amendment questionnaire step synchronously (no Batch API).
#
# This reads prior zoning box-classifier outputs, selects ordinance/amendment boxes,
# and calls OpenAI /v1/responses directly with local validation + retries.
#
# Usage:
#   ./scripts/run_ordinance_questionnaire_openai_sync.sh \
#     <CLASSIFIER_REQUEST_DIR> <CLASSIFIER_RESULTS_DIR>
#
# Or rely on auto-detected latest dirs under newspaper-parsing-local/data/.
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
OUT_DIR="${OUT_DIR:-${OUT_ROOT}/ordinance_questionnaire_sync_openai_${TAG}}"

OPENAI_MODEL="${OPENAI_MODEL:-gpt-5-nano}"
OPENAI_REASONING="${OPENAI_REASONING:-medium}"

INCLUDE_LABELS="${INCLUDE_LABELS:-full_ordinance,amendment_substantial,amendment_targeted}"
MIN_CONFIDENCE="${MIN_CONFIDENCE:-0.0}"

SHARDS="${SHARDS:-}"
MAX_BOXES="${MAX_BOXES:-}"
ALLOW_PARTIAL_RESULTS="${ALLOW_PARTIAL_RESULTS:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
DRY_RUN="${DRY_RUN:-0}"

cmd=(python scripts/run_ordinance_questionnaire_openai_sync.py \
  --classification-request-dir "${CLS_REQUEST_DIR}" \
  --classification-results-dir "${CLS_RESULTS_DIR}" \
  --output-dir "${OUT_DIR}" \
  --questions-xlsx "${QUESTIONS_XLSX}" \
  --prompt-path "${PROMPT_PATH}" \
  --include-labels "${INCLUDE_LABELS}" \
  --min-confidence "${MIN_CONFIDENCE}" \
  --openai-model "${OPENAI_MODEL}" \
  --openai-reasoning-effort "${OPENAI_REASONING}")

if [[ -n "${SHARDS}" ]]; then
  cmd+=(--shards "${SHARDS}")
fi
if [[ -n "${MAX_BOXES}" ]]; then
  cmd+=(--max-boxes "${MAX_BOXES}")
fi
if [[ "${ALLOW_PARTIAL_RESULTS}" == "1" ]]; then
  cmd+=(--allow-partial-results)
fi
if [[ "${SKIP_EXISTING}" == "1" ]]; then
  cmd+=(--skip-existing)
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  cmd+=(--dry-run)
fi

PYTHONUNBUFFERED=1 "${cmd[@]}"

echo ""
echo "Wrote outputs to: ${OUT_DIR}"
echo "Manifest: ${OUT_DIR}/manifest.jsonl"

