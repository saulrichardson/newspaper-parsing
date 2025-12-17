#!/usr/bin/env bash
set -euo pipefail

# Wrapper to build an XLSX workbook with ordinance questionnaire answers.
#
# Usage:
#   ./scripts/export_ordinance_questionnaire_answers_to_excel.sh <OUT_XLSX>
#
# Optional env overrides:
#   QUESTIONS_XLSX=~/Downloads/Questions.xlsx
#   RUN1_REQ_DIR=...
#   RUN1_RES_DIR=...
#   RUN2_REQ_DIR=...
#   RUN2_RES_DIR=...
#   SKIP_RUN2=1  (export run #1 only, even if run #2 dirs exist)
#
# By default this uses the known run #1 + run #2 directories under newspaper-parsing-local/data.

OUT_XLSX="${1:-}"
if [[ -z "${OUT_XLSX}" ]]; then
  echo "Usage: $0 <OUT_XLSX>" >&2
  exit 1
fi

QUESTIONS_XLSX="${QUESTIONS_XLSX:-${HOME}/Downloads/Questions.xlsx}"

RUN1_REQ_DIR="${RUN1_REQ_DIR:-newspaper-parsing-local/data/batch_requests_ordinance_questionnaire_openai_gpt5nano_reasoning_medium_20251217_025258}"
RUN1_RES_DIR="${RUN1_RES_DIR:-newspaper-parsing-local/data/batch_results_ordinance_questionnaire_openai_gpt5nano_reasoning_medium_20251217_025258}"

RUN2_REQ_DIR="${RUN2_REQ_DIR:-newspaper-parsing-local/data/batch_requests_ordinance_questionnaire_openai_gpt5nano_reasoning_medium_20251217_062159_incremental}"
RUN2_RES_DIR="${RUN2_RES_DIR:-newspaper-parsing-local/data/batch_results_ordinance_questionnaire_openai_gpt5nano_reasoning_medium_20251217_062159_incremental}"
SKIP_RUN2="${SKIP_RUN2:-0}"

cmd=(python scripts/export_ordinance_questionnaire_answers_to_excel.py
  --questions-xlsx "${QUESTIONS_XLSX}"
  --out-xlsx "${OUT_XLSX}"
  --run "${RUN1_REQ_DIR}" "${RUN1_RES_DIR}"
)

if [[ "${SKIP_RUN2}" != "1" ]]; then
  if [[ -d "${RUN2_REQ_DIR}" ]] && [[ ! -d "${RUN2_RES_DIR}" ]]; then
    echo "Run #2 request dir exists but results dir is missing:" >&2
    echo "  RUN2_REQ_DIR=${RUN2_REQ_DIR}" >&2
    echo "  RUN2_RES_DIR=${RUN2_RES_DIR}" >&2
    echo "" >&2
    echo "Either:" >&2
    echo "  - set SKIP_RUN2=1 to export run #1 only, OR" >&2
    echo "  - download + normalize run #2, then re-run this script." >&2
    exit 1
  fi
  if [[ -d "${RUN2_REQ_DIR}" ]] && [[ -d "${RUN2_RES_DIR}" ]]; then
    cmd+=(--run "${RUN2_REQ_DIR}" "${RUN2_RES_DIR}")
  fi
fi

PYTHONUNBUFFERED=1 "${cmd[@]}"
