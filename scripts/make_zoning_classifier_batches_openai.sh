#!/usr/bin/env bash
set -euo pipefail

#
# Wrapper: export per-box zoning-classifier OpenAI batch request shards.
#
# Defaults assume you've already produced per-page `*.vlm.json` OCR outputs via the earlier pipeline.
# Override via:
#   PAGES_SPEC=... OUT_DIR=... ./scripts/make_zoning_classifier_batches_openai.sh
#

PAGES_SPEC="${PAGES_SPEC:-newspaper-parsing-local/data/vlm_out_openai_gpt52_reasoning_medium_split_pages_with_ok/*.vlm.json}"

if [[ $# -ge 1 ]] && [[ -n "${1:-}" ]]; then
  PAGES_SPEC="$1"
fi

OUT_ROOT="${OUT_ROOT:-newspaper-parsing-local/data}"
TAG="${TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${OUT_ROOT}/batch_requests_zoning_boxes_openai_gpt5nano_reasoning_medium_${TAG}}"
MIN_CHARS="${MIN_CHARS:-1}"

python scripts/export_zoning_classifier_batch_requests.py \
  --pages "${PAGES_SPEC}" \
  --output-dir "${OUT_DIR}" \
  --provider openai \
  --mode boxes \
  --openai-model gpt-5-nano \
  --openai-reasoning-effort medium \
  --openai-text-format json_schema \
  --requests-per-shard 5000 \
  --max-bytes-per-shard 180000000 \
  --max-boxes 0 \
  --min-chars "${MIN_CHARS}" \
  --skip-empty

echo ""
echo "Wrote OpenAI request shards to: ${OUT_DIR}"
echo ""
echo "Next (submit):"
echo "  python scripts/submit_batch_shards.py --request-dir \"${OUT_DIR}\" --providers openai --openai-endpoint /v1/responses"
