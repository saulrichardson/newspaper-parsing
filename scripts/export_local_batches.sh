#!/usr/bin/env bash
set -euo pipefail

# Export provider-ready batch request JSONLs for the local staged PNG set.
#
# Defaults are tuned to:
#   - OpenAI: gpt-5.2 with reasoning.effort=medium
#   - Gemini: requests compatible with models/gemini-2.5-flash (model chosen at submit time)
#   - JPEG crops downscaled for manageable batch file sizes
#
# Usage:
#   bash scripts/export_local_batches.sh
#   bash scripts/export_local_batches.sh /path/to/output_dir
#
# Optional overrides (env vars):
#   LAYOUTS_FILE                 (default: newspaper-parsing-local/data/staged_manifest/layouts.txt)
#   PNG_ROOT                     (default: newspaper-parsing-local/data/unique_png)
#   OUT_DIR
#   OPENAI_MODEL                 (default: gpt-5.2)
#   OPENAI_REASONING_EFFORT      (default: medium)
#   IMAGE_FORMAT                 (default: jpeg)
#   JPEG_QUALITY                 (default: 80)
#   MAX_CROP_DIM                 (default: 1024)
#   MAX_CROP_MEGAPIXELS          (default: 1.0)
#   BOXES_PER_SHARD              (default: 5000)
#   MAX_BYTES_PER_SHARD          (default: 180000000)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

LAYOUTS_FILE="${LAYOUTS_FILE:-newspaper-parsing-local/data/staged_manifest/layouts.txt}"
PNG_ROOT="${PNG_ROOT:-newspaper-parsing-local/data/unique_png}"

if [[ ! -f "${LAYOUTS_FILE}" ]]; then
  echo "Missing layouts list: ${LAYOUTS_FILE}" >&2
  exit 1
fi
if [[ ! -d "${PNG_ROOT}" ]]; then
  echo "Missing PNG root: ${PNG_ROOT}" >&2
  exit 1
fi

OPENAI_MODEL="${OPENAI_MODEL:-gpt-5.2}"
OPENAI_REASONING_EFFORT="${OPENAI_REASONING_EFFORT:-medium}"

IMAGE_FORMAT="${IMAGE_FORMAT:-jpeg}"
JPEG_QUALITY="${JPEG_QUALITY:-80}"
MAX_CROP_DIM="${MAX_CROP_DIM:-1024}"
MAX_CROP_MEGAPIXELS="${MAX_CROP_MEGAPIXELS:-1.0}"
BOXES_PER_SHARD="${BOXES_PER_SHARD:-5000}"
MAX_BYTES_PER_SHARD="${MAX_BYTES_PER_SHARD:-180000000}"

timestamp="$(date +%Y%m%d_%H%M%S)"
default_out="newspaper-parsing-local/data/batch_requests_all_local16511_openai-${OPENAI_MODEL}_reasoning-${OPENAI_REASONING_EFFORT}_gemini-2.5-flash_jpeg${MAX_CROP_DIM}_mp${MAX_CROP_MEGAPIXELS}_q${JPEG_QUALITY}_max${MAX_BYTES_PER_SHARD}_${timestamp}"

OUT_DIR="${OUT_DIR:-${1:-${default_out}}}"

if [[ -e "${OUT_DIR}" ]]; then
  echo "Refusing to overwrite existing OUT_DIR: ${OUT_DIR}" >&2
  exit 1
fi

echo "Exporting batch requests..."
echo "  layouts:   ${LAYOUTS_FILE}"
echo "  png-root:  ${PNG_ROOT}"
echo "  out-dir:   ${OUT_DIR}"
echo "  providers: openai+gemini"
echo "  openai:    model=${OPENAI_MODEL} reasoning.effort=${OPENAI_REASONING_EFFORT}"
echo "  crops:     ${IMAGE_FORMAT} q=${JPEG_QUALITY} max_dim=${MAX_CROP_DIM} max_mp=${MAX_CROP_MEGAPIXELS}"
echo "  sharding:  boxes_per_shard=${BOXES_PER_SHARD} max_bytes_per_shard=${MAX_BYTES_PER_SHARD}"
echo

python - <<'PY'
import sys
try:
    import PIL  # noqa: F401
    import pydantic  # noqa: F401
except Exception as exc:
    print("Missing Python deps for export (need at least Pillow + pydantic).", file=sys.stderr)
    print("Fix by running: pip install -r requirements.txt", file=sys.stderr)
    raise SystemExit(1) from exc
PY

python scripts/export_batch_requests.py \
  --layouts "@${LAYOUTS_FILE}" \
  --png-root "${PNG_ROOT}" \
  --output-dir "${OUT_DIR}" \
  --provider both \
  --openai-model "${OPENAI_MODEL}" \
  --openai-reasoning-effort "${OPENAI_REASONING_EFFORT}" \
  --image-format "${IMAGE_FORMAT}" \
  --jpeg-quality "${JPEG_QUALITY}" \
  --max-crop-dim "${MAX_CROP_DIM}" \
  --max-crop-megapixels "${MAX_CROP_MEGAPIXELS}" \
  --boxes-per-shard "${BOXES_PER_SHARD}" \
  --max-bytes-per-shard "${MAX_BYTES_PER_SHARD}" \
  --skip-bad-layouts

echo
echo "Done. Wrote batch request shards to:"
echo "  ${OUT_DIR}"
echo
echo "Sanity-check (no API calls):"
echo "  python scripts/submit_batch_shards.py --request-dir \"${OUT_DIR}\" --providers both --dry-run"
echo
echo "Submit tomorrow (calls provider APIs; requires GEMINI_KEY + OPENAI_API_KEY):"
echo "  python scripts/submit_batch_shards.py --request-dir \"${OUT_DIR}\" --providers both --gemini-model models/gemini-2.5-flash"
