#!/usr/bin/env bash
set -euo pipefail

#
# Rsync a local batch request directory to Greene.
#
# This is intended for directories produced by:
#   - scripts/export_batch_requests.py
#   - scripts/export_zoning_classifier_batch_requests.py
#
# Usage (from repo root, locally):
#   bash scripts/push_request_dir_to_greene.sh <local_request_dir>
#
# Optional overrides:
#   DEST_ROOT=/vast/sxr203/newspaper-downloads/dedupe-webp bash scripts/push_request_dir_to_greene.sh <dir>
#   DEST_DIR=/vast/sxr203/.../my_run bash scripts/push_request_dir_to_greene.sh <dir>
#

if [[ $# -lt 1 ]] || [[ -z "${1:-}" ]]; then
  echo "Usage: $0 <local_request_dir>" >&2
  exit 2
fi

SRC_DIR="$1"
if [[ ! -d "${SRC_DIR}" ]]; then
  echo "Source dir not found: ${SRC_DIR}" >&2
  exit 1
fi

DEST_ROOT="${DEST_ROOT:-/vast/sxr203/newspaper-downloads/dedupe-webp}"
BASE="$(basename "${SRC_DIR}")"
DEST_DIR="${DEST_DIR:-${DEST_ROOT}/${BASE}}"

echo "Syncing request dir to Greene:"
echo "  SRC=${SRC_DIR}"
echo "  DEST=greene:${DEST_DIR}"

ssh greene "mkdir -p \"${DEST_DIR}\""

# Use rsync so the transfer is resumable.
# Note: macOS ships an older rsync that may not support --info=progress2.
rsync -av --progress --partial "${SRC_DIR}/" "greene:${DEST_DIR}/"

echo ""
echo "Done."
echo "Greene destination: ${DEST_DIR}"
