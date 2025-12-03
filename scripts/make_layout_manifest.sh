#!/usr/bin/env bash
set -euo pipefail
ROOT=${1:-/vast/sxr203/newspaper-downloads/dedupe-webp}
OUT=${2:-${ROOT}/layouts_all.txt}
find "${ROOT}/unique_outputs" -type f -name '*.json' | sort > "${OUT}"
echo "Wrote ${OUT} with $(wc -l < "${OUT}") layouts"
