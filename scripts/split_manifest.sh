#!/usr/bin/env bash
set -euo pipefail
MANIFEST=${1:-/vast/sxr203/newspaper-downloads/dedupe-webp/layouts_all.txt}
CHUNKS=${2:-200}
PREFIX=${3:-/vast/sxr203/newspaper-downloads/dedupe-webp/layouts_chunk_}
split -n l/${CHUNKS} "${MANIFEST}" "${PREFIX}"
count=$(ls ${PREFIX}* | wc -l)
echo "Split ${MANIFEST} into ${count} chunks with prefix ${PREFIX}"
