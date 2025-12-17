#!/usr/bin/env bash
set -euo pipefail

# === CONFIG ===
REMOTE_USER="sxr203"
REMOTE_HOST="greene.hpc.nyu.edu"
REMOTE_BASE="/scratch/sxr203/newspaper-parsing"
REMOTE_DATA_BASE="/vast/sxr203/newspaper-downloads/dedupe-webp"
DEST_BASE="$PWD/newspaper-parsing-local"

# === LIGHTWEIGHT PULL (code + manifests + cache) ===
mkdir -p "$DEST_BASE"
echo "Syncing repo (excluding venv and large data)…"
rsync -avz --exclude 'venv' --exclude '.git' "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE}/" "$DEST_BASE/"

echo "Pulling dedup manifest and chunk lists (small)…"
mkdir -p "$DEST_BASE/chunks_dedup"
rsync -avz "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE}/chunks_dedup/manifest.txt" "$DEST_BASE/chunks_dedup/"
rsync -avz "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE}/chunks_dedup/chunk_*" "$DEST_BASE/chunks_dedup/"

echo "Pulling upload cache (optional, small)…"
rsync -avz "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DATA_BASE}/gemini_cache/file_uris.json" "$DEST_BASE/gemini_cache.json" || true

echo "Pulling .env (comment this line out if you don’t want secrets copied)…"
rsync -avz "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE}/.env" "$DEST_BASE/.env" || true

# === HEAVY PULL (layouts + PNGs) ===
# WARNING: large transfer (~50k PNGs). Ensure you have bandwidth and disk space.
mkdir -p "$DEST_BASE/data"
echo "Pulling layout JSONs (large)…"
rsync -avz "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DATA_BASE}/unique_outputs/" "$DEST_BASE/data/unique_outputs/"

echo "Pulling PNGs (very large)…"
rsync -avz "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DATA_BASE}/unique_png/" "$DEST_BASE/data/unique_png/"

echo "Done. Everything is in: $DEST_BASE"
