# Newspaper VLM Pipeline

Runs Dell layout crops through a VLM (Gemini 2.5 Flash) via agent-gateway. Target data live on NYU Greene VAST.

## Local artifacts (not committed)
This repo writes intermediate artifacts (batch shards, downloaded results, Excel/PDF reports, etc.) under `newspaper-parsing-local/`.
That directory is intentionally gitignored so you can keep large data + writeups locally without polluting Git.

## Data layout (Greene)
- PNGs: `/vast/sxr203/newspaper-downloads/dedupe-webp/unique_png/`
- Dell layouts: `/vast/sxr203/newspaper-downloads/dedupe-webp/unique_outputs/`
- Outputs: recommended `/vast/sxr203/newspaper-downloads/dedupe-webp/vlm_gemini25/`
- Crops: recommended `/vast/sxr203/newspaper-downloads/dedupe-webp/vlm_crops/`

## Quick start (Greene)
1. Clone with submodule and set up venv:
   ```bash
   git clone --recursive git@github.com:saulrichardson/newspaper-parsing.git
   cd newspaper-parsing
   python -m venv venv && source venv/bin/activate
   pip install -r requirements.txt
   ```
2. Build manifest and shards:
   ```bash
   make manifest VAST_ROOT=/vast/sxr203/newspaper-downloads/dedupe-webp
   make shard VAST_ROOT=/vast/sxr203/newspaper-downloads/dedupe-webp CHUNKS=200
   ```
3. Submit Slurm array (needs `GEMINI_KEY` and `VENV` env vars):
   ```bash
   export GEMINI_KEY=... 
   export VENV=$(pwd)/venv
   make submit VAST_ROOT=/vast/sxr203/newspaper-downloads/dedupe-webp
   ```

## What the Slurm job does
- Starts agent-gateway locally inside each task (Gemini provider).
- Calls `scripts/run_vlm_gateway.py` on that chunk of layouts.
- Writes per-page JSON (`*.vlm.json`) and optional crops, plus an append-only `manifest.jsonl` for audit/resume.

## Local run
1. Start the gateway from repo root (loads `.env`):
   ```bash
   source venv/bin/activate
   uvicorn gateway.app:create_app --factory --host 127.0.0.1 --port 8000
   ```
2. Run a small smoke test:
   ```bash
   python scripts/run_vlm_gateway.py \
     --layouts "newspaper-parsing-local/data/unique_outputs_dedup/*.json" \
     --png-root newspaper-parsing-local/data/unique_png \
     --output-dir newspaper-parsing-local/data/vlm_out \
     --model gemini:gemini-2.5-flash \
     --page-concurrency 2 \
     --max-concurrency 4 \
     --skip-existing
   ```

### One-command local sync
If you prefer not to manage a separate gateway terminal, use:
```bash
python scripts/run_local_sync.py \
  --layouts "newspaper-parsing-local/data/unique_outputs_dedup/*.json" \
  --png-root newspaper-parsing-local/data/unique_png \
  --output-dir newspaper-parsing-local/data/vlm_out \
  --model gemini:gemini-2.5-flash \
  --page-concurrency 2 \
  --max-concurrency 4 \
  --skip-existing \
  --skip-bad-layouts
```

## Batch export (Gemini + OpenAI)
For large-scale runs, export per-box JSONL shards and submit them to provider batch APIs.

1. Export batch request shards:
   ```bash
   python scripts/export_batch_requests.py \
     --layouts "newspaper-parsing-local/data/unique_outputs_dedup/*.json" \
     --png-root newspaper-parsing-local/data/unique_png \
     --output-dir newspaper-parsing-local/data/batch_requests \
     --provider both \
     --boxes-per-shard 5000
   ```
   This writes:
   - `gemini_requests_shardNNN.jsonl`
   - `openai_requests_shardNNN.jsonl`
   - `mapping_shardNNN.jsonl` (provenance for rehydrating results)

2. Submit shards to each provider’s Batch API using their official clients.
   - Gemini expects each line to be `{"key":..., "request":{GenerateContentRequest}}`.
   - OpenAI expects each line to be a Batch wrapper for `POST /v1/responses`.

3. When batch results return, join on `key` / `custom_id` using the matching `mapping_shardNNN.jsonl`
   to write final per-page `*.vlm.json` outputs.

### Download + rehydrate Gemini batch results
After submitting Gemini batches with `scripts/submit_batch_shards.py`, you can download outputs and
rehydrate them into per-page `*.vlm.json` files:
```bash
python scripts/download_gemini_batch_results.py \
  --request-dir newspaper-parsing-local/data/batch_requests_remaining \
  --out-dir newspaper-parsing-local/data/batch_results_gemini_remaining \
  --skip-existing

python scripts/rehydrate_gemini_batch_results.py \
  --request-dir newspaper-parsing-local/data/batch_requests_remaining \
  --results-dir newspaper-parsing-local/data/batch_results_gemini_remaining \
  --output-dir newspaper-parsing-local/data/vlm_out_batch_gemini_remaining \
  --skip-bad-boxes
```

## Page QA (post-OCR)
Once you have per-page `*.vlm.json` outputs (sync or rehydrated), run a second LLM pass to:
- assemble a normalized page transcript
- flag likely OCR issues and whether the page should be rerun

One-command local QA (auto-starts gateway if needed):
```bash
python scripts/run_local_page_qa.py \
  --pages "newspaper-parsing-local/data/vlm_out_batch_gemini_remaining/*.vlm.json" \
  --output-dir newspaper-parsing-local/data/page_qa_gpt5nano \
  --model openai:gpt-5-nano \
  --max-concurrency 4 \
  --skip-existing
```

## Zoning classifier (page-level)
Given a per-page `*.vlm.json`, flatten all box transcripts and classify whether the page contains
zoning-related ordinance/amendment/hearing text.

Default prompt lives in `prompts/zoning_ocr_classifier_prompt_text.txt`.

One-command local run (stages only Gemini pages with real OCR text, then runs `openai:gpt-5-nano`):
```bash
python scripts/run_local_zoning_gpt5nano.py \
  --vlm-dir "newspaper-parsing-local/data/vlm_out_batch_gemini_remaining" \
  --output-dir "newspaper-parsing-local/data/zoning_labels_openai_gpt5nano" \
  --max-concurrency 4
```

## Staging from a PNG manifest
To run only a subset of pages listed in a `png_manifest.json` (with `entries[].slug`):
```bash
python scripts/stage_from_png_manifest.py \
  --png-manifest /path/to/png_manifest.json \
  --layout-root newspaper-parsing-local/data/unique_outputs_dedup \
  --png-root newspaper-parsing-local/data/unique_png \
  --out-dir newspaper-parsing-local/data/staged_manifest
```
Then use `@layouts.txt` for sync or batch:
```bash
python scripts/run_local_sync.py --layouts "@newspaper-parsing-local/data/staged_manifest/layouts.txt" ...
python scripts/export_batch_requests.py --layouts "@newspaper-parsing-local/data/staged_manifest/layouts.txt" ...
```

## Helper scripts
- `scripts/make_layout_manifest.sh [root] [out]`: list all Dell JSONs.
- `scripts/split_manifest.sh [manifest] [chunks] [prefix]`: split manifest for array.

## Notes
- Model must be prefixed with a real provider (`gemini:*`, `openai:*`, or `claude:*`). Test providers like `echo:*` are rejected unless you pass `--allow-test-providers`.
- `--layouts` accepts either a glob (absolute OK) or an `@file` listing one JSON path per line.
- Crop guards: crops are downscaled if they exceed `--max-crop-megapixels` (default 3.0MP) or `--max-crop-dim` (default 2048px). Set either to 0 to disable.
- Concurrency:
  - `--page-concurrency` controls pages in flight (default 1).
  - `--max-concurrency` controls total box requests in flight (default 4).
- `--skip-existing` enables resume without recomputing finished pages.
- `--skip-bad-layouts` skips layouts/boxes that fail to load or crop, and appends details to `bad_layouts.jsonl` in the output dir.
- `manifest.jsonl` is appended to as pages complete; override location with `--manifest-path`.
