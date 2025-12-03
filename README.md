# Newspaper VLM Pipeline

Runs Dell layout crops through a VLM (Gemini 2.5 Flash) via agent-gateway. Target data live on NYU Greene VAST.

## Data layout (Greene)
- PNGs: `/vast/sxr203/newspaper-downloads/dedupe-webp/unique_png/`
- Dell layouts: `/vast/sxr203/newspaper-downloads/dedupe-webp/unique_outputs/`
- Outputs: recommended `/vast/sxr203/newspaper-downloads/dedupe-webp/vlm_gemini25/`
- Crops: recommended `/vast/sxr203/newspaper-downloads/dedupe-webp/vlm_crops/`

## Quick start (Greene)
1. Clone with submodule and set up venv:
   ```bash
   git clone --recursive git@github.com:saulrichardson/newspaper-scrapping.git
   cd newspaper-scrapping
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
- Writes per-page JSON (`*.vlm.json`) and optional crops.

## Helper scripts
- `scripts/make_layout_manifest.sh [root] [out]`: list all Dell JSONs.
- `scripts/split_manifest.sh [manifest] [chunks] [prefix]`: split manifest for array.

## Notes
- Model: Gemini 2.5 Flash (`gemini:gemini-2.5-flash`).
- Concurrency: 4 per task; adjust if rate-limited.
- Timeouts: 120s per call; 1 retry on parse/transport errors.

