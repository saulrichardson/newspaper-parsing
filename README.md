# newspaper-parsing

GPU-first newspaper layout parsing and fusion pipeline for **Paddle(4) + Dell + MinerU**.

This repository is intentionally scoped to one production flow:

- Paddle layout detectors: `PP-DocLayoutV2`, `PP-DocLayoutV3`, `PP-DocLayout_plus-L`
- Paddle VL parser: `doc_parser v1.5` (`layout_det_res` + semantic blocks from `parsing_res_list`)
- Dell American Stories layout parser
- MinerU2.5 layout parser
- Cross-model fusion with anti-noise gating and review pack generation
- Fused-region transcription with Paddle OCR over fused ROIs (reading-order transcript artifacts)

No other parser families are in the active path.

## Why this repo exists

- Run the full parser bag from a folder of newspaper PNGs/TIFFs.
- Keep Torch usage efficient: GPU inference first, CPU fusion/review second.
- Preserve model provenance and labels (including VL1.5 semantic/table payloads).
- Produce visual artifacts you can inspect quickly (`review/pages`, `top20_informative`, `top20_miner_delta`).
- Produce transcript artifacts aligned to fused regions (`outputs/transcription/...`).

## Visual walkthrough

### 1) Raw page input

![Input newspaper page](docs/images/example_input.png)

### 2) Full model comparison board (Paddle4 + Dell + MinerU + fused)

![Full comparison board](docs/images/example_full_board.png)

### 3) Noise-control impact (before vs after fusion noise guard)

![Noise control comparison](docs/images/example_noise_fix.png)

### 4) MinerU contribution view (without MinerU vs with MinerU)

![Miner contribution comparison](docs/images/example_miner_delta.png)

## High-level pipeline

```mermaid
flowchart LR
  A["Input folder (PNG/JPG/TIFF)"] --> B["Manifest builder"]
  B --> C["GPU inference jobs"]
  C --> C1["Paddle layout x3 (L40S)"]
  C --> C2["Paddle VL1.5 doc_parser (L40S)"]
  C --> C3["Dell + MinerU (H200 or L40S)"]
  C1 --> D["Normalized source outputs + label stats"]
  C2 --> D
  C3 --> D
  D --> E["CPU fusion + ranking"]
  E --> F["Review bundle + top20 packs (CPU)"]
  F --> G["GPU transcription from fused regions"]
  G --> H["Ship artifacts: TSVs + PNG boards + JSON + TXT"]
```

## Repository layout

- `src/newsbag/` core package (CLI, runners, fusion, review)
- `configs/pipeline.example.json` local reference config
- `configs/pipeline.torch.json` Torch reference config
- `torch/slurm/` sbatch templates + submit helpers
- `scripts/make_manifest.py` manifest generation utility
- `scripts/monitor_torch_jobs.sh` queue monitor helper
- `docs/output_layout.md` exact artifact contracts
- `docs/torch_hpc.md` Torch operational guidance

## Quick start (local)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

cp configs/pipeline.example.json configs/pipeline.local.json
# edit paths in configs/pipeline.local.json

newsbag run --config configs/pipeline.local.json
```

Run selected stages explicitly:

```bash
newsbag run --config configs/pipeline.local.json \
  --stages paddle_layout,paddle_vl15,dell,mineru,fusion,review,transcription
```

Create a manifest from a scan folder:

```bash
python scripts/make_manifest.py \
  --input /absolute/path/to/scans \
  --output /absolute/path/to/news_manifest.txt
```

## Quick start (Torch, recommended)

### 0) GPU preflight

```bash
cd /scratch/$USER/paddleocr_vl15/newspaper-parsing
bash torch/slurm/preflight_gpu.sh
```

### 1) Submit end-to-end from a folder

```bash
cd /scratch/$USER/paddleocr_vl15/newspaper-parsing

bash torch/slurm/submit_newsbag_from_dir.sh \
  --input-dir /scratch/$USER/paddleocr_vl15/input/your_pages \
  --recursive \
  --gpu split
```

What `--gpu split` does:

- L40S job: `paddle_layout,paddle_vl15`
- H200 job: `dell,mineru`
- CPU job: `fusion,review` (after both GPU jobs succeed)
- L40S job: `transcription` (after fusion/review succeeds)

This is the safest high-throughput default on Torch.

### 2) Monitor

```bash
cd /scratch/$USER/paddleocr_vl15/newspaper-parsing
bash scripts/monitor_torch_jobs.sh
```

## Outputs you get per run

Each run writes under:

- `/scratch/$USER/paddleocr_vl15/runs/<run_name_timestamp>/`

Key outputs:

- `outputs/sources/...` per-model normalized boxes and label aggregates
- `outputs/fusion/variant_leaderboard.tsv`
- `outputs/fusion/source_leaderboard.tsv`
- `outputs/fusion/per_page_variant_metrics.tsv`
- `review/pages/<slug>/06_board.png`
- `review/top20_informative/`
- `review/top20_miner_delta/`
- `outputs/transcription/<variant>/transcription_report.tsv`
- `outputs/transcription/<variant>/<slug>/transcript.txt`
- `outputs/transcription/<variant>/transcript_combined.txt`

For exact schemas and filenames, see `docs/output_layout.md`.

## Fusion variants

- `S1_paddle_best_single`
- `S2_dell_only`
- `S3_mineru_only`
- `P1_paddle_union4`
- `P2_paddle_union4_plus_dell`
- `P3_paddle_union4_plus_mineru`
- `P4_paddle_union4_plus_dell_plus_mineru`

Default recommended variant in Torch config:

- `P4_paddle_union4_plus_dell_plus_mineru`

## Operational notes

- Always request Torch GPUs via `--gres=gpu:<type>:1`.
- Keep fusion/review off GPU partitions (CPU job only) to avoid low-util cancellation.
- As of February 2026 on Torch, Paddle stages are most reliable on `l40s_public`; split mode handles this automatically.
- External-source guardrails are enabled by default:
  - `dell.min_nonempty_pages` (default `1`)
  - `mineru.min_nonempty_pages` (default `1`)
  If an external source returns empty boxes across all processed pages, the run fails fast before fusion/review.

## Mini-Run Example (Abilene + Bakersfield, Corrected Dell, Feb 13, 2026)

This is a real completed run on Torch with two representative pages:

- Run root: `/scratch/$USER/paddleocr_vl15/runs/layout_bagging_20260213_024626`
- Local synced artifacts:
  - `/Users/saulrichardson/Downloads/ad-hoc-newspapers/newspaper-parsing-mini-run-20260213-024626`

Submission command used:

```bash
cd /scratch/$USER/paddleocr_vl15/newspaper-parsing
bash torch/slurm/submit_newsbag_from_dir.sh \
  --input-dir /scratch/$USER/paddleocr_vl15/input/newspaper_parsing_example_20260213 \
  --gpu l40s
```

### Visuals from that run

![Abilene input](docs/images/mini_run/abilene_input.png)

![Abilene full board](docs/images/mini_run/abilene_board.png)

![Bakersfield full board](docs/images/mini_run/bakersfield_board.png)

![Bakersfield with vs without MinerU](docs/images/mini_run/bakersfield_miner_delta.png)

![Abilene Dell-only overlay (corrected)](docs/images/mini_run/abilene_dell_layout.png)

### Actual output metrics (from `variant_leaderboard.tsv`)

| Variant | Mean Base Recall | Mean Text Area | Mean Boxes |
|---|---:|---:|---:|
| `P2_paddle_union4_plus_dell` | 0.997774 | 0.737427 | 329.0 |
| `S1_paddle_best_single` | 0.966463 | 0.698220 | 224.5 |
| `P4_paddle_union4_plus_dell_plus_mineru` | 0.997929 | 0.738797 | 404.5 |

### Dell sanity check (corrected)

- `abilene-reporter-news-apr-29-1946-p-19`: `23` Dell boxes
- `bakersfield-californian-aug-31-1937-p-14`: `28` Dell boxes
- Source files:
  - `outputs/sources/dell/dell_c0005_i010/<slug>/layout_boxes.normalized.json`

### Actual transcription outputs

Transcription is ROI-first: Paddle OCR runs on a cleaned non-redundant subset of fused `text/title`
regions (not full-page OCR), then lines are remapped to page coordinates and emitted in fused reading order.

- `outputs/transcription/P4_paddle_union4_plus_dell_plus_mineru/transcription_report.tsv`
- `outputs/transcription/P4_paddle_union4_plus_dell_plus_mineru/transcript_combined.txt`
- `outputs/transcription/P4_paddle_union4_plus_dell_plus_mineru/<slug>/ocr_regions_overlay.png`
- `outputs/transcription/P4_paddle_union4_plus_dell_plus_mineru/<slug>/ocr_lines_overlay.png`

Transcription report excerpt:

| Page | Fused Boxes | OCR Regions | OCR Lines | Assigned Lines | Transcript Chars |
|---|---:|---:|---:|---:|---:|
| `abilene-reporter-news-apr-29-1946-p-19` | 331 | 54 | 3814 | 3814 | 32396 |
| `bakersfield-californian-aug-31-1937-p-14` | 457 | 112 | 2871 | 2871 | 31760 |

Cleaned OCR-region counts in this rerun:

- Abilene: `331 fused boxes -> 54 OCR regions`
- Bakersfield: `457 fused boxes -> 112 OCR regions`

![Abilene cleaned OCR regions](docs/images/mini_run/abilene_ocr_regions_overlay.png)

![Abilene OCR line overlay](docs/images/mini_run/abilene_ocr_lines_overlay.png)

![Bakersfield cleaned OCR regions](docs/images/mini_run/bakersfield_ocr_regions_overlay.png)

## Licensing note

`MinerU` integration depends on `mineru_vl_utils` (AGPL-3.0). Confirm license compatibility for your deployment/distribution model.

## LaTeX Walkthrough

A full technical walkthrough (with visuals and real run outputs) is available at:

- `docs/newspaper_parsing_pipeline.tex`

Compile from the `docs/` directory:

```bash
cd docs
xelatex newspaper_parsing_pipeline.tex
xelatex newspaper_parsing_pipeline.tex
```
