# Newspaper Layout Bagging Pipeline

This repository contains a clean, end-to-end pipeline for newspaper layout extraction and fusion on Torch.

Scope is intentionally strict:
- Paddle layout models (4 sources):
  - `PP-DocLayoutV2`
  - `PP-DocLayoutV3`
  - `PP-DocLayout_plus-L`
  - `Paddle doc_parser v1.5` layout stream (`layout_det_res`) plus semantic blocks (`parsing_res_list`)
- Dell American Stories layout parser
- MinerU2.5 layout parser

No other layout models are part of the active code path.

## Goals
- GPU-first execution on Torch (`--gres=gpu:*`), with explicit CPU fallback controls.
- Preserve raw model labels and semantic payloads (especially VL1.5 table/html blocks).
- Fuse candidate regions while suppressing noisy oversized boxes.
- Produce organized per-page artifacts for visual review and downstream OCR.

## Repository layout
- `src/newsbag/`: core package (orchestration, runners, fusion, review artifacts)
- `configs/pipeline.example.json`: example run config
- `torch/slurm/`: Torch sbatch templates and submission helper
- `scripts/`: local convenience wrappers
- `docs/`: output and operations notes

## Quickstart (local)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
# for full model stack:
# pip install -e .[paddle,dell,mineru]

cp configs/pipeline.example.json configs/pipeline.local.json
# edit paths in configs/pipeline.local.json

newsbag run --config configs/pipeline.local.json
```

Create a manifest from a scan folder:

```bash
python scripts/make_manifest.py \
  --input /Users/saulrichardson/Downloads/ad-hoc-newspapers \
  --output /Users/saulrichardson/Downloads/ad-hoc-newspapers/news_manifest.txt
```

## Torch usage
Use the templates in `torch/slurm/` and set account/partition/GRES according to cluster policy.

Critical rule for Torch GPU requests:
- Use `--gres=gpu:<type>:1` (not `--gpus=1`)

Examples:
- `l40s_public`: `--gres=gpu:l40s:1`
- `h200_public`: `--gres=gpu:h200:1`

Preflight:

```bash
bash torch/slurm/preflight_gpu.sh
```

Submit:

```bash
# Recommended: submit a directory of scans end-to-end (manifest + config + GPU infer + CPU fuse/review)
bash torch/slurm/submit_newsbag_from_dir.sh --input-dir /scratch/$USER/paddleocr_vl15/input/ad_hoc_newspapers_20260205_190618 --recursive --gpu l40s

# Alternative: two-job chain only (assumes configs/pipeline.torch.json already points at your manifest)
bash torch/slurm/submit_newsbag_full.sh

# Manual (example):
# RUN_DIR="/scratch/$USER/paddleocr_vl15/runs/layout_bagging_$(date +%Y%m%d_%H%M%S)"
# mkdir -p "$RUN_DIR"
# JID="$(sbatch --parsable --export=ALL,RUN_DIR="$RUN_DIR" torch/slurm/newsbag_infer_l40s.sbatch)"
# sbatch --dependency=afterok:$JID --export=ALL,RUN_DIR="$RUN_DIR" torch/slurm/newsbag_fuse_review_cs.sbatch
```

Monitor:

```bash
bash scripts/monitor_torch_jobs.sh
```

## Outputs
Each run writes a timestamped run directory with:
- `sources/`: per-model raw + normalized outputs
- `fusion/`: fused boxes and metrics by variant
- `review/`: per-page PNG boards for manual quality checks
- `reports/`: leaderboard and per-page metrics TSV/JSON

See:
- `docs/output_layout.md` for exact file contracts.
- `docs/torch_hpc.md` for HPC package/runtime guidance.

## Licensing Notes
This repo can be used with optional third-party model utilities that have their own licenses.
In particular, `MinerU` support in this pipeline uses `mineru_vl_utils`, which is licensed under AGPL-3.0.
If you plan to publish this repo or distribute binaries, confirm license compatibility with your intended use.

## Decision Logic (Compact)

Fusion ranks variants using proxy coverage against deduped text-line candidates:

1. Build candidate sets for `S1..S3`, `P1..P4`.
2. Apply dedupe + anti-noise rules:
   - suppress unsupported oversized strips/boxes
   - keep large text boxes only when cross-source support or meaningful new line coverage is present
3. Score variants with:
   - mean line-coverage ratio (higher is better)
   - mean text-area ratio (higher is better)
   - mean box count penalty (lower is better)
4. Emit `outputs/fusion/variant_leaderboard.tsv`.
5. Build `review/top20_informative/` using pages with strongest fused-vs-baseline signal.
