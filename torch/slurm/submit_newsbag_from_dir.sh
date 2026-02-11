#!/bin/bash
# Submit a directory of newspaper scans through the full pipeline on Torch.
#
# This script:
# 1) Creates a run directory (RUN_DIR) under /scratch/$USER/paddleocr_vl15/runs
# 2) Generates a manifest from INPUT_DIR
# 3) Writes a per-run config.json (based on configs/pipeline.torch.json)
# 4) Submits GPU inference + CPU fusion/review (afterok)
#
# Usage (on Torch login):
#   bash torch/slurm/submit_newsbag_from_dir.sh --input-dir /path/to/scans --recursive --gpu l40s
#
# Flags:
#   --input-dir <dir>   Required. Directory containing PNG/JPG/TIFF/etc.
#   --recursive         Optional. Recurse subdirectories.
#   --gpu l40s|h200|split  Optional. Default: l40s.
#   --max-pages N       Optional. Limit the manifest to N images (for smoke tests).
#   --run-dir <dir>     Optional. Explicit run dir (default: /scratch/.../runs/layout_bagging_<ts>)
#   --stages <csv>      Optional. Override stages for the single GPU job mode (l40s/h200).
#
# Optional env:
#   BASE, PROJECT_ROOT, TEMPLATE_CONFIG_JSON

set -euo pipefail

INPUT_DIR=""
RECURSIVE=0
GPU_KIND="l40s"
RUN_DIR=""
MAX_PAGES=0
STAGES=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input-dir)
      INPUT_DIR="${2:-}"
      shift 2
      ;;
    --recursive)
      RECURSIVE=1
      shift 1
      ;;
    --gpu)
      GPU_KIND="${2:-l40s}"
      shift 2
      ;;
    --stages)
      STAGES="${2:-}"
      shift 2
      ;;
    --max-pages)
      MAX_PAGES="${2:-0}"
      shift 2
      ;;
    --run-dir)
      RUN_DIR="${2:-}"
      shift 2
      ;;
    -h|--help)
      sed -n '1,180p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$INPUT_DIR" ]]; then
  echo "ERROR: --input-dir is required" >&2
  exit 2
fi

BASE="${BASE:-/scratch/$USER/paddleocr_vl15}"
PROJECT_ROOT="${PROJECT_ROOT:-$BASE/newspaper-parsing}"
TEMPLATE_CONFIG_JSON="${TEMPLATE_CONFIG_JSON:-$PROJECT_ROOT/configs/pipeline.torch.json}"

STAMP="$(date +%Y%m%d_%H%M%S)"
if [[ -z "$RUN_DIR" ]]; then
  RUN_DIR="$BASE/runs/layout_bagging_$STAMP"
fi

mkdir -p "$RUN_DIR/manifests"

MANIFEST="$RUN_DIR/manifests/images.input.txt"
OUT_CONFIG="$RUN_DIR/manifests/config.input.json"

echo "[submit] input_dir=$INPUT_DIR"
echo "[submit] recursive=$RECURSIVE"
echo "[submit] gpu_kind=$GPU_KIND"
echo "[submit] max_pages=$MAX_PAGES"
echo "[submit] run_dir=$RUN_DIR"
if [[ -n "$STAGES" ]]; then
  echo "[submit] stages=$STAGES"
fi

MAKE_MANIFEST_ARGS=(--input "$INPUT_DIR" --output "$MANIFEST")
if [[ "$RECURSIVE" -eq 1 ]]; then
  MAKE_MANIFEST_ARGS+=(--recursive)
fi

# scripts/make_manifest.py is stdlib-only; use python3 from login node.
python3 "$PROJECT_ROOT/scripts/make_manifest.py" "${MAKE_MANIFEST_ARGS[@]}"

if [[ "$MAX_PAGES" -gt 0 ]]; then
  tmp="$MANIFEST.tmp"
  head -n "$MAX_PAGES" "$MANIFEST" > "$tmp"
  mv "$tmp" "$MANIFEST"
fi

COUNT="$(wc -l < "$MANIFEST" | tr -d ' ')"
echo "[submit] manifest_count=$COUNT"
if [[ "$COUNT" -eq 0 ]]; then
  echo "ERROR: manifest is empty (no images found)" >&2
  exit 2
fi

# Create run-scoped config so this run is self-contained and reproducible.
python3 - <<PY
import json
from pathlib import Path

tmpl=Path("$TEMPLATE_CONFIG_JSON")
payload=json.loads(tmpl.read_text(encoding="utf-8"))
payload["manifest_path"]=str(Path("$MANIFEST"))
payload["run_root"]=str(Path("$BASE")/"runs")
payload["run_name"]="layout_bagging"
Path("$OUT_CONFIG").write_text(json.dumps(payload, indent=2) + "\\n", encoding="utf-8")
print("[submit] wrote config:", "$OUT_CONFIG")
PY

GPU_SCRIPT="$PROJECT_ROOT/torch/slurm/newsbag_infer_l40s.sbatch"
if [[ "$GPU_KIND" == "h200" ]]; then
  GPU_SCRIPT="$PROJECT_ROOT/torch/slurm/newsbag_infer_h200.sbatch"
elif [[ "$GPU_KIND" == "split" ]]; then
  GPU_SCRIPT=""
fi

export RUN_DIR
export CONFIG_JSON="$OUT_CONFIG"
export GPU_SCRIPT

cd "$PROJECT_ROOT"

if [[ "$GPU_KIND" == "split" ]]; then
  # Split-GPU submission: Paddle on L40S, Dell+MinerU on H200, then CPU fuse/review.
  bash torch/slurm/submit_newsbag_split_gpu.sh
  exit 0
fi

if [[ -z "$STAGES" ]]; then
  if [[ "$GPU_KIND" == "h200" ]]; then
    STAGES="dell,mineru"
  else
    STAGES="paddle_layout,paddle_vl15,dell,mineru"
  fi
fi

if [[ "$GPU_KIND" == "h200" ]]; then
  if echo ",$STAGES," | grep -qE ",paddle_layout,|,paddle_vl15,"; then
    echo "[submit] ERROR: Paddle stages requested with --gpu h200 ($STAGES)." >&2
    echo "[submit] Use --gpu split (recommended) or --gpu l40s." >&2
    exit 3
  fi
fi

export STAGES
bash torch/slurm/submit_newsbag_full.sh
