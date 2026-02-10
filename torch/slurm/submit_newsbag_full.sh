#!/bin/bash
# Submit a full end-to-end run as two jobs:
# 1) GPU inference-only (Paddle4 + Dell + MinerU)
# 2) CPU fusion + review bundle
#
# Usage (on Torch login):
#   bash torch/slurm/submit_newsbag_full.sh
#
# Optional env:
# - BASE, PROJECT_ROOT, VENV_DIR, CONFIG_JSON
# - RUN_ROOT, RUN_DIR
# - GPU_SCRIPT (default: torch/slurm/newsbag_infer_l40s.sbatch)
# - CPU_SCRIPT (default: torch/slurm/newsbag_fuse_review_cs.sbatch)

set -euo pipefail

BASE="${BASE:-/scratch/$USER/paddleocr_vl15}"
PROJECT_ROOT="${PROJECT_ROOT:-$BASE/new-ocr}"
VENV_DIR="${VENV_DIR:-$BASE/envs/newsbag}"
CONFIG_JSON="${CONFIG_JSON:-$PROJECT_ROOT/configs/pipeline.torch.json}"
RUN_ROOT="${RUN_ROOT:-$BASE/runs}"

GPU_SCRIPT="${GPU_SCRIPT:-$PROJECT_ROOT/torch/slurm/newsbag_infer_l40s.sbatch}"
CPU_SCRIPT="${CPU_SCRIPT:-$PROJECT_ROOT/torch/slurm/newsbag_fuse_review_cs.sbatch}"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-$RUN_ROOT/layout_bagging_$STAMP}"

mkdir -p "$RUN_DIR"

echo "[submit] project_root=$PROJECT_ROOT"
echo "[submit] config=$CONFIG_JSON"
echo "[submit] run_dir=$RUN_DIR"
echo "[submit] gpu_script=$GPU_SCRIPT"
echo "[submit] cpu_script=$CPU_SCRIPT"

JID_INFER="$(
  sbatch --parsable \
    --export=ALL,PROJECT_ROOT="$PROJECT_ROOT",VENV_DIR="$VENV_DIR",CONFIG_JSON="$CONFIG_JSON",RUN_DIR="$RUN_DIR" \
    "$GPU_SCRIPT"
)"
echo "[submit] infer job: $JID_INFER"

JID_FUSE="$(
  sbatch --parsable --dependency=afterok:$JID_INFER \
    --export=ALL,PROJECT_ROOT="$PROJECT_ROOT",VENV_DIR="$VENV_DIR",CONFIG_JSON="$CONFIG_JSON",RUN_DIR="$RUN_DIR" \
    "$CPU_SCRIPT"
)"
echo "[submit] fuse/review job: $JID_FUSE (afterok:$JID_INFER)"

echo "[submit] Monitor:"
echo "  squeue -u $USER"
echo "  tail -f /scratch/$USER/paddleocr_vl15/logs/newsbag_infer_l40s-$JID_INFER.out"
echo "  tail -f /scratch/$USER/paddleocr_vl15/logs/newsbag_fuse_review-$JID_FUSE.out"
