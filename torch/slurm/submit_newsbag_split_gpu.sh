#!/bin/bash
# Submit a full end-to-end run as four jobs (split across GPU partitions):
# 1) L40S GPU inference-only: Paddle layout + VL1.5 doc_parser
# 2) H200 GPU inference-only: Dell + MinerU
# 3) CPU fusion + review bundle (afterok on both GPU jobs)
# 4) L40S GPU transcription (afterok on CPU fusion/review)
#
# Why split: Paddle currently fails on Torch H200 with CUDA kernel image mismatch errors.
#
# Usage (on Torch login):
#   bash torch/slurm/submit_newsbag_split_gpu.sh
#
# Optional env:
# - BASE, PROJECT_ROOT, VENV_DIR, CONFIG_JSON
# - RUN_ROOT, RUN_DIR
# - GPU_SCRIPT_L40S (default: torch/slurm/newsbag_infer_l40s.sbatch)
# - GPU_SCRIPT_H200 (default: torch/slurm/newsbag_infer_h200.sbatch)
# - CPU_SCRIPT (default: torch/slurm/newsbag_fuse_review_cs.sbatch)
# - TRANSCRIBE_SCRIPT (default: torch/slurm/newsbag_transcribe_l40s.sbatch)
# - STAGES_L40S (default: paddle_layout,paddle_vl15)
# - STAGES_H200 (default: dell,mineru)
# - STAGES_CPU (default: fusion,review)
# - STAGES_TRANSCRIBE (default: transcription)

set -euo pipefail

BASE="${BASE:-/scratch/$USER/paddleocr_vl15}"
PROJECT_ROOT="${PROJECT_ROOT:-$BASE/newspaper-parsing}"
VENV_DIR="${VENV_DIR:-$BASE/envs/mineru25_py310}"
CONFIG_JSON="${CONFIG_JSON:-$PROJECT_ROOT/configs/pipeline.torch.json}"
RUN_ROOT="${RUN_ROOT:-$BASE/runs}"

GPU_SCRIPT_L40S="${GPU_SCRIPT_L40S:-$PROJECT_ROOT/torch/slurm/newsbag_infer_l40s.sbatch}"
GPU_SCRIPT_H200="${GPU_SCRIPT_H200:-$PROJECT_ROOT/torch/slurm/newsbag_infer_h200.sbatch}"
CPU_SCRIPT="${CPU_SCRIPT:-$PROJECT_ROOT/torch/slurm/newsbag_fuse_review_cs.sbatch}"
TRANSCRIBE_SCRIPT="${TRANSCRIBE_SCRIPT:-$PROJECT_ROOT/torch/slurm/newsbag_transcribe_l40s.sbatch}"

STAGES_L40S="${STAGES_L40S:-paddle_layout,paddle_vl15}"
STAGES_H200="${STAGES_H200:-dell,mineru}"
STAGES_CPU="${STAGES_CPU:-fusion,review}"
STAGES_TRANSCRIBE="${STAGES_TRANSCRIBE:-transcription}"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR:-$RUN_ROOT/layout_bagging_$STAMP}"

mkdir -p "$RUN_DIR"

echo "[submit] project_root=$PROJECT_ROOT"
echo "[submit] config=$CONFIG_JSON"
echo "[submit] run_dir=$RUN_DIR"
echo "[submit] gpu_l40s=$GPU_SCRIPT_L40S stages_l40s=$STAGES_L40S"
echo "[submit] gpu_h200=$GPU_SCRIPT_H200 stages_h200=$STAGES_H200"
echo "[submit] cpu=$CPU_SCRIPT stages_cpu=$STAGES_CPU"
echo "[submit] transcribe=$TRANSCRIBE_SCRIPT stages_transcribe=$STAGES_TRANSCRIBE"

JID_L40S="$(
  sbatch --parsable \
    --export=ALL,PROJECT_ROOT="$PROJECT_ROOT",VENV_DIR="$VENV_DIR",CONFIG_JSON="$CONFIG_JSON",RUN_DIR="$RUN_DIR",STAGES="$STAGES_L40S" \
    "$GPU_SCRIPT_L40S"
)"
echo "[submit] l40s infer job: $JID_L40S"

JID_H200="$(
  sbatch --parsable \
    --export=ALL,PROJECT_ROOT="$PROJECT_ROOT",VENV_DIR="$VENV_DIR",CONFIG_JSON="$CONFIG_JSON",RUN_DIR="$RUN_DIR",STAGES="$STAGES_H200" \
    "$GPU_SCRIPT_H200"
)"
echo "[submit] h200 infer job: $JID_H200"

JID_FUSE="$(
  sbatch --parsable --dependency=afterok:$JID_L40S:$JID_H200 \
    --export=ALL,PROJECT_ROOT="$PROJECT_ROOT",VENV_DIR="$VENV_DIR",CONFIG_JSON="$CONFIG_JSON",RUN_DIR="$RUN_DIR",STAGES="$STAGES_CPU" \
    "$CPU_SCRIPT"
)"
echo "[submit] fuse/review job: $JID_FUSE (afterok:$JID_L40S:$JID_H200)"

JID_TRANSCRIBE="$(
  sbatch --parsable --dependency=afterok:$JID_FUSE \
    --export=ALL,PROJECT_ROOT="$PROJECT_ROOT",VENV_DIR="$VENV_DIR",CONFIG_JSON="$CONFIG_JSON",RUN_DIR="$RUN_DIR",STAGES="$STAGES_TRANSCRIBE" \
    "$TRANSCRIBE_SCRIPT"
)"
echo "[submit] transcription job: $JID_TRANSCRIBE (afterok:$JID_FUSE)"

echo "[submit] Monitor:"
echo "  squeue -u $USER"
echo "  tail -f /scratch/$USER/paddleocr_vl15/logs/newsbag_infer_l40s-$JID_L40S.out"
echo "  tail -f /scratch/$USER/paddleocr_vl15/logs/newsbag_infer_h200-$JID_H200.out"
echo "  tail -f /scratch/$USER/paddleocr_vl15/logs/newsbag_fuse_review-$JID_FUSE.out"
echo "  tail -f /scratch/$USER/paddleocr_vl15/logs/newsbag_transcribe_l40s-$JID_TRANSCRIBE.out"
