#!/bin/bash
# GPU inference-only job for Torch H200 (keeps GPU utilization high; no fusion/review here).
# Required env:
# - RUN_DIR: shared run directory used by downstream CPU fusion/review job.
#
# Optional env:
# - PROJECT_ROOT, VENV_DIR, CONFIG_JSON, STAGES, NEWSBAG_PY

#SBATCH -A torch_pr_609_general
#SBATCH -p h200_public
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH -J newsbag_infer_h200
#SBATCH -o /scratch/%u/paddleocr_vl15/logs/%x-%j.out
#SBATCH -e /scratch/%u/paddleocr_vl15/logs/%x-%j.err

set -euo pipefail

BASE="${BASE:-/scratch/$USER/paddleocr_vl15}"
PROJECT_ROOT="${PROJECT_ROOT:-$BASE/new-ocr}"
VENV_DIR="${VENV_DIR:-$BASE/envs/newsbag}"
CONFIG_JSON="${CONFIG_JSON:-$PROJECT_ROOT/configs/pipeline.torch.json}"
# IMPORTANT: Paddle (layout_detection + VL1.5 doc_parser) does not currently run reliably on Torch H200
# due to CUDA kernel image mismatch errors. Keep this job H200-safe by defaulting to Dell+MinerU only.
STAGES="${STAGES:-dell,mineru}"
RUN_DIR="${RUN_DIR:-}"
NEWSBAG_PY="${NEWSBAG_PY:-}"

if [ -z "$RUN_DIR" ]; then
  echo "[newsbag] ERROR: RUN_DIR is required (shared across jobs)." >&2
  exit 2
fi

cd "$PROJECT_ROOT"
if [ -z "$NEWSBAG_PY" ]; then
  source "$VENV_DIR/bin/activate"
  NEWSBAG_PY="$VENV_DIR/bin/python"
fi

if [ ! -x "$NEWSBAG_PY" ]; then
  echo "[newsbag] ERROR: NEWSBAG_PY not found/executable: $NEWSBAG_PY" >&2
  exit 2
fi

# Auto-heal orchestration env drift on shared Torch filesystems.
if ! "$NEWSBAG_PY" -c 'import newsbag' >/dev/null 2>&1; then
  echo "[newsbag] installing package into orchestration env: $PROJECT_ROOT"
  if ! "$NEWSBAG_PY" -m pip --version >/dev/null 2>&1; then
    echo "[newsbag] pip missing in orchestration env; bootstrapping with ensurepip"
    "$NEWSBAG_PY" -m ensurepip --upgrade
  fi
  "$NEWSBAG_PY" -m pip install -e "$PROJECT_ROOT"
fi

echo "[newsbag] host=$(hostname) date=$(date)"
echo "[newsbag] run_dir=$RUN_DIR"
echo "[newsbag] stages=$STAGES"
nvidia-smi

if echo ",$STAGES," | grep -qE ",paddle_layout,|,paddle_vl15,"; then
  echo "[newsbag] ERROR: Paddle stages requested on H200 ($STAGES). Paddle must run on l40s_public." >&2
  echo "[newsbag] Hint: run split GPU: L40S for paddle_layout,paddle_vl15 and H200 for dell,mineru." >&2
  exit 3
fi

"$NEWSBAG_PY" -m newsbag run --config "$CONFIG_JSON" --run-dir "$RUN_DIR" --stages "$STAGES"
