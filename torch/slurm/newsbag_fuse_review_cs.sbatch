#!/bin/bash
# CPU job for fusion + review generation (do NOT run this on GPU partitions).
# Required env:
# - RUN_DIR: run directory created/used by the upstream inference job.
#
# Optional env:
# - PROJECT_ROOT, VENV_DIR, CONFIG_JSON, STAGES

#SBATCH -A torch_pr_609_general
#SBATCH -p cs
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH -J newsbag_fuse_review
#SBATCH -o /scratch/%u/paddleocr_vl15/logs/%x-%j.out
#SBATCH -e /scratch/%u/paddleocr_vl15/logs/%x-%j.err

set -euo pipefail

BASE="${BASE:-/scratch/$USER/paddleocr_vl15}"
PROJECT_ROOT="${PROJECT_ROOT:-$BASE/newspaper-parsing}"
VENV_DIR="${VENV_DIR:-$BASE/envs/mineru25_py310}"
CONFIG_JSON="${CONFIG_JSON:-$PROJECT_ROOT/configs/pipeline.torch.json}"
STAGES="${STAGES:-fusion,review}"
RUN_DIR="${RUN_DIR:-}"
NEWSBAG_PY="${NEWSBAG_PY:-}"

if [ -z "$RUN_DIR" ]; then
  echo "[newsbag] ERROR: RUN_DIR is required (shared across jobs)." >&2
  exit 2
fi

cd "$PROJECT_ROOT"
if [ -z "$NEWSBAG_PY" ]; then
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

"$NEWSBAG_PY" -m newsbag run --config "$CONFIG_JSON" --run-dir "$RUN_DIR" --stages "$STAGES"
