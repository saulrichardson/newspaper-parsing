#!/bin/bash
#SBATCH -A torch_pr_609_general
#SBATCH -p l40s_public
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH -J newsbag_l40s
#SBATCH -o /scratch/%u/paddleocr_vl15/logs/%x-%j.out
#SBATCH -e /scratch/%u/paddleocr_vl15/logs/%x-%j.err

set -euo pipefail

BASE="${BASE:-/scratch/$USER/paddleocr_vl15}"
PROJECT_ROOT="${PROJECT_ROOT:-$BASE/new-ocr}"
VENV_DIR="${VENV_DIR:-$BASE/envs/newsbag}"

# Allow overriding the orchestration python (e.g., a portable conda env).
# Default uses the lightweight newsbag venv.
NEWSBAG_PY="${NEWSBAG_PY:-}"

CONFIG_JSON="${CONFIG_JSON:-$PROJECT_ROOT/configs/pipeline.torch.json}"

# Default GPU job stages: run sources only to keep GPU utilization high.
# Run fusion/review separately (CPU/login) to avoid low-GPU-util cancellation.
STAGES="${STAGES:-paddle_layout,paddle_vl15,dell,mineru}"

# Optional explicit output run directory.
RUN_DIR="${RUN_DIR:-}"

cd "$PROJECT_ROOT"

echo "[newsbag] host=$(hostname) date=$(date)"
nvidia-smi

if [ -z "$NEWSBAG_PY" ]; then
  source "$VENV_DIR/bin/activate"
  NEWSBAG_PY="$VENV_DIR/bin/python"
fi

if [ ! -x "$NEWSBAG_PY" ]; then
  echo "[newsbag] ERROR: NEWSBAG_PY not found/executable: $NEWSBAG_PY" >&2
  exit 2
fi

if ! "$NEWSBAG_PY" -c 'import newsbag' >/dev/null 2>&1; then
  echo "[newsbag] installing package into orchestration env: $PROJECT_ROOT"
  "$NEWSBAG_PY" -m pip install -e "$PROJECT_ROOT"
fi

if [ -n "$RUN_DIR" ]; then
  "$NEWSBAG_PY" -m newsbag run --config "$CONFIG_JSON" --run-dir "$RUN_DIR" --stages "$STAGES"
else
  "$NEWSBAG_PY" -m newsbag run --config "$CONFIG_JSON" --stages "$STAGES"
fi
