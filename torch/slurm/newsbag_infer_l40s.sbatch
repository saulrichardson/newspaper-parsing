#!/bin/bash
# GPU inference-only job for Torch (keeps GPU utilization high; no fusion/review here).
# Required env:
# - RUN_DIR: shared run directory used by downstream CPU fusion/review job.
#
# Optional env:
# - PROJECT_ROOT, VENV_DIR, CONFIG_JSON, STAGES

#SBATCH -A torch_pr_609_general
#SBATCH -p l40s_public
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=12:00:00
#SBATCH -J newsbag_infer_l40s
#SBATCH -o /scratch/%u/paddleocr_vl15/logs/%x-%j.out
#SBATCH -e /scratch/%u/paddleocr_vl15/logs/%x-%j.err

set -euo pipefail

BASE="${BASE:-/scratch/$USER/paddleocr_vl15}"
PROJECT_ROOT="${PROJECT_ROOT:-$BASE/newspaper-parsing}"
VENV_DIR="${VENV_DIR:-$BASE/envs/mineru25_py310}"
CONFIG_JSON="${CONFIG_JSON:-$PROJECT_ROOT/configs/pipeline.torch.json}"
STAGES="${STAGES:-paddle_layout,paddle_vl15,dell,mineru}"
RUN_DIR="${RUN_DIR:-}"
NEWSBAG_PY="${NEWSBAG_PY:-}"
USER_CACHE_ROOT="${USER_CACHE_ROOT:-/scratch/$USER}"

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

# Keep PaddleX/Paddle caches on scratch and auto-heal broken home symlinks.
mkdir -p "$USER_CACHE_ROOT/.cache" "$USER_CACHE_ROOT/.cache/paddle" "$USER_CACHE_ROOT/.paddlex"
if [ -L "/home/$USER/.cache" ] && [ ! -e "/home/$USER/.cache" ]; then
  rm -f "/home/$USER/.cache"
fi
if [ -L "/home/$USER/.paddlex" ] && [ ! -e "/home/$USER/.paddlex" ]; then
  rm -f "/home/$USER/.paddlex"
fi
if [ ! -e "/home/$USER/.cache" ]; then
  ln -s "$USER_CACHE_ROOT/.cache" "/home/$USER/.cache"
fi
if [ ! -e "/home/$USER/.paddlex" ]; then
  ln -s "$USER_CACHE_ROOT/.paddlex" "/home/$USER/.paddlex"
fi

export XDG_CACHE_HOME="$USER_CACHE_ROOT/.cache"
export PADDLE_HOME="$USER_CACHE_ROOT/.cache/paddle"
export PADDLEX_HOME="$USER_CACHE_ROOT/.paddlex"

# Auto-heal orchestration env drift on shared Torch filesystems.
# Reinstall if package is missing OR points outside PROJECT_ROOT (stale editable install).
if ! "$NEWSBAG_PY" - <<PY >/dev/null 2>&1
import pathlib
import newsbag
p = pathlib.Path(newsbag.__file__).resolve()
root = pathlib.Path("$PROJECT_ROOT").resolve()
raise SystemExit(0 if root in p.parents else 1)
PY
then
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
has_stage() {
  echo ",$STAGES," | grep -q ",$1,"
}

HAS_PADDLE_LAYOUT=0
HAS_PADDLE_VL15=0
HAS_DELL=0
HAS_MINERU=0
if has_stage "paddle_layout"; then HAS_PADDLE_LAYOUT=1; fi
if has_stage "paddle_vl15"; then HAS_PADDLE_VL15=1; fi
if has_stage "dell"; then HAS_DELL=1; fi
if has_stage "mineru"; then HAS_MINERU=1; fi

if [ "$HAS_PADDLE_LAYOUT" -eq 1 ]; then
  export NEWSBAG_PADDLE_LAYOUT_WORKERS="${NEWSBAG_PADDLE_LAYOUT_WORKERS:-3}"
  echo "[newsbag] NEWSBAG_PADDLE_LAYOUT_WORKERS=$NEWSBAG_PADDLE_LAYOUT_WORKERS"
fi

if [ "$HAS_PADDLE_VL15" -eq 1 ]; then
  # OCR-only runs can push one extra in-flight page per GPU to raise utilization.
  default_vl15_workers=3
  if [ "$HAS_PADDLE_LAYOUT" -eq 0 ] && [ "$HAS_DELL" -eq 0 ] && [ "$HAS_MINERU" -eq 0 ]; then
    default_vl15_workers=4
  fi
  export NEWSBAG_PADDLE_VL15_WORKERS="${NEWSBAG_PADDLE_VL15_WORKERS:-$default_vl15_workers}"
  echo "[newsbag] NEWSBAG_PADDLE_VL15_WORKERS=$NEWSBAG_PADDLE_VL15_WORKERS"
fi
if [ "$HAS_MINERU" -eq 1 ]; then
  export MINERU_BATCH_PAGES="${MINERU_BATCH_PAGES:-4}"
  echo "[newsbag] MINERU_BATCH_PAGES=$MINERU_BATCH_PAGES"
fi

# Parallel shard jobs should not contend on a shared "latest" convenience symlink.
export NEWSBAG_SKIP_LATEST_LINK="${NEWSBAG_SKIP_LATEST_LINK:-1}"
echo "[newsbag] NEWSBAG_SKIP_LATEST_LINK=$NEWSBAG_SKIP_LATEST_LINK"

nvidia-smi

"$NEWSBAG_PY" -m newsbag run --config "$CONFIG_JSON" --run-dir "$RUN_DIR" --stages "$STAGES"
