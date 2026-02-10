#!/bin/bash
#SBATCH -A torch_pr_609_general
#SBATCH -p l40s_public
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH -J newsbag_canary1
#SBATCH -o /scratch/%u/paddleocr_vl15/logs/%x-%j.out
#SBATCH -e /scratch/%u/paddleocr_vl15/logs/%x-%j.err

set -euo pipefail

BASE="/scratch/$USER/paddleocr_vl15"
REPO="$BASE/new-ocr"
RUN_DIR="$BASE/runs/canary1_$(date +%Y%m%d_%H%M%S)"

# Use the portable conda python directly (not a venv tied to /usr/bin/python3).
NEWSBAG_PY="$BASE/envs/mineru25_py310/bin/python"

echo "[newsbag] host=$(hostname) date=$(date)"
nvidia-smi

mkdir -p "$BASE/input"

# 1 page canary: keep it fast.
MANIFEST="$BASE/input/news_manifest_canary1.txt"
printf "%s\n" \
  "$BASE/input/ad_hoc_newspapers_20260205_190618/cambridge-sentinel-apr-18-1942-p-4.png" \
  > "$MANIFEST"

CONFIG_JSON="$RUN_DIR/config.canary1.json"
$NEWSBAG_PY - <<PY
import json
from pathlib import Path
repo=Path("$REPO")
base=Path("$BASE")
payload=json.loads((repo/"configs/pipeline.torch.json").read_text())
payload["manifest_path"]=str(Path("$MANIFEST"))
payload["run_root"]=str(base/"runs")
payload["run_name"]="canary1"
out_cfg=Path("$RUN_DIR")/"config.canary1.json"
out_cfg.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print("[newsbag] wrote", out_cfg)
PY

echo "[newsbag] Dell providers (this job node):"
set +e
"$NEWSBAG_PY" - <<'PY'
try:
    import onnxruntime as ort
    print("available:", ort.get_available_providers())
except Exception as e:
    print("onnxruntime check failed:", repr(e))
PY
set -e

cd "$REPO"
"$NEWSBAG_PY" -m newsbag run --config "$CONFIG_JSON" --run-dir "$RUN_DIR" --stages "paddle_layout,paddle_vl15,dell,mineru,fusion,review"

echo "[newsbag] canary outputs: $RUN_DIR"
