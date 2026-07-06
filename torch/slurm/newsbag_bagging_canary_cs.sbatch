#!/bin/bash
# CPU Slurm canary for the manifest-driven parser-bagging engine.
#
# Required or defaulted env:
# - BASE: scratch work root, defaults to /scratch/$USER/codex_hpc/parser_bagging
# - PROJECT_ROOT: synced repo, defaults to $BASE/newspaper-parsing
# - RUN_DIR: output run directory, defaults to $BASE/runs/bagging_canary_$SLURM_JOB_ID
# - MANIFEST: parse input JSONL manifest, defaults to $BASE/fixtures/parse_input.jsonl
# - PROFILE: baseline|adaptive|full, defaults to full
# - CONFIG: optional parser-bagging adapter config JSON
# - NEWSBAG_PY: Python executable. If unset, $BASE/envs/newsbag/bin/python is created/reused.

#SBATCH -A torch_pr_609_general
#SBATCH -p cs
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH -J newsbag_bagging_canary
#SBATCH -o /scratch/%u/codex_hpc/parser_bagging/logs/%x-%j.out
#SBATCH -e /scratch/%u/codex_hpc/parser_bagging/logs/%x-%j.err

set -euo pipefail

BASE="${BASE:-/scratch/$USER/codex_hpc/parser_bagging}"
PROJECT_ROOT="${PROJECT_ROOT:-$BASE/newspaper-parsing}"
RUN_DIR="${RUN_DIR:-$BASE/runs/bagging_canary_${SLURM_JOB_ID:-manual}}"
MANIFEST="${MANIFEST:-$BASE/fixtures/parse_input.jsonl}"
PROFILE="${PROFILE:-full}"
CONFIG="${CONFIG:-}"
BOOTSTRAP_PY="${BOOTSTRAP_PY:-}"
VENV_DIR="${VENV_DIR:-$BASE/envs/newsbag_py312}"
NEWSBAG_PY="${NEWSBAG_PY:-$VENV_DIR/bin/python}"

mkdir -p "$BASE/logs" "$BASE/fixtures" "$BASE/runs"

echo "[bagging-canary] host=$(hostname)"
echo "[bagging-canary] date=$(date -Is)"
echo "[bagging-canary] project_root=$PROJECT_ROOT"
echo "[bagging-canary] run_dir=$RUN_DIR"
echo "[bagging-canary] manifest=$MANIFEST"
echo "[bagging-canary] profile=$PROFILE"
if [ -n "$CONFIG" ]; then
  echo "[bagging-canary] config=$CONFIG"
fi

if [ ! -d "$PROJECT_ROOT/src/newsbag" ]; then
  echo "[bagging-canary] ERROR: PROJECT_ROOT does not look like newspaper-parsing: $PROJECT_ROOT" >&2
  exit 2
fi

select_bootstrap_python() {
  if [ -n "$BOOTSTRAP_PY" ]; then
    printf "%s\n" "$BOOTSTRAP_PY"
    return 0
  fi
  for candidate in python3.12 python3.11 python3.10 /scratch/$USER/paddleocr_vl15/envs/mineru25_py310/bin/python python3; do
    if command -v "$candidate" >/dev/null 2>&1 || [ -x "$candidate" ]; then
      if "$candidate" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY
      then
        command -v "$candidate" 2>/dev/null || printf "%s\n" "$candidate"
        return 0
      fi
    fi
  done
  return 1
}

if [ ! -x "$NEWSBAG_PY" ]; then
  BOOTSTRAP_PY="$(select_bootstrap_python)" || {
    echo "[bagging-canary] ERROR: could not find Python >=3.10 on this node" >&2
    exit 2
  }
  echo "[bagging-canary] bootstrap_python=$BOOTSTRAP_PY"
  echo "[bagging-canary] creating orchestration venv: $VENV_DIR"
  "$BOOTSTRAP_PY" -m venv "$VENV_DIR"
fi

if ! "$NEWSBAG_PY" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY
then
  echo "[bagging-canary] ERROR: NEWSBAG_PY must be Python >=3.10: $NEWSBAG_PY" >&2
  "$NEWSBAG_PY" --version >&2 || true
  exit 2
fi

if ! "$NEWSBAG_PY" -m pip --version >/dev/null 2>&1; then
  "$NEWSBAG_PY" -m ensurepip --upgrade
fi

"$NEWSBAG_PY" -m pip install --upgrade pip >/dev/null
"$NEWSBAG_PY" -m pip install -e "$PROJECT_ROOT[dev]" >/dev/null

if [ ! -s "$MANIFEST" ]; then
  echo "[bagging-canary] creating fixture manifest and page image"
  "$NEWSBAG_PY" - <<PY
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

base = Path("$BASE")
page_path = base / "fixtures" / "torch-page-001.png"
manifest = Path("$MANIFEST")
page_path.parent.mkdir(parents=True, exist_ok=True)
image = Image.new("RGB", (420, 640), "white")
draw = ImageDraw.Draw(image)
for x0, x1 in ((42, 188), (232, 378)):
    for y in range(68, 540, 31):
        draw.rectangle([x0, y, x1, y + 9], fill="black")
draw.rectangle([62, 484, 358, 584], outline="black", width=3)
image.save(page_path)
row = {
    "page_id": "torch-page-001",
    "image_path": str(page_path),
    "issue_id": "torch-fixture-issue",
    "page_number": 1,
    "source": {
        "source_system": "torch_fixture",
        "source_id": "torch-page-001",
    },
}
manifest.parent.mkdir(parents=True, exist_ok=True)
manifest.write_text(json.dumps(row, sort_keys=True) + "\\n", encoding="utf-8")
print(f"[bagging-canary] wrote {manifest}")
PY
fi

if [ "$PROFILE" = "legacy_fixture" ]; then
  echo "[bagging-canary] creating legacy layout fixture JSON"
  "$NEWSBAG_PY" - <<PY
from __future__ import annotations

import json
from pathlib import Path

manifest = Path("$MANIFEST")
for line in manifest.read_text(encoding="utf-8").splitlines():
    if not line.strip() or line.lstrip().startswith("#"):
        continue
    row = json.loads(line)
    page_id = str(row["page_id"])
    image_path = Path(row["image_path"])
    legacy_path = image_path.with_suffix(".legacy.json")
    legacy_path.write_text(
        json.dumps(
            {
                "boxes": [
                    {
                        "source_family": "legacy_fixture",
                        "source_model": "legacy_fixture_import",
                        "source_label": "paragraph",
                        "bbox_xyxy": [42, 64, 378, 164],
                        "text": f"legacy fixture transcript for {page_id}",
                        "reading_order": 1,
                    }
                ]
            },
            sort_keys=True,
        )
        + "\\n",
        encoding="utf-8",
    )
    print(f"[bagging-canary] wrote {legacy_path}")
PY
fi

cd "$PROJECT_ROOT"
EXTRA_ARGS=()
if [ -n "$CONFIG" ]; then
  EXTRA_ARGS+=(--config "$CONFIG")
fi
"$NEWSBAG_PY" -m newsbag bagging-canary \
  --manifest "$MANIFEST" \
  --run-dir "$RUN_DIR" \
  --profile "$PROFILE" \
  "${EXTRA_ARGS[@]}"

"$NEWSBAG_PY" -m newsbag validate-run \
  --run-dir "$RUN_DIR" \
  --output-json "$RUN_DIR/reports/validation.json" \
  --json

"$NEWSBAG_PY" - <<PY
from __future__ import annotations

import json
from pathlib import Path

run_dir = Path("$RUN_DIR")
summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
validation = json.loads((run_dir / "reports" / "validation.json").read_text(encoding="utf-8"))
errors = [
    json.loads(line)
    for line in (run_dir / "errors.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
status_ok = summary["performance"]["errors"] == 0 and validation["status"] == "ok"
status = {
    "status": "ok" if status_ok else "error",
    "run_dir": str(run_dir),
    "summary_path": str(run_dir / "summary.json"),
    "performance_path": str(run_dir / "reports" / "performance.json"),
    "validation_path": str(run_dir / "reports" / "validation.json"),
    "validation_status": validation["status"],
    "validation_issue_count": len(validation.get("issues") or []),
    "errors_path": str(run_dir / "errors.jsonl"),
    "pages_completed": summary["performance"]["pages_completed"],
    "errors": errors,
}
(run_dir / "slurm_status.json").write_text(json.dumps(status, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
print(json.dumps(status, indent=2, sort_keys=True))
raise SystemExit(0 if status["status"] == "ok" else 1)
PY
