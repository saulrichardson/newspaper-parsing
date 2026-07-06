#!/bin/bash
# Sync this repo to Torch, submit the parser-bagging Slurm canary, and print
# the structured result summary.

set -euo pipefail

REMOTE="${REMOTE:-torch}"
ACCOUNT="${ACCOUNT:-torch_pr_609_general}"
PARTITION="${PARTITION:-cs}"
PROFILE="${PROFILE:-full}"
CONFIG="${CONFIG:-}"
REMOTE_BASE="${REMOTE_BASE:-}"
WAIT=1
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-900}"
POLL_SECONDS="${POLL_SECONDS:-10}"
PLAN_ONLY=0
SKIP_SYNC=0

usage() {
  sed -n '1,80p' "$0"
  cat <<'TXT'

Flags:
  --remote HOST          SSH host, default: torch
  --remote-base PATH     Scratch root, default: /scratch/$REMOTE_USER/codex_hpc/parser_bagging
  --account ACCOUNT      Slurm account, default: torch_pr_609_general
  --partition PARTITION  Slurm partition, default: cs
  --profile PROFILE      baseline|adaptive|full|command_fixture|legacy_fixture, default: full
  --config PATH          Optional bagging adapter config, relative to repo or absolute on Torch
  --timeout SECONDS      Poll timeout, default: 900
  --poll SECONDS         Poll interval, default: 10
  --no-wait              Submit and print job/run paths without polling
  --skip-sync            Reuse the existing remote repo copy
  --plan-only            Print planned remote commands without executing
TXT
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --remote)
      REMOTE="${2:-}"
      shift 2
      ;;
    --remote-base)
      REMOTE_BASE="${2:-}"
      shift 2
      ;;
    --account)
      ACCOUNT="${2:-}"
      shift 2
      ;;
    --partition)
      PARTITION="${2:-}"
      shift 2
      ;;
    --profile)
      PROFILE="${2:-full}"
      shift 2
      ;;
    --config)
      CONFIG="${2:-}"
      shift 2
      ;;
    --timeout)
      TIMEOUT_SECONDS="${2:-900}"
      shift 2
      ;;
    --poll)
      POLL_SECONDS="${2:-10}"
      shift 2
      ;;
    --no-wait)
      WAIT=0
      shift 1
      ;;
    --skip-sync)
      SKIP_SYNC=1
      shift 1
      ;;
    --plan-only|--dry-run)
      PLAN_ONLY=1
      shift 1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

REMOTE_USER="$(ssh "$REMOTE" 'printf %s "$USER"')"
if [[ -z "$REMOTE_USER" ]]; then
  echo "ERROR: could not determine remote user for $REMOTE" >&2
  exit 2
fi

if [[ -z "$REMOTE_BASE" ]]; then
  REMOTE_BASE="/scratch/$REMOTE_USER/codex_hpc/parser_bagging"
fi

PROJECT_ROOT="$REMOTE_BASE/newspaper-parsing"
RUN_DIR="$REMOTE_BASE/runs/bagging_canary_$(date -u +%Y%m%d_%H%M%S)"
MANIFEST="$REMOTE_BASE/fixtures/parse_input.jsonl"
SCRIPT="torch/slurm/newsbag_bagging_canary_cs.sbatch"
CONFIG_REMOTE=""
if [[ -n "$CONFIG" ]]; then
  if [[ "$CONFIG" = /* ]]; then
    CONFIG_REMOTE="$CONFIG"
  else
    CONFIG_REMOTE="$PROJECT_ROOT/$CONFIG"
  fi
fi

echo "[plan] remote=$REMOTE"
echo "[plan] remote_user=$REMOTE_USER"
echo "[plan] remote_base=$REMOTE_BASE"
echo "[plan] project_root=$PROJECT_ROOT"
echo "[plan] run_dir=$RUN_DIR"
echo "[plan] manifest=$MANIFEST"
echo "[plan] account=$ACCOUNT partition=$PARTITION profile=$PROFILE"
if [[ -n "$CONFIG_REMOTE" ]]; then
  echo "[plan] config=$CONFIG_REMOTE"
fi

if [[ "$PLAN_ONLY" -eq 1 ]]; then
  echo "[plan] would rsync repo unless --skip-sync"
  echo "[plan] would run sbatch --test-only and submit $SCRIPT"
  exit 0
fi

ssh "$REMOTE" "mkdir -p '$REMOTE_BASE/logs' '$REMOTE_BASE/runs' '$REMOTE_BASE/fixtures' '$PROJECT_ROOT'"

if [[ "$SKIP_SYNC" -eq 0 ]]; then
  rsync -az --delete \
    --exclude '.git/' \
    --exclude '.pytest_cache/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude '.venv/' \
    --exclude 'runs/' \
    --exclude 'local-archive/' \
    ./ "$REMOTE:$PROJECT_ROOT/"
fi

ssh "$REMOTE" "cd '$PROJECT_ROOT' && sbatch --test-only -A '$ACCOUNT' -p '$PARTITION' --cpus-per-task=2 --mem=4G --time=00:10:00 --wrap hostname >/dev/null"

JOB_ID="$(
  ssh "$REMOTE" "cd '$PROJECT_ROOT' && sbatch --parsable -A '$ACCOUNT' -p '$PARTITION' \
    --export=ALL,BASE='$REMOTE_BASE',PROJECT_ROOT='$PROJECT_ROOT',RUN_DIR='$RUN_DIR',MANIFEST='$MANIFEST',PROFILE='$PROFILE',CONFIG='$CONFIG_REMOTE' \
    '$SCRIPT'"
)"

echo "[submit] job_id=$JOB_ID"
echo "[submit] run_dir=$RUN_DIR"
echo "[submit] logs=$REMOTE_BASE/logs/newsbag_bagging_canary-$JOB_ID.out"

if [[ "$WAIT" -eq 0 ]]; then
  exit 0
fi

deadline=$((SECONDS + TIMEOUT_SECONDS))
while [[ "$SECONDS" -lt "$deadline" ]]; do
  queued="$(ssh "$REMOTE" "squeue -h -j '$JOB_ID' -o %T 2>/dev/null || true")"
  if [[ -z "$queued" ]]; then
    break
  fi
  echo "[poll] job_id=$JOB_ID state=$queued"
  sleep "$POLL_SECONDS"
done

if [[ "$SECONDS" -ge "$deadline" ]]; then
  echo "ERROR: timed out waiting for job $JOB_ID after $TIMEOUT_SECONDS seconds" >&2
  exit 3
fi

ssh "$REMOTE" "python3 - <<PY
from __future__ import annotations

import json
from pathlib import Path

run_dir = Path('$RUN_DIR')
status_path = run_dir / 'slurm_status.json'
summary_path = run_dir / 'summary.json'
if status_path.exists():
    print(status_path.read_text(encoding='utf-8'), end='')
elif summary_path.exists():
    payload = json.loads(summary_path.read_text(encoding='utf-8'))
    print(json.dumps({'status': 'missing_slurm_status', 'run_dir': str(run_dir), 'summary': payload}, indent=2, sort_keys=True))
else:
    print(json.dumps({'status': 'missing_outputs', 'run_dir': str(run_dir)}, indent=2, sort_keys=True))
    raise SystemExit(1)
PY"
