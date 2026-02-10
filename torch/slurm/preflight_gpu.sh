#!/bin/bash
set -euo pipefail

ACCOUNT="${ACCOUNT:-torch_pr_609_general}"
PARTITION="${PARTITION:-l40s_public}"
GRES="${GRES:-gpu:l40s:1}"

echo "[preflight] account=$ACCOUNT partition=$PARTITION gres=$GRES"
sbatch --test-only -A "$ACCOUNT" -p "$PARTITION" --gres="$GRES" \
  --cpus-per-task=4 --mem=16G --time=00:05:00 --wrap hostname

echo "[preflight] slurm test-only passed"
