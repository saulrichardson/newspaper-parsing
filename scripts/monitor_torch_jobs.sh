#!/bin/bash
set -euo pipefail

REMOTE="${REMOTE:-torch}"
TARGET_USER="${TARGET_USER:-}"

if [ -z "$TARGET_USER" ]; then
  # Default to the remote username (useful when local $USER != Torch $USER).
  TARGET_USER="$(ssh "$REMOTE" "echo \\$USER" 2>/dev/null || true)"
fi

if [ -z "$TARGET_USER" ]; then
  TARGET_USER="$USER"
fi

ssh "$REMOTE" "echo host=\$(hostname) date=\$(date); squeue -u $TARGET_USER -o '%.18i %.12P %.24j %.8T %.10M %.9l %.6D %R'"
