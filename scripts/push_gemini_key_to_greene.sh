#!/bin/bash
#
# Copy GEMINI_KEY from local .env to a protected file on Greene for Slurm jobs to source.
#
# Why this exists:
# - Slurm jobs need GEMINI_KEY on compute nodes.
# - Passing the key inline to `sbatch` leaks it into shell history / logs.
# - This writes a 0600 file under /scratch/sxr203 (not world-traversable).
#
# Usage (from repo root, locally):
#   bash scripts/push_gemini_key_to_greene.sh
#
# Optional overrides:
#   ENV_FILE=.env DEST=/scratch/sxr203/.secrets/gemini.env bash scripts/push_gemini_key_to_greene.sh

set -euo pipefail

ENV_FILE=${ENV_FILE:-.env}
DEST=${DEST:-/scratch/sxr203/.secrets/gemini.env}

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "ENV_FILE not found: ${ENV_FILE}" >&2
  exit 1
fi

gemini_key=$(
  python - <<PY
from pathlib import Path

env_path = Path("${ENV_FILE}")
key = None
for raw in env_path.read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    k, v = line.split("=", 1)
    if k.strip() == "GEMINI_KEY":
        key = v.strip().strip('"').strip("'")
        break
if not key:
    raise SystemExit("GEMINI_KEY not found in env file")
print(key)
PY
)

tmp="$(mktemp)"
trap 'rm -f "${tmp}"' EXIT
chmod 600 "${tmp}"

# Do NOT echo the key; just write it into a file that Slurm jobs can source.
#
# %q produces a shell-escaped form safe for `source`-ing.
printf 'export GEMINI_KEY=%q\n' "${gemini_key}" > "${tmp}"

# Create secrets dir on Greene with restrictive perms, then copy the file.
ssh greene "mkdir -p \"$(dirname "${DEST}")\" && chmod 700 \"$(dirname "${DEST}")\""
scp "${tmp}" "greene:${DEST}"
ssh greene "chmod 600 \"${DEST}\""

echo "Wrote GEMINI_KEY to greene:${DEST} (chmod 600)."
