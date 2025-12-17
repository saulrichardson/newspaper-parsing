#!/bin/bash
#
# Copy OpenAI keys from local .env to a protected file on Greene for Slurm jobs to source.
#
# Why this exists:
# - Slurm jobs need OPENAI_KEY on compute nodes.
# - Passing the key inline to `sbatch` leaks it into shell history / logs.
# - This writes a 0600 file under /scratch/sxr203 (not world-traversable).
#
# Usage (from repo root, locally):
#   bash scripts/push_openai_key_to_greene.sh
#
# Optional overrides:
#   ENV_FILE=.env DEST=/scratch/sxr203/.secrets/openai.env bash scripts/push_openai_key_to_greene.sh

set -euo pipefail

ENV_FILE=${ENV_FILE:-.env}
DEST=${DEST:-/scratch/sxr203/.secrets/openai.env}

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "ENV_FILE not found: ${ENV_FILE}" >&2
  exit 1
fi

keys_json=$(
  python - <<PY
import json
from pathlib import Path

env_path = Path("${ENV_FILE}")
out = {}
for raw in env_path.read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
    k, v = line.split("=", 1)
    key = k.strip()
    if key in {"PROJECT_OPENAI_KEY", "OPENAI_KEY", "OPENAI_API_KEY"}:
        out[key] = v.strip().strip('"').strip("'")
if not out.get("PROJECT_OPENAI_KEY") and not out.get("OPENAI_KEY") and not out.get("OPENAI_API_KEY"):
    raise SystemExit("No OpenAI key found in env file (expected PROJECT_OPENAI_KEY and/or OPENAI_KEY / OPENAI_API_KEY)")
print(json.dumps(out))
PY
)

tmp="$(mktemp)"
trap 'rm -f "${tmp}"' EXIT
chmod 600 "${tmp}"

# Do NOT echo the keys; just write them into a file that Slurm jobs can source.
python - <<PY > "${tmp}"
import json
import shlex

keys = json.loads(r'''${keys_json}''')
for name in ["PROJECT_OPENAI_KEY", "OPENAI_API_KEY", "OPENAI_KEY"]:
    val = (keys.get(name) or "").strip()
    if not val:
        continue
    print(f"export {name}={shlex.quote(val)}")
PY

# Create secrets dir on Greene with restrictive perms, then copy the file.
ssh greene "mkdir -p \"$(dirname "${DEST}")\" && chmod 700 \"$(dirname "${DEST}")\""
scp "${tmp}" "greene:${DEST}"
ssh greene "chmod 600 \"${DEST}\""

echo "Wrote OpenAI key(s) to greene:${DEST} (chmod 600)."
