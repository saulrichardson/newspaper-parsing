#!/bin/bash
# Submit Gemini Batch jobs for the dedup export request shards (Dell text boxes only).
#
# This job CALLS the Gemini API (uploads JSONL + creates a Batch job). It is NOT offline.
#
# Prereqs on Greene:
#   - Request files exist under IN_BASE/part_XXXX:
#       - gemini_requests_shardNNN.jsonl
#       - mapping_shardNNN.jsonl
#   - Python venv has google-genai installed (VENV below)
#   - GEMINI_KEY is available via SECRETS_FILE (see scripts/push_gemini_key_to_greene.sh)
#
# Output:
#   - Per-part submission record: IN_BASE/part_XXXX/submitted_batches.jsonl
#   - Per-part done flag: IN_BASE/part_XXXX/SUBMITTED_GEMINI
#
# Safe to re-run:
#   - submit_batch_shards.py is resumable via submitted_batches.jsonl
#   - done flag prevents redundant work once a part fully succeeded

#SBATCH -J submit-gemini-dedup
#SBATCH -o /vast/sxr203/newspaper-downloads/dedupe-webp/logs/submit_gemini_dedup_%A_%a.out
#SBATCH -e /vast/sxr203/newspaper-downloads/dedupe-webp/logs/submit_gemini_dedup_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --array=0-469%10
#SBATCH -p short

set -euo pipefail

PROJECT_ROOT=/scratch/sxr203/newspaper-parsing-export
VENV=/scratch/sxr203/newspaper-parsing/venv

IN_BASE=${IN_BASE:-/vast/sxr203/newspaper-downloads/dedupe-webp/batch_requests_dedup_jpeg1024_mp1_q80}
SECRETS_FILE=${SECRETS_FILE:-/scratch/sxr203/.secrets/gemini.env}
GEMINI_MODEL=${GEMINI_MODEL:-models/gemini-2.5-flash}

# If DRY_RUN=1, validates shards + prints what would be submitted, but does not call the API.
DRY_RUN=${DRY_RUN:-0}

# Retry/backoff behavior for Gemini 429s.
MAX_ATTEMPTS=${MAX_ATTEMPTS:-20}
INITIAL_SLEEP_SECONDS=${INITIAL_SLEEP_SECONDS:-20}
MAX_SLEEP_SECONDS=${MAX_SLEEP_SECONDS:-600}

idx=$(printf "%04d" "${SLURM_ARRAY_TASK_ID}")
part_dir="${IN_BASE}/part_${idx}"
done_flag="${part_dir}/SUBMITTED_GEMINI"

if [[ ! -d "${part_dir}" ]]; then
  echo "Missing part dir: ${part_dir}" >&2
  exit 1
fi

if [[ "${DRY_RUN}" != "1" ]] && [[ -f "${done_flag}" ]]; then
  echo "Already done: ${done_flag}"
  exit 0
fi

if [[ ! -f "${SECRETS_FILE}" ]]; then
  echo "Missing secrets file: ${SECRETS_FILE}" >&2
  echo "Create it via: scripts/push_gemini_key_to_greene.sh (run locally)" >&2
  exit 1
fi

source "${SECRETS_FILE}"
if [[ -z "${GEMINI_KEY:-}" ]]; then
  echo "GEMINI_KEY was not set after sourcing ${SECRETS_FILE}" >&2
  exit 1
fi

source "${VENV}/bin/activate"

cmd=(python "${PROJECT_ROOT}/scripts/submit_batch_shards.py" \
  --request-dir "${part_dir}" \
  --providers gemini \
  --gemini-model "${GEMINI_MODEL}")

if [[ "${DRY_RUN}" == "1" ]]; then
  cmd+=(--dry-run)
fi

attempt=1
sleep_s="${INITIAL_SLEEP_SECONDS}"
while true; do
  echo "Attempt ${attempt}/${MAX_ATTEMPTS} for ${part_dir} (sleep_s=${sleep_s})"
  tmp_out="$(mktemp)"
  tmp_err="$(mktemp)"
  set +e
  PYTHONUNBUFFERED=1 "${cmd[@]}" >"${tmp_out}" 2>"${tmp_err}"
  rc=$?
  set -e
  cat "${tmp_out}"
  cat "${tmp_err}" >&2

  if [[ "${rc}" == "0" ]]; then
    rm -f "${tmp_out}" "${tmp_err}"
    break
  fi

  if [[ -f "${done_flag}" ]]; then
    # Safety: should not happen because we only write done_flag on success, but don't spin if it does.
    rm -f "${tmp_out}" "${tmp_err}"
    echo "Found done flag despite failure: ${done_flag}" >&2
    exit "${rc}"
  fi

  if grep -q -E "RESOURCE_EXHAUSTED|\\b429\\b|rate limit" "${tmp_err}" 2>/dev/null || grep -q -E "RESOURCE_EXHAUSTED|\\b429\\b|rate limit" "${tmp_out}" 2>/dev/null; then
    if [[ "${attempt}" -ge "${MAX_ATTEMPTS}" ]]; then
      rm -f "${tmp_out}" "${tmp_err}"
      echo "Giving up after ${attempt} attempts due to repeated 429/RESOURCE_EXHAUSTED." >&2
      exit "${rc}"
    fi
    jitter=$((RANDOM % 10))
    wait_s=$((sleep_s + jitter))
    echo "Gemini quota/rate-limit hit; sleeping ${wait_s}s then retrying..." >&2
    rm -f "${tmp_out}" "${tmp_err}"
    sleep "${wait_s}"
    attempt=$((attempt + 1))
    sleep_s=$((sleep_s * 2))
    if [[ "${sleep_s}" -gt "${MAX_SLEEP_SECONDS}" ]]; then
      sleep_s="${MAX_SLEEP_SECONDS}"
    fi
    continue
  fi

  rm -f "${tmp_out}" "${tmp_err}"
  echo "Non-retriable failure (rc=${rc}) for ${part_dir}" >&2
  exit "${rc}"
done

if [[ "${DRY_RUN}" != "1" ]]; then
  touch "${done_flag}"
  echo "Wrote DONE flag: ${done_flag}"
fi
