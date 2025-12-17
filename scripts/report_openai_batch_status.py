#!/usr/bin/env python3
"""
Report status/progress for OpenAI Batch jobs recorded by scripts/submit_batch_shards.py.

Usage:
  python scripts/report_openai_batch_status.py \
    --submitted-record <request_dir>/submitted_batches.jsonl

This script:
  - loads batch_ids from the submitted record
  - queries OpenAI for each batch status + request_counts
  - prints aggregate totals and a short per-status breakdown

Auth:
  Prefers PROJECT_OPENAI_KEY (project key) then falls back to OPENAI_API_KEY / OPENAI_KEY.
  Reads from environment first, then from .env in repo root.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

from openai import OpenAI


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Report status/progress for OpenAI batches in submitted_batches.jsonl.")
    ap.add_argument("--submitted-record", required=True, help="Path to submitted_batches.jsonl")
    ap.add_argument(
        "--shards",
        default=None,
        help="Optional shard selector: e.g. '0-10' or '0,1,2'. Default: all shards in record.",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, only query the first N batches (useful for quick checks).",
    )
    ap.add_argument(
        "--sleep-ms",
        type=int,
        default=0,
        help="Optional delay between API calls (helps avoid rate limits).",
    )
    ap.add_argument(
        "--openai-key-mode",
        choices=["project_first", "openai_first", "openai_only", "project_only"],
        default="project_first",
        help=(
            "Which OpenAI key(s) to use/prefer. "
            "project_first (default) tries PROJECT_OPENAI_KEY then OPENAI_API_KEY/OPENAI_KEY. "
            "openai_only uses ONLY OPENAI_API_KEY/OPENAI_KEY (ignores PROJECT_OPENAI_KEY)."
        ),
    )
    return ap.parse_args()


def _load_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        key = k.strip()
        val = v.strip().strip('"').strip("'")
        if key:
            env[key] = val
    return env


def _find_default_env_file() -> Path | None:
    candidates = [
        Path(".env"),
        Path(__file__).resolve().parents[1] / ".env",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def _get_openai_key_candidates(*, mode: str) -> list[tuple[str, str]]:
    env_path = _find_default_env_file()
    file_env = _load_env_file(env_path) if env_path else {}

    candidates: list[tuple[str, str]] = []
    if mode == "project_first":
        names = ["PROJECT_OPENAI_KEY", "OPENAI_API_KEY", "OPENAI_KEY"]
    elif mode == "openai_first":
        names = ["OPENAI_API_KEY", "OPENAI_KEY", "PROJECT_OPENAI_KEY"]
    elif mode == "openai_only":
        names = ["OPENAI_API_KEY", "OPENAI_KEY"]
    elif mode == "project_only":
        names = ["PROJECT_OPENAI_KEY"]
    else:
        raise ValueError(f"Unknown openai key mode: {mode!r}")

    for name in names:
        val = os.environ.get(name) or file_env.get(name)
        if val:
            candidates.append((name, val))

    seen: set[str] = set()
    uniq: list[tuple[str, str]] = []
    for src, key in candidates:
        if key in seen:
            continue
        seen.add(key)
        uniq.append((src, key))
    return uniq


def _is_insufficient_quota_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return ("insufficient_quota" in msg) or ("exceeded your current quota" in msg)


def _is_not_found_error(exc: Exception) -> bool:
    # Most commonly happens when a batch was created under a different OpenAI project/key.
    msg = str(exc).lower()
    return ("no batch found" in msg) or ("error code: 404" in msg) or ("notfounderror" in msg)


def _parse_shard_selector(selector: str) -> set[int]:
    selector = selector.strip()
    if not selector:
        return set()
    if "-" in selector and "," not in selector:
        a, b = selector.split("-", 1)
        start = int(a)
        end = int(b)
        if end < start:
            raise SystemExit(f"Invalid shard range: {selector}")
        return set(range(start, end + 1))
    parts = [p.strip() for p in selector.split(",") if p.strip()]
    return {int(p) for p in parts}


def _retrieve_batch(client: OpenAI, batch_id: str):
    return client.batches.retrieve(batch_id)


def _retrieve_batch_with_key_fallback(keys: list[tuple[str, str]], batch_id: str):
    last_exc: Exception | None = None
    for src, key in keys:
        client = OpenAI(api_key=key)
        try:
            job = _retrieve_batch(client, batch_id)
            return job, src
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if _is_insufficient_quota_error(exc) or _is_not_found_error(exc):
                continue
            raise
    raise SystemExit(str(last_exc) if last_exc else f"Failed to retrieve batch {batch_id}")


def main() -> None:
    args = _parse_args()
    record_path = Path(args.submitted_record).expanduser().resolve()
    if not record_path.is_file():
        raise SystemExit(f"--submitted-record not found: {record_path}")

    rows: list[dict] = []
    for raw in record_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        obj = json.loads(line)
        if obj.get("provider") != "openai":
            continue
        if not isinstance(obj.get("batch_id"), str):
            continue
        if not isinstance(obj.get("shard"), int):
            continue
        rows.append(obj)

    if args.shards:
        want = _parse_shard_selector(args.shards)
        rows = [r for r in rows if r["shard"] in want]

    rows.sort(key=lambda r: int(r["shard"]))
    if args.limit and int(args.limit) > 0:
        rows = rows[: int(args.limit)]

    if not rows:
        raise SystemExit(f"No OpenAI batch rows found in {record_path}")

    keys = _get_openai_key_candidates(mode=str(args.openai_key_mode))
    if not keys:
        raise SystemExit(
            "Missing OpenAI key. Set PROJECT_OPENAI_KEY (preferred) or OPENAI_API_KEY/OPENAI_KEY in env, "
            "or add them to .env in repo root."
        )

    status_counts: Counter[str] = Counter()
    auth_source_counts: Counter[str] = Counter()
    total = completed = failed = 0

    for i, r in enumerate(rows, start=1):
        batch_id = r["batch_id"]
        shard = int(r["shard"])
        job, auth_src = _retrieve_batch_with_key_fallback(keys, batch_id)
        auth_source_counts[auth_src] += 1

        status = str(getattr(job, "status", "unknown"))
        status_counts[status] += 1

        rc = getattr(job, "request_counts", None)
        if rc is not None:
            try:
                total += int(rc.total or 0)
                completed += int(rc.completed or 0)
                failed += int(rc.failed or 0)
            except Exception:
                pass

        if args.sleep_ms and int(args.sleep_ms) > 0:
            time.sleep(int(args.sleep_ms) / 1000.0)

        if i % 25 == 0:
            print(f"progress {i}/{len(rows)} status_counts={dict(status_counts)}", flush=True)

    print(f"record_path\t{record_path}")
    print(f"auth_sources\t{json.dumps(dict(auth_source_counts), ensure_ascii=False)}")
    print(f"batches_queried\t{len(rows)}")
    print(f"status_counts\t{json.dumps(dict(status_counts), ensure_ascii=False)}")
    if total:
        pct = (completed / total) * 100.0
        print(f"requests_total\t{total}")
        print(f"requests_completed\t{completed}")
        print(f"requests_failed\t{failed}")
        print(f"requests_completed_pct\t{pct:.2f}%")


if __name__ == "__main__":
    main()
