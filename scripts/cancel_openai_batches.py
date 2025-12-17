#!/usr/bin/env python3
"""
Cancel OpenAI Batch jobs recorded in a submitted_batches.jsonl file.

This is useful if you need to stop an in-flight run (e.g., to regenerate inputs
without an option like max_output_tokens).

Typical usage:
  python scripts/cancel_openai_batches.py \
    --submitted-record <request_dir>/submitted_batches.jsonl

Notes:
  - We retrieve each batch first to check status.
  - By default, we SKIP batches already completed/cancelled/failed/expired.
  - Writes an append-only cancellation log JSONL (default: alongside submitted record).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI


@dataclass(frozen=True)
class SubmittedBatch:
    shard: int
    batch_id: str
    display_name: str | None


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Cancel OpenAI batches in submitted_batches.jsonl.")
    ap.add_argument("--submitted-record", required=True, help="Path to submitted_batches.jsonl")
    ap.add_argument(
        "--record-out",
        default=None,
        help="Append-only cancellation log (default: <submitted-record dir>/cancelled_batches.jsonl)",
    )
    ap.add_argument(
        "--shards",
        default=None,
        help="Optional shard selector: e.g. '0-10' or '0,1,2'. Default: all shards in record.",
    )
    ap.add_argument(
        "--include-completed",
        action="store_true",
        help="Attempt to cancel even if batch is completed/cancelled/failed/expired (usually unnecessary).",
    )
    ap.add_argument("--sleep-ms", type=int, default=0, help="Delay between API calls (avoid rate limits).")
    ap.add_argument("--dry-run", action="store_true", help="Print what would be cancelled, but do not call API.")
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


def _get_openai_key_candidates() -> list[tuple[str, str]]:
    env_path = _find_default_env_file()
    file_env = _load_env_file(env_path) if env_path else {}

    candidates: list[tuple[str, str]] = []
    for name in ["PROJECT_OPENAI_KEY", "OPENAI_API_KEY", "OPENAI_KEY"]:
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


def _load_submitted_batches(path: Path) -> list[SubmittedBatch]:
    if not path.is_file():
        raise SystemExit(f"--submitted-record not found: {path}")

    # Keep the *last* record per shard (in case multiple appends happened).
    by_shard: dict[int, dict[str, Any]] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("provider") != "openai":
            continue
        shard = obj.get("shard")
        batch_id = obj.get("batch_id")
        if not isinstance(shard, int) or not isinstance(batch_id, str):
            continue
        by_shard[shard] = obj

    out: list[SubmittedBatch] = []
    for shard in sorted(by_shard):
        obj = by_shard[shard]
        out.append(
            SubmittedBatch(
                shard=int(shard),
                batch_id=str(obj.get("batch_id")),
                display_name=obj.get("display_name"),
            )
        )
    return out


def _append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main() -> None:
    args = _parse_args()
    submitted = Path(args.submitted_record).expanduser().resolve()
    record_out = (
        Path(args.record_out).expanduser().resolve()
        if args.record_out
        else (submitted.parent / "cancelled_batches.jsonl")
    )

    batches = _load_submitted_batches(submitted)
    if args.shards:
        want = _parse_shard_selector(args.shards)
        batches = [b for b in batches if b.shard in want]
    if not batches:
        raise SystemExit(f"No OpenAI batches found in {submitted}")

    keys = _get_openai_key_candidates()
    if not keys:
        raise SystemExit(
            "Missing OpenAI key. Set PROJECT_OPENAI_KEY (preferred) or OPENAI_API_KEY/OPENAI_KEY in env, "
            "or add them to .env in repo root."
        )

    client = OpenAI(api_key=keys[0][1])
    auth_src = keys[0][0]

    terminal_statuses = {"completed", "failed", "expired", "cancelled"}

    cancelled = skipped = attempted = 0
    for b in batches:
        job = client.batches.retrieve(b.batch_id)
        status = str(getattr(job, "status", "unknown"))
        if (not args.include_completed) and (status in terminal_statuses):
            skipped += 1
            _append_jsonl(
                record_out,
                {
                    "provider": "openai",
                    "shard": b.shard,
                    "batch_id": b.batch_id,
                    "display_name": b.display_name,
                    "action": "skip_terminal",
                    "status": status,
                    "auth_source": auth_src,
                    "ts": int(time.time()),
                },
            )
            continue

        attempted += 1
        if args.dry_run:
            print(f"DRY-RUN cancel shard{b.shard:03d} {b.batch_id} status={status}")
            continue

        try:
            job2 = client.batches.cancel(b.batch_id)
            status2 = str(getattr(job2, "status", "unknown"))
            _append_jsonl(
                record_out,
                {
                    "provider": "openai",
                    "shard": b.shard,
                    "batch_id": b.batch_id,
                    "display_name": b.display_name,
                    "action": "cancel",
                    "status_before": status,
                    "status_after": status2,
                    "auth_source": auth_src,
                    "ts": int(time.time()),
                },
            )
            if status2 in {"cancelling", "cancelled"}:
                cancelled += 1
        except Exception as exc:  # noqa: BLE001
            # Some statuses (e.g., 'finalizing') cannot be cancelled and return HTTP 409.
            _append_jsonl(
                record_out,
                {
                    "provider": "openai",
                    "shard": b.shard,
                    "batch_id": b.batch_id,
                    "display_name": b.display_name,
                    "action": "cancel_error",
                    "status_before": status,
                    "error": str(exc),
                    "auth_source": auth_src,
                    "ts": int(time.time()),
                },
            )

        if args.sleep_ms and int(args.sleep_ms) > 0:
            time.sleep(int(args.sleep_ms) / 1000.0)

    print(f"submitted_record\t{submitted}")
    print(f"record_out\t{record_out}")
    print(f"auth_source\t{auth_src}")
    print(f"batches_total\t{len(batches)}")
    print(f"attempted\t{attempted}")
    print(f"cancelled_or_cancelling\t{cancelled}")
    print(f"skipped_terminal\t{skipped}")


if __name__ == "__main__":
    main()
