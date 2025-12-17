#!/usr/bin/env python3
"""
Download Gemini Batch outputs for previously-submitted shards.

Reads a `submitted_batches.jsonl` (created by scripts/submit_batch_shards.py),
queries each Gemini batch job, and downloads the completed output file to disk.

Typical usage:
  python scripts/download_gemini_batch_results.py \
    --request-dir newspaper-parsing-local/data/batch_requests_remaining \
    --out-dir newspaper-parsing-local/data/batch_results_gemini_remaining

Outputs:
  - gemini_results_shardNNN.jsonl: raw Gemini batch responses (one per input line)
  - downloaded_batches.jsonl: append-only download log (resumable)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class SubmittedGeminiBatch:
    shard: int
    batch_name: str
    display_name: str | None
    mapping_path: str | None
    requests_path: str | None


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"Missing required environment variable: {name}")
    return value


def _get_gemini_key() -> str:
    """Return GEMINI_KEY from env, falling back to .env in repo root."""

    key = os.environ.get("GEMINI_KEY")
    if key:
        return key

    env_path = Path(".env")
    if env_path.is_file():
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip() == "GEMINI_KEY":
                return v.strip().strip('"').strip("'")

    raise SystemExit("Missing GEMINI_KEY (set env var or add GEMINI_KEY=... to .env)")


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


def _load_submitted_batches(path: Path) -> list[SubmittedGeminiBatch]:
    if not path.exists():
        raise SystemExit(f"submitted_batches.jsonl not found: {path}")

    # Keep the *last* record per shard in case you appended multiple runs.
    by_shard: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("provider") != "gemini":
                continue
            shard = obj.get("shard")
            batch_name = obj.get("batch_name")
            if not isinstance(shard, int) or not isinstance(batch_name, str):
                continue
            by_shard[shard] = obj

    out: list[SubmittedGeminiBatch] = []
    for shard in sorted(by_shard):
        obj = by_shard[shard]
        out.append(
            SubmittedGeminiBatch(
                shard=shard,
                batch_name=str(obj.get("batch_name")),
                display_name=obj.get("display_name"),
                mapping_path=obj.get("mapping_path"),
                requests_path=obj.get("requests_path"),
            )
        )
    return out


def _append_record(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Download Gemini Batch outputs from submitted_batches.jsonl.")
    ap.add_argument(
        "--request-dir",
        required=True,
        help="Directory containing submitted_batches.jsonl and mapping_shard*.jsonl",
    )
    ap.add_argument(
        "--out-dir",
        required=True,
        help="Directory to write gemini_results_shardNNN.jsonl files into",
    )
    ap.add_argument(
        "--record",
        default=None,
        help="Path to submitted_batches.jsonl (default: <request-dir>/submitted_batches.jsonl)",
    )
    ap.add_argument(
        "--download-record",
        default=None,
        help="Append-only download log (default: <out-dir>/downloaded_batches.jsonl)",
    )
    ap.add_argument(
        "--shards",
        default=None,
        help="Optional shard selector (e.g. '0-16' or '0,1,2'). Default: all shards in record.",
    )
    ap.add_argument("--skip-existing", action="store_true", help="Skip downloads if output file exists")
    ap.add_argument("--dry-run", action="store_true", help="Print what would be downloaded, but do not download")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    request_dir = Path(args.request_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    record_path = Path(args.record).expanduser() if args.record else (request_dir / "submitted_batches.jsonl")
    download_record = (
        Path(args.download_record).expanduser()
        if args.download_record
        else (out_dir / "downloaded_batches.jsonl")
    )

    if not request_dir.is_dir():
        raise SystemExit(f"--request-dir is not a directory: {request_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    batches = _load_submitted_batches(record_path)
    if args.shards:
        want = _parse_shard_selector(args.shards)
        batches = [b for b in batches if b.shard in want]
    if not batches:
        raise SystemExit(f"No Gemini batches found in {record_path}")

    from google import genai  # imported late so the script is importable without deps

    client = genai.Client(api_key=_get_gemini_key())

    for b in batches:
        out_path = out_dir / f"gemini_results_shard{b.shard:03d}.jsonl"
        if args.skip_existing and out_path.exists():
            print(f"skip shard{b.shard:03d} (exists): {out_path}")
            continue

        if args.dry_run:
            print(f"DRY-RUN download shard{b.shard:03d} {b.batch_name} -> {out_path}")
            continue

        t0 = time.time()
        job = client.batches.get(name=b.batch_name)
        state = getattr(job, "state", None)
        if str(state) != "JobState.JOB_STATE_SUCCEEDED":
            _eprint(f"skip shard{b.shard:03d} (state={state}): {b.batch_name}")
            continue

        dest = getattr(job, "dest", None)
        file_name = getattr(dest, "file_name", None) if dest else None
        if not file_name:
            raise SystemExit(f"Batch {b.batch_name} has no dest.file_name; cannot download.")

        blob = client.files.download(file=file_name)
        out_path.write_bytes(blob)

        elapsed = round(time.time() - t0, 2)
        rec: dict[str, Any] = {
            "provider": "gemini",
            "shard": b.shard,
            "batch_name": b.batch_name,
            "display_name": b.display_name,
            "state": str(state),
            "dest_file_name": file_name,
            "out_path": str(out_path),
            "bytes": len(blob),
            "elapsed_seconds": elapsed,
            "downloaded_at": int(time.time()),
            "mapping_path": b.mapping_path,
            "requests_path": b.requests_path,
        }
        _append_record(download_record, rec)
        print(f"downloaded shard{b.shard:03d} -> {out_path} ({len(blob)} bytes)")

    print(f"Done. wrote {download_record}")


if __name__ == "__main__":
    main()
