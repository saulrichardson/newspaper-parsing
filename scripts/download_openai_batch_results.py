#!/usr/bin/env python3
"""
Download OpenAI Batch outputs for previously-submitted shards.

Reads a `submitted_batches.jsonl` (created by scripts/submit_batch_shards.py),
queries each OpenAI batch job, and downloads the completed output + error JSONL
to disk.

This is the OpenAI analogue to scripts/download_gemini_batch_results.py.

Typical usage (single request dir):
  python scripts/download_openai_batch_results.py \
    --request-dir /path/to/openai_request_dir \
    --out-dir /path/to/out_dir

Outputs (per shard):
  - openai_results_shardNNN.jsonl: raw OpenAI batch output JSONL
  - openai_errors_shardNNN.jsonl: raw OpenAI batch error JSONL (may be empty)
  - downloaded_batches.jsonl: append-only download log (resumable)

Environment:
  - OPENAI_API_KEY or OPENAI_KEY
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SubmittedOpenAIBatch:
    shard: int
    batch_id: str
    display_name: str | None
    mapping_path: str | None
    requests_path: str | None


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


def _get_openai_key_candidates(*, mode: str) -> list[tuple[str, str]]:
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

    candidates = [
        Path(".env"),
        Path(__file__).resolve().parents[1] / ".env",
    ]
    file_env: dict[str, str] = {}
    for p in candidates:
        if p.is_file():
            file_env = _load_env_file(p)
            break

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

    out: list[tuple[str, str]] = []
    for name in names:
        val = os.environ.get(name) or file_env.get(name)
        if not val:
            continue
        out.append((name, val))

    seen: set[str] = set()
    uniq: list[tuple[str, str]] = []
    for src, key in out:
        if key in seen:
            continue
        seen.add(key)
        uniq.append((src, key))

    if not uniq:
        raise SystemExit(
            "Missing OpenAI key. Set PROJECT_OPENAI_KEY (preferred) or OPENAI_API_KEY/OPENAI_KEY in env, "
            "or add them to .env in repo root."
        )
    return uniq


def _is_insufficient_quota_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return ("insufficient_quota" in msg) or ("exceeded your current quota" in msg)


def _is_not_found_error(exc: Exception) -> bool:
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


def _load_submitted_batches(path: Path) -> list[SubmittedOpenAIBatch]:
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
            if obj.get("provider") != "openai":
                continue
            shard = obj.get("shard")
            batch_id = obj.get("batch_id")
            if not isinstance(shard, int) or not isinstance(batch_id, str):
                continue
            by_shard[shard] = obj

    out: list[SubmittedOpenAIBatch] = []
    for shard in sorted(by_shard):
        obj = by_shard[shard]
        out.append(
            SubmittedOpenAIBatch(
                shard=shard,
                batch_id=str(obj.get("batch_id")),
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


def _write_openai_file_to_disk(*, client, file_id: str, out_path: Path) -> int:
    def _is_retryable_download_error(exc: Exception) -> bool:
        # OpenAI file downloads can occasionally return transient 5xx errors.
        msg = str(exc).lower()
        if ("error code: 500" in msg) or ("error code: 502" in msg) or ("error code: 503" in msg) or ("error code: 504" in msg):
            return True
        # Some exceptions include a status_code attribute.
        sc = getattr(exc, "status_code", None)
        if isinstance(sc, int) and sc in {500, 502, 503, 504}:
            return True
        return False

    # Avoid loading large JSONL files into memory.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")

    max_attempts = 6
    for attempt in range(1, max_attempts + 1):
        try:
            if tmp_path.exists():
                tmp_path.unlink()

            blob = client.files.content(file_id)

            # openai-python returns HttpxBinaryResponseContent which supports iter_bytes() and read().
            if hasattr(blob, "iter_bytes"):
                n = 0
                with tmp_path.open("wb") as f:
                    for chunk in blob.iter_bytes():
                        if not chunk:
                            continue
                        f.write(chunk)
                        n += len(chunk)
                tmp_path.replace(out_path)
                return n
            if hasattr(blob, "read"):
                data = blob.read()
                tmp_path.write_bytes(data)
                tmp_path.replace(out_path)
                return len(data)
            if hasattr(blob, "content"):
                data = blob.content  # type: ignore[attr-defined]
                tmp_path.write_bytes(data)
                tmp_path.replace(out_path)
                return len(data)
            if hasattr(blob, "text"):
                text = blob.text  # type: ignore[attr-defined]
                tmp_path.write_text(text, encoding="utf-8")
                tmp_path.replace(out_path)
                return len(text.encode("utf-8"))
            raise SystemExit(f"Unknown OpenAI file content response type: {type(blob)}")
        except Exception as exc:  # noqa: BLE001
            if attempt >= max_attempts or (not _is_retryable_download_error(exc)):
                raise
            delay = min(30, 2 * attempt)
            _eprint(
                f"warn: download failed for file_id={file_id} -> {out_path} "
                f"(attempt {attempt}/{max_attempts}): {exc} ; retrying in {delay}s"
            )
            time.sleep(delay)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Download OpenAI Batch outputs from submitted_batches.jsonl.")
    ap.add_argument(
        "--request-dir",
        required=True,
        help="Directory containing submitted_batches.jsonl and mapping_shard*.jsonl",
    )
    ap.add_argument(
        "--out-dir",
        required=True,
        help="Directory to write openai_results_shardNNN.jsonl and openai_errors_shardNNN.jsonl into",
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
    ap.add_argument("--skip-existing", action="store_true", help="Skip downloads if output files exist")
    ap.add_argument(
        "--skip-not-completed",
        action="store_true",
        help="Skip batches whose status is not 'completed' instead of aborting.",
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
        raise SystemExit(f"No OpenAI batches found in {record_path}")

    from openai import OpenAI  # imported late so the script is importable without deps

    keys = _get_openai_key_candidates(mode=str(args.openai_key_mode))

    for b in batches:
        out_path = out_dir / f"openai_results_shard{b.shard:03d}.jsonl"
        err_path = out_dir / f"openai_errors_shard{b.shard:03d}.jsonl"
        if args.skip_existing and out_path.exists() and err_path.exists():
            print(f"skip shard{b.shard:03d} (exists): {out_path} {err_path}")
            continue

        if args.dry_run:
            print(f"DRY-RUN download shard{b.shard:03d} {b.batch_id} -> {out_path} (+errors)")
            continue

        t0 = time.time()
        # A batch can be invisible (404) from a different OpenAI project key than the one that created it.
        # Try keys in preference order until one can retrieve the job.
        job = None
        client = None
        auth_src = None
        last_exc: Exception | None = None
        for src, key in keys:
            client_try = OpenAI(api_key=key)
            try:
                job_try = client_try.batches.retrieve(b.batch_id)
                job = job_try
                client = client_try
                auth_src = src
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if _is_not_found_error(exc) or _is_insufficient_quota_error(exc):
                    continue
                raise
        if job is None or client is None:
            raise SystemExit(str(last_exc) if last_exc else f"Failed to retrieve batch {b.batch_id}")

        status = getattr(job, "status", None)
        if str(status) != "completed":
            if args.skip_not_completed:
                print(f"skip shard{b.shard:03d} (status={status}): {b.batch_id}")
                continue
            raise SystemExit(f"Batch {b.batch_id} is not completed (status={status}); refusing to download.")

        output_file_id = getattr(job, "output_file_id", None)
        error_file_id = getattr(job, "error_file_id", None)
        request_counts = getattr(job, "request_counts", None)

        if not output_file_id:
            # OpenAI can omit output_file_id when *all* requests fail; in that case we still want
            # to download the error_file_id, and we create an empty output JSONL so downstream code
            # can treat it uniformly.
            if error_file_id:
                out_path.write_text("", encoding="utf-8")
                bytes_out = 0
            else:
                raise SystemExit(f"Batch {b.batch_id} has no output_file_id and no error_file_id; cannot download.")
        else:
            bytes_out = _write_openai_file_to_disk(client=client, file_id=str(output_file_id), out_path=out_path)

        if error_file_id:
            bytes_err = _write_openai_file_to_disk(client=client, file_id=str(error_file_id), out_path=err_path)
        else:
            err_path.write_text("", encoding="utf-8")
            bytes_err = 0

        elapsed = round(time.time() - t0, 2)
        rec: dict[str, Any] = {
            "provider": "openai",
            "shard": b.shard,
            "batch_id": b.batch_id,
            "display_name": b.display_name,
            "status": str(status),
            "auth_source": auth_src,
            "output_file_id": str(output_file_id),
            "error_file_id": str(error_file_id) if error_file_id else None,
            "out_path": str(out_path),
            "err_path": str(err_path),
            "bytes_out": bytes_out,
            "bytes_err": bytes_err,
            "elapsed_seconds": elapsed,
            "downloaded_at": int(time.time()),
            "mapping_path": b.mapping_path,
            "requests_path": b.requests_path,
        }
        if request_counts is not None:
            try:
                rec["request_counts"] = {
                    "total": int(getattr(request_counts, "total", 0) or 0),
                    "completed": int(getattr(request_counts, "completed", 0) or 0),
                    "failed": int(getattr(request_counts, "failed", 0) or 0),
                }
            except Exception:
                pass

        _append_record(download_record, rec)
        print(
            f"downloaded shard{b.shard:03d} -> {out_path} ({bytes_out} bytes) "
            f"+ {err_path} ({bytes_err} bytes)"
        )

    print(f"Done. wrote {download_record}")


if __name__ == "__main__":
    main()
