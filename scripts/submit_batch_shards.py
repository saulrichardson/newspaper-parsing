#!/usr/bin/env python3
"""
Submit OpenAI + Gemini Batch jobs for shard JSONL files.

This is intended to work with the outputs of `scripts/export_batch_requests.py`:
  - openai_requests_shardNNN.jsonl (OpenAI Batch format for POST /v1/responses)
  - gemini_requests_shardNNN.jsonl (Google GenAI Batch format: {"key":..., "request":{...}})
  - mapping_shardNNN.jsonl (provenance; used for validation + later rehydration)

This script:
  - discovers shard indices in a request directory
  - validates mapping/request line counts match (to avoid "interrupted export" shards)
  - submits batches (OpenAI and/or Gemini)
  - appends a JSONL record for each successful submission so it is resumable

Environment:
  - OpenAI key: OPENAI_API_KEY or OPENAI_KEY
  - Gemini key: GEMINI_KEY
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal


Provider = Literal["openai", "gemini"]


@dataclass(frozen=True)
class ShardPaths:
    shard: int
    mapping: Path
    openai: Path | None
    gemini: Path | None


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Submit OpenAI + Gemini Batch jobs for shard JSONLs.")
    ap.add_argument(
        "--request-dir",
        required=True,
        help="Directory containing *_requests_shardNNN.jsonl and mapping_shardNNN.jsonl",
    )
    ap.add_argument(
        "--providers",
        choices=["openai", "gemini", "both"],
        default="both",
        help="Which providers to submit",
    )
    ap.add_argument(
        "--shards",
        default=None,
        help="Optional shard selector: e.g. '0-37' or '0,1,2,10'. Default: all discovered.",
    )
    ap.add_argument(
        "--skip-shards",
        default="",
        help="Comma-separated shard indices to skip (e.g. '38,11').",
    )
    ap.add_argument(
        "--gemini-model",
        default="models/gemini-2.5-flash",
        help="Gemini model for batch create",
    )
    ap.add_argument(
        "--openai-endpoint",
        default="/v1/responses",
        choices=["/v1/responses", "/v1/chat/completions", "/v1/embeddings", "/v1/completions", "/v1/moderations"],
        help="OpenAI batch endpoint",
    )
    ap.add_argument(
        "--openai-max-bytes",
        type=int,
        default=209_715_200,
        help=(
            "Fail fast if an OpenAI batch input JSONL exceeds this many bytes "
            "(0 disables). Default matches the common 200MB limit."
        ),
    )
    ap.add_argument(
        "--openai-completion-window",
        default="24h",
        choices=["24h"],
        help="OpenAI Batch completion window",
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
    ap.add_argument(
        "--display-name-prefix",
        default="newsvlm",
        help="Provider display name prefix (used for Gemini display_name and OpenAI metadata)",
    )
    ap.add_argument(
        "--record",
        default=None,
        help="JSONL file to append submission records to (default: <request-dir>/submitted_batches.jsonl)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be submitted, but do not call any APIs.",
    )
    return ap.parse_args()


def _line_count(path: Path) -> int:
    # `wc -l` is significantly faster than Python line iteration for multi-GB files.
    out = subprocess.check_output(["wc", "-l", str(path)], text=True).strip()
    # output format: "<count> <path>"
    count_str = out.split()[0]
    return int(count_str)


def _discover_shards(request_dir: Path, *, providers: Iterable[Provider]) -> dict[int, ShardPaths]:
    want_openai = "openai" in set(providers)
    want_gemini = "gemini" in set(providers)

    mapping: dict[int, Path] = {}
    openai: dict[int, Path] = {}
    gemini: dict[int, Path] = {}

    for p in request_dir.glob("mapping_shard*.jsonl"):
        s = _parse_shard_index(p.name, prefix="mapping_shard", suffix=".jsonl")
        mapping[s] = p
    for p in request_dir.glob("openai_requests_shard*.jsonl"):
        s = _parse_shard_index(p.name, prefix="openai_requests_shard", suffix=".jsonl")
        openai[s] = p
    for p in request_dir.glob("gemini_requests_shard*.jsonl"):
        s = _parse_shard_index(p.name, prefix="gemini_requests_shard", suffix=".jsonl")
        gemini[s] = p

    shards: dict[int, ShardPaths] = {}
    all_indices = sorted(set(mapping) | set(openai) | set(gemini))
    for s in all_indices:
        if s not in mapping:
            raise SystemExit(f"Shard {s:03d} is missing mapping file")
        if want_openai and s not in openai:
            raise SystemExit(f"Shard {s:03d} is missing OpenAI requests file")
        if want_gemini and s not in gemini:
            raise SystemExit(f"Shard {s:03d} is missing Gemini requests file")

        if (not want_openai) and (not want_gemini):
            raise SystemExit("No providers selected")

        if want_openai and want_gemini and (s not in openai or s not in gemini):
            missing: list[str] = []
            if s not in openai:
                missing.append("openai")
            if s not in gemini:
                missing.append("gemini")
            raise SystemExit(f"Shard {s:03d} is missing files: {', '.join(missing)}")

        shards[s] = ShardPaths(
            shard=s,
            mapping=mapping[s],
            openai=openai.get(s),
            gemini=gemini.get(s),
        )

    if not shards:
        raise SystemExit(f"No shard files found in {request_dir}")
    return shards


def _parse_shard_index(filename: str, *, prefix: str, suffix: str) -> int:
    if not (filename.startswith(prefix) and filename.endswith(suffix)):
        raise ValueError(f"Unexpected filename: {filename}")
    mid = filename[len(prefix) : -len(suffix)]
    if not mid.isdigit():
        raise ValueError(f"Unexpected shard token in filename: {filename}")
    return int(mid)


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


def _load_existing_records(record_path: Path) -> set[tuple[Provider, int]]:
    if not record_path.exists():
        return set()
    seen: set[tuple[Provider, int]] = set()
    with record_path.open() as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            provider = obj.get("provider")
            shard = obj.get("shard")
            if provider in {"openai", "gemini"} and isinstance(shard, int):
                seen.add((provider, shard))
    return seen


def _append_record(record_path: Path, obj: dict[str, Any]) -> None:
    record_path.parent.mkdir(parents=True, exist_ok=True)
    with record_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"Missing required environment variable: {name}")
    return value


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
    # Prefer CWD .env, but allow running from elsewhere.
    candidates = [
        Path(".env"),
        Path(__file__).resolve().parents[1] / ".env",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def _get_openai_key_candidates(*, mode: str) -> list[tuple[str, str]]:
    """Return OpenAI keys in preference order.

    Priority is by *variable name* (project key first), and we fall back to .env
    in repo root if env vars aren't set.
    """

    env_path = _find_default_env_file()
    file_env = _load_env_file(env_path) if env_path else {}

    out: list[tuple[str, str]] = []
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
        if not val:
            continue
        out.append((name, val))

    # Deduplicate by value while preserving order.
    seen: set[str] = set()
    uniq: list[tuple[str, str]] = []
    for src, key in out:
        if key in seen:
            continue
        seen.add(key)
        uniq.append((src, key))
    return uniq


def _is_insufficient_quota_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return ("insufficient_quota" in msg) or ("exceeded your current quota" in msg)


def _validate_counts(sp: ShardPaths, *, providers: Iterable[Provider]) -> dict[str, int]:
    counts: dict[str, int] = {}
    counts["mapping"] = _line_count(sp.mapping)
    if "openai" in providers:
        if not sp.openai:
            raise SystemExit(f"Shard {sp.shard:03d} missing OpenAI requests file")
        counts["openai"] = _line_count(sp.openai)
        if counts["openai"] != counts["mapping"]:
            raise SystemExit(
                f"Shard {sp.shard:03d} mismatch: mapping={counts['mapping']} openai={counts['openai']} "
                f"({sp.mapping.name} vs {sp.openai.name})"
            )
    if "gemini" in providers:
        if not sp.gemini:
            raise SystemExit(f"Shard {sp.shard:03d} missing Gemini requests file")
        counts["gemini"] = _line_count(sp.gemini)
        if counts["gemini"] != counts["mapping"]:
            raise SystemExit(
                f"Shard {sp.shard:03d} mismatch: mapping={counts['mapping']} gemini={counts['gemini']} "
                f"({sp.mapping.name} vs {sp.gemini.name})"
            )
    return counts


def _submit_openai(
    *,
    jsonl_path: Path,
    endpoint: str,
    completion_window: str,
    display_name: str,
    line_count: int,
    openai_key_mode: str,
) -> dict[str, Any]:
    from openai import OpenAI

    keys = _get_openai_key_candidates(mode=openai_key_mode)
    if not keys:
        raise SystemExit(
            "Missing OpenAI key. Set PROJECT_OPENAI_KEY (preferred) or OPENAI_API_KEY/OPENAI_KEY in env, "
            "or add them to .env in repo root."
        )

    last_exc: Exception | None = None
    for idx, (src, key) in enumerate(keys, start=1):
        client = OpenAI(api_key=key)
        try:
            with jsonl_path.open("rb") as f:
                file_obj = client.files.create(file=f, purpose="batch")

            batch = client.batches.create(
                input_file_id=file_obj.id,
                endpoint=endpoint,
                completion_window=completion_window,  # type: ignore[arg-type]
                metadata={
                    "display_name": display_name,
                    "requests_path": str(jsonl_path),
                    "line_count": str(line_count),
                },
            )
            return {
                "file_id": file_obj.id,
                "batch_id": batch.id,
                "batch_status": batch.status,
                "auth_source": src,
            }
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if _is_insufficient_quota_error(exc) and idx < len(keys):
                _eprint(f"OpenAI key ({src}) appears out of quota; retrying with fallback key…")
                continue
            raise

    raise SystemExit(str(last_exc) if last_exc else "OpenAI submission failed for unknown reasons.")


def _submit_gemini(*, jsonl_path: Path, model: str, display_name: str) -> dict[str, Any]:
    from google import genai
    from google.genai.types import UploadFileConfig

    client = genai.Client(api_key=_require_env("GEMINI_KEY"))

    req_file = client.files.upload(
        file=jsonl_path,
        config=UploadFileConfig(mime_type="application/jsonl"),
    )
    batch = client.batches.create(
        model=model,
        src={"file_name": req_file.name},
        config={"display_name": display_name},
    )
    # The google-genai client uses "batches/..." name + state.
    return {
        "uploaded_file_name": req_file.name,
        "batch_name": batch.name,
        "batch_state": getattr(batch, "state", None),
    }


def main() -> None:
    args = _parse_args()
    request_dir = Path(args.request_dir).expanduser()
    if not request_dir.is_dir():
        raise SystemExit(f"--request-dir is not a directory: {request_dir}")

    record_path = Path(args.record).expanduser() if args.record else (request_dir / "submitted_batches.jsonl")

    want_providers: list[Provider]
    if args.providers == "both":
        want_providers = ["openai", "gemini"]
    else:
        want_providers = [args.providers]  # type: ignore[list-item]

    if args.openai_max_bytes < 0:
        raise SystemExit("--openai-max-bytes must be >= 0")

    skip = _parse_shard_selector(args.skip_shards)
    shards = _discover_shards(request_dir, providers=want_providers)
    if args.shards:
        selected = _parse_shard_selector(args.shards)
        shards = {s: sp for s, sp in shards.items() if s in selected}
    shards = {s: sp for s, sp in shards.items() if s not in skip}

    existing = _load_existing_records(record_path)

    submitted = {"openai": 0, "gemini": 0}
    for shard_idx in sorted(shards):
        sp = shards[shard_idx]
        display = f"{args.display_name_prefix}-shard{sp.shard:03d}"

        counts = _validate_counts(sp, providers=want_providers)
        mapping_count = counts["mapping"]

        for provider in want_providers:
            if (provider, sp.shard) in existing:
                print(f"skip {provider} shard{sp.shard:03d} (already recorded in {record_path})")
                continue

            if args.dry_run:
                print(f"DRY-RUN submit {provider} shard{sp.shard:03d} ({mapping_count} requests)")
                continue

            t0 = time.time()
            if provider == "openai":
                if args.openai_max_bytes and sp.openai and sp.openai.stat().st_size > args.openai_max_bytes:
                    raise SystemExit(
                        f"Refusing to submit OpenAI shard {sp.shard:03d}: "
                        f"{sp.openai.name} is {sp.openai.stat().st_size} bytes "
                        f"(limit {args.openai_max_bytes}). Split the shard smaller."
                    )
                result = _submit_openai(
                    jsonl_path=sp.openai,
                    endpoint=args.openai_endpoint,
                    completion_window=args.openai_completion_window,
                    display_name=display,
                    line_count=mapping_count,
                    openai_key_mode=str(args.openai_key_mode),
                )
            elif provider == "gemini":
                result = _submit_gemini(
                    jsonl_path=sp.gemini,
                    model=args.gemini_model,
                    display_name=display,
                )
            else:
                raise AssertionError(f"Unhandled provider: {provider}")

            elapsed = round(time.time() - t0, 2)
            rec: dict[str, Any] = {
                "provider": provider,
                "shard": sp.shard,
                "display_name": display,
                "request_dir": str(request_dir),
                "mapping_path": str(sp.mapping),
                "mapping_lines": mapping_count,
                "created_at": int(time.time()),
                "elapsed_seconds": elapsed,
                **result,
            }
            if provider == "openai":
                rec["requests_path"] = str(sp.openai)
                rec["endpoint"] = args.openai_endpoint
                rec["completion_window"] = args.openai_completion_window
            if provider == "gemini":
                rec["requests_path"] = str(sp.gemini)
                rec["model"] = args.gemini_model

            _append_record(record_path, rec)
            submitted[provider] += 1
            print(f"submitted {provider} shard{sp.shard:03d} -> recorded in {record_path}")

    print(
        "Done. "
        + " ".join([f"{p}={submitted[p]}" for p in sorted(submitted)])
        + f" record={record_path}"
    )


if __name__ == "__main__":
    main()
