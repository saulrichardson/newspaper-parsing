#!/usr/bin/env python3
"""
Convert Gemini batch request shards into OpenAI Batch (/v1/responses) request shards.

Why:
- We already exported Gemini request JSONLs (with base64 inline images) on Greene/VAST.
- OpenAI Batch expects a different JSONL envelope.
- Re-cropping images on the HPC is expensive; conversion is purely JSON reshaping.

Input directory format (per shard NNN):
  - gemini_requests_shardNNN.jsonl
  - mapping_shardNNN.jsonl

Output directory format (split + upload-ready):
  - openai_requests_shardMMM.jsonl
  - mapping_shardMMM.jsonl
  - split_manifest.jsonl

Notes:
- Output shards are split to respect OpenAI file upload limits (typically ~200MB).
- `mapping_shardMMM.jsonl` stays aligned line-for-line with the corresponding OpenAI shard.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class InputShard:
    shard: int
    mapping: Path
    gemini: Path


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Convert Gemini batch shards to OpenAI batch shards.")
    ap.add_argument("--in-dir", required=True, help="Directory containing mapping_shardNNN.jsonl + gemini_requests_shardNNN.jsonl")
    ap.add_argument("--out-dir", required=True, help="Directory to write OpenAI shards into (must be empty or non-existent)")
    ap.add_argument("--openai-model", default="gpt-5.2", help="OpenAI model name for /v1/responses")
    ap.add_argument(
        "--openai-reasoning-effort",
        choices=["none", "low", "medium", "high", "xhigh"],
        default="medium",
        help="OpenAI Responses API reasoning.effort value to include in each request body",
    )
    ap.add_argument(
        "--lines-per-shard",
        type=int,
        default=0,
        help="Max lines per output shard (0 = no line limit; byte limit may still split)",
    )
    ap.add_argument(
        "--max-openai-bytes",
        type=int,
        default=180_000_000,
        help=(
            "Max bytes for each openai_requests_shardMMM.jsonl file "
            "(0 = no byte limit). Default targets <200MB."
        ),
    )
    ap.add_argument("--skip-shards", default="", help="Comma-separated input shard indices to skip (e.g. '38')")
    ap.add_argument("--dry-run", action="store_true", help="Discover and validate inputs, but do not write outputs")
    return ap.parse_args()


def _parse_int_set(spec: str) -> set[int]:
    spec = spec.strip()
    if not spec:
        return set()
    return {int(p.strip()) for p in spec.split(",") if p.strip()}


def _parse_shard_index(filename: str, *, prefix: str, suffix: str) -> int:
    if not (filename.startswith(prefix) and filename.endswith(suffix)):
        raise ValueError(f"Unexpected filename: {filename}")
    token = filename[len(prefix) : -len(suffix)]
    if not token.isdigit():
        raise ValueError(f"Unexpected shard token in filename: {filename}")
    return int(token)


def _discover_inputs(in_dir: Path, *, skip: set[int]) -> list[InputShard]:
    mapping: dict[int, Path] = {}
    gemini: dict[int, Path] = {}

    for p in in_dir.glob("mapping_shard*.jsonl"):
        s = _parse_shard_index(p.name, prefix="mapping_shard", suffix=".jsonl")
        mapping[s] = p
    for p in in_dir.glob("gemini_requests_shard*.jsonl"):
        s = _parse_shard_index(p.name, prefix="gemini_requests_shard", suffix=".jsonl")
        gemini[s] = p

    indices = sorted(set(mapping) | set(gemini))
    out: list[InputShard] = []
    for s in indices:
        if s in skip:
            continue
        if s not in mapping:
            raise SystemExit(f"Missing mapping_shard{s:03d}.jsonl in {in_dir}")
        if s not in gemini:
            raise SystemExit(f"Missing gemini_requests_shard{s:03d}.jsonl in {in_dir}")
        out.append(InputShard(shard=s, mapping=mapping[s], gemini=gemini[s]))
    if not out:
        raise SystemExit(f"No input shards discovered in {in_dir}")
    return out


def _open_out_files(out_dir: Path, out_shard: int):
    mapping_path = out_dir / f"mapping_shard{out_shard:03d}.jsonl"
    openai_path = out_dir / f"openai_requests_shard{out_shard:03d}.jsonl"
    mapping_f = mapping_path.open("wb")
    openai_f = openai_path.open("wb")
    return mapping_f, openai_f


def _extract_prompt_and_inline_data(gemini_line: dict[str, Any], *, ctx: str) -> tuple[str, str, str, str]:
    try:
        key = gemini_line["key"]
        contents = gemini_line["request"]["contents"]
        parts = contents[0]["parts"]
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"{ctx}: malformed Gemini line structure: {exc}") from exc

    if not isinstance(key, str) or not key:
        raise SystemExit(f"{ctx}: gemini key must be a non-empty string")

    texts = [p.get("text") for p in parts if isinstance(p, dict) and "text" in p]
    inline = [p.get("inline_data") for p in parts if isinstance(p, dict) and "inline_data" in p]
    if len(texts) != 1:
        raise SystemExit(f"{ctx}: expected exactly 1 text part, got {len(texts)}")
    if len(inline) != 1:
        raise SystemExit(f"{ctx}: expected exactly 1 inline_data part, got {len(inline)}")

    prompt = texts[0]
    inline_data = inline[0]
    if not isinstance(prompt, str) or not prompt:
        raise SystemExit(f"{ctx}: prompt must be a non-empty string")

    try:
        mime_type = inline_data["mime_type"]
        b64 = inline_data["data"]
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"{ctx}: inline_data missing mime_type/data: {exc}") from exc
    if not isinstance(mime_type, str) or not mime_type.startswith("image/"):
        raise SystemExit(f"{ctx}: inline_data mime_type must look like image/*, got {mime_type!r}")
    if not isinstance(b64, str) or not b64:
        raise SystemExit(f"{ctx}: inline_data data must be non-empty base64 string")

    data_url = f"data:{mime_type};base64,{b64}"
    return key, prompt, mime_type, data_url


def _gemini_to_openai_request(
    *,
    key: str,
    prompt: str,
    data_url: str,
    model: str,
    reasoning_effort: str | None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": data_url},
                ],
            }
        ],
    }
    if reasoning_effort is not None:
        body["reasoning"] = {"effort": reasoning_effort}
    return {
        "custom_id": key,
        "method": "POST",
        "url": "/v1/responses",
        "body": body,
    }


def main() -> None:
    args = _parse_args()
    in_dir = Path(args.in_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    skip = _parse_int_set(args.skip_shards)

    if not in_dir.is_dir():
        raise SystemExit(f"--in-dir is not a directory: {in_dir}")
    if args.lines_per_shard < 0:
        raise SystemExit("--lines-per-shard must be >= 0")
    if args.max_openai_bytes < 0:
        raise SystemExit("--max-openai-bytes must be >= 0")

    max_lines = args.lines_per_shard if args.lines_per_shard > 0 else None
    max_openai_bytes = args.max_openai_bytes if args.max_openai_bytes > 0 else None
    if not max_lines and not max_openai_bytes:
        raise SystemExit("At least one of --lines-per-shard or --max-openai-bytes must be set (non-zero)")

    inputs = _discover_inputs(in_dir, skip=skip)

    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"--out-dir must be empty or non-existent: {out_dir}")
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "split_manifest.jsonl"
    out_shard = 0

    if args.dry_run:
        limits: list[str] = []
        if max_lines:
            limits.append(f"{max_lines} lines")
        if max_openai_bytes:
            limits.append(f"{max_openai_bytes} bytes")
        limits_str = " and ".join(limits)
        print(f"Would convert+split {len(inputs)} input shards from {in_dir} into OpenAI shards limited by {limits_str}")
        return

    reasoning_effort: str | None = args.openai_reasoning_effort
    if reasoning_effort is not None and reasoning_effort.strip() == "":
        reasoning_effort = None

    with manifest_path.open("w", encoding="utf-8") as manifest_f:
        for inp in inputs:
            with inp.mapping.open("rb") as mapping_f, inp.gemini.open("rb") as gemini_f:
                lines_in_current = 0
                openai_bytes_in_current = 0
                mapping_out = None
                openai_out = None
                start_line = 1
                line_idx = 0

                def close_and_record(*, end_line: int) -> None:
                    nonlocal out_shard, mapping_out, openai_out, lines_in_current, openai_bytes_in_current
                    if mapping_out is None or openai_out is None:
                        return
                    mapping_out.close()
                    openai_out.close()
                    rec = {
                        "out_shard": out_shard,
                        "in_shard": inp.shard,
                        "in_mapping": str(inp.mapping),
                        "in_gemini": str(inp.gemini),
                        "start_line": start_line,
                        "end_line": end_line,
                        "lines": lines_in_current,
                        "openai_bytes": openai_bytes_in_current,
                    }
                    manifest_f.write(json.dumps(rec) + "\n")
                    out_shard += 1
                    mapping_out = None
                    openai_out = None
                    lines_in_current = 0
                    openai_bytes_in_current = 0

                for mapping_line, gemini_line in zip_longest(mapping_f, gemini_f):
                    line_idx += 1
                    if mapping_line is None or gemini_line is None:
                        raise SystemExit(
                            f"Input shard {inp.shard:03d} line count mismatch between "
                            f"{inp.mapping.name} and {inp.gemini.name}"
                        )

                    ctx = f"{inp.gemini.name} line {line_idx}"
                    try:
                        gemini_obj = json.loads(gemini_line)
                    except Exception as exc:  # noqa: BLE001
                        raise SystemExit(f"{ctx}: invalid JSON: {exc}") from exc

                    key, prompt, _mime, data_url = _extract_prompt_and_inline_data(gemini_obj, ctx=ctx)
                    openai_obj = _gemini_to_openai_request(
                        key=key,
                        prompt=prompt,
                        data_url=data_url,
                        model=args.openai_model,
                        reasoning_effort=reasoning_effort,
                    )
                    openai_line_bytes = (
                        json.dumps(openai_obj, ensure_ascii=False, separators=(",", ":")) + "\n"
                    ).encode("utf-8")

                    if max_openai_bytes and len(openai_line_bytes) > max_openai_bytes:
                        raise SystemExit(
                            f"{ctx}: single OpenAI request line exceeds max bytes "
                            f"({len(openai_line_bytes)} > {max_openai_bytes})"
                        )

                    should_roll = False
                    if max_lines and lines_in_current >= max_lines:
                        should_roll = True
                    if (
                        max_openai_bytes
                        and openai_bytes_in_current > 0
                        and openai_bytes_in_current + len(openai_line_bytes) > max_openai_bytes
                    ):
                        should_roll = True

                    if should_roll:
                        close_and_record(end_line=line_idx - 1)

                    if mapping_out is None or openai_out is None:
                        mapping_out, openai_out = _open_out_files(out_dir, out_shard)
                        start_line = line_idx

                    mapping_out.write(mapping_line)
                    openai_out.write(openai_line_bytes)
                    lines_in_current += 1
                    openai_bytes_in_current += len(openai_line_bytes)

                if mapping_out is not None and openai_out is not None:
                    close_and_record(end_line=line_idx)

    print(f"Wrote {out_shard} OpenAI shards to {out_dir}")
    print(f"Wrote manifest {manifest_path}")


if __name__ == "__main__":
    main()

