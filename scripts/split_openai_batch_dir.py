#!/usr/bin/env python3
"""
Split oversized OpenAI Batch JSONL shards into smaller shards.

Why:
OpenAI Files API rejects multi-GB JSONL uploads (HTTP 413). If the exported
`openai_requests_shardNNN.jsonl` files are too large, split them into smaller
files while keeping `mapping_shardNNN.jsonl` aligned line-for-line.

Additionally, OpenAI Batch enforces a per-model *input file size* limit
(often ~200MB). This script can split by a target byte size to keep each
OpenAI JSONL upload under that limit.

Input directory must contain, per shard:
  - mapping_shardNNN.jsonl
  - openai_requests_shardNNN.jsonl

Output directory will contain:
  - mapping_shardMMM.jsonl
  - openai_requests_shardMMM.jsonl
  - split_manifest.jsonl (one line per output shard)

Notes:
- This script does not touch Gemini files.
- Output shard numbering is sequential and independent of input numbering.
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
    openai: Path


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Split OpenAI batch JSONL shards into smaller files.")
    ap.add_argument("--in-dir", required=True, help="Directory containing mapping_shardNNN.jsonl + openai_requests_shardNNN.jsonl")
    ap.add_argument("--out-dir", required=True, help="Directory to write split shards into")
    ap.add_argument(
        "--lines-per-shard",
        type=int,
        default=0,
        help="Max lines per output shard (0 = no line limit; byte limit may still split)",
    )
    ap.add_argument(
        "--max-openai-bytes",
        type=int,
        default=195_000_000,
        help=(
            "Max bytes for each openai_requests_shardMMM.jsonl file "
            "(0 = no byte limit). Default targets <200MB."
        ),
    )
    ap.add_argument("--skip-shards", default="", help="Comma-separated input shard indices to skip (e.g. '38')")
    ap.add_argument("--dry-run", action="store_true", help="Discover and validate inputs, but do not write output files")
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
    openai: dict[int, Path] = {}

    for p in in_dir.glob("mapping_shard*.jsonl"):
        s = _parse_shard_index(p.name, prefix="mapping_shard", suffix=".jsonl")
        mapping[s] = p
    for p in in_dir.glob("openai_requests_shard*.jsonl"):
        s = _parse_shard_index(p.name, prefix="openai_requests_shard", suffix=".jsonl")
        openai[s] = p

    indices = sorted(set(mapping) | set(openai))
    out: list[InputShard] = []
    for s in indices:
        if s in skip:
            continue
        if s not in mapping:
            raise SystemExit(f"Missing mapping_shard{s:03d}.jsonl in {in_dir}")
        if s not in openai:
            raise SystemExit(f"Missing openai_requests_shard{s:03d}.jsonl in {in_dir}")
        out.append(InputShard(shard=s, mapping=mapping[s], openai=openai[s]))
    if not out:
        raise SystemExit(f"No input shards discovered in {in_dir}")
    return out


def _open_out_files(out_dir: Path, out_shard: int) -> tuple[Any, Any]:
    mapping_path = out_dir / f"mapping_shard{out_shard:03d}.jsonl"
    openai_path = out_dir / f"openai_requests_shard{out_shard:03d}.jsonl"
    mapping_f = mapping_path.open("wb")
    openai_f = openai_path.open("wb")
    return mapping_f, openai_f


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
        print(f"Would split {len(inputs)} input shards from {in_dir} into shards limited by {limits_str}")
        return

    with manifest_path.open("w", encoding="utf-8") as manifest_f:
        for inp in inputs:
            with inp.mapping.open("rb") as mapping_f, inp.openai.open("rb") as openai_f:
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
                        "in_openai": str(inp.openai),
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

                for mapping_line, openai_line in zip_longest(mapping_f, openai_f):
                    line_idx += 1
                    if mapping_line is None or openai_line is None:
                        raise SystemExit(
                            f"Input shard {inp.shard:03d} line count mismatch between "
                            f"{inp.mapping.name} and {inp.openai.name}"
                        )

                    if max_openai_bytes and len(openai_line) > max_openai_bytes:
                        raise SystemExit(
                            f"Single OpenAI request line exceeds max bytes ({len(openai_line)} > {max_openai_bytes}) "
                            f"in shard {inp.shard:03d} at input line {line_idx}"
                        )

                    # Roll over if adding this line would exceed limits (but don't create empty shards).
                    should_roll = False
                    if max_lines and lines_in_current >= max_lines:
                        should_roll = True
                    if max_openai_bytes and openai_bytes_in_current > 0 and openai_bytes_in_current + len(openai_line) > max_openai_bytes:
                        should_roll = True

                    if should_roll:
                        close_and_record(end_line=line_idx - 1)

                    if mapping_out is None or openai_out is None:
                        mapping_out, openai_out = _open_out_files(out_dir, out_shard)
                        start_line = line_idx

                    mapping_out.write(mapping_line)
                    openai_out.write(openai_line)
                    lines_in_current += 1
                    openai_bytes_in_current += len(openai_line)

                # Close any partial shard.
                if mapping_out is not None and openai_out is not None:
                    close_and_record(end_line=line_idx)

    print(f"Wrote {out_shard} split shards to {out_dir}")
    print(f"Wrote manifest {manifest_path}")


if __name__ == "__main__":
    main()
