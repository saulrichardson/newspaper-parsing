#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from glob import glob
from os.path import expanduser
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from newsvlm.pipeline import load_layout, transcribe_pages  # noqa: E402


def collect_layout_paths(spec: str) -> list[Path]:
    """Collect layout JSON paths from a glob or @file list.

    - If spec starts with "@", the remainder is treated as a text file listing
      one JSON path per line (absolute or relative).
    - Otherwise spec is treated as a glob; absolute globs are supported.
    """
    spec = spec.strip()
    if spec.startswith("@"):
        list_path = Path(expanduser(spec[1:]))
        if not list_path.is_file():
            raise SystemExit(f"Layouts file not found: {list_path}")
        paths: list[Path] = []
        for raw in list_path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            p = Path(expanduser(line))
            if not p.is_file():
                raise SystemExit(f"Layout JSON listed but not found: {p}")
            paths.append(p)
        return sorted(paths)

    expanded = expanduser(spec)
    return sorted(Path(p) for p in glob(expanded))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Send Dell layout crops to a VLM via agent-gateway.")
    p.add_argument(
        "--layouts",
        required=True,
        help='Glob for Dell layout JSONs (absolute OK) or "@file" listing JSON paths',
    )
    p.add_argument("--output-dir", required=True, help="Where to write VLM JSON outputs")
    p.add_argument("--model", required=True, help="Model name, e.g., gemini:gemini-2.5-flash")
    p.add_argument("--gateway-url", default="http://127.0.0.1:8000", help="Gateway base URL")
    p.add_argument("--png-root", default=None, help="Optional directory to find PNGs by basename")
    p.add_argument("--max-concurrency", type=int, default=4, help="Parallel crop requests per task")
    p.add_argument("--page-concurrency", type=int, default=1, help="Pages to process concurrently")
    p.add_argument("--timeout", type=float, default=60.0, help="Gateway timeout seconds per call")
    p.add_argument("--max-retries", type=int, default=2, help="Retries per bbox on parse/transport error")
    p.add_argument("--save-crops-dir", default=None, help="Optional directory to save bbox crops for inspection")
    p.add_argument(
        "--max-crop-megapixels",
        type=float,
        default=3.0,
        help="Downscale crops larger than this many megapixels (0 to disable)",
    )
    p.add_argument(
        "--max-crop-dim",
        type=int,
        default=2048,
        help="Downscale crops whose longest edge exceeds this (0 to disable)",
    )
    p.add_argument("--skip-existing", action="store_true", help="Skip layouts with existing .vlm.json outputs")
    p.add_argument(
        "--manifest-path",
        default=None,
        help="Append per-page JSONL manifest here (default: <output-dir>/manifest.jsonl)",
    )
    p.add_argument(
        "--allow-test-providers",
        action="store_true",
        help="Allow non-real providers (e.g., echo) for local debugging",
    )
    p.add_argument(
        "--skip-bad-layouts",
        action="store_true",
        help="Skip layouts that fail to load (missing PNG, no boxes) instead of aborting",
    )
    return p.parse_args()


async def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest_path) if args.manifest_path else (out_dir / "manifest.jsonl")
    manifest_lock = asyncio.Lock()

    def _out_path_for(layout_path: Path) -> Path:
        return out_dir / f"{layout_path.stem}.vlm.json"

    layout_paths = collect_layout_paths(args.layouts)
    if not layout_paths:
        raise SystemExit(f"No layout JSONs matched pattern: {args.layouts}")

    if args.skip_existing:
        layout_paths = [p for p in layout_paths if not _out_path_for(p).exists()]
        if not layout_paths:
            print("All layouts already have outputs; nothing to do.")
            return

    png_root = Path(args.png_root) if args.png_root else None

    bad_layouts: list[dict] = []

    def iter_items():
        for p in layout_paths:
            try:
                yield load_layout(p, png_root=png_root)
            except Exception as exc:
                if not args.skip_bad_layouts:
                    raise
                bad_layouts.append({"layout_path": str(p), "error": str(exc)})

    items = iter_items()

    async def on_page(page) -> None:
        layout_path = Path(page.layout_path)
        out_path = _out_path_for(layout_path)
        out_path.write_text(json.dumps(page.model_dump(mode="json"), indent=2))
        print(f"Wrote {out_path}")

        counts = Counter(b.status for b in page.boxes)
        line = {
            "page_id": page.page_id,
            "png_path": page.png_path,
            "layout_path": page.layout_path,
            "output_path": str(out_path),
            "model": page.model,
            "started_at": page.started_at,
            "finished_at": page.finished_at,
            "status_counts": dict(counts),
        }
        async with manifest_lock:
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with manifest_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(line) + "\n")

    await transcribe_pages(
        items=items,
        model=args.model,
        gateway_url=args.gateway_url,
        max_concurrency=args.max_concurrency,
        page_concurrency=args.page_concurrency,
        timeout=args.timeout,
        max_retries=args.max_retries,
        crops_dir=Path(args.save_crops_dir) if args.save_crops_dir else None,
        max_crop_megapixels=args.max_crop_megapixels,
        max_crop_dim=args.max_crop_dim,
        allow_test_providers=args.allow_test_providers,
        on_page=on_page,
    )

    if bad_layouts:
        bad_path = out_dir / "bad_layouts.jsonl"
        with bad_path.open("a", encoding="utf-8") as f:
            for entry in bad_layouts:
                f.write(json.dumps(entry) + "\n")
        print(f"Skipped {len(bad_layouts)} bad layouts; details in {bad_path}")


if __name__ == "__main__":
    asyncio.run(main())
