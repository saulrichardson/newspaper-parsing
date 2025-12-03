#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from newsvlm.pipeline import load_layout, transcribe_pages  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Send Dell layout crops to a VLM via agent-gateway.")
    p.add_argument("--layouts", required=True, help="Glob for Dell layout JSONs")
    p.add_argument("--output-dir", required=True, help="Where to write VLM JSON outputs")
    p.add_argument("--model", required=True, help="Model name, e.g., openai:gpt-4o or echo:echo")
    p.add_argument("--gateway-url", default="http://127.0.0.1:8000", help="Gateway base URL")
    p.add_argument("--png-root", default=None, help="Optional directory to find PNGs by basename")
    p.add_argument("--max-concurrency", type=int, default=4, help="Parallel crop requests per task")
    p.add_argument("--timeout", type=float, default=60.0, help="Gateway timeout seconds per call")
    p.add_argument("--max-retries", type=int, default=2, help="Retries per bbox on parse/transport error")
    p.add_argument("--save-crops-dir", default=None, help="Optional directory to save bbox crops for inspection")
    return p.parse_args()


async def main() -> None:
    args = parse_args()
    layout_paths = sorted(Path().glob(args.layouts))
    if not layout_paths:
        raise SystemExit(f"No layout JSONs matched pattern: {args.layouts}")

    png_root = Path(args.png_root) if args.png_root else None
    items = [load_layout(p, png_root=png_root) for p in layout_paths]
    results = await transcribe_pages(
        items=items,
        model=args.model,
        gateway_url=args.gateway_url,
        max_concurrency=args.max_concurrency,
        timeout=args.timeout,
        max_retries=args.max_retries,
        crops_dir=Path(args.save_crops_dir) if args.save_crops_dir else None,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for page in results:
        name = Path(page.layout_path).stem + ".vlm.json"
        out_path = out_dir / name
        out_path.write_text(json.dumps(page.model_dump(mode="json"), indent=2))
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
