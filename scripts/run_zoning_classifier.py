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
from newsvlm.zoning_classifier import PageZoningResult, classify_pages_zoning  # noqa: E402


def collect_page_paths(spec: str) -> list[Path]:
    """Collect per-page .vlm.json paths from a glob or @file list."""
    spec = spec.strip()
    if spec.startswith("@"):
        list_path = Path(expanduser(spec[1:]))
        if not list_path.is_file():
            raise SystemExit(f"Pages file not found: {list_path}")
        paths: list[Path] = []
        for raw in list_path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            p = Path(expanduser(line))
            if not p.is_file():
                raise SystemExit(f"Page JSON listed but not found: {p}")
            paths.append(p)
        return sorted(paths)

    expanded = expanduser(spec)
    return sorted(Path(p) for p in glob(expanded))


def _page_id_from_path(p: Path) -> str:
    name = p.name
    if name.endswith(".vlm.json"):
        return name[: -len(".vlm.json")]
    return p.stem


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Classify pages for zoning-related content using an LLM.")
    ap.add_argument(
        "--pages",
        required=True,
        help='Glob for per-page *.vlm.json files (absolute OK) or "@file" listing those paths',
    )
    ap.add_argument("--output-dir", required=True, help="Directory to write per-page classifier outputs")
    ap.add_argument("--model", required=True, help="Classifier model, e.g. openai:gpt-5-nano")
    ap.add_argument("--gateway-url", default="http://127.0.0.1:8000", help="Gateway base URL")
    ap.add_argument(
        "--prompt-path",
        default="prompts/zoning_ocr_classifier_prompt_text.txt",
        help="Path to the zoning classifier prompt text file",
    )
    ap.add_argument("--max-concurrency", type=int, default=4, help="Pages to classify concurrently")
    ap.add_argument("--timeout", type=float, default=120.0, help="Gateway timeout seconds per page classification call")
    ap.add_argument("--max-retries", type=int, default=2, help="Retries per page on parse/transport error")
    ap.add_argument("--skip-existing", action="store_true", help="Skip pages with existing classifier output")
    ap.add_argument(
        "--manifest-path",
        default=None,
        help="Append per-page JSONL manifest here (default: <output-dir>/manifest.jsonl)",
    )
    ap.add_argument(
        "--allow-test-providers",
        action="store_true",
        help="Allow non-real providers (e.g., echo) for local debugging",
    )
    return ap.parse_args()


async def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = Path(expanduser(args.prompt_path))

    manifest_path = Path(args.manifest_path) if args.manifest_path else (out_dir / "manifest.jsonl")
    manifest_lock = asyncio.Lock()

    page_paths = collect_page_paths(args.pages)
    if not page_paths:
        raise SystemExit(f"No page JSONs matched pattern: {args.pages}")

    def out_path_for(page_path: Path) -> Path:
        page_id = _page_id_from_path(page_path)
        return out_dir / f"{page_id}.zoning.json"

    if args.skip_existing:
        page_paths = [p for p in page_paths if not out_path_for(p).exists()]
        if not page_paths:
            print("All pages already have zoning classifier outputs; nothing to do.")
            return

    async def on_page(res: PageZoningResult) -> None:
        out_path = out_dir / f"{res.page_id}.zoning.json"
        out_path.write_text(json.dumps(res.model_dump(mode="json"), indent=2, ensure_ascii=False))
        print(f"Wrote {out_path}")

        label = res.classification.label if res.classification else None
        severity_counts = Counter()
        line = {
            "page_id": res.page_id,
            "output_path": str(out_path),
            "source_model": res.source_model,
            "classifier_model": res.classifier_model,
            "label": label,
            "confidence": (res.classification.confidence if res.classification else None),
            "present": (res.classification.present.model_dump() if res.classification else None),
            "started_at": res.started_at,
            "finished_at": res.finished_at,
            "attempts": res.attempts,
            "duration_ms": res.duration_ms,
            "error": res.error,
            "input_stats": res.input_stats,
        }
        async with manifest_lock:
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with manifest_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")

    await classify_pages_zoning(
        page_paths=page_paths,
        model=args.model,
        gateway_url=args.gateway_url,
        prompt_path=prompt_path,
        max_concurrency=args.max_concurrency,
        timeout=args.timeout,
        max_retries=args.max_retries,
        allow_test_providers=args.allow_test_providers,
        on_page=on_page,
    )


if __name__ == "__main__":
    asyncio.run(main())

