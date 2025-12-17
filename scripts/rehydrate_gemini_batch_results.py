#!/usr/bin/env python3
"""
Rehydrate Gemini Batch outputs into per-page *.vlm.json files.

Inputs (from export + download steps):
  - mapping_shardNNN.jsonl (from scripts/export_batch_requests.py)
  - gemini_results_shardNNN.jsonl (downloaded Gemini batch output JSONL)

Each output line contains a `key` like "<page_id>:<box_id>" and a raw `response`.
We parse the model text (which we asked to be JSON) and produce per-page outputs
compatible with the sync pipeline's PageResult schema.

Typical usage:
  python scripts/rehydrate_gemini_batch_results.py \
    --request-dir newspaper-parsing-local/data/batch_requests_remaining \
    --results-dir newspaper-parsing-local/data/batch_results_gemini_remaining \
    --output-dir newspaper-parsing-local/data/vlm_out_batch_gemini_remaining
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from newsvlm.models import BoxResult, PageResult  # noqa: E402
from newsvlm.pipeline import _coerce_json  # noqa: E402


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Rehydrate Gemini batch JSONL outputs into per-page .vlm.json files.")
    ap.add_argument("--request-dir", required=True, help="Directory containing mapping_shardNNN.jsonl files")
    ap.add_argument("--results-dir", required=True, help="Directory containing gemini_results_shardNNN.jsonl files")
    ap.add_argument("--output-dir", required=True, help="Directory to write per-page *.vlm.json outputs into")
    ap.add_argument(
        "--shards",
        default=None,
        help="Optional shard selector (e.g. '0-16' or '0,1,2'). Default: all shards found in results-dir.",
    )
    ap.add_argument("--skip-existing", action="store_true", help="Skip writing pages that already exist")
    ap.add_argument(
        "--skip-bad-boxes",
        action="store_true",
        help="Record per-box errors instead of aborting on parse/lookup failures",
    )
    ap.add_argument(
        "--manifest-path",
        default=None,
        help="Append per-page JSONL manifest here (default: <output-dir>/manifest.jsonl)",
    )
    return ap.parse_args()


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


def _discover_shards(results_dir: Path) -> list[int]:
    shards: set[int] = set()
    for p in results_dir.glob("gemini_results_shard*.jsonl"):
        token = p.name.removeprefix("gemini_results_shard").removesuffix(".jsonl")
        if token.isdigit():
            shards.add(int(token))
    return sorted(shards)


def _extract_model_text(result_obj: dict[str, Any]) -> str:
    resp = result_obj.get("response") or {}
    try:
        return resp["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        return ""


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _load_existing_page(path: Path) -> PageResult | None:
    if not path.exists():
        return None
    try:
        return PageResult.model_validate(json.loads(path.read_text()))
    except Exception:
        return None


def main() -> None:
    args = _parse_args()
    request_dir = Path(args.request_dir).expanduser()
    results_dir = Path(args.results_dir).expanduser()
    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.manifest_path).expanduser() if args.manifest_path else (out_dir / "manifest.jsonl")

    if not request_dir.is_dir():
        raise SystemExit(f"--request-dir is not a directory: {request_dir}")
    if not results_dir.is_dir():
        raise SystemExit(f"--results-dir is not a directory: {results_dir}")

    shards = _discover_shards(results_dir)
    if args.shards:
        want = _parse_shard_selector(args.shards)
        shards = [s for s in shards if s in want]
    if not shards:
        raise SystemExit(f"No gemini_results_shard*.jsonl files found in {results_dir}")

    pages_written = 0
    boxes_total = 0
    boxes_error = 0

    # Accumulate per-page boxes while iterating shards; pages can span shard boundaries.
    pages: dict[str, dict[str, Any]] = {}

    for shard in shards:
        mapping_path = request_dir / f"mapping_shard{shard:03d}.jsonl"
        results_path = results_dir / f"gemini_results_shard{shard:03d}.jsonl"
        if not mapping_path.is_file():
            raise SystemExit(f"Missing mapping file for shard {shard:03d}: {mapping_path}")
        if not results_path.is_file():
            raise SystemExit(f"Missing results file for shard {shard:03d}: {results_path}")

        # Load results into a dict keyed by "<page_id>:<box_id>".
        results_by_key: dict[str, dict[str, Any]] = {}
        with results_path.open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                obj = json.loads(line)
                key = obj.get("key")
                if isinstance(key, str):
                    results_by_key[key] = obj

        with mapping_path.open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                m = json.loads(line)
                key = m.get("id")
                page_id = m.get("page_id")
                if not isinstance(key, str) or not isinstance(page_id, str):
                    raise SystemExit(f"Bad mapping line in {mapping_path}: missing id/page_id")

                boxes_total += 1
                rec = pages.get(page_id)
                if rec is None:
                    rec = pages[page_id] = {
                        "page_id": page_id,
                        "png_path": m.get("png_path"),
                        "layout_path": m.get("layout_path"),
                        "model": "gemini:gemini-2.5-flash",
                        "prompt": "class-aware (batch)",
                        "boxes": [],
                    }

                r = results_by_key.get(key)
                if r is None:
                    boxes_error += 1
                    if not args.skip_bad_boxes:
                        raise SystemExit(f"Missing result for key {key} in {results_path}")
                    status = "error"
                    transcript = None
                    err = {"message": "missing result line for key"}
                else:
                    model_text = _extract_model_text(r)
                    if not model_text:
                        boxes_error += 1
                        if not args.skip_bad_boxes:
                            raise SystemExit(f"Empty model response text for key {key} in {results_path}")
                        status = "error"
                        transcript = None
                        err = {"message": "empty model response text"}
                    else:
                        try:
                            parsed = _coerce_json(model_text)
                            status = parsed.get("status")
                            transcript = parsed.get("transcript")
                            if status == "ok" and (transcript is None or not str(transcript).strip()):
                                status = "unreadable"
                                transcript = ""
                            if status not in {"ok", "unreadable"}:
                                raise ValueError("status must be ok or unreadable")
                            err = None
                        except Exception as exc:
                            boxes_error += 1
                            if not args.skip_bad_boxes:
                                raise
                            status = "error"
                            transcript = None
                            err = {"message": str(exc)}

                box_res = BoxResult.model_validate(
                    {
                        "id": int(m.get("box_id")),
                        "class": m.get("class"),
                        "bbox": m.get("bbox"),
                        "legibility": m.get("legibility"),
                        "status": status,
                        "transcript": transcript,
                        "model": rec["model"],
                        "prompt": m.get("prompt") or "",
                        "attempts": 1,
                        "duration_ms": 0.0,
                        "orig_size": m.get("orig_size"),
                        "sent_size": m.get("sent_size"),
                        "scale_factor": m.get("scale_factor"),
                        "bytes_sent": m.get("bytes_sent"),
                        "error": err,
                    }
                )
                rec["boxes"].append(box_res)

    # Write all pages once, merging if already exists.
    now = datetime.now(timezone.utc).isoformat()
    for page_id, rec in sorted(pages.items()):
        out_path = out_dir / f"{page_id}.vlm.json"
        if args.skip_existing and out_path.exists():
            continue

        existing = _load_existing_page(out_path)
        if existing is not None:
            # Merge by box id; later wins (shouldn't conflict).
            by_id: dict[int, BoxResult] = {b.id: b for b in existing.boxes}
            for b in rec["boxes"]:
                by_id[b.id] = b
            boxes = [by_id[i] for i in sorted(by_id)]
            started_at = existing.started_at
        else:
            boxes = sorted(rec["boxes"], key=lambda b: b.id)
            started_at = now

        page = PageResult.model_validate(
            {
                "page_id": page_id,
                "png_path": rec.get("png_path") or "",
                "layout_path": rec.get("layout_path") or "",
                "model": rec.get("model") or "gemini:gemini-2.5-flash",
                "prompt": rec.get("prompt") or "class-aware (batch)",
                "started_at": started_at,
                "finished_at": now,
                "boxes": [b.model_dump(mode="json") for b in boxes],
            }
        )

        _atomic_write_text(out_path, json.dumps(page.model_dump(mode="json"), indent=2, ensure_ascii=False))
        pages_written += 1

        counts = Counter(b.status for b in page.boxes)
        manifest_line = {
            "page_id": page.page_id,
            "png_path": page.png_path,
            "layout_path": page.layout_path,
            "output_path": str(out_path),
            "model": page.model,
            "finished_at": page.finished_at,
            "status_counts": dict(counts),
        }
        with manifest_path.open("a", encoding="utf-8") as mf:
            mf.write(json.dumps(manifest_line, ensure_ascii=False) + "\n")

    print(
        "Done. "
        f"pages_written={pages_written} boxes_total={boxes_total} boxes_error={boxes_error} "
        f"out_dir={out_dir} manifest={manifest_path}"
    )


if __name__ == "__main__":
    t0 = time.time()
    main()
    dt = round(time.time() - t0, 2)
    print(f"elapsed_seconds={dt}")

