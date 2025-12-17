#!/usr/bin/env python3
"""
Rehydrate OpenAI Batch OCR outputs into per-page *.vlm.json files (PageResult schema).

Why this exists:
  - OpenAI batch downloads are stored as JSONL lines keyed by `custom_id = "<page_id>:<box_id>"`.
  - The downstream zoning classifier / QA tools in this repo operate on per-page `*.vlm.json`
    (see `src/newsvlm/models.py` and `src/newsvlm/zoning_classifier.py`).

Inputs:
  - Dell layout JSONs (for bbox/class/legibility):
      <layout_root>/<page_id>.json
  - Downloaded OpenAI batch result JSONLs:
      <results_root>/part_*/<openai_subdir>/results/openai_results_shard*.jsonl
      <results_root>/part_*/<openai_subdir>/results/openai_errors_shard*.jsonl

Output:
  - <output_dir>/<page_id>.vlm.json (PageResult)
  - <output_dir>/manifest.jsonl (one JSON object per written/skipped page)

This script does NOT call any APIs; it only reshapes existing on-disk artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from os.path import expanduser
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from newsvlm.models import TEXT_CLASSES  # noqa: E402


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Rehydrate OpenAI OCR batch JSONLs into per-page .vlm.json files.")
    ap.add_argument(
        "--results-root",
        default="newspaper-parsing-local/data/greene_dedupe_webp_results",
        help="Root containing part_XXXX directories with downloaded OpenAI result/error JSONLs",
    )
    ap.add_argument(
        "--openai-subdir",
        default="openai_gpt52_reasoning_medium_split",
        help="Subdirectory under each part containing OpenAI results (default matches existing downloads)",
    )
    ap.add_argument(
        "--layout-root",
        default="newspaper-parsing-local/data/unique_outputs_dedup",
        help="Directory containing Dell layout JSONs named <page_id>.json",
    )
    ap.add_argument("--output-dir", required=True, help="Directory to write per-page *.vlm.json outputs into")
    ap.add_argument(
        "--page-ids",
        default=None,
        help='Optional "@file" listing page_ids to include (one per line). If omitted, includes all pages encountered.',
    )
    ap.add_argument("--require-ok", action="store_true", help="Only write pages that have >=1 ok box transcript")
    ap.add_argument("--skip-existing", action="store_true", help="Skip pages whose output file already exists")
    ap.add_argument(
        "--skip-missing-layout",
        action="store_true",
        help="Skip pages missing <layout_root>/<page_id>.json instead of failing",
    )
    ap.add_argument("--max-pages", type=int, default=None, help="Stop after writing this many pages (smoke testing)")
    return ap.parse_args()


def _load_page_id_allowlist(spec: str | None) -> set[str] | None:
    if not spec:
        return None
    spec = spec.strip()
    if not spec.startswith("@"):
        raise SystemExit("--page-ids must be an @file (one page_id per line)")
    p = Path(expanduser(spec[1:])).resolve()
    if not p.is_file():
        raise SystemExit(f"Page id list not found: {p}")
    out: set[str] = set()
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        out.add(line)
    return out


def _extract_output_text(body: dict) -> str:
    out = body.get("output")
    if not isinstance(out, list):
        return ""
    parts: list[str] = []
    for item in out:
        if not isinstance(item, dict):
            continue
        content = item.get("content") or []
        if not isinstance(content, list):
            continue
        for c in content:
            if isinstance(c, dict) and c.get("type") == "output_text":
                parts.append(str(c.get("text") or ""))
    return "".join(parts)


def _parse_custom_id(custom_id: str) -> tuple[str, int]:
    if ":" not in custom_id:
        raise ValueError(f"custom_id missing ':' separator: {custom_id!r}")
    page_id, box_id_s = custom_id.rsplit(":", 1)
    if not page_id:
        raise ValueError(f"custom_id missing page_id: {custom_id!r}")
    if not box_id_s.isdigit():
        raise ValueError(f"custom_id box_id is not an int: {custom_id!r}")
    return page_id, int(box_id_s)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _atomic_write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _load_layout_boxes(layout_path: Path) -> tuple[str, dict[int, dict[str, Any]]]:
    data = json.loads(layout_path.read_text(encoding="utf-8"))
    scan_url = data.get("scan_url") or data.get("page_path") or data.get("image_path") or ""
    boxes_by_id: dict[int, dict[str, Any]] = {}
    for b in data.get("bboxes", []) or []:
        if not isinstance(b, dict):
            continue
        if b.get("class") not in TEXT_CLASSES:
            continue
        bid = b.get("id")
        if not isinstance(bid, int):
            continue
        bbox = b.get("bbox")
        if not isinstance(bbox, dict):
            continue
        # Normalize to required bbox keys.
        try:
            x0 = int(bbox["x0"])
            y0 = int(bbox["y0"])
            x1 = int(bbox["x1"])
            y1 = int(bbox["y1"])
        except Exception:
            continue
        boxes_by_id[bid] = {
            "id": bid,
            "class": b.get("class"),
            "bbox": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
            "legibility": b.get("legibility") or "Unknown",
        }
    return str(scan_url), boxes_by_id


def main() -> None:
    args = _parse_args()

    results_root = Path(expanduser(args.results_root)).resolve()
    layout_root = Path(expanduser(args.layout_root)).resolve()
    out_dir = Path(expanduser(args.output_dir)).resolve()
    allow = _load_page_id_allowlist(args.page_ids)

    if not results_root.is_dir():
        raise SystemExit(f"--results-root is not a directory: {results_root}")
    if not layout_root.is_dir():
        raise SystemExit(f"--layout-root is not a directory: {layout_root}")
    out_dir.mkdir(parents=True, exist_ok=True)

    openai_subdir = args.openai_subdir.strip()
    part_dirs = sorted(p for p in results_root.glob("part_*") if p.is_dir())
    if not part_dirs:
        raise SystemExit(f"No part_* directories found under: {results_root}")

    manifest_path = out_dir / "manifest.jsonl"
    manifest_f = manifest_path.open("a", encoding="utf-8")

    pages_written = 0
    pages_skipped = 0
    pages_error = 0
    t0 = time.perf_counter()

    try:
        for part in part_dirs:
            res_dir = part / openai_subdir / "results"
            if not res_dir.is_dir():
                continue

            result_files = sorted(res_dir.glob("openai_results_shard*.jsonl"))
            error_files = sorted(res_dir.glob("openai_errors_shard*.jsonl"))
            if not result_files and not error_files:
                continue

            # Accumulate per-page box-level outputs for this part.
            boxes_by_page: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
            models_by_page: dict[str, Counter[str]] = defaultdict(Counter)
            status_counts_by_page: dict[str, Counter[str]] = defaultdict(Counter)

            for fp in result_files:
                with fp.open("r", encoding="utf-8") as f:
                    for raw in f:
                        line = raw.strip()
                        if not line:
                            continue
                        obj = json.loads(line)
                        cid = str(obj.get("custom_id") or "")
                        try:
                            page_id, box_id = _parse_custom_id(cid)
                        except Exception:
                            continue
                        if allow is not None and page_id not in allow:
                            continue

                        body = ((obj.get("response") or {}).get("body") or {})
                        if not isinstance(body, dict):
                            boxes_by_page[page_id][box_id] = {
                                "status": "error",
                                "transcript": None,
                                "model": None,
                                "error": {"message": "missing response body"},
                            }
                            status_counts_by_page[page_id]["error"] += 1
                            continue

                        model_name = body.get("model")
                        if isinstance(model_name, str) and model_name.strip():
                            models_by_page[page_id][model_name.strip()] += 1

                        output_text = _extract_output_text(body)
                        try:
                            parsed = json.loads(output_text)
                        except Exception:
                            boxes_by_page[page_id][box_id] = {
                                "status": "error",
                                "transcript": None,
                                "model": model_name,
                                "error": {
                                    "message": "parsefail: model output was not valid JSON",
                                    "output_sha256": _sha256(output_text),
                                },
                            }
                            status_counts_by_page[page_id]["parsefail"] += 1
                            continue

                        st = parsed.get("status")
                        tx = parsed.get("transcript")
                        if st == "ok":
                            if tx is None or not str(tx).strip():
                                st = "unreadable"
                                tx = ""
                        if st not in {"ok", "unreadable"}:
                            boxes_by_page[page_id][box_id] = {
                                "status": "error",
                                "transcript": None,
                                "model": model_name,
                                "error": {"message": f"invalid status in model JSON: {st!r}"},
                            }
                            status_counts_by_page[page_id]["invalid_status"] += 1
                            continue

                        boxes_by_page[page_id][box_id] = {
                            "status": st,
                            "transcript": str(tx) if tx is not None else "",
                            "model": model_name,
                            "error": None,
                        }
                        status_counts_by_page[page_id][st] += 1

            for fp in error_files:
                with fp.open("r", encoding="utf-8") as f:
                    for raw in f:
                        line = raw.strip()
                        if not line:
                            continue
                        obj = json.loads(line)
                        cid = str(obj.get("custom_id") or "")
                        try:
                            page_id, box_id = _parse_custom_id(cid)
                        except Exception:
                            continue
                        if allow is not None and page_id not in allow:
                            continue

                        resp = obj.get("response") or {}
                        body = (resp.get("body") or {}) if isinstance(resp, dict) else {}
                        err = body.get("error") if isinstance(body, dict) else None

                        boxes_by_page[page_id][box_id] = {
                            "status": "error",
                            "transcript": None,
                            "model": None,
                            "error": {
                                "status_code": resp.get("status_code") if isinstance(resp, dict) else None,
                                "error": err if isinstance(err, dict) else {"message": "unknown error"},
                            },
                        }
                        status_counts_by_page[page_id]["error"] += 1

            # Write out pages found in this part.
            for page_id, box_map in sorted(boxes_by_page.items()):
                if args.max_pages is not None and pages_written >= int(args.max_pages):
                    break
                if args.require_ok and status_counts_by_page[page_id].get("ok", 0) <= 0:
                    pages_skipped += 1
                    manifest_f.write(
                        json.dumps(
                            {
                                "page_id": page_id,
                                "status": "skipped",
                                "reason": "require_ok but page has zero ok boxes",
                                "counts": dict(status_counts_by_page[page_id]),
                                "part": part.name,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    continue

                out_path = out_dir / f"{page_id}.vlm.json"
                if args.skip_existing and out_path.exists():
                    pages_skipped += 1
                    manifest_f.write(
                        json.dumps(
                            {
                                "page_id": page_id,
                                "status": "skipped",
                                "reason": "output exists",
                                "output_path": str(out_path),
                                "counts": dict(status_counts_by_page[page_id]),
                                "part": part.name,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    continue

                layout_path = layout_root / f"{page_id}.json"
                if not layout_path.is_file():
                    msg = f"Missing layout: {layout_path}"
                    if args.skip_missing_layout:
                        pages_error += 1
                        manifest_f.write(
                            json.dumps(
                                {
                                    "page_id": page_id,
                                    "status": "error",
                                    "error": {"message": msg},
                                    "counts": dict(status_counts_by_page[page_id]),
                                    "part": part.name,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        continue
                    raise SystemExit(msg)

                try:
                    png_path, layout_boxes = _load_layout_boxes(layout_path)
                except Exception as exc:
                    pages_error += 1
                    manifest_f.write(
                        json.dumps(
                            {
                                "page_id": page_id,
                                "status": "error",
                                "error": {"message": f"Failed to parse layout JSON: {exc}"},
                                "layout_path": str(layout_path),
                                "counts": dict(status_counts_by_page[page_id]),
                                "part": part.name,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    continue

                # Pick a representative OCR model name for the page (best effort).
                model_counts = models_by_page.get(page_id) or Counter()
                best_model = model_counts.most_common(1)[0][0] if model_counts else "unknown"
                page_model = f"openai:{best_model}" if best_model != "unknown" else "openai:unknown"

                now = datetime.now(timezone.utc).isoformat()
                boxes_out: list[dict[str, Any]] = []

                missing_layout_box_ids: list[int] = []
                for box_id in sorted(box_map.keys()):
                    layout_box = layout_boxes.get(int(box_id))
                    if layout_box is None:
                        missing_layout_box_ids.append(int(box_id))
                        continue
                    rec = box_map[int(box_id)]
                    st = rec.get("status")
                    tx = rec.get("transcript")
                    model_name = rec.get("model")
                    box_model = (
                        f"openai:{model_name.strip()}" if isinstance(model_name, str) and model_name.strip() else page_model
                    )

                    boxes_out.append(
                        {
                            "id": int(box_id),
                            "class": layout_box["class"],
                            "bbox": layout_box["bbox"],
                            "legibility": layout_box["legibility"],
                            "status": st,
                            "transcript": tx,
                            "model": box_model,
                            "prompt": "class-aware (batch)",
                            "attempts": 1,
                            "duration_ms": 0.0,
                            "orig_size": None,
                            "sent_size": None,
                            "scale_factor": None,
                            "bytes_sent": None,
                            "error": rec.get("error"),
                        }
                    )

                if not boxes_out:
                    pages_error += 1
                    manifest_f.write(
                        json.dumps(
                            {
                                "page_id": page_id,
                                "status": "error",
                                "error": {
                                    "message": "No boxes could be emitted (no matching layout boxes).",
                                    "missing_layout_box_ids": missing_layout_box_ids,
                                },
                                "layout_path": str(layout_path),
                                "counts": dict(status_counts_by_page[page_id]),
                                "part": part.name,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    continue

                page_obj = {
                    "page_id": page_id,
                    "png_path": png_path,
                    "layout_path": str(layout_path),
                    "model": page_model,
                    "prompt": "class-aware (batch)",
                    "started_at": now,
                    "finished_at": now,
                    "boxes": boxes_out,
                }
                _atomic_write_json(out_path, page_obj)

                pages_written += 1
                manifest_f.write(
                    json.dumps(
                        {
                            "page_id": page_id,
                            "status": "ok",
                            "output_path": str(out_path),
                            "layout_path": str(layout_path),
                            "model": page_model,
                            "counts": dict(status_counts_by_page[page_id]),
                            "missing_layout_box_ids_count": len(missing_layout_box_ids),
                            "part": part.name,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            if args.max_pages is not None and pages_written >= int(args.max_pages):
                break

    finally:
        manifest_f.close()

    elapsed = time.perf_counter() - t0
    print(f"pages_written={pages_written} pages_skipped={pages_skipped} pages_error={pages_error} elapsed_s={elapsed:.1f}")


if __name__ == "__main__":
    main()

