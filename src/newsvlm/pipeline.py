from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image
from gateway.client import GatewayAgentClient, build_user_message

from .models import BBox, BoxResult, PageResult, TEXT_CLASSES


def load_layout(path: Path, *, png_root: Path | None = None) -> tuple[Path, Path, list[BBox]]:
    data = json.loads(path.read_text())
    scan_url = data.get("scan_url") or data.get("page_path") or data.get("image_path")
    if not scan_url:
        raise ValueError(f"No scan_url/page_path in {path}")
    png_path = Path(scan_url)
    if png_root and not png_path.is_file():
        candidate = png_root / png_path.name
        if candidate.is_file():
            png_path = candidate
    boxes: list[BBox] = []
    for entry in data.get("bboxes", []):
        try:
            cls = entry.get("class")
        except Exception:
            continue
        if cls not in TEXT_CLASSES:
            continue
        boxes.append(BBox.model_validate(entry))
    if not png_path.is_file():
        raise FileNotFoundError(f"PNG not found: {png_path} (from {path})")
    if not boxes:
        raise ValueError(f"No text boxes after filtering in {path}")
    return path, png_path, boxes


def build_prompt(cls: str) -> str:
    base = "Transcribe all visible text exactly as printed. Preserve punctuation and line breaks. Do not summarize."
    if cls == "headline":
        return base + " Keep headline casing and line breaks."
    if cls == "image_caption":
        return base + " Include every word even if fragmented; keep line breaks."
    if cls == "table":
        return base + " Read left-to-right, top-to-bottom; separate columns with a tab character."
    return base


async def transcribe_pages(
    *,
    items: Iterable[tuple[Path, Path, list[BBox]]],
    model: str,
    gateway_url: str,
    max_concurrency: int = 4,
    timeout: float = 60.0,
    max_retries: int = 2,
    crops_dir: Path | None = None,
) -> list[PageResult]:
    sem = asyncio.Semaphore(max_concurrency)
    results: list[PageResult] = []

    async with GatewayAgentClient(base_url=gateway_url, timeout=timeout) as client:

        async def _one_page(layout_path: Path, png_path: Path, boxes: list[BBox]) -> None:
            page_img = Image.open(png_path).convert("RGB")
            box_results: list[BoxResult] = []
            started_at = datetime.now(timezone.utc).isoformat()

            async def _one_box(b: BBox) -> None:
                prompt = build_prompt(b.cls)
                attempts = 0
                while True:
                    attempts += 1
                    start = time.perf_counter()
                    async with sem:
                        msg = build_user_message(
                            prompt + ' Respond ONLY with JSON: {"status":"ok"|"unreadable","transcript":string}. If unreadable, set transcript to "".',
                            image_bytes=[_maybe_save_crop(page_img, b, crops_dir)],
                        )
                        resp = await client.complete_response(model=model, input_messages=[msg])
                    elapsed_ms = (time.perf_counter() - start) * 1000.0
                    try:
                        parsed = _coerce_json(resp["text"])
                        status = parsed.get("status")
                        transcript = parsed.get("transcript")
                        if status not in {"ok", "unreadable"}:
                            raise ValueError("status must be ok or unreadable")
                        box_res = BoxResult.model_validate(
                            {
                                "id": b.id,
                                "class": b.cls,
                                "bbox": b.bbox,
                                "legibility": b.legibility,
                                "status": status,
                                "transcript": transcript,
                                "model": model,
                                "prompt": prompt,
                                "attempts": attempts,
                                "duration_ms": elapsed_ms,
                                "error": None,
                            }
                        )
                        box_results.append(box_res)
                        break
                    except Exception as exc:
                        if attempts > max_retries + 1:
                            raise
                        await asyncio.sleep(0.5 * attempts)
                return

            await asyncio.gather(*(_one_box(b) for b in boxes))
            finished_at = datetime.now(timezone.utc).isoformat()
            page_id = png_path.stem
            page_res = PageResult.model_validate(
                {
                    "page_id": page_id,
                    "png_path": str(png_path),
                    "layout_path": str(layout_path),
                    "model": model,
                    "prompt": "class-aware",
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "boxes": [br.model_dump() for br in box_results],
                }
            )
            results.append(page_res)

        for layout_path, png_path, boxes in items:
            await _one_page(layout_path, png_path, boxes)

    return results


def _crop_to_png_bytes(img: Image.Image, xyxy: Sequence[int]) -> bytes:
    crop = img.crop(tuple(xyxy))
    buf = BytesIO()
    crop.save(buf, format="PNG")
    return buf.getvalue()


def _maybe_save_crop(img: Image.Image, bbox: BBox, crops_dir: Path | None) -> bytes:
    data = _crop_to_png_bytes(img, bbox.xyxy)
    if crops_dir:
        crops_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{bbox.id}_{bbox.cls}.png"
        (crops_dir / fname).write_bytes(data)
    return data


def _coerce_json(text: str) -> dict:
    """Accept raw JSON or JSON wrapped in ```json fences."""

    if text is None:
        raise ValueError("empty response text")
    stripped = text.strip()
    if stripped.startswith("```"):
        # remove leading ``` and language tag
        lines = stripped.splitlines()
        # drop first fence line
        lines = lines[1:]
        # drop trailing fence if present
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError as exc:
        # Some providers return raw newlines in strings; try escaping them and retry.
        if "Invalid control character" in str(exc):
            escaped = stripped.replace("\n", "\\n")
            return json.loads(escaped)
        raise
