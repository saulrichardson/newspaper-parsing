from __future__ import annotations

import asyncio
import json
import math
import time
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Awaitable, Callable, Iterable, Sequence

from PIL import Image
from gateway.client import GatewayAgentClient, build_user_message

from .models import BBox, BoxResult, PageResult, TEXT_CLASSES

_PNG_INDEX_CACHE: dict[str, dict[str, Path]] = {}


def _build_png_index(root: Path) -> dict[str, Path]:
    """Build a basename->path index for PNGs under a root."""
    idx: dict[str, Path] = {}
    for p in root.rglob("*.png"):
        name = p.name
        if name not in idx:
            idx[name] = p
    return idx


def _resolve_png_path(scan_url: str, png_root: Path | None) -> Path:
    """Resolve a scan_url into an on-disk PNG path.

    Handles:
      - absolute paths embedded in layouts (e.g., old VAST paths)
      - local png_root with nested subdirectories
      - basename lookups via a cached index
    """
    png_path = Path(scan_url)
    if png_root is None:
        return png_path
    if png_path.is_file():
        return png_path

    # 1) Direct basename under png_root.
    candidate = png_root / png_path.name
    if candidate.is_file():
        return candidate

    # 2) Preserve tail after a 'unique_png' segment if present.
    try:
        parts = png_path.parts
        if "unique_png" in parts:
            idx = parts.index("unique_png")
            tail = Path(*parts[idx + 1 :])
            candidate = png_root / tail
            if candidate.is_file():
                return candidate
    except Exception:
        pass

    # 3) Cached recursive basename index.
    key = str(png_root)
    index = _PNG_INDEX_CACHE.get(key)
    if index is None:
        index = _PNG_INDEX_CACHE[key] = _build_png_index(png_root)
    found = index.get(png_path.name)
    if found and found.is_file():
        return found

    return png_path

REAL_PROVIDERS = {"gemini", "openai", "claude"}


def load_layout(path: Path, *, png_root: Path | None = None) -> tuple[Path, Path, list[BBox]]:
    data = json.loads(path.read_text())
    scan_url = data.get("scan_url") or data.get("page_path") or data.get("image_path")
    if not scan_url:
        raise ValueError(f"No scan_url/page_path in {path}")
    png_path = _resolve_png_path(scan_url, png_root)
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
    if cls == "author":
        return base + " This is a byline; preserve capitalization and any affiliations exactly."
    if cls == "article":
        return (
            base
            + " Text may be multi-column. Read columns left-to-right, top-to-bottom, preserving line breaks."
        )
    if cls == "image_caption":
        return base + " Captions may be small or fragmented; include every word and keep line breaks."
    if cls == "newspaper_header":
        return base + " This is a masthead/header; preserve styling, spacing, and line breaks."
    if cls == "table":
        return (
            base
            + " Read left-to-right, top-to-bottom. Preserve each row on its own line, and separate columns with a tab character."
        )
    return base


def _validate_model_provider(model: str, *, allow_test_providers: bool) -> None:
    if ":" not in model:
        if allow_test_providers:
            return
        raise ValueError(
            f"Model '{model}' must be prefixed with a real provider (one of {sorted(REAL_PROVIDERS)})."
        )
    provider = model.split(":", 1)[0].lower()
    if provider not in REAL_PROVIDERS and not allow_test_providers:
        raise ValueError(
            f"Provider '{provider}' is not allowed for transcription. Use one of {sorted(REAL_PROVIDERS)}."
        )


async def transcribe_pages(
    *,
    items: Iterable[tuple[Path, Path, list[BBox]]],
    model: str,
    gateway_url: str,
    max_concurrency: int = 4,
    page_concurrency: int = 1,
    timeout: float = 60.0,
    max_retries: int = 2,
    crops_dir: Path | None = None,
    max_crop_megapixels: float | None = None,
    max_crop_dim: int | None = None,
    allow_test_providers: bool = False,
    on_page: Callable[[PageResult], Awaitable[None]] | None = None,
) -> list[PageResult]:
    sem_boxes = asyncio.Semaphore(max_concurrency)
    results: list[PageResult] = []
    results_lock = asyncio.Lock()

    _validate_model_provider(model, allow_test_providers=allow_test_providers)

    async with GatewayAgentClient(base_url=gateway_url, timeout=timeout) as client:

        async def _one_page(layout_path: Path, png_path: Path, boxes: list[BBox]) -> PageResult:
            with Image.open(png_path) as _im:
                page_img = _im.convert("RGB")
            box_results: list[BoxResult] = []
            started_at = datetime.now(timezone.utc).isoformat()
            page_id = png_path.stem

            async def _one_box(b: BBox) -> None:
                prompt = build_prompt(b.cls)
                crop_bytes, crop_meta = _maybe_save_crop(
                    page_img,
                    b,
                    crops_dir,
                    page_id,
                    max_megapixels=max_crop_megapixels,
                    max_dim=max_crop_dim,
                )

                attempts = 0
                last_error: Exception | None = None
                while attempts < max_retries + 1:
                    attempts += 1
                    start = time.perf_counter()
                    try:
                        async with sem_boxes:
                            msg = build_user_message(
                                prompt
                                + ' Respond ONLY with JSON: {"status":"ok"|"unreadable","transcript":string}. If unreadable, set transcript to "".',
                                image_bytes=[crop_bytes],
                            )
                            resp = await client.complete_response(model=model, input_messages=[msg])
                        elapsed_ms = (time.perf_counter() - start) * 1000.0

                        parsed = _coerce_json(resp["text"])
                        status = parsed.get("status")
                        transcript = parsed.get("transcript")
                        # Normalize empty transcript: if model says ok but gave empty, treat as unreadable.
                        if status == "ok" and (transcript is None or not str(transcript).strip()):
                            status = "unreadable"
                            transcript = ""
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
                                **crop_meta,
                                "error": None,
                            }
                        )
                        box_results.append(box_res)
                        return
                    except Exception as exc:
                        last_error = exc
                        if attempts >= max_retries + 1:
                            break
                        await asyncio.sleep(0.5 * attempts)

                # Record a per-box error instead of aborting the whole page/chunk.
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                box_res = BoxResult.model_validate(
                    {
                        "id": b.id,
                        "class": b.cls,
                        "bbox": b.bbox,
                        "legibility": b.legibility,
                        "status": "error",
                        "transcript": None,
                        "model": model,
                        "prompt": prompt,
                        "attempts": attempts,
                        "duration_ms": elapsed_ms,
                        **crop_meta,
                        "error": {"message": str(last_error) if last_error else "unknown error"},
                    }
                )
                box_results.append(box_res)
                return

            await asyncio.gather(*(_one_box(b) for b in boxes))
            finished_at = datetime.now(timezone.utc).isoformat()
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
            if on_page:
                await on_page(page_res)
            async with results_lock:
                results.append(page_res)
            return page_res

        page_concurrency = max(1, int(page_concurrency))
        in_flight: set[asyncio.Task[PageResult]] = set()

        async def _spawn(item: tuple[Path, Path, list[BBox]]) -> None:
            task = asyncio.create_task(_one_page(*item))
            in_flight.add(task)
            task.add_done_callback(lambda t: in_flight.discard(t))  # type: ignore[arg-type]

        try:
            for item in items:
                while len(in_flight) >= page_concurrency:
                    done, _ = await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED)
                    for d in done:
                        d.result()
                await _spawn(item)

            if in_flight:
                done, _ = await asyncio.wait(in_flight)
                for d in done:
                    d.result()
        except Exception:
            for t in in_flight:
                t.cancel()
            raise

    return results


def _crop_to_png_bytes(
    img: Image.Image,
    xyxy: Sequence[int],
    *,
    max_megapixels: float | None = None,
    max_dim: int | None = None,
) -> tuple[bytes, dict]:
    crop = img.crop(tuple(xyxy))
    orig_w, orig_h = crop.size

    scale = 1.0
    if max_dim and max_dim > 0:
        max_edge = max(orig_w, orig_h)
        if max_edge > max_dim:
            scale = min(scale, max_dim / float(max_edge))

    if max_megapixels and max_megapixels > 0:
        max_pixels = max_megapixels * 1_000_000.0
        pixels = float(orig_w * orig_h)
        if pixels > max_pixels:
            scale = min(scale, math.sqrt(max_pixels / pixels))

    if scale < 1.0:
        new_w = max(1, int(round(orig_w * scale)))
        new_h = max(1, int(round(orig_h * scale)))
        crop = crop.resize((new_w, new_h), resample=Image.LANCZOS)

    buf = BytesIO()
    crop.save(buf, format="PNG", optimize=True)
    data = buf.getvalue()

    meta = {
        "orig_size": (orig_w, orig_h),
        "sent_size": crop.size,
        "scale_factor": float(scale),
        "bytes_sent": len(data),
    }
    return data, meta


def _maybe_save_crop(
    img: Image.Image,
    bbox: BBox,
    crops_dir: Path | None,
    page_id: str,
    *,
    max_megapixels: float | None = None,
    max_dim: int | None = None,
) -> tuple[bytes, dict]:
    data, meta = _crop_to_png_bytes(img, bbox.xyxy, max_megapixels=max_megapixels, max_dim=max_dim)
    if crops_dir:
        crops_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{page_id}_{bbox.id}_{bbox.cls}.png"
        (crops_dir / fname).write_bytes(data)
    return data, meta


def _coerce_json(text: str) -> dict:
    """Accept raw JSON or JSON wrapped in ```json fences, scrub control chars."""

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
        # First pass parse
        return json.loads(stripped)
    except json.JSONDecodeError as exc:
        # Scrub ASCII control characters (except tab/newline/carriage return) then retry.
        import re

        cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", stripped)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Some providers return raw newlines in strings; try escaping them and retry.
            if "Invalid control character" in str(exc):
                escaped = cleaned.replace("\n", "\\n")
                return json.loads(escaped)
        raise
