from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable, Iterable, Literal

from gateway.client import GatewayAgentClient, build_user_message
from pydantic import BaseModel, Field

from .models import PageResult
from .pipeline import _coerce_json, _validate_model_provider


class ZoningPresent(BaseModel):
    full_ordinance: bool
    amendment_substantial: bool
    amendment_targeted: bool
    public_hearing: bool


ZoningLabel = Literal[
    "full_ordinance",
    "amendment_substantial",
    "amendment_targeted",
    "public_hearing",
    "unrelated",
]


class ZoningClassifierOutput(BaseModel):
    label: ZoningLabel
    confidence: float = Field(ge=0.0, le=1.0)
    present: ZoningPresent
    rationale: str


class PageZoningResult(BaseModel):
    page_id: str
    source_page_path: str
    source_model: str | None
    classifier_model: str
    prompt_path: str
    started_at: str
    finished_at: str
    input_stats: dict
    classification: ZoningClassifierOutput | None = None
    error: dict | None = None
    attempts: int
    duration_ms: float


def load_page_result(path: Path) -> PageResult:
    data = json.loads(path.read_text())
    return PageResult.model_validate(data)


def _box_sort_key(b) -> tuple[int, int, int]:
    bbox = getattr(b, "bbox", None) or {}
    return (int(bbox.get("y0", 0)), int(bbox.get("x0", 0)), int(getattr(b, "id", 0)))


def page_text_from_boxes(page: PageResult) -> str:
    """Flatten all box transcripts into a single page text block.

    The user prompt expects OCR-extracted TEXT only. We do not inject metadata,
    but we do order by bbox reading order for legibility.
    """

    # Sort by bbox, but only include OCR transcripts from "ok" boxes.
    boxes = sorted(page.boxes, key=_box_sort_key)
    parts: list[str] = []
    for b in boxes:
        if getattr(b, "status", None) != "ok":
            continue
        t = (getattr(b, "transcript", None) or "").strip()
        if not t:
            continue
        parts.append(t)
    return "\n\n".join(parts).strip()


def load_prompt_text(prompt_path: Path) -> str:
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8").strip()


def build_zoning_prompt(*, prompt_text: str, page_text: str) -> str:
    # The prompt expects to receive OCR-extracted TEXT of the page. Keep the
    # input section strictly limited to that text (no bbox metadata, no labels).
    return f"{prompt_text}\n\n{page_text}\n"


def _require_gemini_source(page: PageResult) -> None:
    model = (page.model or "").strip()
    if ":" not in model:
        raise ValueError(f"Expected page.model to be gemini:* but got '{model}' (missing provider prefix)")
    provider = model.split(":", 1)[0].lower()
    if provider != "gemini":
        raise ValueError(f"Expected page.model to be gemini:* but got '{model}'")


async def classify_pages_zoning(
    *,
    page_paths: Iterable[Path],
    model: str,
    gateway_url: str,
    prompt_path: Path,
    max_concurrency: int = 4,
    timeout: float = 120.0,
    max_retries: int = 2,
    allow_test_providers: bool = False,
    on_page: Callable[[PageZoningResult], Awaitable[None]] | None = None,
) -> list[PageZoningResult]:
    _validate_model_provider(model, allow_test_providers=allow_test_providers)
    max_concurrency = max(1, int(max_concurrency))

    prompt_text = load_prompt_text(prompt_path)

    results: list[PageZoningResult] = []
    lock = asyncio.Lock()

    async with GatewayAgentClient(base_url=gateway_url, timeout=timeout) as client:

        async def _one(path: Path) -> PageZoningResult:
            started_at = datetime.now(timezone.utc).isoformat()
            attempts = 0
            last_error: Exception | None = None
            t0 = time.perf_counter()

            page_id = path.stem.replace(".vlm", "")
            source_model: str | None = None
            input_stats: dict = {"page_path": str(path)}

            try:
                page = load_page_result(path)
                page_id = page.page_id
                source_model = page.model
                _require_gemini_source(page)

                page_text = page_text_from_boxes(page)
                if not page_text:
                    raise ValueError("No OCR text found in any ok box; refusing to classify empty input.")

                page_text_sha = hashlib.sha256(page_text.encode("utf-8")).hexdigest()
                input_stats = {
                    "page_path": str(path),
                    "box_count": len(page.boxes),
                    "status_counts": dict(Counter(b.status for b in page.boxes)),
                    "page_text_chars": len(page_text),
                    "page_text_sha256": page_text_sha,
                }
                prompt = build_zoning_prompt(prompt_text=prompt_text, page_text=page_text)
            except Exception as exc:
                last_error = exc
                finished_at = datetime.now(timezone.utc).isoformat()
                res = PageZoningResult(
                    page_id=page_id,
                    source_page_path=str(path),
                    source_model=source_model,
                    classifier_model=model,
                    prompt_path=str(prompt_path),
                    started_at=started_at,
                    finished_at=finished_at,
                    input_stats=input_stats,
                    classification=None,
                    error={"message": str(last_error)},
                    attempts=attempts,
                    duration_ms=(time.perf_counter() - t0) * 1000.0,
                )
                if on_page:
                    await on_page(res)
                async with lock:
                    results.append(res)
                return res

            while attempts < max_retries + 1:
                attempts += 1
                try:
                    msg = build_user_message(prompt)
                    resp = await client.complete_response(model=model, input_messages=[msg])
                    parsed = _coerce_json(resp["text"])
                    classification = ZoningClassifierOutput.model_validate(parsed)

                    finished_at = datetime.now(timezone.utc).isoformat()
                    res = PageZoningResult(
                        page_id=page_id,
                        source_page_path=str(path),
                        source_model=source_model,
                        classifier_model=model,
                        prompt_path=str(prompt_path),
                        started_at=started_at,
                        finished_at=finished_at,
                        input_stats=input_stats,
                        classification=classification,
                        error=None,
                        attempts=attempts,
                        duration_ms=(time.perf_counter() - t0) * 1000.0,
                    )
                    if on_page:
                        await on_page(res)
                    async with lock:
                        results.append(res)
                    return res
                except Exception as exc:
                    last_error = exc
                    if attempts >= max_retries + 1:
                        break
                    await asyncio.sleep(0.5 * attempts)

            finished_at = datetime.now(timezone.utc).isoformat()
            res = PageZoningResult(
                page_id=page_id,
                source_page_path=str(path),
                source_model=source_model,
                classifier_model=model,
                prompt_path=str(prompt_path),
                started_at=started_at,
                finished_at=finished_at,
                input_stats=input_stats,
                classification=None,
                error={"message": str(last_error) if last_error else "unknown error"},
                attempts=attempts,
                duration_ms=(time.perf_counter() - t0) * 1000.0,
            )
            if on_page:
                await on_page(res)
            async with lock:
                results.append(res)
            return res

        queue: asyncio.Queue[Path | None] = asyncio.Queue()

        async def worker() -> None:
            while True:
                item = await queue.get()
                try:
                    if item is None:
                        return
                    await _one(item)
                finally:
                    queue.task_done()

        workers = [asyncio.create_task(worker()) for _ in range(max_concurrency)]
        try:
            for p in page_paths:
                await queue.put(p)
            for _ in range(max_concurrency):
                await queue.put(None)
            await queue.join()
        finally:
            for w in workers:
                w.cancel()
            await asyncio.gather(*workers, return_exceptions=True)

    return results
