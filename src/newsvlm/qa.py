from __future__ import annotations

import asyncio
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


class QaIssue(BaseModel):
    severity: Literal["low", "medium", "high"]
    type: str
    message: str
    box_ids: list[int] = Field(default_factory=list)


class QaLLMOutput(BaseModel):
    status: Literal["ok", "needs_rerun"]
    quality_score: int = Field(ge=0, le=100)
    issues: list[QaIssue] = Field(default_factory=list)
    normalized_page_text: str = ""
    rerun_model: str | None = None
    rerun_reason: str | None = None


class PageQaResult(BaseModel):
    page_id: str
    source_model: str | None = None
    qa_model: str
    started_at: str
    finished_at: str
    input_stats: dict
    qa: QaLLMOutput | None = None
    error: dict | None = None
    attempts: int
    duration_ms: float


def load_page_result(path: Path) -> PageResult:
    data = json.loads(path.read_text())
    return PageResult.model_validate(data)


def _box_sort_key(b: dict) -> tuple[int, int, int]:
    bbox = b.get("bbox") or {}
    return (int(bbox.get("y0", 0)), int(bbox.get("x0", 0)), int(b.get("id", 0)))


def build_page_qa_prompt(page: PageResult) -> str:
    boxes = []
    for b in sorted([br.model_dump(mode="json") for br in page.boxes], key=_box_sort_key):
        boxes.append(
            {
                "id": b.get("id"),
                "cls": b.get("cls"),
                "bbox": b.get("bbox"),
                "status": b.get("status"),
                "transcript": b.get("transcript") or "",
            }
        )

    status_counts = Counter(b.status for b in page.boxes)

    payload = {
        "page_id": page.page_id,
        "status_counts": dict(status_counts),
        "boxes": boxes,
    }
    payload_json = json.dumps(payload, ensure_ascii=False)

    return (
        "You are performing QA on OCR transcripts for a single newspaper page.\n"
        "You are given ONLY OCR box transcripts + bounding boxes (no image). Do NOT hallucinate missing text.\n"
        "\n"
        "Tasks:\n"
        "1) Produce a normalized page text in reading order (left-to-right, top-to-bottom). "
        "You may join hyphenated line breaks when it is obvious, but do not invent words.\n"
        "2) Identify OCR quality issues (gibberish, duplicated content across boxes, obvious truncation, missing/blank boxes).\n"
        "3) Decide if the page likely needs a re-run with a stronger OCR model.\n"
        "\n"
        "Input JSON:\n"
        f"{payload_json}\n"
        "\n"
        "Respond ONLY with JSON (no markdown, no commentary) with this schema:\n"
        "{\n"
        '  "status": "ok" | "needs_rerun",\n'
        '  "quality_score": 0-100,\n'
        '  "issues": [\n'
        '    {"severity":"low"|"medium"|"high","type":string,"message":string,"box_ids":[int,...]}\n'
        "  ],\n"
        '  "normalized_page_text": string,\n'
        '  "rerun_model": string|null,\n'
        '  "rerun_reason": string|null\n'
        "}\n"
    )


async def qa_pages(
    *,
    page_paths: Iterable[Path],
    model: str,
    gateway_url: str,
    max_concurrency: int = 4,
    timeout: float = 120.0,
    max_retries: int = 2,
    allow_test_providers: bool = False,
    on_page: Callable[[PageQaResult], Awaitable[None]] | None = None,
) -> list[PageQaResult]:
    _validate_model_provider(model, allow_test_providers=allow_test_providers)
    max_concurrency = max(1, int(max_concurrency))

    results: list[PageQaResult] = []
    lock = asyncio.Lock()

    async with GatewayAgentClient(base_url=gateway_url, timeout=timeout) as client:

        async def _one(path: Path) -> PageQaResult:
            started_at = datetime.now(timezone.utc).isoformat()
            page: PageResult | None = None
            attempts = 0
            last_error: Exception | None = None
            t0 = time.perf_counter()

            try:
                page = load_page_result(path)
            except Exception as exc:
                finished_at = datetime.now(timezone.utc).isoformat()
                res = PageQaResult(
                    page_id=path.name,
                    source_model=None,
                    qa_model=model,
                    started_at=started_at,
                    finished_at=finished_at,
                    input_stats={},
                    qa=None,
                    error={"message": f"failed to load page json: {exc}"},
                    attempts=1,
                    duration_ms=(time.perf_counter() - t0) * 1000.0,
                )
                if on_page:
                    await on_page(res)
                async with lock:
                    results.append(res)
                return res

            prompt = build_page_qa_prompt(page)
            input_stats = {
                "page_path": str(path),
                "box_count": len(page.boxes),
                "status_counts": dict(Counter(b.status for b in page.boxes)),
            }

            while attempts < max_retries + 1:
                attempts += 1
                try:
                    msg = build_user_message(prompt)
                    resp = await client.complete_response(model=model, input_messages=[msg])
                    parsed = _coerce_json(resp["text"])
                    qa = QaLLMOutput.model_validate(parsed)
                    finished_at = datetime.now(timezone.utc).isoformat()
                    res = PageQaResult(
                        page_id=page.page_id,
                        source_model=page.model,
                        qa_model=model,
                        started_at=started_at,
                        finished_at=finished_at,
                        input_stats=input_stats,
                        qa=qa,
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
            res = PageQaResult(
                page_id=page.page_id,
                source_model=page.model,
                qa_model=model,
                started_at=started_at,
                finished_at=finished_at,
                input_stats=input_stats,
                qa=None,
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
