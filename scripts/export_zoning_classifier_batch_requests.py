#!/usr/bin/env python3
"""
Export batch-request JSONLs for zoning classification from per-page *.vlm.json OCR outputs.

This is for the "zoning OCR classifier" prompt:
  prompts/zoning_ocr_classifier_prompt_text.txt

Modes:
  - page:          1 request per page (input is concatenated OCR text from all ok boxes)
  - boxes:         1 request per box (input is that box transcript only)
  - page+box_ids:  1 request per page, asks model to return page classification + box ids likely containing zoning text

Outputs (sharded JSONL):
  - mapping_shardNNN.jsonl
  - openai_requests_shardNNN.jsonl (optional)
  - gemini_requests_shardNNN.jsonl (optional)

These are compatible with:
  - scripts/submit_batch_shards.py (submission)
  - scripts/download_openai_batch_results.py / scripts/download_gemini_batch_results.py (download)

Note: This script intentionally does NOT submit anything; it only writes request shards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from glob import glob
from os.path import expanduser
from pathlib import Path
from typing import IO, Any, Iterable, Literal

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from newsvlm.zoning_classifier import load_page_result, load_prompt_text, page_text_from_boxes  # noqa: E402


Provider = Literal["openai", "gemini", "both"]
Mode = Literal["page", "boxes", "page+box_ids"]
BoxSelection = Literal["reading", "longest"]
OpenAITextFormat = Literal["json_object", "json_schema", "none"]


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Export batch JSONL requests for zoning classification.")
    ap.add_argument(
        "--pages",
        required=True,
        help='Glob for per-page *.vlm.json files (absolute OK) or "@file" listing those paths',
    )
    ap.add_argument("--output-dir", required=True, help="Directory to write request shards into")
    ap.add_argument(
        "--provider",
        choices=["openai", "gemini", "both"],
        default="openai",
        help="Which provider request JSONLs to emit",
    )
    ap.add_argument(
        "--mode",
        choices=["page", "boxes", "page+box_ids"],
        default="page",
        help="How to structure requests",
    )
    ap.add_argument(
        "--prompt-path",
        default="prompts/zoning_ocr_classifier_prompt_text.txt",
        help="Prompt text file path",
    )
    ap.add_argument(
        "--openai-model",
        default="gpt-5.2",
        help="OpenAI model name for /v1/responses batch body",
    )
    ap.add_argument(
        "--openai-reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        default="medium",
        help="OpenAI reasoning.effort for Responses API (set 'none' to omit)",
    )
    ap.add_argument(
        "--openai-max-output-tokens",
        type=int,
        default=None,
        help=(
            "OpenAI Responses API max_output_tokens. "
            "If omitted (recommended), we do not set it at all to avoid causing `status=incomplete` "
            "due to reasoning-token budget constraints."
        ),
    )
    ap.add_argument(
        "--openai-text-format",
        choices=["json_object", "json_schema", "none"],
        default="json_object",
        help="OpenAI Responses text.format enforcement (json_schema is strictest; json_object is simpler).",
    )
    ap.add_argument(
        "--requests-per-shard",
        type=int,
        default=5000,
        help="Maximum requests per shard file",
    )
    ap.add_argument(
        "--max-bytes-per-shard",
        type=int,
        default=0,
        help=(
            "Max bytes per request shard file (applies to provider request file(s); mapping is not capped). "
            "0 disables byte-based splitting. Recommended for OpenAI: ~180_000_000."
        ),
    )
    ap.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="If set, only export up to this many pages (useful for smoke tests)",
    )
    ap.add_argument(
        "--skip-empty",
        action="store_true",
        help="Skip pages/boxes with no usable OCR text instead of erroring",
    )

    # Box-related knobs (mode=boxes or page+box_ids)
    ap.add_argument(
        "--max-boxes",
        type=int,
        default=40,
        help="Max boxes per page to include (box modes only). Use 0 to include ALL ok boxes.",
    )
    ap.add_argument(
        "--box-selection",
        choices=["reading", "longest"],
        default="longest",
        help="How to pick boxes when mode requires a subset",
    )
    ap.add_argument("--min-chars", type=int, default=40, help="Skip boxes shorter than this many chars")
    ap.add_argument(
        "--strict-box-zoning-terms",
        action="store_true",
        help=(
            "EXPERIMENTAL: require explicit zoning terms for box-level decisions. "
            "Reduces false positives on generic ordinances/hearings."
        ),
    )

    return ap.parse_args()


def _collect_page_paths(spec: str) -> list[Path]:
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


def _box_sort_key(b) -> tuple[int, int, int]:
    bbox = getattr(b, "bbox", None) or {}
    return (int(bbox.get("y0", 0)), int(bbox.get("x0", 0)), int(getattr(b, "id", 0)))


def _iter_ok_boxes(page) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for b in sorted(page.boxes, key=_box_sort_key):
        if getattr(b, "status", None) != "ok":
            continue
        t = (getattr(b, "transcript", None) or "").strip()
        if not t:
            continue
        out.append(
            {
                "box_id": int(getattr(b, "id", -1)),
                "cls": getattr(b, "cls", None),
                "bbox": getattr(b, "bbox", None),
                "transcript": t,
            }
        )
    return out


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _open_shard_files(
    out_dir: Path, shard_idx: int, *, want_openai: bool, want_gemini: bool
) -> tuple[IO[str] | None, IO[str] | None, IO[str]]:
    gemini_f = None
    openai_f = None
    if want_gemini:
        gemini_f = (out_dir / f"gemini_requests_shard{shard_idx:03d}.jsonl").open("w", encoding="utf-8")
    if want_openai:
        openai_f = (out_dir / f"openai_requests_shard{shard_idx:03d}.jsonl").open("w", encoding="utf-8")
    mapping_f = (out_dir / f"mapping_shard{shard_idx:03d}.jsonl").open("w", encoding="utf-8")
    return gemini_f, openai_f, mapping_f


def _close_files(files: Iterable[IO[str] | None]) -> None:
    for f in files:
        if f:
            f.close()


def _openai_text_config_json_object() -> dict[str, Any]:
    return {"format": {"type": "json_object"}}


def _zoning_classifier_openai_json_schema() -> dict[str, Any]:
    # OpenAI's strict JSON-schema mode requires `additionalProperties: false` for objects.
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "label": {
                "type": "string",
                "enum": [
                    "full_ordinance",
                    "amendment_substantial",
                    "amendment_targeted",
                    "public_hearing",
                    "unrelated",
                ],
            },
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "present": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "full_ordinance": {"type": "boolean"},
                    "amendment_substantial": {"type": "boolean"},
                    "amendment_targeted": {"type": "boolean"},
                    "public_hearing": {"type": "boolean"},
                },
                "required": [
                    "full_ordinance",
                    "amendment_substantial",
                    "amendment_targeted",
                    "public_hearing",
                ],
            },
            "rationale": {"type": "string"},
        },
        "required": ["label", "confidence", "present", "rationale"],
    }


def _openai_text_config(fmt: OpenAITextFormat) -> dict[str, Any] | None:
    if fmt == "none":
        return None
    if fmt == "json_object":
        return _openai_text_config_json_object()
    if fmt == "json_schema":
        return {
            "format": {
                "type": "json_schema",
                "name": "zoning_classifier_output",
                "schema": _zoning_classifier_openai_json_schema(),
                "strict": True,
            }
        }
    raise ValueError(f"Unknown openai_text_format: {fmt}")


def _build_page_prompt(prompt_text: str, *, page_text: str) -> str:
    return f"{prompt_text}\n\n{page_text}\n"


def _with_strict_box_terms(prompt_text: str) -> str:
    return (
        f"{prompt_text}\n\n"
        "EXTRA CONSTRAINT (box-level): Only return a zoning-related label if the text explicitly "
        "mentions zoning/rezoning/rezone/zoning district/zoning map/variance/conditional use/"
        "board of zoning appeals/planning commission (or similarly unmistakable zoning-language). "
        "If the text is a generic ordinance, street vacation, ditch hearing, road work notice, etc. "
        "and does not clearly mention zoning, return 'unrelated'."
    )


def _build_box_ids_prompt(prompt_text: str, *, boxes: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for b in boxes:
        lines.append(f"BOX {b['box_id']}:")
        lines.append(b["transcript"])
        lines.append("")
    box_block = "\n".join(lines).strip() + "\n"

    return (
        f"{prompt_text}\n\n"
        "The OCR text below is split into bounding boxes from the same page.\n"
        "Each box starts with a line like 'BOX <id>:' followed by the OCR transcript for that box.\n\n"
        "Do the same zoning-presence classification for the FULL PAGE by considering all boxes.\n"
        "Additionally, identify which box ids contain qualifying zoning ordinance/amendment/hearing text.\n\n"
        "Output JSON only with this schema:\n"
        "{\n"
        '  "page": {\n'
        '    "label": "full_ordinance | amendment_substantial | amendment_targeted | public_hearing | unrelated",\n'
        '    "confidence": 0.0-1.0,\n'
        '    "present": {\n'
        '      "full_ordinance": true/false,\n'
        '      "amendment_substantial": true/false,\n'
        '      "amendment_targeted": true/false,\n'
        '      "public_hearing": true/false\n'
        "    },\n"
        '    "rationale": "1–2 sentences."\n'
        "  },\n"
        '  "boxes_with_zoning": [<box_id>, ...]\n'
        "}\n\n"
        f"{box_block}"
    )


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = Path(expanduser(args.prompt_path))
    prompt_text = load_prompt_text(prompt_path)

    pages = _collect_page_paths(args.pages)
    if not pages:
        raise SystemExit(f"No pages matched: {args.pages}")

    max_pages = int(args.max_pages) if args.max_pages is not None else None
    mode: Mode = args.mode
    provider: Provider = args.provider  # type: ignore[assignment]
    want_openai = provider in {"openai", "both"}
    want_gemini = provider in {"gemini", "both"}

    shard_idx = 0
    reqs_in_shard = 0
    bytes_in_shard_openai = 0
    bytes_in_shard_gemini = 0
    gemini_f, openai_f, mapping_f = _open_shard_files(out_dir, shard_idx, want_openai=want_openai, want_gemini=want_gemini)

    pages_seen = 0
    reqs_written = 0
    skipped_empty = 0

    for page_path in pages:
        pages_seen += 1
        if max_pages is not None and pages_seen > max_pages:
            break

        page = load_page_result(page_path)
        page_id = page.page_id
        ok_boxes = _iter_ok_boxes(page)

        def rotate_if_needed(next_openai_line: str | None, next_gemini_line: str | None) -> None:
            nonlocal shard_idx, reqs_in_shard, bytes_in_shard_openai, bytes_in_shard_gemini, gemini_f, openai_f, mapping_f
            if reqs_in_shard <= 0:
                return
            if reqs_in_shard >= int(args.requests_per_shard):
                pass
            else:
                limit = int(args.max_bytes_per_shard)
                if limit > 0:
                    if next_openai_line and (bytes_in_shard_openai + len(next_openai_line.encode("utf-8")) > limit):
                        pass
                    elif next_gemini_line and (bytes_in_shard_gemini + len(next_gemini_line.encode("utf-8")) > limit):
                        pass
                    else:
                        return
                else:
                    return

            _close_files([gemini_f, openai_f, mapping_f])
            shard_idx += 1
            reqs_in_shard = 0
            bytes_in_shard_openai = 0
            bytes_in_shard_gemini = 0
            gemini_f, openai_f, mapping_f = _open_shard_files(
                out_dir, shard_idx, want_openai=want_openai, want_gemini=want_gemini
            )

        def write_one(*, key: str, prompt: str, mapping_line: dict[str, Any]) -> None:
            nonlocal reqs_in_shard, bytes_in_shard_openai, bytes_in_shard_gemini, reqs_written

            mapping_line_s = json.dumps(mapping_line, ensure_ascii=False) + "\n"

            gemini_line_s = None
            if gemini_f:
                gemini_line_s = json.dumps(
                    {
                        "key": key,
                        "request": {
                            "contents": [
                                {"role": "user", "parts": [{"text": prompt}]},
                            ]
                        },
                    },
                    ensure_ascii=False,
                ) + "\n"

            openai_line_s = None
            if openai_f:
                body: dict[str, Any] = {
                    "model": args.openai_model,
                    "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
                }
                if args.openai_max_output_tokens is not None and int(args.openai_max_output_tokens) > 0:
                    body["max_output_tokens"] = int(args.openai_max_output_tokens)
                text_cfg = _openai_text_config(args.openai_text_format)
                if text_cfg is not None:
                    body["text"] = text_cfg
                if args.openai_reasoning_effort and args.openai_reasoning_effort != "none":
                    body["reasoning"] = {"effort": args.openai_reasoning_effort}
                openai_line_s = json.dumps(
                    {
                        "custom_id": key,
                        "method": "POST",
                        "url": "/v1/responses",
                        "body": body,
                    },
                    ensure_ascii=False,
                ) + "\n"

            rotate_if_needed(openai_line_s, gemini_line_s)

            mapping_f.write(mapping_line_s)
            if gemini_f and gemini_line_s:
                gemini_f.write(gemini_line_s)
                bytes_in_shard_gemini += len(gemini_line_s.encode("utf-8"))
            if openai_f and openai_line_s:
                openai_f.write(openai_line_s)
                bytes_in_shard_openai += len(openai_line_s.encode("utf-8"))

            reqs_in_shard += 1
            reqs_written += 1

        if mode == "page":
            page_text = page_text_from_boxes(page)
            if not page_text:
                if args.skip_empty:
                    skipped_empty += 1
                    continue
                raise SystemExit(f"Empty page_text (no ok transcripts) for: {page_path}")

            prompt = _build_page_prompt(prompt_text, page_text=page_text)
            mapping_line = {
                "id": page_id,
                "page_id": page_id,
                "page_path": str(page_path),
                "source_model": page.model,
                "mode": "page",
                "prompt_path": str(prompt_path),
                "page_text_chars": len(page_text),
                "page_text_sha256": _sha256(page_text),
            }
            write_one(key=page_id, prompt=prompt, mapping_line=mapping_line)
            continue

        if mode in {"boxes", "page+box_ids"}:
            # Pre-filter
            ok_boxes = [b for b in ok_boxes if len(b["transcript"]) >= int(args.min_chars)]
            if not ok_boxes:
                if args.skip_empty:
                    skipped_empty += 1
                    continue
                raise SystemExit(f"No ok boxes with text after filtering for: {page_path}")

            if args.box_selection == "reading":
                if int(args.max_boxes) <= 0:
                    selected = ok_boxes
                else:
                    selected = ok_boxes[: int(args.max_boxes)]
            elif args.box_selection == "longest":
                ordered = sorted(ok_boxes, key=lambda b: len(b["transcript"]), reverse=True)
                if int(args.max_boxes) <= 0:
                    selected = ordered
                else:
                    selected = ordered[: int(args.max_boxes)]
            else:
                raise AssertionError(f"Unhandled box_selection: {args.box_selection}")

            if mode == "boxes":
                effective_prompt_text = _with_strict_box_terms(prompt_text) if args.strict_box_zoning_terms else prompt_text
                for b in selected:
                    key = f"{page_id}:{b['box_id']}"
                    prompt = _build_page_prompt(effective_prompt_text, page_text=b["transcript"])
                    mapping_line = {
                        "id": key,
                        "page_id": page_id,
                        "page_path": str(page_path),
                        "source_model": page.model,
                        "mode": "boxes",
                        "box_id": b["box_id"],
                        "cls": b.get("cls"),
                        "bbox": b.get("bbox"),
                        "prompt_path": str(prompt_path),
                        "strict_box_zoning_terms": bool(args.strict_box_zoning_terms),
                        "box_text_chars": len(b["transcript"]),
                        "box_text_sha256": _sha256(b["transcript"]),
                    }
                    write_one(key=key, prompt=prompt, mapping_line=mapping_line)
                continue

            if mode == "page+box_ids":
                effective_prompt_text = _with_strict_box_terms(prompt_text) if args.strict_box_zoning_terms else prompt_text
                prompt = _build_box_ids_prompt(effective_prompt_text, boxes=selected)
                mapping_line = {
                    "id": page_id,
                    "page_id": page_id,
                    "page_path": str(page_path),
                    "source_model": page.model,
                    "mode": "page+box_ids",
                    "prompt_path": str(prompt_path),
                    "strict_box_zoning_terms": bool(args.strict_box_zoning_terms),
                    "boxes_included": [b["box_id"] for b in selected],
                    "box_count_included": len(selected),
                }
                write_one(key=page_id, prompt=prompt, mapping_line=mapping_line)
                continue

        raise AssertionError(f"Unhandled mode: {mode}")

    _close_files([gemini_f, openai_f, mapping_f])

    print(
        "Done. "
        f"pages_seen={pages_seen} reqs_written={reqs_written} skipped_empty={skipped_empty} "
        f"out_dir={out_dir}"
    )


if __name__ == "__main__":
    main()
