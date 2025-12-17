#!/usr/bin/env python3
"""
Small experiment harness: zoning classifier (page-level vs per-box) using OpenAI Responses API.

This is intentionally NOT a full batch pipeline. It's meant to help decide:
  - page-level classification only
  - per-box classification (one call per box)
  - a hybrid "page + identify boxes" prompt variant

It uses the existing prompt text file:
  prompts/zoning_ocr_classifier_prompt_text.txt

Examples:
  # Page-level classification
  python scripts/experiment_zoning_classifier_openai.py \
    --page newspaper-parsing-local/data/vlm_out_manifest_openai/albert-lea-evening-tribune-jul-01-1966-p-14.vlm.json \
    --mode page

  # Classify a few boxes (largest transcripts) with the SAME prompt
  python scripts/experiment_zoning_classifier_openai.py \
    --page newspaper-parsing-local/data/vlm_out_manifest_openai/albert-lea-evening-tribune-jul-01-1966-p-14.vlm.json \
    --mode boxes --max-boxes 5 --box-selection longest

  # One call: classify page + return box ids likely containing zoning text
  python scripts/experiment_zoning_classifier_openai.py \
    --page newspaper-parsing-local/data/vlm_out_manifest_openai/albert-lea-evening-tribune-jul-01-1966-p-14.vlm.json \
    --mode page+box_ids --max-boxes 40

Auth:
  - Prefers OPENAI_API_KEY / OPENAI_KEY from env
  - Otherwise tries to read OPENAI_KEY from --env-file (default: .env at repo root)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from openai import OpenAI

import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from newsvlm.pipeline import _coerce_json  # noqa: E402
from newsvlm.zoning_classifier import (  # noqa: E402
    ZoningClassifierOutput,
    load_page_result,
    load_prompt_text,
    page_text_from_boxes,
)


Provider = Literal["openai", "gemini"]
Mode = Literal["page", "boxes", "page+box_ids"]
BoxSelection = Literal["reading", "longest"]
TextFormat = Literal["json_schema", "json_object", "none"]


@dataclass(frozen=True)
class BoxForClassification:
    box_id: int
    transcript: str
    bbox: dict[str, int] | None


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Experiment: zoning classifier (page vs boxes) using OpenAI.")
    ap.add_argument("--page", required=True, help="Path to a per-page *.vlm.json file")
    ap.add_argument(
        "--prompt-path",
        default="prompts/zoning_ocr_classifier_prompt_text.txt",
        help="Prompt text file (page-level zoning classifier)",
    )
    ap.add_argument(
        "--provider",
        choices=["openai", "gemini"],
        default="openai",
        help="Which API to call (OpenAI Responses API or Gemini generate_content).",
    )
    ap.add_argument(
        "--mode",
        choices=["page", "boxes", "page+box_ids"],
        default="page",
        help="Experiment mode",
    )
    ap.add_argument(
        "--model",
        default=None,
        help="Model name. Default: gpt-5.2 (OpenAI) or gemini-2.5-flash (Gemini).",
    )
    ap.add_argument(
        "--reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high"],
        default="minimal",
        help="OpenAI reasoning.effort for Responses API (set 'none' to omit)",
    )
    ap.add_argument(
        "--text-format",
        choices=["json_schema", "json_object", "none"],
        default="json_schema",
        help="Ask the provider to enforce JSON output (OpenAI supports json_schema/json_object; Gemini uses response_mime_type).",
    )
    ap.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    ap.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help=(
            "Upper bound on model output tokens (OpenAI: max_output_tokens). "
            "If omitted (recommended), we do not set it to avoid `status=incomplete` due to reasoning-token limits."
        ),
    )
    ap.add_argument(
        "--max-boxes",
        type=int,
        default=8,
        help="When --mode uses boxes, classify at most this many boxes",
    )
    ap.add_argument(
        "--box-selection",
        choices=["reading", "longest"],
        default="longest",
        help="How to pick boxes when not specifying --box-ids",
    )
    ap.add_argument(
        "--box-ids",
        default="",
        help="Comma-separated box ids to classify (overrides --box-selection). Example: '3,12,99'",
    )
    ap.add_argument(
        "--min-chars",
        type=int,
        default=40,
        help="Skip boxes whose transcript is shorter than this (after stripping).",
    )
    ap.add_argument(
        "--strict-box-zoning-terms",
        action="store_true",
        help=(
            "EXPERIMENTAL: when classifying individual boxes (or identifying box ids), "
            "require explicit zoning terms (zoning/rezone/rezoning/variance/etc). "
            "This reduces false positives on generic ordinances/hearings."
        ),
    )
    ap.add_argument(
        "--env-file",
        default=".env",
        help="Env file to read OPENAI_KEY from if not present in environment",
    )
    ap.add_argument("--dry-run", action="store_true", help="Do not call the API; just print what would be sent")
    return ap.parse_args()


def _load_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        key = k.strip()
        val = v.strip().strip('"').strip("'")
        if key:
            env[key] = val
    return env


def _get_openai_key(env_file: Path) -> str:
    key = os.environ.get("PROJECT_OPENAI_KEY") or os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_KEY")
    if key:
        return key

    env = _load_env_file(env_file)
    key = env.get("PROJECT_OPENAI_KEY") or env.get("OPENAI_API_KEY") or env.get("OPENAI_KEY")
    if key:
        return key

    raise SystemExit(
        "Missing OpenAI API key.\n"
        "Set PROJECT_OPENAI_KEY (preferred) or OPENAI_API_KEY/OPENAI_KEY in your environment, "
        "or add PROJECT_OPENAI_KEY=... to --env-file."
    )


def _get_gemini_key(env_file: Path) -> str:
    key = os.environ.get("GEMINI_KEY")
    if key:
        return key
    env = _load_env_file(env_file)
    key = env.get("GEMINI_KEY")
    if key:
        return key
    raise SystemExit(
        "Missing Gemini API key.\n"
        "Set GEMINI_KEY in your environment, or add GEMINI_KEY=... to --env-file."
    )


def _box_sort_key(b) -> tuple[int, int, int]:
    bbox = getattr(b, "bbox", None) or {}
    return (int(bbox.get("y0", 0)), int(bbox.get("x0", 0)), int(getattr(b, "id", 0)))


def _collect_ok_boxes(page) -> list[BoxForClassification]:
    boxes = sorted(page.boxes, key=_box_sort_key)
    out: list[BoxForClassification] = []
    for b in boxes:
        if getattr(b, "status", None) != "ok":
            continue
        t = (getattr(b, "transcript", None) or "").strip()
        if not t:
            continue
        out.append(BoxForClassification(box_id=int(getattr(b, "id", -1)), transcript=t, bbox=getattr(b, "bbox", None)))
    return out


def _parse_box_ids(spec: str) -> list[int]:
    spec = spec.strip()
    if not spec:
        return []
    out: list[int] = []
    for part in spec.split(","):
        token = part.strip()
        if not token:
            continue
        out.append(int(token))
    # preserve order but dedupe
    seen: set[int] = set()
    uniq: list[int] = []
    for i in out:
        if i in seen:
            continue
        seen.add(i)
        uniq.append(i)
    return uniq


def _build_page_prompt(prompt_text: str, *, page_text: str) -> str:
    # Keep the input as "OCR-extracted TEXT only" as the prompt requests.
    return f"{prompt_text}\n\n{page_text}\n"


def _with_strict_box_terms(prompt_text: str) -> str:
    return (
        f"{prompt_text}\n\n"
        "EXTRA CONSTRAINT (box-level ONLY): This constraint applies ONLY when you are judging an individual box "
        "or when selecting 'boxes_with_zoning'. It does NOT change the page-level decision rules above.\n"
        "- Only treat a box as zoning-related (or include it in boxes_with_zoning) if that box's text explicitly "
        "mentions zoning/rezoning/rezone/zoning district/zoning map/variance/conditional use/"
        "board of zoning appeals/planning commission (or similarly unmistakable zoning-language).\n"
        "- If a box is a generic ordinance, street vacation, ditch hearing, road work notice, etc. and it does not "
        "clearly mention zoning terms, do NOT include it in boxes_with_zoning; and if classifying that box alone, "
        "return 'unrelated'."
    )


def _build_box_ids_prompt(
    prompt_text: str,
    *,
    boxes: list[BoxForClassification],
    strict_box_terms: bool,
) -> str:
    # This is a deliberate prompt *variant* for experimentation:
    # we add minimal structure (box headers) so the model can point to box ids.
    box_block_lines: list[str] = []
    for b in boxes:
        box_block_lines.append(f"BOX {b.box_id}:")
        box_block_lines.append(b.transcript)
        box_block_lines.append("")  # blank line separator

    box_block = "\n".join(box_block_lines).strip() + "\n"

    strict_lines = ""
    if strict_box_terms:
        strict_lines = (
            "\nWhen producing boxes_with_zoning:\n"
            "- Only include a box id if that box's text explicitly mentions zoning/rezoning/rezone/"
            "zoning district/zoning map/variance/conditional use/board of zoning appeals/planning commission "
            "(or similarly unmistakable zoning-language).\n"
            "- Do NOT include boxes that are generic ordinances (e.g., street vacations), ditch/road hearings, "
            "or other municipal notices without explicit zoning language.\n"
        )

    return (
        f"{prompt_text}\n\n"
        "The OCR text below is split into bounding boxes from the same page.\n"
        "Each box starts with a line like 'BOX <id>:' followed by the OCR transcript for that box.\n\n"
        "Do the same zoning-presence classification for the FULL PAGE by considering all boxes.\n"
        "Additionally, identify which box ids contain qualifying zoning ordinance/amendment/hearing text.\n\n"
        f"{strict_lines}\n"
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


def _openai_text_config(fmt: TextFormat, *, schema: dict[str, Any] | None) -> dict[str, Any] | None:
    if fmt == "none":
        return None
    if fmt == "json_object":
        return {"format": {"type": "json_object"}}
    if fmt == "json_schema":
        if schema is None:
            raise ValueError("json_schema selected but schema is None")
        return {
            "format": {
                "type": "json_schema",
                "name": "zoning_classifier_output",
                "schema": schema,
                "strict": True,
            }
        }
    raise ValueError(f"Unknown text format: {fmt}")


def _zoning_classifier_openai_json_schema() -> dict[str, Any]:
    # OpenAI's strict JSON-schema mode requires `additionalProperties: false` for objects.
    # Building this manually avoids surprises from Pydantic's default JSON schema output.
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


def _page_with_box_ids_openai_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "page": _zoning_classifier_openai_json_schema(),
            "boxes_with_zoning": {"type": "array", "items": {"type": "integer"}},
        },
        "required": ["page", "boxes_with_zoning"],
    }


def _call_openai(
    *,
    client: OpenAI,
    model: str,
    reasoning_effort: str,
    text_format: TextFormat,
    prompt: str,
    temperature: float,
    max_output_tokens: int | None,
    schema: dict[str, Any] | None,
) -> tuple[str, dict[str, Any]]:
    kwargs: dict[str, Any] = {
        "model": model,
        "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
    }
    if max_output_tokens is not None and int(max_output_tokens) > 0:
        kwargs["max_output_tokens"] = int(max_output_tokens)
    # Some models reject `temperature` entirely; only send it if non-None.
    if temperature is not None:
        kwargs["temperature"] = float(temperature)
    if reasoning_effort and reasoning_effort != "none":
        kwargs["reasoning"] = {"effort": reasoning_effort}

    text_cfg = _openai_text_config(text_format, schema=schema)
    if text_cfg is not None:
        kwargs["text"] = text_cfg

    resp = None
    for attempt in range(2):
        try:
            resp = client.responses.create(**kwargs)
            break
        except Exception as exc:
            msg = str(exc)
            if "insufficient_quota" in msg:
                raise SystemExit(
                    "OpenAI request failed with `insufficient_quota` (HTTP 429).\n"
                    "This means the API key being used does not currently have billable quota in its project/org.\n"
                    "Fix: set a funded key via OPENAI_API_KEY (preferred) or OPENAI_KEY, or update --env-file."
                )
            # Some models reject `temperature` entirely; retry once without it.
            if attempt == 0 and "Unsupported parameter: 'temperature'" in msg and "temperature" in kwargs:
                kwargs.pop("temperature", None)
                continue
            raise
    if resp is None:
        raise RuntimeError("OpenAI call did not return a response (unexpected).")
    raw_text = resp.output_text
    if not raw_text or not raw_text.strip():
        output_types = []
        try:
            output_types = [getattr(o, "type", None) for o in (resp.output or [])]
        except Exception:
            output_types = []
        status = getattr(resp, "status", None)
        incomplete = getattr(resp, "incomplete_details", None)
        raise SystemExit(
            "OpenAI returned an empty output text.\n"
            f"status={status} output_types={output_types} incomplete_details={incomplete}\n"
            "If you set --max-output-tokens, it may have been too small relative to the model's reasoning budget.\n"
            "Try omitting --max-output-tokens entirely, increasing it, and/or set --reasoning-effort minimal."
        )
    parsed = _coerce_json(raw_text)
    return raw_text, parsed


def _call_gemini(
    *,
    api_key: str,
    model: str,
    prompt: str,
    temperature: float,
    max_output_tokens: int,
    want_json: bool,
    json_schema: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    from google import genai

    client = genai.Client(api_key=api_key)
    config: dict[str, Any] = {
        "temperature": float(temperature),
        "max_output_tokens": int(max_output_tokens),
        # Avoid Gemini spending most of the output budget on hidden "thoughts",
        # which can truncate the visible JSON response.
        "thinking_config": {"thinking_budget": 0},
    }
    if want_json:
        # Best-effort hint; prompt still says "Output JSON only".
        config["response_mime_type"] = "application/json"
        if json_schema is not None:
            config["response_json_schema"] = json_schema

    resp = client.models.generate_content(model=model, contents=prompt, config=config)
    raw_text = getattr(resp, "text", None) or ""
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise RuntimeError("Gemini returned empty response text")
    parsed = _coerce_json(raw_text)
    return raw_text, parsed


def main() -> None:
    args = _parse_args()
    page_path = Path(args.page).expanduser()
    prompt_path = Path(args.prompt_path).expanduser()
    env_file = Path(args.env_file).expanduser()

    if not page_path.is_file():
        raise SystemExit(f"--page not found: {page_path}")
    if not prompt_path.is_file():
        raise SystemExit(f"--prompt-path not found: {prompt_path}")

    page = load_page_result(page_path)
    prompt_text = load_prompt_text(prompt_path)
    ok_boxes_all = _collect_ok_boxes(page)

    provider: Provider = args.provider
    model = args.model
    if not model:
        model = "gpt-5.2" if provider == "openai" else "gemini-2.5-flash"

    print(f"page_id={page.page_id}")
    print(f"page_model={page.model}")
    print(f"boxes_total={len(page.boxes)} boxes_ok_with_text={len(ok_boxes_all)}")
    print(f"classifier_provider={provider} classifier_model={model}")
    if provider == "openai":
        print(f"reasoning_effort={args.reasoning_effort} text_format={args.text_format}")
    else:
        print(f"text_format={args.text_format} (gemini uses response_mime_type when enabled)")
    print(f"prompt_path={prompt_path}")

    if args.dry_run:
        print("DRY_RUN=1 (no API calls)")
        return

    openai_client: OpenAI | None = None
    openai_key: str | None = None
    gemini_key: str | None = None

    if provider == "openai":
        openai_key = _get_openai_key(env_file)
        openai_client = OpenAI(api_key=openai_key)
    else:
        gemini_key = _get_gemini_key(env_file)

    mode: Mode = args.mode
    started = time.time()

    if mode == "page":
        page_text = page_text_from_boxes(page)
        prompt = _build_page_prompt(prompt_text, page_text=page_text)
        if provider == "openai":
            schema = _zoning_classifier_openai_json_schema() if args.text_format == "json_schema" else None
            raw_text, parsed = _call_openai(
                client=openai_client,  # type: ignore[arg-type]
                model=model,
                reasoning_effort=args.reasoning_effort,
                text_format=args.text_format,
                prompt=prompt,
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
                schema=schema,
            )
        else:
            schema = _zoning_classifier_openai_json_schema() if args.text_format == "json_schema" else None
            raw_text, parsed = _call_gemini(
                api_key=gemini_key or "",
                model=model,
                prompt=prompt,
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
                want_json=args.text_format != "none",
                json_schema=schema,
            )
        out = ZoningClassifierOutput.model_validate(parsed)
        print(json.dumps(out.model_dump(mode="json"), indent=2, ensure_ascii=False))
        print(f"elapsed_s={time.time() - started:.2f}")
        return

    if mode == "boxes":
        want_ids = set(_parse_box_ids(args.box_ids))

        filtered = [b for b in ok_boxes_all if len(b.transcript) >= int(args.min_chars)]
        if want_ids:
            by_id = {b.box_id: b for b in filtered}
            selected = []
            missing = []
            for i in _parse_box_ids(args.box_ids):
                b = by_id.get(i)
                if b is None:
                    missing.append(i)
                else:
                    selected.append(b)
            if missing:
                raise SystemExit(f"--box-ids referenced ids not found (or filtered by --min-chars): {missing}")
        else:
            if args.box_selection == "reading":
                selected = filtered[: int(args.max_boxes)]
            elif args.box_selection == "longest":
                selected = sorted(filtered, key=lambda b: len(b.transcript), reverse=True)[: int(args.max_boxes)]
            else:
                raise AssertionError(f"Unhandled --box-selection: {args.box_selection}")

        if not selected:
            raise SystemExit("No boxes selected for classification (try lowering --min-chars).")

        schema = _zoning_classifier_openai_json_schema() if args.text_format == "json_schema" else None
        for idx, b in enumerate(selected, start=1):
            effective_prompt_text = _with_strict_box_terms(prompt_text) if args.strict_box_zoning_terms else prompt_text
            prompt = _build_page_prompt(effective_prompt_text, page_text=b.transcript)
            if provider == "openai":
                raw_text, parsed = _call_openai(
                    client=openai_client,  # type: ignore[arg-type]
                    model=model,
                    reasoning_effort=args.reasoning_effort,
                    text_format=args.text_format,
                    prompt=prompt,
                    temperature=args.temperature,
                    max_output_tokens=args.max_output_tokens,
                    schema=schema,
                )
            else:
                raw_text, parsed = _call_gemini(
                    api_key=gemini_key or "",
                    model=model,
                    prompt=prompt,
                    temperature=args.temperature,
                    max_output_tokens=args.max_output_tokens,
                    want_json=args.text_format != "none",
                    json_schema=schema if args.text_format == "json_schema" else None,
                )
            out = ZoningClassifierOutput.model_validate(parsed)
            print(f"\nBOX_RESULT {idx}/{len(selected)} box_id={b.box_id} chars={len(b.transcript)}")
            print(json.dumps(out.model_dump(mode="json"), indent=2, ensure_ascii=False))

        print(f"\nelapsed_s={time.time() - started:.2f}")
        return

    if mode == "page+box_ids":
        # Limit the boxes we send to keep the prompt bounded.
        filtered = [b for b in ok_boxes_all if len(b.transcript) >= int(args.min_chars)]
        if args.box_selection == "reading":
            selected = filtered[: int(args.max_boxes)]
        elif args.box_selection == "longest":
            selected = sorted(filtered, key=lambda b: len(b.transcript), reverse=True)[: int(args.max_boxes)]
        else:
            raise AssertionError(f"Unhandled --box-selection: {args.box_selection}")

        if not selected:
            raise SystemExit("No boxes selected for page+box_ids (try lowering --min-chars).")

        prompt = _build_box_ids_prompt(prompt_text, boxes=selected, strict_box_terms=args.strict_box_zoning_terms)
        if provider == "openai":
            # Unlike the single-page classifier, this output schema includes both the page-level
            # classification and a list of box ids. Enforce it with json_schema when requested.
            if args.text_format == "json_schema":
                tf: TextFormat = "json_schema"
                schema = _page_with_box_ids_openai_json_schema()
            elif args.text_format == "none":
                tf = "none"
                schema = None
            else:
                tf = "json_object"
                schema = None

            raw_text, parsed = _call_openai(
                client=openai_client,  # type: ignore[arg-type]
                model=model,
                reasoning_effort=args.reasoning_effort,
                text_format=tf,
                prompt=prompt,
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
                schema=schema,
            )
        else:
            schema = _zoning_classifier_openai_json_schema() if args.text_format == "json_schema" else None
            raw_text, parsed = _call_gemini(
                api_key=gemini_key or "",
                model=model,
                prompt=prompt,
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
                want_json=args.text_format != "none",
                json_schema=schema,
            )

        page_obj = parsed.get("page")
        if not isinstance(page_obj, dict):
            raise SystemExit("Expected JSON output to include object field 'page'")
        page_out = ZoningClassifierOutput.model_validate(page_obj)

        box_ids = parsed.get("boxes_with_zoning")
        if not isinstance(box_ids, list) or not all(isinstance(i, int) for i in box_ids):
            preview = json.dumps(parsed, ensure_ascii=False)[:500]
            raise SystemExit(
                "Expected JSON output field 'boxes_with_zoning' to be a list of integers.\n"
                f"Got: {type(box_ids)}\n"
                f"parsed_preview={preview}"
            )

        print("PAGE_CLASSIFICATION:")
        print(json.dumps(page_out.model_dump(mode="json"), indent=2, ensure_ascii=False))
        print(f"boxes_with_zoning={sorted(set(box_ids))}")
        print(f"elapsed_s={time.time() - started:.2f}")
        return

    raise AssertionError(f"Unhandled mode: {mode}")


if __name__ == "__main__":
    main()
