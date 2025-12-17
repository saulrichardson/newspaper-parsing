#!/usr/bin/env python3
"""
Export per-box batch request files for Gemini and/or OpenAI.

This script mirrors the synchronous VLM pipeline:
  - loads Dell layout JSONs
  - crops text bboxes from local PNGs
  - applies the same crop downscaling guards
  - builds one request per bbox with class-aware prompt

Outputs:
  - <provider>_requests_shardNNN.jsonl: provider-specific batch inputs
  - mapping_shardNNN.jsonl: provenance mapping to rehydrate per-page outputs

Run from repo root:
  python scripts/export_batch_requests.py \
    --layouts "newspaper-parsing-local/data/unique_outputs_dedup/*.json" \
    --png-root newspaper-parsing-local/data/unique_png \
    --output-dir newspaper-parsing-local/data/batch_requests \
    --provider both
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import sys
from glob import glob
from io import BytesIO
from os.path import expanduser
from pathlib import Path
from typing import IO, Iterable

from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from newsvlm.pipeline import build_prompt, load_layout  # noqa: E402


JSON_INSTRUCTION = (
    ' Respond ONLY with JSON: {"status":"ok"|"unreadable","transcript":string}. '
    'If unreadable, set transcript to "".'
)


def collect_layout_paths(spec: str) -> list[Path]:
    """Collect layout JSON paths from a glob or @file list.

    Mirrors scripts/run_vlm_gateway.py:
    - "@file": one JSON path per line.
    - otherwise: glob pattern (absolute OK).
    """
    spec = spec.strip()
    if spec.startswith("@"):
        list_path = Path(expanduser(spec[1:]))
        if not list_path.is_file():
            raise SystemExit(f"Layouts file not found: {list_path}")
        paths: list[Path] = []
        for raw in list_path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            p = Path(expanduser(line))
            if not p.is_file():
                raise SystemExit(f"Layout JSON listed but not found: {p}")
            paths.append(p)
        return sorted(paths)

    expanded = expanduser(spec)
    return sorted(Path(p) for p in glob(expanded))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export batch JSONL requests from layouts + PNGs.")
    p.add_argument(
        "--layouts",
        required=True,
        help='Glob for Dell layout JSONs (absolute OK) or "@file" listing JSON paths',
    )
    p.add_argument("--png-root", default=None, help="Optional directory to find PNGs by basename")
    p.add_argument("--output-dir", required=True, help="Directory to write batch shards")
    p.add_argument(
        "--provider",
        choices=["gemini", "openai", "both"],
        default="both",
        help="Which provider batch files to emit",
    )
    p.add_argument(
        "--openai-model",
        default="gpt-4o-mini",
        help="OpenAI model name for /v1/responses batch body",
    )
    p.add_argument(
        "--openai-reasoning-effort",
        choices=["none", "low", "medium", "high", "xhigh"],
        default=None,
        help=(
            "Optional OpenAI reasoning.effort value for Responses API. "
            "If set, adds `reasoning: {effort: ...}` to the OpenAI request body."
        ),
    )
    p.add_argument(
        "--image-format",
        choices=["png", "jpeg"],
        default="png",
        help="Image encoding for each bbox crop (jpeg is usually much smaller than png).",
    )
    p.add_argument(
        "--jpeg-quality",
        type=int,
        default=85,
        help="JPEG quality (1-100) when --image-format=jpeg",
    )
    p.add_argument(
        "--boxes-per-shard",
        type=int,
        default=5000,
        help="Number of bbox requests per shard (tune for batch file size limits)",
    )
    p.add_argument(
        "--max-bytes-per-shard",
        type=int,
        default=0,
        help=(
            "Max bytes per request shard file (applies to gemini_requests_shardNNN.jsonl and/or "
            "openai_requests_shardNNN.jsonl depending on --provider). "
            "0 disables byte-based splitting. Recommended for OpenAI: ~180_000_000."
        ),
    )
    p.add_argument(
        "--max-crop-megapixels",
        type=float,
        default=3.0,
        help="Downscale crops larger than this many megapixels (0 to disable)",
    )
    p.add_argument(
        "--max-crop-dim",
        type=int,
        default=2048,
        help="Downscale crops whose longest edge exceeds this (0 to disable)",
    )
    p.add_argument(
        "--save-crops-dir",
        default=None,
        help="Optional directory to save scaled bbox crops for inspection",
    )
    p.add_argument(
        "--skip-bad-layouts",
        action="store_true",
        help="Skip layouts/boxes that error instead of aborting the export",
    )
    return p.parse_args()


def _mime_type_for_format(fmt: str) -> str:
    if fmt == "png":
        return "image/png"
    if fmt == "jpeg":
        return "image/jpeg"
    raise ValueError(f"Unsupported image format: {fmt}")


def _ext_for_format(fmt: str) -> str:
    if fmt == "png":
        return "png"
    if fmt == "jpeg":
        return "jpg"
    raise ValueError(f"Unsupported image format: {fmt}")


def _crop_to_bytes(
    img: Image.Image,
    xyxy: Iterable[int],
    *,
    image_format: str,
    jpeg_quality: int,
    max_megapixels: float | None,
    max_dim: int | None,
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
    if image_format == "png":
        crop.save(buf, format="PNG", optimize=True)
    elif image_format == "jpeg":
        if not (1 <= int(jpeg_quality) <= 100):
            raise ValueError("--jpeg-quality must be in [1, 100]")
        crop = crop.convert("RGB")
        crop.save(buf, format="JPEG", quality=int(jpeg_quality), optimize=True)
    else:
        raise ValueError(f"Unsupported image format: {image_format}")

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
    xyxy: Iterable[int],
    crops_dir: Path | None,
    *,
    page_id: str,
    box_id: int,
    box_cls: str,
    image_format: str,
    jpeg_quality: int,
    max_megapixels: float | None,
    max_dim: int | None,
) -> tuple[bytes, dict]:
    data, meta = _crop_to_bytes(
        img,
        xyxy,
        image_format=image_format,
        jpeg_quality=jpeg_quality,
        max_megapixels=max_megapixels,
        max_dim=max_dim,
    )
    if crops_dir:
        crops_dir.mkdir(parents=True, exist_ok=True)
        ext = _ext_for_format(image_format)
        fname = f"{page_id}_{box_id}_{box_cls}.{ext}"
        (crops_dir / fname).write_bytes(data)
    return data, meta


def _open_shard_files(
    out_dir: Path,
    shard_idx: int,
    *,
    want_gemini: bool,
    want_openai: bool,
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


def main() -> None:
    args = parse_args()
    layout_paths = collect_layout_paths(args.layouts)
    if not layout_paths:
        raise SystemExit(f"No layout JSONs matched pattern: {args.layouts}")

    if args.image_format == "jpeg" and not (1 <= int(args.jpeg_quality) <= 100):
        raise SystemExit("--jpeg-quality must be between 1 and 100")
    if args.max_bytes_per_shard < 0:
        raise SystemExit("--max-bytes-per-shard must be >= 0")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    png_root = Path(args.png_root) if args.png_root else None
    crops_dir = Path(args.save_crops_dir) if args.save_crops_dir else None

    want_gemini = args.provider in {"gemini", "both"}
    want_openai = args.provider in {"openai", "both"}
    mime_type = _mime_type_for_format(args.image_format)

    shard_idx = 0
    boxes_in_shard = 0
    bytes_in_shard_gemini = 0
    bytes_in_shard_openai = 0
    gemini_f, openai_f, mapping_f = _open_shard_files(
        out_dir, shard_idx, want_gemini=want_gemini, want_openai=want_openai
    )

    pages_ok = 0
    pages_bad = 0
    boxes_ok = 0
    boxes_bad = 0

    for lp in layout_paths:
        try:
            layout_path, png_path, boxes = load_layout(lp, png_root=png_root)
        except Exception as exc:
            pages_bad += 1
            msg = f"[layout error] {lp}: {exc}"
            if args.skip_bad_layouts:
                print(msg, file=sys.stderr)
                continue
            raise

        page_id = png_path.stem
        try:
            with Image.open(png_path) as _im:
                page_img = _im.convert("RGB")
        except Exception as exc:
            pages_bad += 1
            msg = f"[image error] {png_path} (layout {lp}): {exc}"
            if args.skip_bad_layouts:
                print(msg, file=sys.stderr)
                continue
            raise

        pages_ok += 1

        for b in boxes:
            try:
                crop_bytes, crop_meta = _maybe_save_crop(
                    page_img,
                    b.xyxy,
                    crops_dir,
                    page_id=page_id,
                    box_id=b.id,
                    box_cls=b.cls,
                    image_format=args.image_format,
                    jpeg_quality=int(args.jpeg_quality),
                    max_megapixels=args.max_crop_megapixels if args.max_crop_megapixels > 0 else None,
                    max_dim=args.max_crop_dim if args.max_crop_dim > 0 else None,
                )
            except Exception as exc:
                boxes_bad += 1
                msg = f"[crop error] {lp} box {getattr(b, 'id', '?')}: {exc}"
                if args.skip_bad_layouts:
                    print(msg, file=sys.stderr)
                    continue
                raise

            b64 = base64.b64encode(crop_bytes).decode("utf-8")
            prompt = build_prompt(b.cls) + JSON_INSTRUCTION
            key = f"{page_id}:{b.id}"

            mapping_line = {
                "id": key,
                "page_id": page_id,
                "layout_path": str(layout_path),
                "png_path": str(png_path),
                "box_id": b.id,
                "class": b.cls,
                "bbox": b.bbox,
                "legibility": b.legibility,
                "prompt": prompt,
                **crop_meta,
            }
            mapping_line_s = json.dumps(mapping_line) + "\n"

            gemini_line_s = None
            openai_line_s = None

            if gemini_f:
                gemini_line = {
                    "key": key,
                    "request": {
                        "contents": [
                            {
                                "role": "user",
                                "parts": [
                                    {"text": prompt},
                                    {"inline_data": {"mime_type": mime_type, "data": b64}},
                                ],
                            }
                        ]
                    },
                }
                gemini_line_s = json.dumps(gemini_line) + "\n"

            if openai_f:
                data_url = f"data:{mime_type};base64,{b64}"
                openai_body = {
                    "model": args.openai_model,
                    "input": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": prompt},
                                {"type": "input_image", "image_url": data_url},
                            ],
                        }
                    ],
                }
                if args.openai_reasoning_effort:
                    openai_body["reasoning"] = {"effort": args.openai_reasoning_effort}
                openai_line = {
                    "custom_id": key,
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": openai_body,
                }
                openai_line_s = json.dumps(openai_line) + "\n"

            # Split shards based on either:
            #   - line count (--boxes-per-shard)
            #   - byte size (--max-bytes-per-shard)
            #
            # For byte limits, we conservatively enforce the threshold separately per provider request file,
            # so both OpenAI and Gemini shards stay under the limit when --provider=both.
            max_bytes = int(args.max_bytes_per_shard) if int(args.max_bytes_per_shard) > 0 else None
            gemini_line_bytes = len(gemini_line_s.encode("utf-8")) if gemini_line_s else 0
            openai_line_bytes = len(openai_line_s.encode("utf-8")) if openai_line_s else 0

            if max_bytes:
                if gemini_line_s and gemini_line_bytes > max_bytes:
                    raise SystemExit(
                        f"Single Gemini request line exceeds --max-bytes-per-shard "
                        f"({gemini_line_bytes} > {max_bytes}) for key={key}"
                    )
                if openai_line_s and openai_line_bytes > max_bytes:
                    raise SystemExit(
                        f"Single OpenAI request line exceeds --max-bytes-per-shard "
                        f"({openai_line_bytes} > {max_bytes}) for key={key}"
                    )

            should_roll = False
            if boxes_in_shard >= args.boxes_per_shard:
                should_roll = True
            if max_bytes:
                if gemini_line_s and bytes_in_shard_gemini > 0 and (bytes_in_shard_gemini + gemini_line_bytes > max_bytes):
                    should_roll = True
                if openai_line_s and bytes_in_shard_openai > 0 and (bytes_in_shard_openai + openai_line_bytes > max_bytes):
                    should_roll = True

            if should_roll:
                _close_files([gemini_f, openai_f, mapping_f])
                shard_idx += 1
                boxes_in_shard = 0
                bytes_in_shard_gemini = 0
                bytes_in_shard_openai = 0
                gemini_f, openai_f, mapping_f = _open_shard_files(
                    out_dir, shard_idx, want_gemini=want_gemini, want_openai=want_openai
                )

            if gemini_f and gemini_line_s:
                gemini_f.write(gemini_line_s)
                bytes_in_shard_gemini += gemini_line_bytes

            if openai_f and openai_line_s:
                openai_f.write(openai_line_s)
                bytes_in_shard_openai += openai_line_bytes

            mapping_f.write(mapping_line_s)

            boxes_ok += 1
            boxes_in_shard += 1

        try:
            page_img.close()
        except Exception:
            pass

    _close_files([gemini_f, openai_f, mapping_f])

    print(
        f"Done. pages_ok={pages_ok} pages_bad={pages_bad} "
        f"boxes_ok={boxes_ok} boxes_bad={boxes_bad} shards={shard_idx + 1}"
    )
    print(f"Wrote shards to {out_dir}")


if __name__ == "__main__":
    main()
