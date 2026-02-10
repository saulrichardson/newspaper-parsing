#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image, ImageDraw
from mineru_vl_utils import MinerUClient
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run MinerU2.5 layout extraction over a manifest.")
    p.add_argument("--manifest", required=True)
    p.add_argument("--output_root", required=True)
    p.add_argument("--model_id", default="opendatalab/MinerU2.5-2509-1.2B")
    p.add_argument("--max_pages", type=int, default=0)
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip pages with existing *_mineru_layout_boxes.json and *_mineru_raw.json (regenerate overlay if missing).",
    )
    return p.parse_args()


def to_jsonable(x: Any) -> Any:
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    return str(x)


def clamp_box(box: List[float], w: int, h: int) -> Optional[List[float]]:
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = max(0.0, min(float(w), x1))
    x2 = max(0.0, min(float(w), x2))
    y1 = max(0.0, min(float(h), y1))
    y2 = max(0.0, min(float(h), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def maybe_normalized(box: List[float]) -> bool:
    return max(box) <= 1.5 and min(box) >= -0.1


def parse_bbox_obj(obj: Any, w: int, h: int) -> Optional[List[float]]:
    box = None
    if isinstance(obj, (list, tuple)) and len(obj) == 4 and all(
        isinstance(v, (int, float)) for v in obj
    ):
        box = [float(v) for v in obj]
    elif isinstance(obj, dict):
        if all(k in obj for k in ("x1", "y1", "x2", "y2")):
            box = [float(obj["x1"]), float(obj["y1"]), float(obj["x2"]), float(obj["y2"])]
        elif all(k in obj for k in ("x0", "y0", "x1", "y1")):
            box = [float(obj["x0"]), float(obj["y0"]), float(obj["x1"]), float(obj["y1"])]
        elif all(k in obj for k in ("left", "top", "right", "bottom")):
            box = [float(obj["left"]), float(obj["top"]), float(obj["right"]), float(obj["bottom"])]

    if box is None:
        return None

    if box[2] <= box[0] or box[3] <= box[1]:
        if box[2] > 0 and box[3] > 0:
            box = [box[0], box[1], box[0] + box[2], box[1] + box[3]]

    if maybe_normalized(box):
        box = [box[0] * w, box[1] * h, box[2] * w, box[3] * h]

    return clamp_box(box, w, h)


def extract_boxes(obj: Any, w: int, h: int, parent_label: Optional[str] = None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    def walk(x: Any, parent: Optional[str]) -> None:
        if isinstance(x, dict):
            label = (
                x.get("label")
                or x.get("type")
                or x.get("block_type")
                or x.get("category")
                or x.get("class")
                or x.get("name")
                or parent
                or "text"
            )
            score = x.get("score")
            if score is None:
                score = x.get("confidence")

            bbox = None
            for k in ("bbox_xyxy", "bbox", "box", "coordinate", "rect"):
                if k in x:
                    bbox = parse_bbox_obj(x[k], w, h)
                    if bbox:
                        break

            if bbox is None:
                for k in ("polygon", "poly", "points", "quad"):
                    if (
                        k in x
                        and isinstance(x[k], list)
                        and x[k]
                        and isinstance(x[k][0], (list, tuple))
                    ):
                        pts = x[k]
                        xs = [float(p[0]) for p in pts if len(p) >= 2]
                        ys = [float(p[1]) for p in pts if len(p) >= 2]
                        if xs and ys:
                            bbox = parse_bbox_obj([min(xs), min(ys), max(xs), max(ys)], w, h)
                            if bbox:
                                break

            if bbox is not None:
                out.append(
                    {
                        "source": "mineru2.5",
                        "label": str(label),
                        "bbox_xyxy": bbox,
                        "score": float(score) if isinstance(score, (int, float)) else None,
                        "reading_order": None,
                        "text": x.get("text") or x.get("content") or x.get("md") or x.get("markdown"),
                    }
                )

            for v in x.values():
                walk(v, str(label))
        elif isinstance(x, list):
            for y in x:
                walk(y, parent)

    walk(obj, parent_label)

    dedup: List[Dict[str, Any]] = []
    seen = set()
    for b in out:
        bb = b["bbox_xyxy"]
        key = (
            b["label"].lower(),
            round(bb[0] / 8),
            round(bb[1] / 8),
            round(bb[2] / 8),
            round(bb[3] / 8),
        )
        if key in seen:
            continue
        seen.add(key)
        dedup.append(b)
    return dedup


def class_color(label: str) -> Tuple[int, int, int]:
    x = (label or "").lower()
    if "table" in x:
        return (220, 140, 30)
    if "title" in x or "header" in x or "headline" in x:
        return (40, 80, 220)
    if "image" in x or "figure" in x or "photo" in x or "picture" in x:
        return (180, 40, 140)
    return (40, 160, 40)


def main() -> None:
    args = parse_args()
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    report_tsv = out_root / "run_report.tsv"

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else "auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model_id, use_fast=True)
    client = MinerUClient(backend="transformers", model=model, processor=processor)

    run_meta = {
        "model_id": args.model_id,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "hf_device_map": to_jsonable(getattr(model, "hf_device_map", None)),
    }
    (out_root / "run_meta.json").write_text(
        json.dumps(run_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        x.strip()
        for x in Path(args.manifest).read_text(encoding="utf-8").splitlines()
        if x.strip() and not x.strip().startswith("#")
    ]
    if args.max_pages and args.max_pages > 0:
        lines = lines[: args.max_pages]

    with report_tsv.open("w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f, delimiter="\t")
        wr.writerow(
            [
                "image",
                "status",
                "seconds",
                "page_dir",
                "overlay",
                "boxes_json",
                "raw_json",
                "n_boxes",
                "model_id",
            ]
        )

        for image_path in lines:
            t0 = time.time()
            img_path = Path(image_path)
            slug = img_path.stem
            page_dir = out_root / slug
            page_dir.mkdir(parents=True, exist_ok=True)

            overlay_path = page_dir / f"{slug}_mineru_overlay.png"
            boxes_json = page_dir / f"{slug}_mineru_layout_boxes.json"
            raw_json = page_dir / f"{slug}_mineru_raw.json"

            status = "ok"
            n_boxes = 0

            if args.resume and boxes_json.exists() and raw_json.exists():
                try:
                    existing = json.loads(boxes_json.read_text(encoding="utf-8"))
                    existing_boxes = existing.get("boxes") or []
                    if isinstance(existing_boxes, list):
                        n_boxes = len(existing_boxes)
                        if not overlay_path.exists():
                            image = Image.open(img_path).convert("RGB")
                            draw = ImageDraw.Draw(image)
                            for b in existing_boxes:
                                bb = b.get("bbox_xyxy")
                                if not bb or len(bb) != 4:
                                    continue
                                x1, y1, x2, y2 = bb
                                lbl = str(b.get("label") or "")
                                color = class_color(lbl)
                                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                                draw.text((x1 + 2, max(0, y1 - 14)), lbl, fill=color)
                            image.save(overlay_path)

                        dt = int(time.time() - t0)
                        wr.writerow(
                            [
                                image_path,
                                "resume",
                                dt,
                                str(page_dir),
                                str(overlay_path),
                                str(boxes_json),
                                str(raw_json),
                                n_boxes,
                                args.model_id,
                            ]
                        )
                        continue
                except Exception:
                    # Corrupt/partial files should be regenerated.
                    pass
            try:
                image = Image.open(img_path).convert("RGB")
                extracted = client.two_step_extract(image)
                raw_json.write_text(
                    json.dumps(to_jsonable(extracted), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

                boxes = extract_boxes(extracted, image.width, image.height)
                n_boxes = len(boxes)

                draw = ImageDraw.Draw(image)
                for b in boxes:
                    x1, y1, x2, y2 = b["bbox_xyxy"]
                    lbl = b["label"]
                    color = class_color(lbl)
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                    draw.text((x1 + 2, max(0, y1 - 14)), str(lbl), fill=color)
                image.save(overlay_path)

                boxes_json.write_text(
                    json.dumps(
                        {
                            "slug": slug,
                            "image": str(img_path),
                            "model_id": args.model_id,
                            "boxes": boxes,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            except Exception as e:  # noqa: BLE001 - keep run going across pages
                status = f"fail:{e}"
                boxes_json.write_text(
                    json.dumps(
                        {
                            "slug": slug,
                            "image": str(img_path),
                            "model_id": args.model_id,
                            "error": str(e),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            dt = int(time.time() - t0)
            wr.writerow(
                [
                    image_path,
                    status,
                    dt,
                    str(page_dir),
                    str(overlay_path),
                    str(boxes_json),
                    str(raw_json),
                    n_boxes,
                    args.model_id,
                ]
            )


if __name__ == "__main__":
    main()
