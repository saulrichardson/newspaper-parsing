#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run Dell AmericanStories layout ONNX over a manifest (layout only)."
    )
    p.add_argument("--manifest", required=True, help="txt file with absolute image path per line")
    p.add_argument("--model", required=True, help="path to layout_model_new.onnx")
    p.add_argument("--label_map", required=True, help="path to label_map_layout.json")
    p.add_argument(
        "--repo_src",
        required=True,
        help="path to AmericanStories src directory (used for proven preprocessing + NMS ops)",
    )
    p.add_argument("--output_root", required=True)
    p.add_argument("--conf", type=float, default=0.01)
    p.add_argument("--iou", type=float, default=0.10)
    p.add_argument("--max_det", type=int, default=2000)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--provider", choices=["auto", "cpu"], default="cpu")
    p.add_argument("--max_pages", type=int, default=0)
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip pages with existing <slug>_dell_layout_boxes.json (and regenerate overlay if missing).",
    )
    return p.parse_args()


def clamp_box(box: list[float], w: int, h: int) -> list[float] | None:
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(float(w), float(x1)))
    x2 = max(0.0, min(float(w), float(x2)))
    y1 = max(0.0, min(float(h), float(y1)))
    y2 = max(0.0, min(float(h), float(y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def main() -> None:
    args = parse_args()
    sys.path.insert(0, args.repo_src)

    try:
        from stages.images_to_layouts import letterbox  # type: ignore # noqa: E402
        from effocr.engines.yolov8_ops import non_max_suppression as nms_yolov8  # type: ignore # noqa: E402
    except Exception as exc:
        raise RuntimeError(
            "Failed to import Dell preprocessing/NMS ops from --repo_src. "
            "Use a valid AmericanStories src checkout."
        ) from exc

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    with open(args.label_map, "r", encoding="utf-8") as f:
        label_map_raw = json.load(f)
    label_map = {int(k): v for k, v in label_map_raw.items()}
    num_classes = len(label_map)

    providers_available = ort.get_available_providers()
    if args.provider == "auto" and "CUDAExecutionProvider" in providers_available:
        providers_requested = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers_requested = ["CPUExecutionProvider"]
    sess = ort.InferenceSession(args.model, providers=providers_requested)
    providers_used = sess.get_providers()
    input_name = sess.get_inputs()[0].name

    report: dict[str, object] = {
        "model": args.model,
        "providers_available": providers_available,
        "providers_requested": providers_requested,
        "providers_used": providers_used,
        "torch_cuda_available": torch.cuda.is_available(),
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "pages": [],
    }

    with open(args.manifest, "r", encoding="utf-8") as f:
        image_paths = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    if args.max_pages and args.max_pages > 0:
        image_paths = image_paths[: args.max_pages]

    for img_path in image_paths:
        img_p = Path(img_path)
        slug = img_p.stem
        page_dir = output_root / slug
        page_dir.mkdir(parents=True, exist_ok=True)

        overlay_path = page_dir / f"{slug}_dell_layout_overlay.png"
        boxes_path = page_dir / f"{slug}_dell_layout_boxes.json"

        if args.resume and boxes_path.exists():
            try:
                existing = json.loads(boxes_path.read_text(encoding="utf-8"))
                existing_boxes = existing.get("boxes") or []
                if isinstance(existing_boxes, list):
                    # Keep overlay around for convenient visual debug; regenerate if needed.
                    if not overlay_path.exists():
                        pil = Image.open(img_p).convert("RGB")
                        draw = ImageDraw.Draw(pil)
                        color_map = {
                            "article": (220, 40, 40),
                            "headline": (30, 80, 220),
                            "table": (220, 140, 20),
                            "photograph": (130, 40, 180),
                            "image_caption": (40, 130, 180),
                            "author": (20, 150, 120),
                            "cartoon_or_advertisement": (180, 60, 160),
                            "masthead": (20, 120, 220),
                            "newspaper_header": (20, 120, 220),
                            "page_number": (90, 90, 90),
                        }
                        for b in existing_boxes:
                            bb = b.get("bbox_xyxy")
                            if not bb or len(bb) != 4:
                                continue
                            x1, y1, x2, y2 = bb
                            lbl = str(b.get("label") or "")
                            score = b.get("score")
                            color = color_map.get(lbl, (120, 120, 120))
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                            if isinstance(score, (int, float)):
                                draw.text((x1 + 2, max(0, y1 - 14)), f"{lbl}:{float(score):.2f}", fill=color)
                            else:
                                draw.text((x1 + 2, max(0, y1 - 14)), lbl, fill=color)
                        pil.save(overlay_path)

                    report["pages"].append(
                        {
                            "slug": slug,
                            "image": str(img_p),
                            "status": "resume",
                            "box_count": len(existing_boxes),
                            "overlay": str(overlay_path),
                            "boxes_json": str(boxes_path),
                        }
                    )
                    continue
            except Exception:
                # Corrupt/partial files should be regenerated.
                pass

        img = cv2.imread(str(img_p), cv2.IMREAD_COLOR)
        if img is None:
            report["pages"].append({"slug": slug, "image": str(img_p), "status": "read_fail"})
            continue
        h, w = img.shape[:2]

        padded, ratio, (dw, dh) = letterbox(img, (args.imgsz, args.imgsz), auto=False)
        x = padded.transpose((2, 0, 1))[::-1]
        x = np.expand_dims(np.ascontiguousarray(x), axis=0).astype(np.float32) / 255.0

        out = sess.run(None, {input_name: x})
        pred = torch.from_numpy(out[0])
        det = nms_yolov8(
            pred,
            conf_thres=args.conf,
            iou_thres=args.iou,
            max_det=args.max_det,
            agnostic=True,
            nc=num_classes,
        )[0]

        boxes = []
        if det is not None and det.numel() > 0:
            for i, row in enumerate(det):
                x1, y1, x2, y2, score, cls_id = row[:6].tolist()
                x1 = (x1 - dw) / ratio[0]
                x2 = (x2 - dw) / ratio[0]
                y1 = (y1 - dh) / ratio[1]
                y2 = (y2 - dh) / ratio[1]
                bb = clamp_box([x1, y1, x2, y2], w, h)
                if bb is None:
                    continue
                cls_id_i = int(round(cls_id))
                boxes.append(
                    {
                        "id": i,
                        "label": label_map.get(cls_id_i, f"class_{cls_id_i}"),
                        "class_id": cls_id_i,
                        "score": float(score),
                        "bbox_xyxy": bb,
                    }
                )

        pil = Image.open(img_p).convert("RGB")
        draw = ImageDraw.Draw(pil)
        color_map = {
            "article": (220, 40, 40),
            "headline": (30, 80, 220),
            "table": (220, 140, 20),
            "photograph": (130, 40, 180),
            "image_caption": (40, 130, 180),
            "author": (20, 150, 120),
            "cartoon_or_advertisement": (180, 60, 160),
            "masthead": (20, 120, 220),
            "newspaper_header": (20, 120, 220),
            "page_number": (90, 90, 90),
        }
        for b in boxes:
            x1, y1, x2, y2 = b["bbox_xyxy"]
            lbl = b["label"]
            score = b["score"]
            color = color_map.get(lbl, (120, 120, 120))
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            draw.text((x1 + 2, max(0, y1 - 14)), f"{lbl}:{score:.2f}", fill=color)

        pil.save(overlay_path)

        with open(boxes_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "slug": slug,
                    "image": str(img_p),
                    "width": w,
                    "height": h,
                    "providers_used": providers_used,
                    "boxes": boxes,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        report["pages"].append(
            {
                "slug": slug,
                "image": str(img_p),
                "status": "ok",
                "box_count": len(boxes),
                "overlay": str(overlay_path),
                "boxes_json": str(boxes_path),
            }
        )

    with open(output_root / "run_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
