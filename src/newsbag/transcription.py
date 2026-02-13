from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PIL import Image, ImageDraw

from newsbag.utils.proc import run_cmd


def _area(bb: List[float]) -> float:
    return max(0.0, bb[2] - bb[0]) * max(0.0, bb[3] - bb[1])


def _inter(a: List[float], b: List[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return (x2 - x1) * (y2 - y1)


def _line_overlap_ratio(line_bb: List[float], box_bb: List[float]) -> float:
    la = _area(line_bb)
    if la <= 0:
        return 0.0
    return _inter(line_bb, box_bb) / la


def _iou(a: List[float], b: List[float]) -> float:
    ia = _inter(a, b)
    if ia <= 0:
        return 0.0
    ua = _area(a) + _area(b) - ia
    if ua <= 0:
        return 0.0
    return ia / ua


def _coverage(inner: List[float], outer: List[float]) -> float:
    ia = _inter(inner, outer)
    if ia <= 0:
        return 0.0
    aa = _area(inner)
    if aa <= 0:
        return 0.0
    return ia / aa


def _poly_to_bbox(poly: Any) -> Optional[List[float]]:
    if not isinstance(poly, list) or not poly:
        return None
    pts: List[Tuple[float, float]] = []
    for p in poly:
        if not isinstance(p, list) or len(p) < 2:
            continue
        try:
            x = float(p[0])
            y = float(p[1])
        except Exception:
            continue
        pts.append((x, y))
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    bb = [min(xs), min(ys), max(xs), max(ys)]
    if bb[2] <= bb[0] or bb[3] <= bb[1]:
        return None
    return bb


def _parse_ocr_lines(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    obj = payload.get("res", payload)
    lines: List[Dict[str, Any]] = []

    def add_line(text: Any, score: Any, poly: Any, idx: int) -> None:
        if not isinstance(text, str):
            return
        t = text.strip()
        if not t:
            return
        bb = None
        if isinstance(poly, list) and poly and isinstance(poly[0], list):
            bb = _poly_to_bbox(poly)
        elif (
            isinstance(poly, list)
            and len(poly) == 4
            and all(isinstance(x, (int, float)) for x in poly)
        ):
            bb = [float(poly[0]), float(poly[1]), float(poly[2]), float(poly[3])]
            if bb[2] <= bb[0] or bb[3] <= bb[1]:
                bb = None
        if not bb:
            return
        s = None
        try:
            if score is not None:
                s = float(score)
        except Exception:
            s = None
        lines.append({"index": idx, "text": t, "score": s, "bbox_xyxy": bb})

    def parse_item(item: Any) -> None:
        if isinstance(item, dict):
            rec_texts = item.get("rec_texts")
            rec_scores = item.get("rec_scores")
            polys = item.get("dt_polys") or item.get("rec_polys") or item.get("polys")
            if isinstance(rec_texts, list) and isinstance(polys, list):
                n = min(len(rec_texts), len(polys))
                for i in range(n):
                    score = rec_scores[i] if isinstance(rec_scores, list) and i < len(rec_scores) else None
                    add_line(rec_texts[i], score, polys[i], len(lines))

            if "text" in item and any(k in item for k in ("dt_poly", "poly", "bbox", "bbox_xyxy")):
                poly = item.get("dt_poly") or item.get("poly") or item.get("bbox_xyxy") or item.get("bbox")
                add_line(item.get("text"), item.get("score"), poly, len(lines))

            for k in ("ocr_res", "ocr_results", "ocr_res_list", "result", "results", "pages"):
                v = item.get(k)
                if isinstance(v, list):
                    for x in v:
                        parse_item(x)
                elif isinstance(v, dict):
                    parse_item(v)

        elif isinstance(item, list):
            for x in item:
                parse_item(x)

    parse_item(obj)

    deduped: List[Dict[str, Any]] = []
    seen = set()
    for ln in lines:
        bb = ln["bbox_xyxy"]
        key = (
            ln["text"],
            round(bb[0] / 4),
            round(bb[1] / 4),
            round(bb[2] / 4),
            round(bb[3] / 4),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(ln)

    deduped.sort(key=lambda x: (x["bbox_xyxy"][1], x["bbox_xyxy"][0]))
    for i, ln in enumerate(deduped, 1):
        ln["reading_order"] = i
    return deduped


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _normalize_fused_boxes(boxes: Iterable[Dict[str, Any]], labels: set[str]) -> List[Dict[str, Any]]:
    out = []
    for b in boxes:
        try:
            bb = [float(x) for x in (b.get("bbox_xyxy") or [])]
        except Exception:
            continue
        if len(bb) != 4 or bb[2] <= bb[0] or bb[3] <= bb[1]:
            continue
        nl = str(b.get("norm_label") or "other")
        if nl not in labels:
            continue
        out.append(
            {
                "norm_label": nl,
                "bbox_xyxy": bb,
                "reading_order": b.get("reading_order"),
                "score": b.get("score"),
            }
        )
    out.sort(
        key=lambda x: (
            x["reading_order"] if isinstance(x.get("reading_order"), int) else math.inf,
            x["bbox_xyxy"][1],
            x["bbox_xyxy"][0],
        )
    )
    for i, b in enumerate(out, 1):
        if not isinstance(b.get("reading_order"), int):
            b["reading_order"] = i
    return out


def _dedupe_boxes(
    candidates: List[Dict[str, Any]],
    *,
    cover_drop_threshold: float,
    iou_drop_threshold: float,
    prefer_larger_area: bool,
) -> List[Dict[str, Any]]:
    if prefer_larger_area:
        cands = sorted(
            candidates,
            key=lambda b: (
                -_area(b["bbox_xyxy"]),
                -(float(b.get("score") or 0.0)),
                int(b.get("reading_order") or 10**9),
            ),
        )
    else:
        cands = sorted(
            candidates,
            key=lambda b: (
                _area(b["bbox_xyxy"]),
                -(float(b.get("score") or 0.0)),
                int(b.get("reading_order") or 10**9),
            ),
        )

    kept: List[Dict[str, Any]] = []
    for cand in cands:
        bb = cand["bbox_xyxy"]
        if _area(bb) <= 0:
            continue
        redundant = False
        for k in kept:
            kbb = k["bbox_xyxy"]
            if _coverage(bb, kbb) >= cover_drop_threshold:
                redundant = True
                break
            if _iou(bb, kbb) >= iou_drop_threshold:
                redundant = True
                break
        if not redundant:
            kept.append(cand)
    return kept


def _select_ocr_boxes(fused_boxes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # OCR needs compact, non-redundant regions. We intentionally keep larger text
    # containers and suppress nested/duplicate overlaps before crop OCR.
    title_boxes = [b for b in fused_boxes if b.get("norm_label") == "title"]
    text_boxes = [b for b in fused_boxes if b.get("norm_label") == "text"]

    title_keep = _dedupe_boxes(
        title_boxes,
        cover_drop_threshold=0.92,
        iou_drop_threshold=0.85,
        prefer_larger_area=False,
    )
    text_keep = _dedupe_boxes(
        text_boxes,
        cover_drop_threshold=0.85,
        iou_drop_threshold=0.75,
        prefer_larger_area=True,
    )

    selected = title_keep + text_keep
    selected.sort(
        key=lambda x: (
            int(x.get("reading_order") or 10**9),
            x["bbox_xyxy"][1],
            x["bbox_xyxy"][0],
        )
    )
    for i, b in enumerate(selected, 1):
        b["box_index"] = i
    return selected


def _to_int_crop(
    bb: List[float],
    w: int,
    h: int,
    pad: int = 2,
    min_width: int = 24,
    min_height: int = 18,
) -> Optional[Tuple[int, int, int, int]]:
    x1 = max(0, int(math.floor(float(bb[0])) - pad))
    y1 = max(0, int(math.floor(float(bb[1])) - pad))
    x2 = min(w, int(math.ceil(float(bb[2])) + pad))
    y2 = min(h, int(math.ceil(float(bb[3])) + pad))
    if x2 <= x1 or y2 <= y1:
        return None
    if (x2 - x1) < min_width or (y2 - y1) < min_height:
        return None
    return (x1, y1, x2, y2)


def _translate_lines(
    lines: List[Dict[str, Any]],
    ox: float,
    oy: float,
    box_index: int,
    box_order: int,
    target_bb: List[float],
    min_overlap: float,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for ln in lines:
        bb = ln.get("bbox_xyxy")
        if not isinstance(bb, list) or len(bb) != 4:
            continue
        try:
            x1, y1, x2, y2 = [float(v) for v in bb]
        except Exception:
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        row = dict(ln)
        page_bb = [x1 + ox, y1 + oy, x2 + ox, y2 + oy]
        if min_overlap > 0 and _line_overlap_ratio(page_bb, target_bb) < min_overlap:
            continue
        row["bbox_xyxy"] = page_bb
        row["box_index"] = int(box_index)
        row["box_reading_order"] = int(box_order)
        out.append(row)
    return out


def _boxes_from_direct_assignment(fused_boxes: List[Dict[str, Any]], lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_box: Dict[int, List[Dict[str, Any]]] = {}
    for ln in lines:
        try:
            idx = int(ln.get("box_index"))
        except Exception:
            continue
        by_box.setdefault(idx, []).append(ln)

    boxes: List[Dict[str, Any]] = []
    for i, b in enumerate(fused_boxes, 1):
        lines_i = by_box.get(i, [])
        lines_i.sort(key=lambda x: (x["bbox_xyxy"][1], x["bbox_xyxy"][0]))
        boxes.append(
            {
                "box_index": i,
                "reading_order": int(b.get("reading_order") or i),
                "norm_label": b.get("norm_label"),
                "bbox_xyxy": b["bbox_xyxy"],
                "lines": lines_i,
                "text": "\n".join(x["text"] for x in lines_i if x.get("text")),
            }
        )
    boxes.sort(key=lambda x: x["reading_order"])
    return boxes


def _draw_transcription_overlays(
    image_path: Path,
    ocr_boxes: List[Dict[str, Any]],
    ocr_lines: List[Dict[str, Any]],
    regions_out: Path,
    lines_out: Path,
) -> None:
    with Image.open(image_path).convert("RGB") as base:
        img_regions = base.copy()
        draw_regions = ImageDraw.Draw(img_regions)
        for b in ocr_boxes:
            bb = b.get("bbox_xyxy") or []
            if not isinstance(bb, list) or len(bb) != 4:
                continue
            x1, y1, x2, y2 = [float(v) for v in bb]
            if x2 <= x1 or y2 <= y1:
                continue
            lbl = str(b.get("norm_label") or "text")
            color = (29, 161, 242) if lbl == "text" else (235, 87, 87)
            draw_regions.rectangle([x1, y1, x2, y2], outline=color, width=3)
            draw_regions.text((x1 + 2, max(0, y1 - 14)), f"{int(b.get('box_index') or 0)}:{lbl}", fill=color)
        regions_out.parent.mkdir(parents=True, exist_ok=True)
        img_regions.save(regions_out)

        img_lines = base.copy()
        draw_lines = ImageDraw.Draw(img_lines)
        for ln in ocr_lines:
            bb = ln.get("bbox_xyxy") or []
            if not isinstance(bb, list) or len(bb) != 4:
                continue
            x1, y1, x2, y2 = [float(v) for v in bb]
            if x2 <= x1 or y2 <= y1:
                continue
            box_idx = int(ln.get("box_index") or 0)
            r = (box_idx * 67) % 255
            g = (box_idx * 41) % 255
            bl = (box_idx * 97) % 255
            draw_lines.rectangle([x1, y1, x2, y2], outline=(r, g, bl), width=1)
        lines_out.parent.mkdir(parents=True, exist_ok=True)
        img_lines.save(lines_out)


def run_transcription(
    run_dir: Path,
    paddleocr_bin: str,
    variant: str = "",
    labels: Optional[List[str]] = None,
    min_overlap: float = 0.30,
    device: str = "gpu:0",
    cpu_threads: int = 8,
    timeout_sec: int = 3600,
    max_pages: int = 0,
    resume: bool = True,
) -> Path:
    run_dir = run_dir.expanduser().resolve()
    fusion_root = run_dir / "outputs" / "fusion"
    summary_path = fusion_root / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing fusion summary: {summary_path}")

    summary = _load_json(summary_path)
    resolved_variant = str(variant).strip() or str(summary.get("recommended_variant") or "")
    if not resolved_variant:
        raise ValueError("Could not resolve transcription variant; set transcription.variant or fusion.recommended_variant.")

    labels_list = labels or ["text", "title"]
    labels_set = {x.strip().lower() for x in labels_list if str(x).strip()}
    if not labels_set:
        raise ValueError("No labels selected for transcription.")

    out_root = run_dir / "outputs" / "transcription" / resolved_variant
    out_root.mkdir(parents=True, exist_ok=True)
    report_tsv = out_root / "transcription_report.tsv"

    pages = list((summary.get("pages") or {}).items())
    if max_pages and max_pages > 0:
        pages = pages[: max_pages]

    with report_tsv.open("w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f, delimiter="\t")
        wr.writerow(
            [
                "slug",
                "status",
                "variant",
                "ocr_device",
                "fused_box_count",
                "ocr_box_count",
                "ocr_line_count",
                "assigned_line_count",
                "transcript_chars",
                "ocr_json",
                "transcript_json",
                "transcript_txt",
                "ocr_log",
            ]
        )

        combined_lines: List[str] = []
        for slug, pmeta in pages:
            image_path = Path(str((pmeta or {}).get("image", ""))).expanduser()
            fused_path = fusion_root / resolved_variant / slug / "fused_boxes.json"
            page_dir = out_root / slug
            page_dir.mkdir(parents=True, exist_ok=True)
            ocr_log = page_dir / "ocr.log"
            ocr_raw_json = page_dir / "ocr_raw.json"
            ocr_lines_json = page_dir / "ocr_lines.json"
            transcript_json = page_dir / "transcript_boxes.json"
            transcript_txt = page_dir / "transcript.txt"
            ocr_regions_overlay = page_dir / "ocr_regions_overlay.png"
            ocr_lines_overlay = page_dir / "ocr_lines_overlay.png"

            if not image_path.exists():
                wr.writerow([slug, "missing_image", resolved_variant, device, 0, 0, 0, 0, 0, "", "", "", str(ocr_log)])
                continue
            if not fused_path.exists():
                wr.writerow([slug, "missing_fused", resolved_variant, device, 0, 0, 0, 0, 0, "", "", "", str(ocr_log)])
                continue

            fused_payload = _load_json(fused_path)
            fused_boxes = _normalize_fused_boxes(fused_payload.get("boxes") or [], labels=labels_set)
            ocr_boxes = _select_ocr_boxes(fused_boxes)
            if not ocr_boxes:
                wr.writerow(
                    [slug, "no_ocr_boxes", resolved_variant, device, len(fused_boxes), 0, 0, 0, 0, "", "", "", str(ocr_log)]
                )
                continue

            if resume and ocr_lines_json.exists() and ocr_raw_json.exists() and transcript_json.exists():
                ocr_raw_payload = _load_json(ocr_raw_json)
                lines = (_load_json(ocr_lines_json).get("lines") or [])
                boxed = (_load_json(transcript_json).get("boxes") or [])
                ocr_boxes = (ocr_raw_payload.get("ocr_boxes") or ocr_boxes)
                status = "resume"
            else:
                # ROI-first OCR: run Paddle OCR on fused text/title crops, then remap
                # OCR line boxes back to page coordinates.
                ocr_out_dir = page_dir / "ocr_raw_dir"
                ocr_out_dir.mkdir(parents=True, exist_ok=True)
                crops_dir = ocr_out_dir / "crops"
                crops_dir.mkdir(parents=True, exist_ok=True)

                crop_rows: List[Dict[str, Any]] = []
                with Image.open(image_path).convert("RGB") as im:
                    w, h = im.size
                    for i, b in enumerate(ocr_boxes, 1):
                        crop_xyxy = _to_int_crop(b["bbox_xyxy"], w, h, pad=2)
                        if not crop_xyxy:
                            continue
                        x1, y1, x2, y2 = crop_xyxy
                        crop_stem = f"crop_{i:04d}"
                        crop_path = crops_dir / f"{crop_stem}.png"
                        im.crop((x1, y1, x2, y2)).save(crop_path)
                        crop_rows.append(
                            {
                                "box_index": i,
                                "box_reading_order": int(b.get("reading_order") or i),
                                "norm_label": b.get("norm_label"),
                                "crop_stem": crop_stem,
                                "crop_path": str(crop_path),
                                "page_bbox_xyxy": b["bbox_xyxy"],
                                "crop_bbox_xyxy": [x1, y1, x2, y2],
                            }
                        )

                lines = []
                if crop_rows:
                    cmd = [
                        paddleocr_bin,
                        "ocr",
                        "-i",
                        str(crops_dir),
                        "--save_path",
                        str(ocr_out_dir),
                        "--device",
                        str(device),
                        "--cpu_threads",
                        str(cpu_threads),
                        "--enable_mkldnn",
                        "False",
                    ]
                    res = run_cmd(
                        cmd,
                        ocr_log,
                        timeout_sec=timeout_sec,
                        env={"PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True"},
                    )
                    if res.rc != 0:
                        wr.writerow(
                            [
                                slug,
                                f"ocr_fail({res.rc})",
                                resolved_variant,
                                device,
                                len(fused_boxes),
                                len(ocr_boxes),
                                0,
                                0,
                                0,
                                "",
                                "",
                                "",
                                str(ocr_log),
                            ]
                        )
                        continue
                    for row in crop_rows:
                        raw_path = ocr_out_dir / f"{row['crop_stem']}_res.json"
                        if not raw_path.exists():
                            alt = list(ocr_out_dir.glob(f"{row['crop_stem']}*_res.json"))
                            raw_path = alt[0] if alt else raw_path
                        row["raw_json"] = str(raw_path) if raw_path.exists() else ""
                        if not raw_path.exists():
                            row["ocr_line_count"] = 0
                            continue
                        payload = _load_json(raw_path)
                        crop_lines = _parse_ocr_lines(payload)
                        row["ocr_line_count"] = len(crop_lines)
                        cb = row["crop_bbox_xyxy"]
                        lines.extend(
                            _translate_lines(
                                crop_lines,
                                ox=float(cb[0]),
                                oy=float(cb[1]),
                                box_index=int(row["box_index"]),
                                box_order=int(row["box_reading_order"]),
                                target_bb=[float(v) for v in row["page_bbox_xyxy"]],
                                min_overlap=float(min_overlap),
                            )
                        )

                lines.sort(
                    key=lambda x: (
                        int(x.get("box_reading_order") or 10**9),
                        x["bbox_xyxy"][1],
                        x["bbox_xyxy"][0],
                    )
                )
                boxed = _boxes_from_direct_assignment(fused_boxes=ocr_boxes, lines=lines)
                _write_json(
                    ocr_raw_json,
                    {
                        "slug": slug,
                        "mode": "roi_fused",
                        "variant": resolved_variant,
                        "device": device,
                        "fused_box_count": len(fused_boxes),
                        "ocr_box_count": len(ocr_boxes),
                        "crop_count": len(crop_rows),
                        "ocr_boxes": ocr_boxes,
                        "crops": crop_rows,
                    },
                )
                _write_json(
                    ocr_lines_json,
                    {"slug": slug, "mode": "roi_fused", "line_count": len(lines), "lines": lines},
                )
                status = "ok"

            _write_json(
                transcript_json,
                {
                    "slug": slug,
                    "variant": resolved_variant,
                    "mode": "roi_fused",
                    "labels": sorted(labels_set),
                    "fused_box_count": len(fused_boxes),
                    "ocr_box_count": len(ocr_boxes),
                    "box_count": len(ocr_boxes),
                    "ocr_line_count": len(lines),
                    "boxes": boxed,
                },
            )

            _draw_transcription_overlays(
                image_path=image_path,
                ocr_boxes=ocr_boxes,
                ocr_lines=lines,
                regions_out=ocr_regions_overlay,
                lines_out=ocr_lines_overlay,
            )

            text_blocks = [b.get("text", "").strip() for b in boxed if str(b.get("text", "")).strip()]
            page_text = "\n\n".join(text_blocks).strip()
            transcript_txt.write_text(page_text + ("\n" if page_text else ""), encoding="utf-8")

            assigned = sum(len(b.get("lines", [])) for b in boxed)
            wr.writerow(
                [
                    slug,
                    status,
                    resolved_variant,
                    device,
                    len(fused_boxes),
                    len(ocr_boxes),
                    len(lines),
                    assigned,
                    len(page_text),
                    str(ocr_raw_json),
                    str(transcript_json),
                    str(transcript_txt),
                    str(ocr_log),
                ]
            )

            combined_lines.extend([f"## {slug}", page_text, ""])

    combined_txt = out_root / "transcript_combined.txt"
    combined_txt.write_text("\n".join(combined_lines).strip() + "\n", encoding="utf-8")
    return out_root
