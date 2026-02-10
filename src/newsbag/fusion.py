from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw


def area(b: List[float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def inter(a: List[float], b: List[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return (x2 - x1) * (y2 - y1)


def iou(a: List[float], b: List[float]) -> float:
    ia = inter(a, b)
    if ia <= 0:
        return 0.0
    u = area(a) + area(b) - ia
    return ia / u if u > 0 else 0.0


def ioa(a: List[float], b: List[float]) -> float:
    aa = area(a)
    if aa <= 0:
        return 0.0
    return inter(a, b) / aa


def coverage(line_box: List[float], box: List[float]) -> float:
    la = area(line_box)
    if la <= 0:
        return 0.0
    return inter(line_box, box) / la


def best_line_cov(line_box: List[float], boxes: List[Dict[str, Any]]) -> float:
    return max([coverage(line_box, b["bbox_xyxy"]) for b in boxes], default=0.0)


def source_family(source: Dict[str, Any]) -> str:
    fam = source.get("source_family")
    if fam:
        return str(fam)
    model = str(source.get("source_model") or "")
    if model.startswith("pld_") or model.startswith("pvl15"):
        return "paddle"
    if model.startswith("dell"):
        return "dell"
    if model.startswith("miner"):
        return "mineru"
    return model


def cross_source_support_count(cand: Dict[str, Any], all_cands: List[Dict[str, Any]], min_iou: float = 0.20) -> int:
    bb = cand["bbox_xyxy"]
    fams = {source_family(cand)}
    for other in all_cands:
        if other is cand:
            continue
        if other.get("norm_label") != cand.get("norm_label"):
            continue
        ob = other["bbox_xyxy"]
        if iou(bb, ob) >= min_iou or ioa(bb, ob) >= 0.60 or ioa(ob, bb) >= 0.60:
            fams.add(source_family(other))
            if len(fams) >= 3:
                break
    return len(fams)


def approx_union_area_ratio(boxes: List[Dict[str, Any]], w: int, h: int, include_labels=("text", "title"), scale: int = 8) -> float:
    wb = max(1, int(np.ceil(w / scale)))
    hb = max(1, int(np.ceil(h / scale)))
    mask = np.zeros((hb, wb), dtype=np.uint8)
    for b in boxes:
        if b.get("norm_label") not in include_labels:
            continue
        x1, y1, x2, y2 = b["bbox_xyxy"]
        ix1 = max(0, min(wb, int(x1 // scale)))
        ix2 = max(0, min(wb, int(np.ceil(x2 / scale))))
        iy1 = max(0, min(hb, int(y1 // scale)))
        iy2 = max(0, min(hb, int(np.ceil(y2 / scale))))
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        mask[iy1:iy2, ix1:ix2] = 1
    return float(mask.sum()) / float(mask.size)


def raster_mask(boxes: List[Dict[str, Any]], w: int, h: int, include_labels=("text", "title"), scale: int = 8) -> np.ndarray:
    wb = max(1, int(np.ceil(w / scale)))
    hb = max(1, int(np.ceil(h / scale)))
    mask = np.zeros((hb, wb), dtype=np.uint8)
    for b in boxes:
        if b.get("norm_label") not in include_labels:
            continue
        x1, y1, x2, y2 = b["bbox_xyxy"]
        ix1 = max(0, min(wb, int(x1 // scale)))
        ix2 = max(0, min(wb, int(np.ceil(x2 / scale))))
        iy1 = max(0, min(hb, int(y1 // scale)))
        iy2 = max(0, min(hb, int(np.ceil(y2 / scale))))
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        mask[iy1:iy2, ix1:ix2] = 1
    return mask


def dedupe_line_boxes(boxes: List[Dict[str, Any]], w: int, h: int, max_lines: int = 120) -> List[Dict[str, Any]]:
    lines: List[Dict[str, Any]] = []
    seen = set()
    page_area = float(max(1, w * h))
    ordered = sorted(
        [b for b in boxes if b.get("norm_label") in ("text", "title")],
        key=lambda b: (float(b.get("score") or 0.0), area(b["bbox_xyxy"])),
        reverse=True,
    )
    for b in ordered:
        bb = b["bbox_xyxy"]
        bw = bb[2] - bb[0]
        bh = bb[3] - bb[1]
        if bw <= 1 or bh <= 1:
            continue
        area_ratio = area(bb) / page_area
        if area_ratio > 0.08:
            continue
        if bw >= 0.90 * w and bh <= 0.06 * h:
            continue
        if bh >= 0.90 * h and bw <= 0.06 * w:
            continue
        key = (round(bb[0] / 8), round(bb[1] / 8), round(bb[2] / 8), round(bb[3] / 8))
        if key in seen:
            continue
        seen.add(key)
        lines.append({"bbox_xyxy": bb, "text": "x", "score": 1.0})
        if len(lines) >= max_lines:
            break
    return lines


def dedupe_boxes(
    candidates: List[Dict[str, Any]],
    pseudo_lines: List[Dict[str, Any]],
    page_area: float,
    page_w: int,
    page_h: int,
    line_cover_threshold: float,
) -> List[Dict[str, Any]]:
    def _clamp(bb: List[float]) -> List[float]:
        x1, y1, x2, y2 = bb
        x1 = max(0.0, min(float(page_w), float(x1)))
        x2 = max(0.0, min(float(page_w), float(x2)))
        y1 = max(0.0, min(float(page_h), float(y1)))
        y2 = max(0.0, min(float(page_h), float(y2)))
        if x2 <= x1 or y2 <= y1:
            return [0.0, 0.0, 0.0, 0.0]
        return [x1, y1, x2, y2]

    def _expand(bb: List[float], pad: float) -> List[float]:
        return _clamp([bb[0] - pad, bb[1] - pad, bb[2] + pad, bb[3] + pad])

    def _synthetic_recovery_boxes(
        parent_bb: List[float],
        score: float,
        max_boxes: int = 32,
    ) -> List[Dict[str, Any]]:
        """Replace a large unsupported text box with smaller boxes around uncovered pseudo-lines.

        This preserves recall while avoiding huge noisy blocks that confuse OCR/reading order.
        """

        uncovered: List[List[float]] = []
        for ln in pseudo_lines:
            if coverage(ln["bbox_xyxy"], parent_bb) < line_cover_threshold:
                continue
            if best_line_cov(ln["bbox_xyxy"], selected) >= line_cover_threshold:
                continue
            uncovered.append(ln["bbox_xyxy"])

        if not uncovered:
            return []

        # Tight bbox around uncovered pseudo-lines.
        x1 = min(b[0] for b in uncovered)
        y1 = min(b[1] for b in uncovered)
        x2 = max(b[2] for b in uncovered)
        y2 = max(b[3] for b in uncovered)
        tight = _clamp([max(parent_bb[0], x1), max(parent_bb[1], y1), min(parent_bb[2], x2), min(parent_bb[3], y2)])

        parent_a = max(1.0, area(parent_bb))
        tight_ratio = area(tight) / parent_a

        def mk(bb: List[float]) -> Dict[str, Any]:
            return {
                "source_family": "synthetic",
                "source_model": "synthetic_line_recovery",
                "source_label": "text_recovery",
                "norm_label": "text",
                "bbox_xyxy": bb,
                "score": max(0.10, min(0.95, score)),
                "reading_order": None,
                "text": None,
            }

        # If the tight bbox is meaningfully smaller, use it as a single recovery block.
        # Otherwise fall back to a capped set of per-line blocks.
        if tight[2] > tight[0] and tight[3] > tight[1] and tight_ratio <= 0.55:
            return [mk(_expand(tight, pad=6.0))]

        out: List[Dict[str, Any]] = []
        seen = set()
        for ln_bb in uncovered:
            bb = _expand(ln_bb, pad=6.0)
            key = (round(bb[0] / 8), round(bb[1] / 8), round(bb[2] / 8), round(bb[3] / 8))
            if key in seen:
                continue
            seen.add(key)
            out.append(mk(bb))
            if len(out) >= max_boxes:
                break
        return out

    cands = sorted(candidates, key=lambda x: (float(x.get("score") or 0.0), area(x["bbox_xyxy"])), reverse=True)
    selected: List[Dict[str, Any]] = []
    for c in cands:
        bb = c["bbox_xyxy"]
        bw = bb[2] - bb[0]
        bh = bb[3] - bb[1]
        if bw <= 1 or bh <= 1:
            continue
        area_ratio = area(bb) / max(1.0, page_area)
        score = float(c.get("score") or 0.0)
        is_textlike = c.get("norm_label") in ("text", "title")
        is_wide_strip = bw >= 0.65 * page_w and bh <= 0.06 * page_h
        is_tall_strip = bh >= 0.65 * page_h and bw <= 0.06 * page_w
        is_large = area_ratio >= 0.08 or is_wide_strip or is_tall_strip
        support = 1
        if is_textlike and is_large:
            support = cross_source_support_count(c, cands)

        if is_textlike and is_wide_strip and support < 2:
            continue
        if is_textlike and is_tall_strip and support < 2:
            continue
        if is_textlike and area_ratio >= 0.22 and support < 2 and score < 0.75:
            continue
        if is_textlike and area_ratio >= 0.30 and support < 3:
            continue

        same = [s for s in selected if s["norm_label"] == c["norm_label"] and iou(s["bbox_xyxy"], bb) >= 0.85]
        if not same:
            if is_textlike and is_large and support < 2:
                total_hits = 0
                new_hits = 0
                for ln in pseudo_lines:
                    cov = coverage(ln["bbox_xyxy"], bb)
                    if cov >= line_cover_threshold:
                        total_hits += 1
                        if best_line_cov(ln["bbox_xyxy"], selected) < line_cover_threshold:
                            new_hits += 1
                if new_hits < 2:
                    continue
                if total_hits > 0 and (new_hits / total_hits) < 0.40:
                    continue

                # If this is a large, unsupported text block, prefer smaller synthetic recovery boxes
                # over keeping the huge region.
                if area_ratio >= 0.12:
                    synth = _synthetic_recovery_boxes(bb, score=score)
                    if synth:
                        selected.extend(synth)
                        continue
            selected.append(c)
            continue

        max_inter = max(inter(s["bbox_xyxy"], bb) for s in same)
        unc_ratio = max(0.0, area(bb) - max_inter) / max(1.0, page_area)
        total_hits = 0
        new_hits = 0
        for ln in pseudo_lines:
            cov = coverage(ln["bbox_xyxy"], bb)
            if cov >= line_cover_threshold:
                total_hits += 1
                if best_line_cov(ln["bbox_xyxy"], selected) < line_cover_threshold:
                    new_hits += 1
        if is_textlike and is_large and support < 2:
            if new_hits < 2:
                continue
            if total_hits > 0 and (new_hits / total_hits) < 0.40:
                continue
        if unc_ratio > 0.02 or new_hits > 0:
            selected.append(c)

    selected.sort(key=lambda b: (b["bbox_xyxy"][1], b["bbox_xyxy"][0]))
    for i, b in enumerate(selected, 1):
        b["reading_order"] = i
    return selected


def variant_metrics(
    pseudo_lines: List[Dict[str, Any]],
    fused: List[Dict[str, Any]],
    w: int,
    h: int,
    line_cover_threshold: float,
    base_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    covered = [ln for ln in pseudo_lines if best_line_cov(ln["bbox_xyxy"], fused) >= line_cover_threshold]
    coverage_ratio = (len(covered) / len(pseudo_lines)) if pseudo_lines else 0.0
    uncovered = len(pseudo_lines) - len(covered)
    text_area = approx_union_area_ratio(fused, w, h)
    base_recall_ratio = None
    if base_mask is not None:
        denom = int(base_mask.sum())
        if denom > 0:
            m = raster_mask(fused, w, h)
            base_recall_ratio = float((m & base_mask).sum()) / float(denom)
        else:
            base_recall_ratio = 0.0
    return {
        "line_coverage_ratio": coverage_ratio,
        "uncovered_lines": uncovered,
        "covered_lines": len(covered),
        "total_lines": len(pseudo_lines),
        "box_count": len(fused),
        "text_area_ratio": text_area,
        "base_recall_ratio": base_recall_ratio,
    }


def draw_boxes(image_path: Path, boxes: List[Dict[str, Any]], out_path: Path, title: str) -> None:
    img = Image.open(image_path).convert("RGB")
    d = ImageDraw.Draw(img)
    palette = {
        "text": (40, 160, 40),
        "title": (30, 80, 220),
        "table": (220, 120, 20),
        "image": (180, 40, 140),
        "other": (120, 120, 120),
    }
    for b in boxes:
        x1, y1, x2, y2 = b["bbox_xyxy"]
        lbl = b.get("norm_label", "other")
        color = palette.get(lbl, (160, 160, 160))
        d.rectangle([x1, y1, x2, y2], outline=color, width=3)
    d.rectangle([0, 0, img.width, 24], fill=(255, 255, 255))
    d.text((8, 5), title, fill=(20, 20, 20))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def _load_boxes(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("boxes") or []


def choose_best_variant(variants: Dict[str, List[Dict[str, Any]]], w: int, h: int) -> Tuple[Optional[str], List[Dict[str, Any]], Dict[str, Any]]:
    if not variants:
        return None, [], {"score": 0.0, "box_count": 0}
    best_name = None
    best_boxes: List[Dict[str, Any]] = []
    best_score = -1e9
    stats = {}
    for name, boxes in variants.items():
        text_count = sum(1 for b in boxes if b.get("norm_label") in ("text", "title"))
        text_area = approx_union_area_ratio(boxes, w, h)
        score = text_count * 0.02 + text_area - (0.0002 * len(boxes))
        stats[name] = {
            "text_count": text_count,
            "text_area_ratio": text_area,
            "box_count": len(boxes),
            "score": score,
        }
        if score > best_score:
            best_score = score
            best_name = name
            best_boxes = boxes
    return best_name, best_boxes, {"score": best_score, "all": stats}


def run_fusion(
    images: Iterable[Path],
    run_dir: Path,
    paddle_layout_variant_ids: List[str],
    paddle_vl15_variant_id: str,
    dell_variant_id: str,
    mineru_variant_id: str,
    line_cover_threshold: float,
    preferred_recommended_variant: str,
) -> Path:
    out_root = run_dir / "outputs" / "fusion"
    out_root.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    source_rows: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {
        "pages": {},
        "leaderboard": {},
        "preferred_recommended_variant": preferred_recommended_variant,
        "best_variant_by_score": None,
        "recommended_variant": None,
    }

    for i, img in enumerate(images, 1):
        slug = img.stem
        with Image.open(img) as im:
            w, h = im.size
        page_area = float(max(1, w * h))

        paddle_variants: Dict[str, List[Dict[str, Any]]] = {}
        for vid in paddle_layout_variant_ids:
            p = run_dir / "outputs" / "sources" / "paddle_layout" / vid / slug / "layout_boxes.normalized.json"
            boxes = _load_boxes(p)
            if boxes:
                paddle_variants[vid] = boxes

        # include VL1.5 as the fourth Paddle source
        pvl = run_dir / "outputs" / "sources" / "paddle_vl15" / paddle_vl15_variant_id / slug / "layout_boxes.normalized.json"
        pvl_boxes = _load_boxes(pvl)
        if pvl_boxes:
            paddle_variants[paddle_vl15_variant_id] = pvl_boxes

        if not paddle_variants:
            continue

        best_paddle_name, best_paddle_boxes, paddle_stats = choose_best_variant(paddle_variants, w, h)
        paddle_union4 = [b for arr in paddle_variants.values() for b in arr]

        dell_boxes = _load_boxes(
            run_dir / "outputs" / "sources" / "dell" / dell_variant_id / slug / "layout_boxes.normalized.json"
        )
        mineru_boxes = _load_boxes(
            run_dir / "outputs" / "sources" / "mineru" / mineru_variant_id / slug / "layout_boxes.normalized.json"
        )

        consensus_lines = dedupe_line_boxes(paddle_union4 + dell_boxes + mineru_boxes, w, h)
        base_mask = raster_mask(paddle_union4 + dell_boxes + mineru_boxes, w, h)

        variants = {
            "S1_paddle_best_single": best_paddle_boxes,
            "S2_dell_only": dell_boxes,
            "S3_mineru_only": mineru_boxes,
            "P1_paddle_union4": paddle_union4,
            "P2_paddle_union4_plus_dell": paddle_union4 + dell_boxes,
            "P3_paddle_union4_plus_mineru": paddle_union4 + mineru_boxes,
            "P4_paddle_union4_plus_dell_plus_mineru": paddle_union4 + dell_boxes + mineru_boxes,
        }

        # Per-source metrics: helps rank which single parser is strongest on this batch,
        # and which ones are adding meaningful coverage vs just adding noise.
        sources: Dict[str, List[Dict[str, Any]]] = {}
        for k, v in paddle_variants.items():
            sources[k] = v
        sources[dell_variant_id] = dell_boxes
        sources[mineru_variant_id] = mineru_boxes

        page_summary: Dict[str, Any] = {
            "image": str(img),
            "width": w,
            "height": h,
            "best_paddle_variant": best_paddle_name,
            "paddle_variants": list(paddle_variants.keys()),
            "source_counts": {
                "paddle_union4_boxes": len(paddle_union4),
                "dell_boxes": len(dell_boxes),
                "mineru_boxes": len(mineru_boxes),
            },
            "paddle_variant_stats": paddle_stats,
            "variants": {},
            "sources": {},
        }

        for vname, candidates in variants.items():
            fused = dedupe_boxes(candidates, consensus_lines, page_area, w, h, line_cover_threshold)
            metrics = variant_metrics(consensus_lines, fused, w, h, line_cover_threshold, base_mask=base_mask)
            vdir = out_root / vname / slug
            vdir.mkdir(parents=True, exist_ok=True)
            (vdir / "fused_boxes.json").write_text(
                json.dumps({"slug": slug, "variant": vname, "boxes": fused}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (vdir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
            page_summary["variants"][vname] = dict(metrics)
            page_summary["variants"][vname]["dir"] = str(vdir)
            rows.append(
                {
                    "slug": slug,
                    "variant": vname,
                    "line_coverage_ratio": metrics["line_coverage_ratio"],
                    "uncovered_lines": metrics["uncovered_lines"],
                    "box_count": metrics["box_count"],
                    "text_area_ratio": metrics["text_area_ratio"],
                    "base_recall_ratio": metrics["base_recall_ratio"],
                }
            )

        for sname, candidates in sources.items():
            fused = dedupe_boxes(candidates, consensus_lines, page_area, w, h, line_cover_threshold)
            metrics = variant_metrics(consensus_lines, fused, w, h, line_cover_threshold, base_mask=base_mask)
            page_summary["sources"][sname] = dict(metrics)
            source_rows.append(
                {
                    "slug": slug,
                    "source": sname,
                    "line_coverage_ratio": metrics["line_coverage_ratio"],
                    "uncovered_lines": metrics["uncovered_lines"],
                    "box_count": metrics["box_count"],
                    "text_area_ratio": metrics["text_area_ratio"],
                    "base_recall_ratio": metrics["base_recall_ratio"],
                }
            )

        summary["pages"][slug] = page_summary

    by_variant: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_slug_variant: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in rows:
        by_variant[r["variant"]].append(r)
        by_slug_variant[(r["slug"], r["variant"])] = r

    leaderboard: Dict[str, Dict[str, Any]] = {}
    for variant, vr in by_variant.items():
        cov = [x["line_coverage_ratio"] for x in vr]
        txta = [x["text_area_ratio"] for x in vr]
        bxs = [x["box_count"] for x in vr]
        br = [x.get("base_recall_ratio") for x in vr if x.get("base_recall_ratio") is not None]
        mean_cov = sum(cov) / len(cov) if cov else 0.0
        mean_ta = sum(txta) / len(txta) if txta else 0.0
        mean_bx = sum(bxs) / len(bxs) if bxs else 0.0
        mean_br = sum(br) / len(br) if br else 0.0
        improved = 0
        for x in vr:
            p1 = by_slug_variant.get((x["slug"], "S1_paddle_best_single"))
            if p1 and (x.get("base_recall_ratio") or 0.0) > (p1.get("base_recall_ratio") or 0.0):
                improved += 1
        score = mean_br + (0.25 * mean_ta) - (0.00025 * mean_bx)
        leaderboard[variant] = {
            "pages": len(vr),
            "mean_line_coverage_ratio": mean_cov,
            "median_line_coverage_ratio": statistics.median(cov) if cov else 0.0,
            "mean_text_area_ratio": mean_ta,
            "mean_box_count": mean_bx,
            "mean_base_recall_ratio": mean_br,
            "improved_pages_vs_s1": improved,
            "score": score,
        }

    summary["leaderboard"] = leaderboard

    best_by_score = None
    if leaderboard:
        best_by_score = sorted(leaderboard.items(), key=lambda kv: kv[1]["score"], reverse=True)[0][0]
    summary["best_variant_by_score"] = best_by_score

    if preferred_recommended_variant in leaderboard:
        summary["recommended_variant"] = preferred_recommended_variant
    else:
        summary["recommended_variant"] = best_by_score

    rec = summary.get("recommended_variant")
    if rec:
        for slug, meta in summary.get("pages", {}).items():
            img = Path(meta["image"])
            vmeta = meta["variants"].get(rec)
            if not vmeta:
                continue
            vdir = Path(vmeta["dir"])
            fb = vdir / "fused_boxes.json"
            if not fb.exists():
                continue
            fused = json.loads(fb.read_text(encoding="utf-8")).get("boxes", [])
            draw_boxes(img, fused, vdir / "fused_layout.png", f"{rec}: {slug}")
            draw_boxes(img, fused, vdir / "consensus_coverage_overlay.png", f"{rec} consensus coverage: {slug}")

    (out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    with (out_root / "variant_leaderboard.tsv").open("w", encoding="utf-8", newline="") as f:
        wv = csv.writer(f, delimiter="\t")
        wv.writerow(
            [
                "variant",
                "pages",
                "mean_base_recall_ratio",
                "mean_line_coverage_ratio",
                "median_line_coverage_ratio",
                "mean_text_area_ratio",
                "mean_box_count",
                "improved_pages_vs_s1",
                "score",
            ]
        )
        for variant, m in sorted(leaderboard.items(), key=lambda kv: kv[1]["score"], reverse=True):
            wv.writerow(
                [
                    variant,
                    m["pages"],
                    f"{m.get('mean_base_recall_ratio', 0.0):.6f}",
                    f"{m['mean_line_coverage_ratio']:.6f}",
                    f"{m['median_line_coverage_ratio']:.6f}",
                    f"{m['mean_text_area_ratio']:.6f}",
                    f"{m['mean_box_count']:.3f}",
                    m["improved_pages_vs_s1"],
                    f"{m['score']:.6f}",
                ]
            )

    by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in source_rows:
        by_source[r["source"]].append(r)

    source_leaderboard: Dict[str, Dict[str, Any]] = {}
    for source, sr in by_source.items():
        cov = [x["line_coverage_ratio"] for x in sr]
        txta = [x["text_area_ratio"] for x in sr]
        bxs = [x["box_count"] for x in sr]
        br = [x.get("base_recall_ratio") for x in sr if x.get("base_recall_ratio") is not None]
        mean_cov = sum(cov) / len(cov) if cov else 0.0
        mean_ta = sum(txta) / len(txta) if txta else 0.0
        mean_bx = sum(bxs) / len(bxs) if bxs else 0.0
        mean_br = sum(br) / len(br) if br else 0.0
        # Score is intentionally similar to variant leaderboard, so we can compare quickly.
        score = mean_br + (0.25 * mean_ta) - (0.00025 * mean_bx)
        source_leaderboard[source] = {
            "pages": len(sr),
            "mean_base_recall_ratio": mean_br,
            "mean_line_coverage_ratio": mean_cov,
            "mean_text_area_ratio": mean_ta,
            "mean_box_count": mean_bx,
            "score": score,
        }

    with (out_root / "source_leaderboard.tsv").open("w", encoding="utf-8", newline="") as f:
        ws = csv.writer(f, delimiter="\t")
        ws.writerow(
            [
                "source",
                "pages",
                "mean_base_recall_ratio",
                "mean_line_coverage_ratio",
                "mean_text_area_ratio",
                "mean_box_count",
                "score",
            ]
        )
        for source, m in sorted(source_leaderboard.items(), key=lambda kv: kv[1]["score"], reverse=True):
            ws.writerow(
                [
                    source,
                    m["pages"],
                    f"{m.get('mean_base_recall_ratio', 0.0):.6f}",
                    f"{m['mean_line_coverage_ratio']:.6f}",
                    f"{m['mean_text_area_ratio']:.6f}",
                    f"{m['mean_box_count']:.3f}",
                    f"{m['score']:.6f}",
                ]
            )

    with (out_root / "per_page_source_metrics.tsv").open("w", encoding="utf-8", newline="") as f:
        ws = csv.writer(f, delimiter="\t")
        ws.writerow(
            [
                "slug",
                "source",
                "base_recall_ratio",
                "line_coverage_ratio",
                "uncovered_lines",
                "box_count",
                "text_area_ratio",
            ]
        )
        for r in sorted(source_rows, key=lambda x: (x["slug"], x["source"])):
            ws.writerow(
                [
                    r["slug"],
                    r["source"],
                    f"{float(r.get('base_recall_ratio') or 0.0):.6f}",
                    f"{r['line_coverage_ratio']:.6f}",
                    r["uncovered_lines"],
                    r["box_count"],
                    f"{r['text_area_ratio']:.6f}",
                ]
            )

    with (out_root / "per_page_variant_metrics.tsv").open("w", encoding="utf-8", newline="") as f:
        wp = csv.writer(f, delimiter="\t")
        wp.writerow(
            [
                "slug",
                "variant",
                "base_recall_ratio",
                "line_coverage_ratio",
                "uncovered_lines",
                "box_count",
                "text_area_ratio",
            ]
        )
        for r in sorted(rows, key=lambda x: (x["slug"], x["variant"])):
            wp.writerow(
                [
                    r["slug"],
                    r["variant"],
                    f"{float(r.get('base_recall_ratio') or 0.0):.6f}",
                    f"{r['line_coverage_ratio']:.6f}",
                    r["uncovered_lines"],
                    r["box_count"],
                    f"{r['text_area_ratio']:.6f}",
                ]
            )

    return out_root
