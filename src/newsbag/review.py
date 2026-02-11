from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from PIL import Image, ImageDraw

from newsbag.fusion import draw_boxes
from newsbag.utils.io import read_json


def _board(images: List[Tuple[str, Path]], out_path: Path, title: str, cols: int = 3, tile_w: int = 860, tile_h: int = 1200) -> None:
    rows = (len(images) + cols - 1) // cols
    margin = 20
    label_h = 46
    canvas_w = margin + cols * tile_w + (cols - 1) * margin + margin
    canvas_h = margin + rows * (tile_h + label_h) + (rows - 1) * margin + margin + 56
    canvas = Image.new("RGB", (canvas_w, canvas_h), (250, 250, 250))
    d = ImageDraw.Draw(canvas)
    d.text((margin, 14), title, fill=(20, 20, 20))

    for idx, (label, path) in enumerate(images):
        r = idx // cols
        c = idx % cols
        x = margin + c * (tile_w + margin)
        y = margin + 56 + r * (tile_h + label_h + margin)

        tile = Image.new("RGB", (tile_w, tile_h), (255, 255, 255))
        if path.exists():
            im = Image.open(path).convert("RGB")
            im.thumbnail((tile_w, tile_h))
            ox = (tile_w - im.width) // 2
            oy = (tile_h - im.height) // 2
            tile.paste(im, (ox, oy))
        canvas.paste(tile, (x, y))
        d.rectangle([x, y + tile_h, x + tile_w, y + tile_h + label_h], fill=(238, 242, 247), outline=(210, 215, 220))
        d.text((x + 10, y + tile_h + 14), label, fill=(30, 30, 30))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def _top_informative_rows(summary: dict, rec_variant: str, top_k: int) -> List[dict]:
    rows = []
    for slug, page in (summary.get("pages", {}) or {}).items():
        v = (page or {}).get("variants", {}) or {}
        s1 = v.get("S1_paddle_best_single") or {}
        r = v.get(rec_variant) or {}
        s1_br = float(s1.get("base_recall_ratio") or 0.0)
        r_br = float(r.get("base_recall_ratio") or 0.0)
        score = (
            3.0 * (r_br - s1_br)
            + 2.0 * (float(r.get("text_area_ratio", 0.0)) - float(s1.get("text_area_ratio", 0.0)))
            + 0.7 * (1.0 - s1_br)
        )
        rows.append(
            {
                "slug": slug,
                "score": score,
                "cov_gain": r_br - s1_br,
                "ta_gain": float(r.get("text_area_ratio", 0.0)) - float(s1.get("text_area_ratio", 0.0)),
            }
        )
    rows.sort(key=lambda x: x["score"], reverse=True)
    return rows[: max(0, int(top_k))]


def _top_miner_delta_rows(summary: dict, top_k: int) -> List[dict]:
    # Compare the canonical "no MinerU" vs "with MinerU" variants.
    v_no = "P2_paddle_union4_plus_dell"
    v_with = "P4_paddle_union4_plus_dell_plus_mineru"

    rows = []
    for slug, page in (summary.get("pages", {}) or {}).items():
        v = (page or {}).get("variants", {}) or {}
        a = v.get(v_no) or {}
        b = v.get(v_with) or {}
        if not a or not b:
            continue
        a_br = float(a.get("base_recall_ratio") or 0.0)
        b_br = float(b.get("base_recall_ratio") or 0.0)
        a_ta = float(a.get("text_area_ratio") or 0.0)
        b_ta = float(b.get("text_area_ratio") or 0.0)
        score = 3.5 * (b_br - a_br) + 1.5 * (b_ta - a_ta) + 0.5 * (1.0 - a_br)
        rows.append(
            {
                "slug": slug,
                "score": score,
                "br_gain": b_br - a_br,
                "ta_gain": b_ta - a_ta,
            }
        )

    rows.sort(key=lambda x: x["score"], reverse=True)
    return rows[: max(0, int(top_k))]


def build_review_bundle(
    images: Iterable[Path],
    run_dir: Path,
    paddle_layout_variant_ids: List[str],
    paddle_vl15_variant_id: str,
    dell_variant_id: str,
    mineru_variant_id: str,
    fusion_root: Path,
    mode: str = "all",
    top_k_informative: int = 20,
    top_k_miner_delta: int = 20,
) -> Path:
    summary = read_json(fusion_root / "summary.json")
    rec = summary.get("recommended_variant")
    review_root = run_dir / "review"
    pages_root = review_root / "pages"
    main_png = review_root / "main_png"
    pages_root.mkdir(parents=True, exist_ok=True)
    main_png.mkdir(parents=True, exist_ok=True)

    images_list = list(images)
    slug_to_image = {p.stem: p for p in images_list}

    if mode not in ("all", "top20"):
        raise ValueError(f"Unknown review mode: {mode}. Expected: all|top20")

    slugs_to_render = list(slug_to_image.keys())
    if mode == "top20":
        slugs: set[str] = set()
        if rec:
            slugs.update([r["slug"] for r in _top_informative_rows(summary, rec, top_k_informative)])
        slugs.update([r["slug"] for r in _top_miner_delta_rows(summary, top_k_miner_delta)])
        if not slugs:
            slugs = set(slugs_to_render[: min(5, len(slugs_to_render))])
        slugs_to_render = [s for s in slugs_to_render if s in slugs]

    for slug in slugs_to_render:
        image = slug_to_image[slug]
        page = pages_root / slug
        page.mkdir(parents=True, exist_ok=True)

        input_png = page / "01_input.png"
        shutil.copy2(image, input_png)

        paddle_imgs = []
        for idx, vid in enumerate(paddle_layout_variant_ids, 1):
            src = run_dir / "outputs" / "sources" / "paddle_layout" / vid / slug / "layout_boxes.normalized.json"
            out = page / f"02{chr(ord('a') + idx - 1)}_{vid}.png"
            if src.exists():
                boxes = read_json(src).get("boxes", [])
                draw_boxes(image, boxes, out, f"{vid}: {slug}")
            paddle_imgs.append((f"Paddle {vid}", out))

        pvl_src = run_dir / "outputs" / "sources" / "paddle_vl15" / paddle_vl15_variant_id / slug / "layout_boxes.normalized.json"
        pvl_out = page / "02d_paddle_vl15.png"
        if pvl_src.exists():
            draw_boxes(image, read_json(pvl_src).get("boxes", []), pvl_out, f"{paddle_vl15_variant_id}: {slug}")

        # Dedicated Paddle-only board for full visibility into the four Paddle sources.
        _board(
            paddle_imgs + [(f"Paddle {paddle_vl15_variant_id}", pvl_out)],
            page / "02e_paddle4_board.png",
            f"{slug} | Paddle4 sources",
            cols=2,
            tile_w=980,
            tile_h=1200,
        )

        dell_src = run_dir / "outputs" / "sources" / "dell" / dell_variant_id / slug / "layout_boxes.normalized.json"
        dell_out = page / "03_dell_layout.png"
        if dell_src.exists():
            draw_boxes(image, read_json(dell_src).get("boxes", []), dell_out, f"{dell_variant_id}: {slug}")

        miner_src = run_dir / "outputs" / "sources" / "mineru" / mineru_variant_id / slug / "layout_boxes.normalized.json"
        miner_out = page / "04_mineru_layout.png"
        if miner_src.exists():
            draw_boxes(image, read_json(miner_src).get("boxes", []), miner_out, f"{mineru_variant_id}: {slug}")

        fused_out = page / "05_fused_layout.png"
        if rec:
            rec_boxes = fusion_root / rec / slug / "fused_boxes.json"
            if rec_boxes.exists():
                draw_boxes(image, read_json(rec_boxes).get("boxes", []), fused_out, f"{rec}: {slug}")
                shutil.copy2(fused_out, main_png / f"{slug}_fused.png")

        # Explicit miner contribution panel: with vs without MinerU.
        no_miner_out = page / "05a_fused_no_miner.png"
        with_miner_out = page / "05b_fused_with_miner.png"
        v_no_miner = fusion_root / "P2_paddle_union4_plus_dell" / slug / "fused_boxes.json"
        v_with_miner = fusion_root / "P4_paddle_union4_plus_dell_plus_mineru" / slug / "fused_boxes.json"
        if v_no_miner.exists():
            draw_boxes(
                image,
                read_json(v_no_miner).get("boxes", []),
                no_miner_out,
                f"P2 paddle4+dell (no MinerU): {slug}",
            )
        if v_with_miner.exists():
            draw_boxes(
                image,
                read_json(v_with_miner).get("boxes", []),
                with_miner_out,
                f"P4 paddle4+dell+MinerU: {slug}",
            )
        if no_miner_out.exists() or with_miner_out.exists():
            _board(
                [("No MinerU (P2)", no_miner_out), ("With MinerU (P4)", with_miner_out)],
                page / "07_with_vs_without_miner.png",
                f"{slug} | MinerU contribution",
                cols=2,
                tile_w=980,
                tile_h=1200,
            )

        board_images = [("Input", input_png)] + paddle_imgs + [
            (f"Paddle {paddle_vl15_variant_id}", pvl_out),
            ("Dell", dell_out),
            ("MinerU", miner_out),
            (f"Fused ({rec})", fused_out),
        ]
        _board(board_images, page / "06_board.png", f"{slug} | layout bagging", cols=3)

        readme = page / "README.txt"
        readme.write_text(
            "\n".join(
                [
                    f"Page: {slug}",
                    f"Recommended fused variant: {rec}",
                    "Files:",
                    "- 01_input.png: original scan",
                    "- 02a..02d: each Paddle source overlay (3 layout detectors + VL1.5)",
                    "- 02e_paddle4_board.png: Paddle-only comparison board",
                    "- 03_dell_layout.png: Dell overlay",
                    "- 04_mineru_layout.png: MinerU overlay",
                    "- 05_fused_layout.png: fused output",
                    "- 06_board.png: compact comparison board",
                    "- 07_with_vs_without_miner.png: direct P2 vs P4 comparison",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    _build_top20_pack(run_dir, fusion_root, pages_root, top_k=top_k_informative)
    _build_top20_miner_delta_pack(run_dir, fusion_root, pages_root, top_k=top_k_miner_delta)
    return review_root


def _build_top20_pack(run_dir: Path, fusion_root: Path, pages_root: Path, top_k: int = 20) -> None:
    summary = read_json(fusion_root / "summary.json")
    rec = summary.get("recommended_variant")
    if not rec:
        return

    top = _top_informative_rows(summary, rec, top_k=top_k)

    pack = run_dir / "review" / "top20_informative"
    pack_pages = pack / "pages"
    pack_pages.mkdir(parents=True, exist_ok=True)

    ranking = pack / "ranking_top20.tsv"
    with ranking.open("w", encoding="utf-8") as f:
        f.write("rank\tslug\tscore\tcov_gain\tta_gain\n")
        for idx, row in enumerate(top, 1):
            f.write(f"{idx}\t{row['slug']}\t{row['score']:.6f}\t{row['cov_gain']:.6f}\t{row['ta_gain']:.6f}\n")
            src = pages_root / row["slug"]
            dst = pack_pages / f"{idx:02d}_{row['slug']}"
            if dst.exists():
                shutil.rmtree(dst)
            if src.exists():
                shutil.copytree(src, dst)

    (pack / "README.txt").write_text(
        "\n".join(
            [
                "Top 20 most informative pages for manual layout review.",
                "Score favors fused-vs-best-single gains and hard baseline pages:",
                "score = 3.0 * base_recall_gain + 2.0 * text_area_gain + 0.7 * (1 - baseline_base_recall)",
                f"Recommended fused variant in this run: {rec}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _build_top20_miner_delta_pack(run_dir: Path, fusion_root: Path, pages_root: Path, top_k: int = 20) -> None:
    summary = read_json(fusion_root / "summary.json")

    # Compare the canonical "no MinerU" vs "with MinerU" variants.
    v_no = "P2_paddle_union4_plus_dell"
    v_with = "P4_paddle_union4_plus_dell_plus_mineru"

    top = _top_miner_delta_rows(summary, top_k=top_k)

    pack = run_dir / "review" / "top20_miner_delta"
    pack_pages = pack / "pages"
    pack_pages.mkdir(parents=True, exist_ok=True)

    ranking = pack / "ranking_top20.tsv"
    with ranking.open("w", encoding="utf-8") as f:
        f.write("rank\tslug\tscore\tbase_recall_gain\ttext_area_gain\n")
        for idx, row in enumerate(top, 1):
            f.write(f"{idx}\t{row['slug']}\t{row['score']:.6f}\t{row['br_gain']:.6f}\t{row['ta_gain']:.6f}\n")
            src = pages_root / row["slug"]
            dst = pack_pages / f"{idx:02d}_{row['slug']}"
            if dst.exists():
                shutil.rmtree(dst)
            if src.exists():
                shutil.copytree(src, dst)

    (pack / "README.txt").write_text(
        "\n".join(
            [
                "Top 20 pages where MinerU changes the fused layout the most (P4 vs P2).",
                "score = 3.5 * base_recall_gain + 1.5 * text_area_gain + 0.5 * (1 - baseline_base_recall)",
                f"no_miner_variant={v_no}",
                f"with_miner_variant={v_with}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
