from __future__ import annotations

import csv
from collections import Counter
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from newsbag.config import PaddleLayoutVariant, PaddleVL15Config, PipelineConfig
from newsbag.labels import label_counts, normalize_label
from newsbag.utils.io import read_json, write_json
from newsbag.utils.proc import run_cmd


def _normalize_box_record(raw: Dict[str, Any], source_model: str) -> Optional[Dict[str, Any]]:
    coord = raw.get("coordinate") or raw.get("bbox_xyxy")
    if not coord or len(coord) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in coord]
    except Exception:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    source_label = str(raw.get("label") or "text")
    return {
        "source_family": "paddle",
        "source_model": source_model,
        "source_label": source_label,
        "norm_label": normalize_label(source_label),
        "bbox_xyxy": [x1, y1, x2, y2],
        "score": float(raw.get("score") or 0.0),
        "reading_order": raw.get("order"),
        "text": None,
    }


def _load_layout_boxes_from_res(path: Path, source_model: str) -> List[Dict[str, Any]]:
    payload = read_json(path)
    obj = payload.get("res") or payload
    boxes = obj.get("boxes") or []
    out: List[Dict[str, Any]] = []
    for b in boxes:
        row = _normalize_box_record(b, source_model)
        if row:
            out.append(row)
    return out


def run_paddle_layout_variants(cfg: PipelineConfig, images: Iterable[Path], run_dir: Path) -> Dict[str, Dict[str, Path]]:
    source_root = run_dir / "outputs" / "sources" / "paddle_layout"
    report_dir = run_dir / "reports"
    log_dir = run_dir / "logs"
    source_root.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    outputs: Dict[str, Dict[str, Path]] = {}

    for variant in cfg.paddle_layout_variants:
        variant_root = source_root / variant.variant_id
        variant_root.mkdir(parents=True, exist_ok=True)
        report_tsv = report_dir / f"paddle_layout_{variant.variant_id}.tsv"
        agg_source_labels: Counter[str] = Counter()
        agg_norm_labels: Counter[str] = Counter()
        with report_tsv.open("w", encoding="utf-8", newline="") as f:
            wr = csv.writer(f, delimiter="\t")
            wr.writerow(["image", "variant", "device", "status", "rc", "seconds", "json", "overlay", "log"])

            per_page: Dict[str, Path] = {}
            for image in images:
                slug = image.stem
                page_dir = variant_root / slug
                raw_json = page_dir / f"{slug}_res.json"
                raw_png = page_dir / f"{slug}_res.png"
                normalized_json = page_dir / "layout_boxes.normalized.json"
                labels_json = page_dir / "labels_source_counts.json"

                # If we have the raw JSON already, re-normalize without re-running Paddle.
                if cfg.resume and raw_json.exists() and not normalized_json.exists():
                    normalized = _load_layout_boxes_from_res(raw_json, source_model=variant.variant_id)
                    write_json(normalized_json, {"slug": slug, "variant": variant.variant_id, "boxes": normalized})
                    write_json(
                        labels_json,
                        {
                            "slug": slug,
                            "variant": variant.variant_id,
                            "source_label_counts": label_counts(normalized),
                        },
                    )

                if cfg.resume and normalized_json.exists() and raw_json.exists():
                    try:
                        nb = read_json(normalized_json).get("boxes") or []
                        agg_source_labels.update([str(b.get("source_label") or "") for b in nb])
                        agg_norm_labels.update([str(b.get("norm_label") or "") for b in nb])
                    except Exception:
                        pass
                    per_page[slug] = normalized_json
                    wr.writerow([str(image), variant.variant_id, "resume", "ok", 0, 0, str(raw_json), str(raw_png), ""])
                    continue

                rc = 1
                used_device = ""
                log_file = log_dir / f"paddle_layout_{variant.variant_id}_{slug}.log"

                # Paddle layout_detection writes outputs into --save_path. Use per-page directories to avoid
                # collisions and keep the output contract stable.
                if page_dir.exists():
                    shutil.rmtree(page_dir)
                page_dir.mkdir(parents=True, exist_ok=True)

                for idx, device in enumerate(cfg.device_order):
                    used_device = device
                    cmd = [
                        cfg.paddleocr_bin,
                        "layout_detection",
                        "-i",
                        str(image),
                        "--save_path",
                        str(page_dir),
                        "--device",
                        device,
                        "--cpu_threads",
                        str(cfg.cpu_threads),
                        "--enable_mkldnn",
                        "False",
                        "--model_name",
                        variant.model_name,
                        "--threshold",
                        str(variant.threshold),
                    ]
                    result = run_cmd(
                        cmd,
                        log_file,
                        timeout_sec=variant.timeout_sec,
                        env={"PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True"},
                    )
                    rc = result.rc
                    status = "ok" if rc == 0 and raw_json.exists() else f"fail({rc})"
                    if status == "ok":
                        break
                    if idx == 0 and device.startswith("gpu") and cfg.allow_gpu_to_cpu_fallback:
                        continue
                    break

                if rc != 0 or not raw_json.exists():
                    wr.writerow([str(image), variant.variant_id, used_device, f"fail({rc})", rc, 0, str(raw_json), str(raw_png), str(log_file)])
                    continue

                normalized = _load_layout_boxes_from_res(raw_json, source_model=variant.variant_id)
                write_json(normalized_json, {"slug": slug, "variant": variant.variant_id, "boxes": normalized})
                write_json(labels_json, {"slug": slug, "variant": variant.variant_id, "source_label_counts": label_counts(normalized)})
                agg_source_labels.update([str(b.get("source_label") or "") for b in normalized])
                agg_norm_labels.update([str(b.get("norm_label") or "") for b in normalized])
                per_page[slug] = normalized_json
                wr.writerow([str(image), variant.variant_id, used_device, "ok", 0, result.seconds, str(raw_json), str(raw_png), str(log_file)])

        outputs[variant.variant_id] = per_page
        write_json(
            report_dir / f"paddle_layout_{variant.variant_id}_labels_aggregate.json",
            {
                "variant": variant.variant_id,
                "source_family": "paddle",
                "total_boxes": int(sum(agg_source_labels.values())),
                "source_label_counts": dict(sorted(agg_source_labels.items(), key=lambda kv: (-kv[1], kv[0]))),
                "norm_label_counts": dict(sorted(agg_norm_labels.items(), key=lambda kv: (-kv[1], kv[0]))),
            },
        )

    return outputs


def _extract_vl15_layout_boxes(res_json: Dict[str, Any], variant_id: str) -> List[Dict[str, Any]]:
    lres = (res_json.get("layout_det_res") or {}).get("boxes") or []
    out: List[Dict[str, Any]] = []
    for b in lres:
        row = _normalize_box_record(b, source_model=variant_id)
        if row:
            out.append(row)
    return out


def run_paddle_vl15_docparser(cfg: PipelineConfig, pvl_cfg: PaddleVL15Config, images: Iterable[Path], run_dir: Path) -> Dict[str, Path]:
    variant_root = run_dir / "outputs" / "sources" / "paddle_vl15" / pvl_cfg.variant_id
    variant_root.mkdir(parents=True, exist_ok=True)
    report_tsv = run_dir / "reports" / f"paddle_vl15_{pvl_cfg.variant_id}.tsv"
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    per_page: Dict[str, Path] = {}
    agg_source_labels: Counter[str] = Counter()
    agg_norm_labels: Counter[str] = Counter()
    agg_block_labels: Counter[str] = Counter()
    with report_tsv.open("w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f, delimiter="\t")
        wr.writerow(["image", "variant", "device", "status", "rc", "seconds", "json", "layout_png", "md", "log"])

        for image in images:
            slug = image.stem
            page_dir = variant_root / slug
            page_dir.mkdir(parents=True, exist_ok=True)
            raw_dir = page_dir / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)

            normalized_json = page_dir / "layout_boxes.normalized.json"
            labels_json = page_dir / "labels_source_counts.json"
            parsing_json = page_dir / "parsing_blocks.json"
            table_html = page_dir / "table_blocks.html"

            if cfg.resume and normalized_json.exists() and parsing_json.exists():
                try:
                    nb = read_json(normalized_json).get("boxes") or []
                    agg_source_labels.update([str(b.get("source_label") or "") for b in nb])
                    agg_norm_labels.update([str(b.get("norm_label") or "") for b in nb])
                    pb = read_json(parsing_json).get("parsing_res_list") or []
                    agg_block_labels.update([str(x.get("block_label") or "") for x in pb if isinstance(x, dict)])
                except Exception:
                    pass
                per_page[slug] = normalized_json
                wr.writerow([str(image), pvl_cfg.variant_id, "resume", "ok", 0, 0, str(raw_dir / f"{slug}_res.json"), str(raw_dir / f"{slug}_layout_det_res.png"), str(raw_dir / f"{slug}.md"), ""])
                continue

            flat_json = variant_root / f"{slug}_res.json"
            flat_png = variant_root / f"{slug}_layout_det_res.png"
            flat_md = variant_root / f"{slug}.md"
            flat_docx = variant_root / f"{slug}.docx"
            flat_tex = variant_root / f"{slug}.tex"

            for p in [flat_json, flat_png, flat_md, flat_docx, flat_tex]:
                if p.exists():
                    p.unlink()

            rc = 1
            used_device = ""
            log_file = log_dir / f"paddle_vl15_{slug}.log"

            for idx, device in enumerate(cfg.device_order):
                used_device = device
                cmd = [
                    cfg.paddleocr_bin,
                    "doc_parser",
                    "-i",
                    str(image),
                    "--save_path",
                    str(variant_root),
                    "--pipeline_version",
                    "v1.5",
                    "--layout_detection_model_name",
                    pvl_cfg.layout_model,
                    "--layout_threshold",
                    str(pvl_cfg.layout_threshold),
                    "--use_doc_orientation_classify",
                    "False",
                    "--use_doc_unwarping",
                    "False",
                    "--use_layout_detection",
                    "True",
                    "--use_chart_recognition",
                    "False",
                    "--use_seal_recognition",
                    "False",
                    "--use_ocr_for_image_block",
                    "False",
                    "--format_block_content",
                    "False",
                    "--merge_layout_blocks",
                    "True",
                    "--device",
                    device,
                    "--cpu_threads",
                    str(cfg.cpu_threads),
                    "--enable_mkldnn",
                    "False",
                ]
                result = run_cmd(
                    cmd,
                    log_file,
                    timeout_sec=pvl_cfg.timeout_sec,
                    env={"PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True"},
                )
                rc = result.rc
                status = "ok" if rc == 0 and flat_json.exists() else f"fail({rc})"
                if status == "ok":
                    break
                if idx == 0 and device.startswith("gpu") and cfg.allow_gpu_to_cpu_fallback:
                    continue
                break

            status = "ok" if rc == 0 and flat_json.exists() else f"fail({rc})"
            if status != "ok":
                wr.writerow([str(image), pvl_cfg.variant_id, used_device, status, rc, 0, str(flat_json), str(flat_png), str(flat_md), str(log_file)])
                continue

            res_obj = read_json(flat_json)
            layout_boxes = _extract_vl15_layout_boxes(res_obj, pvl_cfg.variant_id)
            parsing_blocks = res_obj.get("parsing_res_list") or []
            table_chunks = []
            for block in parsing_blocks:
                if str(block.get("block_label", "")).lower() == "table":
                    content = block.get("block_content")
                    if isinstance(content, str) and content.strip():
                        table_chunks.append(content.strip())

            write_json(normalized_json, {"slug": slug, "variant": pvl_cfg.variant_id, "boxes": layout_boxes})
            write_json(labels_json, {"slug": slug, "variant": pvl_cfg.variant_id, "source_label_counts": label_counts(layout_boxes)})
            write_json(parsing_json, {"slug": slug, "variant": pvl_cfg.variant_id, "parsing_res_list": parsing_blocks})
            table_html.write_text("\n\n".join(table_chunks), encoding="utf-8")
            agg_source_labels.update([str(b.get("source_label") or "") for b in layout_boxes])
            agg_norm_labels.update([str(b.get("norm_label") or "") for b in layout_boxes])
            agg_block_labels.update([str(x.get("block_label") or "") for x in parsing_blocks if isinstance(x, dict)])

            for src in [flat_json, flat_png, flat_md, flat_docx, flat_tex]:
                if src.exists():
                    shutil.move(str(src), str(raw_dir / src.name))

            per_page[slug] = normalized_json
            wr.writerow([
                str(image),
                pvl_cfg.variant_id,
                used_device,
                "ok",
                0,
                result.seconds,
                str(raw_dir / f"{slug}_res.json"),
                str(raw_dir / f"{slug}_layout_det_res.png"),
                str(raw_dir / f"{slug}.md"),
                str(log_file),
            ])

    _write_vl15_aggregate_reports(
        run_dir=run_dir,
        variant_id=pvl_cfg.variant_id,
        agg_source_labels=agg_source_labels,
        agg_norm_labels=agg_norm_labels,
        agg_block_labels=agg_block_labels,
    )
    return per_page


def _write_vl15_aggregate_reports(run_dir: Path, variant_id: str, agg_source_labels: Counter[str], agg_norm_labels: Counter[str], agg_block_labels: Counter[str]) -> None:
    report_dir = run_dir / "reports"
    write_json(
        report_dir / f"paddle_vl15_{variant_id}_labels_aggregate.json",
        {
            "variant": variant_id,
            "source_family": "paddle",
            "total_layout_boxes": int(sum(agg_source_labels.values())),
            "layout_source_label_counts": dict(sorted(agg_source_labels.items(), key=lambda kv: (-kv[1], kv[0]))),
            "layout_norm_label_counts": dict(sorted(agg_norm_labels.items(), key=lambda kv: (-kv[1], kv[0]))),
            "parsing_block_label_counts": dict(sorted(agg_block_labels.items(), key=lambda kv: (-kv[1], kv[0]))),
        },
    )
