from __future__ import annotations

import csv
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from newsbag.config import PaddleLayoutVariant, PaddleVL15Config, PipelineConfig
from newsbag.labels import label_counts, normalize_label
from newsbag.utils.io import read_json, write_json
from newsbag.utils.proc import run_cmd


@dataclass
class _LayoutImageResult:
    slug: str
    normalized_json: Optional[Path]
    row: List[Any]
    source_labels: List[str]
    norm_labels: List[str]


@dataclass
class _VL15ImageResult:
    slug: str
    normalized_json: Optional[Path]
    row: List[Any]
    source_labels: List[str]
    norm_labels: List[str]
    block_labels: List[str]


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


def _resolve_layout_worker_count(cfg: PipelineConfig) -> int:
    # Default to 3 workers on GPU. 4+ can exceed memory on large pages in Torch L40S jobs.
    # Users can override with NEWSBAG_PADDLE_LAYOUT_WORKERS.
    default_workers = 3 if any(str(d).startswith("gpu") for d in cfg.device_order) else 1
    raw = os.environ.get("NEWSBAG_PADDLE_LAYOUT_WORKERS", str(default_workers)).strip()
    try:
        workers = int(raw)
    except Exception:
        workers = default_workers
    return max(1, workers)


def _resolve_vl15_worker_count(cfg: PipelineConfig) -> int:
    # Default to 3 workers on GPU. VL1.5 is heavier than layout_detection, but 3 workers
    # is a better throughput/utilization tradeoff for L40S shards than the prior default 2.
    default_workers = 3 if any(str(d).startswith("gpu") for d in cfg.device_order) else 1
    raw = os.environ.get("NEWSBAG_PADDLE_VL15_WORKERS", str(default_workers)).strip()
    try:
        workers = int(raw)
    except Exception:
        workers = default_workers
    return max(1, workers)


def _run_one_paddle_layout_image(
    cfg: PipelineConfig,
    variant: PaddleLayoutVariant,
    image: Path,
    variant_root: Path,
    log_dir: Path,
    worker_cpu_threads: int,
) -> _LayoutImageResult:
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
            source_labels = [str(b.get("source_label") or "") for b in nb]
            norm_labels = [str(b.get("norm_label") or "") for b in nb]
        except Exception:
            source_labels, norm_labels = [], []
        return _LayoutImageResult(
            slug=slug,
            normalized_json=normalized_json,
            row=[str(image), variant.variant_id, "resume", "ok", 0, 0, str(raw_json), str(raw_png), ""],
            source_labels=source_labels,
            norm_labels=norm_labels,
        )

    rc = 1
    used_device = ""
    elapsed_sec = 0.0
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
            str(worker_cpu_threads),
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
        elapsed_sec = result.seconds
        status = "ok" if rc == 0 and raw_json.exists() else f"fail({rc})"
        if status == "ok":
            break
        if idx == 0 and device.startswith("gpu") and cfg.allow_gpu_to_cpu_fallback:
            continue
        break

    if rc != 0 or not raw_json.exists():
        return _LayoutImageResult(
            slug=slug,
            normalized_json=None,
            row=[str(image), variant.variant_id, used_device, f"fail({rc})", rc, 0, str(raw_json), str(raw_png), str(log_file)],
            source_labels=[],
            norm_labels=[],
        )

    normalized = _load_layout_boxes_from_res(raw_json, source_model=variant.variant_id)
    write_json(normalized_json, {"slug": slug, "variant": variant.variant_id, "boxes": normalized})
    write_json(labels_json, {"slug": slug, "variant": variant.variant_id, "source_label_counts": label_counts(normalized)})
    return _LayoutImageResult(
        slug=slug,
        normalized_json=normalized_json,
        row=[str(image), variant.variant_id, used_device, "ok", 0, elapsed_sec, str(raw_json), str(raw_png), str(log_file)],
        source_labels=[str(b.get("source_label") or "") for b in normalized],
        norm_labels=[str(b.get("norm_label") or "") for b in normalized],
    )


def run_paddle_layout_variants(cfg: PipelineConfig, images: Iterable[Path], run_dir: Path) -> Dict[str, Dict[str, Path]]:
    source_root = run_dir / "outputs" / "sources" / "paddle_layout"
    report_dir = run_dir / "reports"
    log_dir = run_dir / "logs"
    source_root.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    outputs: Dict[str, Dict[str, Path]] = {}
    image_list = list(images)
    workers = _resolve_layout_worker_count(cfg)
    worker_cpu_threads = max(1, int(cfg.cpu_threads) // workers)

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

            def handle_result(res: _LayoutImageResult) -> None:
                wr.writerow(res.row)
                if res.normalized_json is not None:
                    per_page[res.slug] = res.normalized_json
                agg_source_labels.update(res.source_labels)
                agg_norm_labels.update(res.norm_labels)

            if workers == 1:
                for image in image_list:
                    handle_result(
                        _run_one_paddle_layout_image(
                            cfg,
                            variant,
                            image,
                            variant_root,
                            log_dir,
                            worker_cpu_threads,
                        )
                    )
            else:
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    futs = [
                        ex.submit(
                            _run_one_paddle_layout_image,
                            cfg,
                            variant,
                            image,
                            variant_root,
                            log_dir,
                            worker_cpu_threads,
                        )
                        for image in image_list
                    ]
                    for fut in as_completed(futs):
                        handle_result(fut.result())

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


def _run_one_paddle_vl15_image(
    cfg: PipelineConfig,
    pvl_cfg: PaddleVL15Config,
    image: Path,
    variant_root: Path,
    log_dir: Path,
    worker_cpu_threads: int,
) -> _VL15ImageResult:
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
        source_labels: List[str] = []
        norm_labels: List[str] = []
        block_labels: List[str] = []
        try:
            nb = read_json(normalized_json).get("boxes") or []
            source_labels = [str(b.get("source_label") or "") for b in nb]
            norm_labels = [str(b.get("norm_label") or "") for b in nb]
            pb = read_json(parsing_json).get("parsing_res_list") or []
            block_labels = [str(x.get("block_label") or "") for x in pb if isinstance(x, dict)]
        except Exception:
            pass
        return _VL15ImageResult(
            slug=slug,
            normalized_json=normalized_json,
            row=[str(image), pvl_cfg.variant_id, "resume", "ok", 0, 0, str(raw_dir / f"{slug}_res.json"), str(raw_dir / f"{slug}_layout_det_res.png"), str(raw_dir / f"{slug}.md"), ""],
            source_labels=source_labels,
            norm_labels=norm_labels,
            block_labels=block_labels,
        )

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
    elapsed_sec = 0.0
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
            str(worker_cpu_threads),
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
        elapsed_sec = result.seconds
        status = "ok" if rc == 0 and flat_json.exists() else f"fail({rc})"
        if status == "ok":
            break
        if idx == 0 and device.startswith("gpu") and cfg.allow_gpu_to_cpu_fallback:
            continue
        break

    status = "ok" if rc == 0 and flat_json.exists() else f"fail({rc})"
    if status != "ok":
        return _VL15ImageResult(
            slug=slug,
            normalized_json=None,
            row=[str(image), pvl_cfg.variant_id, used_device, status, rc, 0, str(flat_json), str(flat_png), str(flat_md), str(log_file)],
            source_labels=[],
            norm_labels=[],
            block_labels=[],
        )

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

    for src in [flat_json, flat_png, flat_md, flat_docx, flat_tex]:
        if src.exists():
            shutil.move(str(src), str(raw_dir / src.name))

    return _VL15ImageResult(
        slug=slug,
        normalized_json=normalized_json,
        row=[
            str(image),
            pvl_cfg.variant_id,
            used_device,
            "ok",
            0,
            elapsed_sec,
            str(raw_dir / f"{slug}_res.json"),
            str(raw_dir / f"{slug}_layout_det_res.png"),
            str(raw_dir / f"{slug}.md"),
            str(log_file),
        ],
        source_labels=[str(b.get("source_label") or "") for b in layout_boxes],
        norm_labels=[str(b.get("norm_label") or "") for b in layout_boxes],
        block_labels=[str(x.get("block_label") or "") for x in parsing_blocks if isinstance(x, dict)],
    )


def run_paddle_vl15_docparser(cfg: PipelineConfig, pvl_cfg: PaddleVL15Config, images: Iterable[Path], run_dir: Path) -> Dict[str, Path]:
    variant_root = run_dir / "outputs" / "sources" / "paddle_vl15" / pvl_cfg.variant_id
    variant_root.mkdir(parents=True, exist_ok=True)
    report_tsv = run_dir / "reports" / f"paddle_vl15_{pvl_cfg.variant_id}.tsv"
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    image_list = list(images)
    workers = _resolve_vl15_worker_count(cfg)
    worker_cpu_threads = max(1, int(cfg.cpu_threads) // workers)

    per_page: Dict[str, Path] = {}
    agg_source_labels: Counter[str] = Counter()
    agg_norm_labels: Counter[str] = Counter()
    agg_block_labels: Counter[str] = Counter()
    with report_tsv.open("w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f, delimiter="\t")
        wr.writerow(["image", "variant", "device", "status", "rc", "seconds", "json", "layout_png", "md", "log"])

        def handle_result(res: _VL15ImageResult) -> None:
            wr.writerow(res.row)
            if res.normalized_json is not None:
                per_page[res.slug] = res.normalized_json
            agg_source_labels.update(res.source_labels)
            agg_norm_labels.update(res.norm_labels)
            agg_block_labels.update(res.block_labels)

        if workers == 1:
            for image in image_list:
                handle_result(
                    _run_one_paddle_vl15_image(
                        cfg=cfg,
                        pvl_cfg=pvl_cfg,
                        image=image,
                        variant_root=variant_root,
                        log_dir=log_dir,
                        worker_cpu_threads=worker_cpu_threads,
                    )
                )
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = [
                    ex.submit(
                        _run_one_paddle_vl15_image,
                        cfg,
                        pvl_cfg,
                        image,
                        variant_root,
                        log_dir,
                        worker_cpu_threads,
                    )
                    for image in image_list
                ]
                for fut in as_completed(futs):
                    handle_result(fut.result())

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
