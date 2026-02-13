from __future__ import annotations

import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

from newsbag.config import DellConfig, MinerConfig
from newsbag.labels import label_counts, normalize_label
from newsbag.utils.io import read_json, write_json
from newsbag.utils.proc import run_cmd


def _normalize_external_boxes(boxes: List[dict], source_family: str, source_model: str) -> List[dict]:
    out = []
    for b in boxes:
        bb = b.get("bbox_xyxy")
        if not bb or len(bb) != 4:
            continue
        try:
            x1, y1, x2, y2 = [float(v) for v in bb]
        except Exception:
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        source_label = str(b.get("label") or "text")
        out.append(
            {
                "source_family": source_family,
                "source_model": source_model,
                "source_label": source_label,
                "norm_label": normalize_label(source_label),
                "bbox_xyxy": [x1, y1, x2, y2],
                "score": float(b.get("score") or 0.0) if b.get("score") is not None else None,
                "reading_order": b.get("reading_order"),
                "text": b.get("text"),
            }
        )
    return out


def _load_manifest_slugs(manifest: Path) -> Dict[str, Path]:
    rows: Dict[str, Path] = {}
    for ln in manifest.read_text(encoding="utf-8").splitlines():
        p = os.path.expandvars(ln.strip())
        if not p or p.startswith("#"):
            continue
        ip = Path(p).expanduser()
        rows[ip.stem] = ip
    return rows


def run_dell(dcfg: DellConfig, manifest: Path, run_dir: Path, resume: bool = True) -> Dict[str, Path]:
    if not dcfg.model_path:
        raise ValueError("dell.model_path is required when dell.enabled=true")
    if not dcfg.label_map_path:
        raise ValueError("dell.label_map_path is required when dell.enabled=true")

    out_root = run_dir / "outputs" / "sources" / "dell" / dcfg.variant_id
    out_root.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / "logs" / f"dell_{dcfg.variant_id}.log"

    cmd = [
        dcfg.python_bin,
        dcfg.script_path,
        "--manifest",
        str(manifest),
        "--model",
        dcfg.model_path,
        "--label_map",
        dcfg.label_map_path,
        "--output_root",
        str(out_root),
        "--conf",
        str(dcfg.conf),
        "--iou",
        str(dcfg.iou),
        "--imgsz",
        str(dcfg.imgsz),
        "--provider",
        dcfg.provider,
    ]
    # Back-compat: older Dell scripts required --repo_src for helper imports.
    # Our current runner doesn't need it, but accept/pass-through when configured.
    if dcfg.repo_src:
        cmd.extend(["--repo_src", dcfg.repo_src])
    if resume:
        cmd.append("--resume")
    result = run_cmd(cmd, log_file, timeout_sec=12 * 3600)

    run_report = out_root / "run_report.json"
    report_obj = read_json(run_report) if run_report.exists() else {}
    providers_used = report_obj.get("providers_used") or []
    has_cuda_provider = any("CUDA" in str(x) for x in providers_used)
    if dcfg.require_cuda_provider and not has_cuda_provider:
        raise RuntimeError(
            f"Dell run did not use CUDAExecutionProvider (providers_used={providers_used}). "
            "This would likely cause low GPU utilization / cancellation on Torch."
        )

    report_tsv = run_dir / "reports" / f"dell_{dcfg.variant_id}.tsv"
    per_page: Dict[str, Path] = {}
    agg_source_labels: Counter[str] = Counter()
    agg_norm_labels: Counter[str] = Counter()
    ok_pages = 0
    nonempty_pages = 0
    manifest_slugs = _load_manifest_slugs(manifest)
    page_report_rows = report_obj.get("pages") or []
    page_report_map = {
        Path(str(r.get("image", ""))).stem: r for r in page_report_rows if isinstance(r, dict)
    }

    with report_tsv.open("w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f, delimiter="\t")
        wr.writerow(
            [
                "slug",
                "status",
                "boxes_json",
                "normalized_json",
                "providers_used",
                "log",
                "rc",
                "seconds",
            ]
        )

        for slug in sorted(manifest_slugs):
            page_dir = out_root / slug
            boxes_json = page_dir / f"{slug}_dell_layout_boxes.json"
            row_meta = page_report_map.get(slug, {})
            row_status = str(row_meta.get("status") or "")
            status_ok = row_status in ("ok", "resume")
            status = "ok" if status_ok and boxes_json.exists() else (row_status or "missing")
            norm_path = page_dir / "layout_boxes.normalized.json"

            if status == "ok":
                payload = read_json(boxes_json)
                boxes = payload.get("boxes") or []
                normalized = _normalize_external_boxes(boxes, "dell", dcfg.variant_id)
                labels_path = page_dir / "labels_source_counts.json"
                write_json(
                    norm_path,
                    {"slug": slug, "variant": dcfg.variant_id, "boxes": normalized},
                )
                write_json(
                    labels_path,
                    {
                        "slug": slug,
                        "variant": dcfg.variant_id,
                        "source_label_counts": label_counts(normalized),
                    },
                )
                per_page[slug] = norm_path
                agg_source_labels.update([str(b.get("source_label") or "") for b in normalized])
                agg_norm_labels.update([str(b.get("norm_label") or "") for b in normalized])
                ok_pages += 1
                if normalized:
                    nonempty_pages += 1

            wr.writerow(
                [
                    slug,
                    status,
                    str(boxes_json),
                    str(norm_path if norm_path.exists() else ""),
                    json.dumps(providers_used, ensure_ascii=False),
                    str(log_file),
                    result.rc,
                    result.seconds,
                ]
            )

    write_json(
        run_dir / "reports" / f"dell_{dcfg.variant_id}_labels_aggregate.json",
        {
            "variant": dcfg.variant_id,
            "source_family": "dell",
            "total_boxes": int(sum(agg_source_labels.values())),
            "source_label_counts": dict(sorted(agg_source_labels.items(), key=lambda kv: (-kv[1], kv[0]))),
            "norm_label_counts": dict(sorted(agg_norm_labels.items(), key=lambda kv: (-kv[1], kv[0]))),
            "providers_used": providers_used,
        },
    )
    if ok_pages > 0 and nonempty_pages < int(dcfg.min_nonempty_pages):
        raise RuntimeError(
            "Dell produced too few non-empty pages: "
            f"nonempty_pages={nonempty_pages}, ok_pages={ok_pages}, "
            f"required_min_nonempty_pages={dcfg.min_nonempty_pages}. "
            "This usually indicates a runner/config regression. "
            f"Check {run_dir / 'reports' / f'dell_{dcfg.variant_id}.tsv'} and {log_file}."
        )
    return per_page


def run_mineru(mcfg: MinerConfig, manifest: Path, run_dir: Path, resume: bool = True) -> Dict[str, Path]:
    out_root = run_dir / "outputs" / "sources" / "mineru" / mcfg.variant_id
    out_root.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / "logs" / f"mineru_{mcfg.variant_id}.log"

    cmd = [
        mcfg.python_bin,
        mcfg.script_path,
        "--manifest",
        str(manifest),
        "--output_root",
        str(out_root),
        "--model_id",
        mcfg.model_id,
    ]
    if resume:
        cmd.append("--resume")
    if mcfg.max_pages and mcfg.max_pages > 0:
        cmd.extend(["--max_pages", str(mcfg.max_pages)])

    result = run_cmd(cmd, log_file, timeout_sec=24 * 3600)

    run_meta = out_root / "run_meta.json"
    meta_obj = read_json(run_meta) if run_meta.exists() else {}
    used_cuda = bool(meta_obj.get("cuda_available", False))
    if mcfg.require_cuda and not used_cuda:
        raise RuntimeError(
            "MinerU run reports cuda_available=false. This run should not be on a GPU partition."
        )

    report_tsv = run_dir / "reports" / f"mineru_{mcfg.variant_id}.tsv"
    per_page: Dict[str, Path] = {}
    agg_source_labels: Counter[str] = Counter()
    agg_norm_labels: Counter[str] = Counter()
    ok_pages = 0
    nonempty_pages = 0
    manifest_slugs = _load_manifest_slugs(manifest)
    report_src = out_root / "run_report.tsv"
    report_rows: Dict[str, Dict[str, str]] = {}
    if report_src.exists():
        with report_src.open("r", encoding="utf-8") as rf:
            rr = csv.DictReader(rf, delimiter="\t")
            for row in rr:
                image = row.get("image", "")
                slug = Path(image).stem if image else ""
                if slug:
                    report_rows[slug] = row

    with report_tsv.open("w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f, delimiter="\t")
        wr.writerow(
            ["slug", "status", "boxes_json", "normalized_json", "cuda_available", "log", "rc", "seconds"]
        )

        for slug in sorted(manifest_slugs):
            page_dir = out_root / slug
            boxes_json = page_dir / f"{slug}_mineru_layout_boxes.json"
            row_meta = report_rows.get(slug, {})
            row_status = str(row_meta.get("status") or "")
            status_ok = row_status in ("ok", "resume")
            status = "ok" if status_ok and boxes_json.exists() else (row_status or "missing")
            norm_path = page_dir / "layout_boxes.normalized.json"

            if status == "ok":
                payload = read_json(boxes_json)
                boxes = payload.get("boxes") or []
                normalized = _normalize_external_boxes(boxes, "mineru", mcfg.variant_id)
                labels_path = page_dir / "labels_source_counts.json"
                write_json(
                    norm_path,
                    {"slug": slug, "variant": mcfg.variant_id, "boxes": normalized},
                )
                write_json(
                    labels_path,
                    {
                        "slug": slug,
                        "variant": mcfg.variant_id,
                        "source_label_counts": label_counts(normalized),
                    },
                )
                per_page[slug] = norm_path
                agg_source_labels.update([str(b.get("source_label") or "") for b in normalized])
                agg_norm_labels.update([str(b.get("norm_label") or "") for b in normalized])
                ok_pages += 1
                if normalized:
                    nonempty_pages += 1

            wr.writerow(
                [
                    slug,
                    status,
                    str(boxes_json),
                    str(norm_path if norm_path.exists() else ""),
                    int(used_cuda),
                    str(log_file),
                    result.rc,
                    result.seconds,
                ]
            )

    write_json(
        run_dir / "reports" / f"mineru_{mcfg.variant_id}_labels_aggregate.json",
        {
            "variant": mcfg.variant_id,
            "source_family": "mineru",
            "total_boxes": int(sum(agg_source_labels.values())),
            "source_label_counts": dict(sorted(agg_source_labels.items(), key=lambda kv: (-kv[1], kv[0]))),
            "norm_label_counts": dict(sorted(agg_norm_labels.items(), key=lambda kv: (-kv[1], kv[0]))),
            "cuda_available": bool(used_cuda),
            "model_id": meta_obj.get("model_id"),
        },
    )
    if ok_pages > 0 and nonempty_pages < int(mcfg.min_nonempty_pages):
        raise RuntimeError(
            "MinerU produced too few non-empty pages: "
            f"nonempty_pages={nonempty_pages}, ok_pages={ok_pages}, "
            f"required_min_nonempty_pages={mcfg.min_nonempty_pages}. "
            "This usually indicates a runner/config regression. "
            f"Check {run_dir / 'reports' / f'mineru_{mcfg.variant_id}.tsv'} and {log_file}."
        )
    return per_page
