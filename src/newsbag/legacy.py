from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

from newsbag.labels import normalize_label

BOX_LIST_KEYS = ("regions", "boxes", "candidates", "layout_boxes", "detections", "blocks")
NESTED_PAYLOAD_KEYS = ("res", "result", "payload", "data")
PAGE_ID_KEYS = ("page_id", "slug", "id", "image_id")


def read_legacy_json(path: Path) -> Any:
    return json.loads(path.expanduser().read_text(encoding="utf-8"))


def _number(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _bbox_from_points(points: list[Any]) -> list[float] | None:
    xs: list[float] = []
    ys: list[float] = []
    for point in points:
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            x = _number(point[0])
            y = _number(point[1])
            if x is not None and y is not None:
                xs.append(x)
                ys.append(y)
    if not xs or not ys:
        return None
    return [min(xs), min(ys), max(xs), max(ys)]


def coerce_bbox_xyxy(raw: Any) -> list[float] | None:
    if isinstance(raw, dict):
        for keys in (("x1", "y1", "x2", "y2"), ("left", "top", "right", "bottom")):
            if all(key in raw for key in keys):
                values = [_number(raw[key]) for key in keys]
                if all(value is not None for value in values):
                    return _valid_bbox([float(value) for value in values if value is not None])
        if all(key in raw for key in ("left", "top", "width", "height")):
            left = _number(raw.get("left"))
            top = _number(raw.get("top"))
            width = _number(raw.get("width"))
            height = _number(raw.get("height"))
            if None not in (left, top, width, height):
                return _valid_bbox([left, top, left + width, top + height])  # type: ignore[operator]
        return None

    if isinstance(raw, (list, tuple)):
        values = list(raw)
        if len(values) == 4 and not any(isinstance(item, (list, tuple, dict)) for item in values):
            numbers = [_number(item) for item in values]
            if all(value is not None for value in numbers):
                return _valid_bbox([float(value) for value in numbers if value is not None])
        if len(values) >= 4 and all(isinstance(item, (list, tuple)) for item in values):
            return _valid_bbox(_bbox_from_points(values) or [])
        if len(values) >= 8 and not any(isinstance(item, (list, tuple, dict)) for item in values):
            numbers = [_number(item) for item in values]
            if all(value is not None for value in numbers):
                xs = [float(value) for value in numbers[0::2] if value is not None]
                ys = [float(value) for value in numbers[1::2] if value is not None]
                return _valid_bbox([min(xs), min(ys), max(xs), max(ys)])
    return None


def _valid_bbox(bbox: list[float]) -> list[float] | None:
    if len(bbox) != 4:
        return None
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return None
    return [float(x1), float(y1), float(x2), float(y2)]


def _box_candidates_from_page(payload: dict[str, Any], page_id: str = "") -> list[dict[str, Any]]:
    pages = payload.get("pages")
    if not isinstance(pages, list):
        return []

    selected: list[dict[str, Any]] = []
    for page in pages:
        if not isinstance(page, dict):
            continue
        if page_id:
            identifiers = {str(page.get(key) or "") for key in PAGE_ID_KEYS}
            image_stem = Path(str(page.get("image") or page.get("image_path") or "")).stem
            if page_id not in identifiers and page_id != image_stem:
                continue
        selected.extend(extract_legacy_box_records(page, page_id=""))
    return selected


def extract_legacy_box_records(payload: Any, *, page_id: str = "") -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []

    page_boxes = _box_candidates_from_page(payload, page_id=page_id)
    if page_boxes:
        return page_boxes

    for key in BOX_LIST_KEYS:
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]

    for key in NESTED_PAYLOAD_KEYS:
        value = payload.get(key)
        if isinstance(value, (dict, list)):
            nested = extract_legacy_box_records(value, page_id=page_id)
            if nested:
                return nested
    return []


def _raw_bbox(raw: dict[str, Any]) -> list[float] | None:
    for key in ("bbox_xyxy", "bbox", "box", "coordinate", "rect", "dt_poly", "poly", "polygon"):
        if key in raw:
            bbox = coerce_bbox_xyxy(raw.get(key))
            if bbox is not None:
                return bbox
    return None


def _source_label(raw: dict[str, Any]) -> str:
    for key in ("source_label", "label", "category", "type", "class_name", "class", "block_label"):
        value = raw.get(key)
        if value not in (None, ""):
            return str(value)
    norm_label = raw.get("norm_label") or raw.get("normalized_label")
    return str(norm_label or "text")


def _text(raw: dict[str, Any]) -> str:
    for key in ("text", "ocr_text", "transcript", "content"):
        value = raw.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _confidence(raw: dict[str, Any], default_confidence: float) -> float:
    for key in ("confidence", "score", "prob", "probability"):
        value = raw.get(key)
        if value not in (None, ""):
            parsed = _number(value)
            if parsed is not None:
                return max(0.0, min(1.0, parsed))
    return max(0.0, min(1.0, float(default_confidence)))


def legacy_payload_to_regions(
    payload: Any,
    *,
    page_id: str,
    model_id: str,
    source_family: str = "",
    default_confidence: float = 0.5,
    start_index: int = 1,
) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    for offset, raw in enumerate(extract_legacy_box_records(payload, page_id=page_id), start=start_index):
        bbox = _raw_bbox(raw)
        if bbox is None:
            continue

        source_label = _source_label(raw)
        label = str(raw.get("norm_label") or raw.get("normalized_label") or normalize_label(source_label))
        source_model = str(raw.get("source_model") or raw.get("variant") or model_id)
        raw_source_family = str(raw.get("source_family") or source_family or "")
        reading_order = raw.get("reading_order", raw.get("order"))
        region_metadata: dict[str, Any] = {
            "legacy_source_family": raw_source_family,
            "legacy_source_model": source_model,
            "legacy_source_label": source_label,
            "legacy_norm_label": label,
            "legacy_payload_keys": sorted(str(key) for key in raw.keys()),
        }
        if isinstance(raw.get("metadata"), dict):
            region_metadata["legacy_metadata"] = raw["metadata"]

        provenance = [model_id]
        for item in (raw_source_family, source_model):
            if item and item not in provenance:
                provenance.append(item)

        regions.append(
            {
                "region_id": str(raw.get("region_id") or f"{page_id}:{model_id}:{offset:04d}"),
                "bbox_xyxy": bbox,
                "label": label,
                "confidence": _confidence(raw, default_confidence),
                "source_model": source_model,
                "text": _text(raw),
                "reading_order": int(reading_order) if reading_order not in (None, "") else offset,
                "provenance": provenance,
                "metadata": region_metadata,
            }
        )
    return regions


def build_legacy_model_output(
    *,
    page_id: str,
    model_id: str,
    regions: list[dict[str, Any]],
    source_family: str = "",
    resource_class: str = "cpu",
    input_paths: list[str] | None = None,
    started: float | None = None,
    default_confidence: float = 0.5,
    input_region_count: int | None = None,
    runtime_status: str = "ok",
    missing_input_paths: list[str] | None = None,
) -> dict[str, Any]:
    elapsed = time.perf_counter() - started if started is not None else 0.0
    paths = input_paths or []
    observed_count = len(regions) if input_region_count is None else int(input_region_count)
    missing_paths = missing_input_paths or []
    return {
        "page_id": page_id,
        "model_id": model_id,
        "regions": regions,
        "runtime": {
            "seconds": round(elapsed, 6),
            "resource_class": resource_class,
            "status": runtime_status,
            "metadata": {
                "source_adapter_kind": "legacy_layout_import",
                "source_family": source_family,
                "input_paths": paths,
                "missing_input_paths": missing_paths,
                "input_region_count": observed_count,
                "emitted_region_count": len(regions),
                "skipped_region_count": max(0, observed_count - len(regions)),
                "default_confidence": default_confidence,
            },
        },
        "metadata": {
            "legacy_layout_import": True,
            "source_family": source_family,
            "input_paths": paths,
        },
    }
