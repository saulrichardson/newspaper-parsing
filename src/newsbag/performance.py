from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from newsbag.contracts import read_json, write_json


SUCCESS_STATUSES = frozenset({"ok", "skipped"})


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(payload)
    return rows


def _seconds(row: dict[str, Any]) -> float:
    try:
        value = float(row.get("seconds") or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) and value >= 0.0 else 0.0


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _timing_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    durations = [_seconds(row) for row in rows]
    status_counts = Counter(str(row.get("status") or "unknown") for row in rows)
    completed = sum(count for status, count in status_counts.items() if status in SUCCESS_STATUSES)
    seconds_total = sum(durations)
    return {
        "count": len(rows),
        "completed": completed,
        "failed": len(rows) - completed,
        "status_counts": dict(sorted(status_counts.items())),
        "seconds_total": round(seconds_total, 6),
        "seconds_mean": round(seconds_total / len(durations), 6) if durations else 0.0,
        "seconds_p50": round(_percentile(durations, 0.50), 6),
        "seconds_p95": round(_percentile(durations, 0.95), 6),
        "seconds_max": round(max(durations), 6) if durations else 0.0,
        "completed_per_minute_serialized": (
            round((completed / seconds_total) * 60.0, 6) if seconds_total > 0.0 else 0.0
        ),
    }


def _group_rows(
    rows: list[dict[str, Any]],
    key: Callable[[dict[str, Any]], str],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        group_name = key(row).strip() or "unknown"
        grouped[group_name].append(row)
    return {name: _timing_summary(grouped[name]) for name in sorted(grouped)}


def _planned_pairs(plan_rows: list[dict[str, Any]]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for row in plan_rows:
        page_id = str(row.get("page_id") or "")
        models = row.get("models") if isinstance(row.get("models"), list) else []
        for model in models:
            if not isinstance(model, dict):
                continue
            model_id = str(model.get("model_id") or "")
            if page_id and model_id:
                pairs.add((page_id, model_id))
    return pairs


def _model_metadata(plan_rows: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    metadata: dict[str, dict[str, str]] = {}
    for row in plan_rows:
        models = row.get("models") if isinstance(row.get("models"), list) else []
        for model in models:
            if not isinstance(model, dict):
                continue
            model_id = str(model.get("model_id") or "")
            if model_id:
                metadata[model_id] = {
                    "family": str(model.get("family") or "unknown"),
                    "resource_class": str(model.get("resource_class") or "unknown"),
                }
    return metadata


def _pair_samples(pairs: set[tuple[str, str]], limit: int = 25) -> list[dict[str, str]]:
    return [
        {"page_id": page_id, "model_id": model_id}
        for page_id, model_id in sorted(pairs)[:limit]
    ]


def summarize_bagging_performance(run_dir: Path) -> dict[str, Any]:
    root = run_dir.expanduser().resolve()
    summary = read_json(root / "summary.json")
    headline = read_json(root / "reports" / "performance.json")
    rows = _read_jsonl(root / "reports" / "performance.jsonl")
    plan_rows = _read_jsonl(root / "manifests" / "model_plan.jsonl")

    adapter_rows = [row for row in rows if str(row.get("stage") or "") == "model_adapter"]
    planned_pairs = _planned_pairs(plan_rows)
    observed_pairs = {
        (str(row.get("page_id") or ""), str(row.get("model_id") or ""))
        for row in adapter_rows
        if str(row.get("page_id") or "") and str(row.get("model_id") or "")
    }
    successful_pairs = {
        (str(row.get("page_id") or ""), str(row.get("model_id") or ""))
        for row in adapter_rows
        if str(row.get("status") or "") in SUCCESS_STATUSES
        and str(row.get("page_id") or "")
        and str(row.get("model_id") or "")
    }
    failed_pairs = observed_pairs - successful_pairs
    missing_pairs = planned_pairs - observed_pairs
    unexpected_pairs = observed_pairs - planned_pairs

    wall_seconds = float(headline.get("seconds_total") or 0.0)
    pages_completed = int(headline.get("pages_completed") or 0)
    slowest = sorted(rows, key=_seconds, reverse=True)[:10]
    model_metadata = _model_metadata(plan_rows)
    by_model = _group_rows(adapter_rows, lambda row: str(row.get("model_id") or "unknown"))
    for model_id, timing in by_model.items():
        timing.update(model_metadata.get(model_id, {"family": "unknown", "resource_class": "unknown"}))
    slowest_rows = [
        {
            key: row.get(key)
            for key in (
                "page_id",
                "stage",
                "model_id",
                "family",
                "resource_class",
                "estimated_complexity",
                "seconds",
                "status",
            )
        }
        for row in slowest
    ]

    return {
        "contract": "parser-performance-summary-v1",
        "run_id": str(summary.get("run_id") or root.name),
        "run_dir": str(root),
        "profile": str(summary.get("profile") or ""),
        "wall_time_seconds": round(wall_seconds, 6),
        "pages_attempted": int(headline.get("pages_attempted") or 0),
        "pages_completed": pages_completed,
        "pages_failed": int(headline.get("pages_failed") or 0),
        "pages_per_minute_wall": (
            round((pages_completed / wall_seconds) * 60.0, 6) if wall_seconds > 0.0 else 0.0
        ),
        "raw_rows": len(rows),
        "adapter_invocations": _timing_summary(adapter_rows),
        "coverage": {
            "planned_invocations": len(planned_pairs),
            "observed_invocations": len(observed_pairs),
            "successful_invocations": len(successful_pairs & planned_pairs),
            "failed_invocations": len(failed_pairs & planned_pairs),
            "missing_invocations": len(missing_pairs),
            "unexpected_invocations": len(unexpected_pairs),
            "coverage_ratio": (
                round(len(observed_pairs & planned_pairs) / len(planned_pairs), 6) if planned_pairs else 1.0
            ),
            "missing_sample": _pair_samples(missing_pairs),
            "unexpected_sample": _pair_samples(unexpected_pairs),
        },
        "by_stage": _group_rows(rows, lambda row: str(row.get("stage") or "unknown")),
        "by_model": by_model,
        "by_model_family": _group_rows(adapter_rows, lambda row: str(row.get("family") or "unknown")),
        "by_resource_class": _group_rows(
            adapter_rows, lambda row: str(row.get("resource_class") or "unknown")
        ),
        "by_page_complexity": _group_rows(
            rows, lambda row: str(row.get("estimated_complexity") or "unknown")
        ),
        "slowest_rows": slowest_rows,
    }


def write_bagging_performance_summary(
    run_dir: Path,
    output_path: Path | None = None,
) -> dict[str, Any]:
    root = run_dir.expanduser().resolve()
    report = summarize_bagging_performance(root)
    destination = output_path.expanduser().resolve() if output_path else root / "reports" / "performance_summary.json"
    write_json(destination, report)
    return report
