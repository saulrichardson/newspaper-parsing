from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def _read_json(path: Path, issues: list[dict[str, Any]]) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        issues.append({"level": "error", "code": "missing_file", "path": str(path), "message": "required file is missing"})
    except json.JSONDecodeError as exc:
        issues.append(
            {
                "level": "error",
                "code": "invalid_json",
                "path": str(path),
                "message": f"invalid JSON at line {exc.lineno}: {exc.msg}",
            }
        )
    return None


def _read_jsonl(path: Path, issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                issues.append(
                    {
                        "level": "error",
                        "code": "invalid_jsonl",
                        "path": str(path),
                        "line": line_number,
                        "message": f"invalid JSONL row: {exc.msg}",
                    }
                )
                continue
            if not isinstance(row, dict):
                issues.append(
                    {
                        "level": "error",
                        "code": "invalid_jsonl_row",
                        "path": str(path),
                        "line": line_number,
                        "message": "JSONL row must be an object",
                    }
                )
                continue
            rows.append(row)
    except FileNotFoundError:
        issues.append({"level": "error", "code": "missing_file", "path": str(path), "message": "required file is missing"})
    return rows


def _path_from_summary(run_dir: Path, summary: dict[str, Any], output_key: str, fallback: Path) -> Path:
    outputs = summary.get("outputs") if isinstance(summary.get("outputs"), dict) else {}
    raw = outputs.get(output_key)
    if raw:
        return Path(str(raw)).expanduser()
    return fallback


def _issue(
    issues: list[dict[str, Any]],
    *,
    level: str,
    code: str,
    message: str,
    path: Path | str | None = None,
    page_id: str = "",
    model_id: str = "",
) -> None:
    row: dict[str, Any] = {"level": level, "code": code, "message": message}
    if path is not None:
        row["path"] = str(path)
    if page_id:
        row["page_id"] = page_id
    if model_id:
        row["model_id"] = model_id
    issues.append(row)


def _validate_region(
    raw: Any,
    *,
    issues: list[dict[str, Any]],
    path: Path,
    page_id: str,
    model_id: str = "",
    index: int,
) -> bool:
    if not isinstance(raw, dict):
        _issue(
            issues,
            level="error",
            code="invalid_region",
            message=f"region {index} must be an object",
            path=path,
            page_id=page_id,
            model_id=model_id,
        )
        return False

    ok = True
    bbox = raw.get("bbox_xyxy")
    if not isinstance(bbox, list) or len(bbox) != 4:
        _issue(
            issues,
            level="error",
            code="invalid_bbox",
            message=f"region {index} bbox_xyxy must contain four numbers",
            path=path,
            page_id=page_id,
            model_id=model_id,
        )
        ok = False
    else:
        try:
            x1, y1, x2, y2 = [float(value) for value in bbox]
        except (TypeError, ValueError):
            _issue(
                issues,
                level="error",
                code="invalid_bbox",
                message=f"region {index} bbox_xyxy contains non-numeric values",
                path=path,
                page_id=page_id,
                model_id=model_id,
            )
            ok = False
        else:
            if not all(math.isfinite(value) for value in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
                _issue(
                    issues,
                    level="error",
                    code="invalid_bbox",
                    message=f"region {index} bbox_xyxy is degenerate or non-finite",
                    path=path,
                    page_id=page_id,
                    model_id=model_id,
                )
                ok = False

    if not str(raw.get("label") or "").strip():
        _issue(
            issues,
            level="error",
            code="missing_region_label",
            message=f"region {index} is missing label",
            path=path,
            page_id=page_id,
            model_id=model_id,
        )
        ok = False

    confidence = raw.get("confidence")
    if confidence is not None:
        try:
            parsed = float(confidence)
        except (TypeError, ValueError):
            parsed = math.nan
        if not math.isfinite(parsed) or parsed < 0.0 or parsed > 1.0:
            _issue(
                issues,
                level="error",
                code="invalid_confidence",
                message=f"region {index} confidence must be between 0 and 1",
                path=path,
                page_id=page_id,
                model_id=model_id,
            )
            ok = False
    return ok


def _validate_model_output(path: Path, *, page_id: str, model_id: str, issues: list[dict[str, Any]]) -> dict[str, Any]:
    payload = _read_json(path, issues)
    if not isinstance(payload, dict):
        return {"regions": 0, "status": "missing"}

    if str(payload.get("page_id") or "") != page_id:
        _issue(
            issues,
            level="error",
            code="model_output_page_mismatch",
            message=f"model output page_id does not match {page_id}",
            path=path,
            page_id=page_id,
            model_id=model_id,
        )
    if str(payload.get("model_id") or "") != model_id:
        _issue(
            issues,
            level="error",
            code="model_output_model_mismatch",
            message=f"model output model_id does not match {model_id}",
            path=path,
            page_id=page_id,
            model_id=model_id,
        )

    runtime = payload.get("runtime")
    runtime_status = ""
    if not isinstance(runtime, dict):
        _issue(
            issues,
            level="error",
            code="missing_runtime",
            message="model output is missing runtime object",
            path=path,
            page_id=page_id,
            model_id=model_id,
        )
    else:
        runtime_status = str(runtime.get("status") or "").strip()
        if not runtime_status:
            _issue(
                issues,
                level="error",
                code="missing_runtime_status",
                message="model output runtime is missing status",
                path=path,
                page_id=page_id,
                model_id=model_id,
            )

    regions = payload.get("regions")
    if not isinstance(regions, list):
        _issue(
            issues,
            level="error",
            code="invalid_regions",
            message="model output regions must be a list",
            path=path,
            page_id=page_id,
            model_id=model_id,
        )
        return {"regions": 0, "status": runtime_status or "invalid"}

    valid_regions = 0
    for index, region in enumerate(regions, start=1):
        if _validate_region(region, issues=issues, path=path, page_id=page_id, model_id=model_id, index=index):
            valid_regions += 1
    return {"regions": valid_regions, "status": runtime_status or "unknown"}


def _validate_fused_page(
    path: Path,
    *,
    transcript_path: Path,
    page_id: str,
    model_ids: list[str],
    issues: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = _read_json(path, issues)
    if not isinstance(payload, dict):
        return {"regions": 0, "has_transcript": False}

    if str(payload.get("page_id") or "") != page_id:
        _issue(
            issues,
            level="error",
            code="fused_page_mismatch",
            message=f"fused page_id does not match {page_id}",
            path=path,
            page_id=page_id,
        )
    fused_model_ids = payload.get("model_ids")
    if fused_model_ids != model_ids:
        _issue(
            issues,
            level="error",
            code="fused_model_ids_mismatch",
            message="fused model_ids do not match summary model_ids",
            path=path,
            page_id=page_id,
        )

    transcript = payload.get("transcript")
    if not isinstance(transcript, str):
        _issue(
            issues,
            level="error",
            code="invalid_transcript",
            message="fused transcript must be a string",
            path=path,
            page_id=page_id,
        )
        transcript = ""

    has_transcript_file = transcript_path.exists()
    if not has_transcript_file:
        _issue(
            issues,
            level="error",
            code="missing_transcript_file",
            message="transcript file is missing",
            path=transcript_path,
            page_id=page_id,
        )
    else:
        on_disk = transcript_path.read_text(encoding="utf-8").rstrip("\n")
        if on_disk != transcript:
            _issue(
                issues,
                level="error",
                code="transcript_mismatch",
                message="transcript file does not match fused page transcript",
                path=transcript_path,
                page_id=page_id,
            )

    regions = payload.get("regions")
    if not isinstance(regions, list):
        _issue(
            issues,
            level="error",
            code="invalid_fused_regions",
            message="fused regions must be a list",
            path=path,
            page_id=page_id,
        )
        return {"regions": 0, "has_transcript": has_transcript_file}

    valid_regions = 0
    for index, region in enumerate(regions, start=1):
        if _validate_region(region, issues=issues, path=path, page_id=page_id, index=index):
            valid_regions += 1
    return {"regions": valid_regions, "has_transcript": has_transcript_file}


def validate_bagging_run(run_dir: Path) -> dict[str, Any]:
    root = run_dir.expanduser().resolve()
    issues: list[dict[str, Any]] = []
    summary_path = root / "summary.json"
    summary = _read_json(summary_path, issues)
    if not isinstance(summary, dict):
        return {
            "status": "error",
            "run_dir": str(root),
            "summary_path": str(summary_path),
            "counts": {"errors": 1, "warnings": 0},
            "issues": issues,
        }

    manifest_rows = _read_jsonl(root / "manifests" / "parse_input.jsonl", issues)
    errors_rows = _read_jsonl(root / "errors.jsonl", issues)
    performance_rows = _read_jsonl(root / "reports" / "performance.jsonl", issues)
    performance_json = _read_json(root / "reports" / "performance.json", issues)
    provenance_json = _read_json(root / "provenance.json", issues)

    model_ids = [str(item) for item in summary.get("model_ids", []) if str(item)]
    page_ids = [str(row.get("page_id") or "") for row in manifest_rows if str(row.get("page_id") or "")]
    page_error_ids = {str(row.get("page_id") or "") for row in errors_rows if str(row.get("page_id") or "")}
    expected_completed_pages = [page_id for page_id in page_ids if page_id not in page_error_ids]
    model_outputs_dir = _path_from_summary(root, summary, "model_outputs", root / "outputs" / "model_outputs")
    fused_pages_dir = _path_from_summary(root, summary, "fused_pages", root / "outputs" / "fused_pages")
    transcripts_dir = _path_from_summary(root, summary, "transcripts", root / "outputs" / "transcripts")

    if int(summary.get("page_count") or -1) != len(page_ids):
        _issue(
            issues,
            level="error",
            code="page_count_mismatch",
            message=f"summary page_count={summary.get('page_count')} but manifest has {len(page_ids)} pages",
            path=summary_path,
        )

    performance = summary.get("performance") if isinstance(summary.get("performance"), dict) else {}
    if performance.get("pages_attempted") != len(page_ids):
        _issue(
            issues,
            level="error",
            code="pages_attempted_mismatch",
            message=f"performance pages_attempted={performance.get('pages_attempted')} but manifest has {len(page_ids)} pages",
            path=summary_path,
        )
    if performance.get("pages_completed") != len(expected_completed_pages):
        _issue(
            issues,
            level="error",
            code="pages_completed_mismatch",
            message=(
                f"performance pages_completed={performance.get('pages_completed')} but "
                f"{len(expected_completed_pages)} pages have no page-level error"
            ),
            path=summary_path,
        )
    if performance.get("errors") != len(errors_rows):
        _issue(
            issues,
            level="error",
            code="error_count_mismatch",
            message=f"performance errors={performance.get('errors')} but errors.jsonl has {len(errors_rows)} rows",
            path=summary_path,
        )
    if errors_rows:
        _issue(
            issues,
            level="error",
            code="run_has_page_errors",
            message=f"run reports {len(errors_rows)} page-level errors",
            path=root / "errors.jsonl",
        )

    if not isinstance(performance_json, dict):
        _issue(
            issues,
            level="error",
            code="invalid_performance_report",
            message="reports/performance.json must be an object",
            path=root / "reports" / "performance.json",
        )
    if not isinstance(provenance_json, dict):
        _issue(
            issues,
            level="error",
            code="invalid_provenance",
            message="provenance.json must be an object",
            path=root / "provenance.json",
        )

    fused_count = 0
    transcript_count = 0
    model_output_count = 0
    model_region_count = 0
    fused_region_count = 0
    runtime_status_counts: dict[str, int] = {}

    for page_id in expected_completed_pages:
        fused_path = fused_pages_dir / f"{page_id}.json"
        transcript_path = transcripts_dir / f"{page_id}.txt"
        fused_stats = _validate_fused_page(
            fused_path,
            transcript_path=transcript_path,
            page_id=page_id,
            model_ids=model_ids,
            issues=issues,
        )
        if fused_path.exists():
            fused_count += 1
        if fused_stats["has_transcript"]:
            transcript_count += 1
        fused_region_count += int(fused_stats["regions"])

        for model_id in model_ids:
            model_path = model_outputs_dir / model_id / f"{page_id}.json"
            model_stats = _validate_model_output(model_path, page_id=page_id, model_id=model_id, issues=issues)
            if model_path.exists():
                model_output_count += 1
            model_region_count += int(model_stats["regions"])
            status = str(model_stats["status"] or "unknown")
            runtime_status_counts[status] = runtime_status_counts.get(status, 0) + 1

    if fused_count != len(expected_completed_pages):
        _issue(
            issues,
            level="error",
            code="fused_page_count_mismatch",
            message=f"found {fused_count} fused pages for {len(expected_completed_pages)} completed pages",
            path=fused_pages_dir,
        )
    if transcript_count != len(expected_completed_pages):
        _issue(
            issues,
            level="error",
            code="transcript_count_mismatch",
            message=f"found {transcript_count} transcript files for {len(expected_completed_pages)} completed pages",
            path=transcripts_dir,
        )
    expected_model_outputs = len(expected_completed_pages) * len(model_ids)
    if model_output_count != expected_model_outputs:
        _issue(
            issues,
            level="error",
            code="model_output_count_mismatch",
            message=f"found {model_output_count} model outputs, expected {expected_model_outputs}",
            path=model_outputs_dir,
        )

    error_count = sum(1 for issue in issues if issue.get("level") == "error")
    warning_count = sum(1 for issue in issues if issue.get("level") == "warning")
    status = "error" if error_count else ("warning" if warning_count else "ok")
    return {
        "status": status,
        "run_dir": str(root),
        "summary_path": str(summary_path),
        "counts": {
            "pages_manifest": len(page_ids),
            "pages_completed": len(expected_completed_pages),
            "models": len(model_ids),
            "model_outputs": model_output_count,
            "model_regions": model_region_count,
            "fused_pages": fused_count,
            "fused_regions": fused_region_count,
            "transcripts": transcript_count,
            "performance_rows": len(performance_rows),
            "page_errors": len(errors_rows),
            "errors": error_count,
            "warnings": warning_count,
        },
        "runtime_status_counts": dict(sorted(runtime_status_counts.items())),
        "paths": {
            "model_outputs": str(model_outputs_dir),
            "fused_pages": str(fused_pages_dir),
            "transcripts": str(transcripts_dir),
            "performance_json": str(root / "reports" / "performance.json"),
            "performance_jsonl": str(root / "reports" / "performance.jsonl"),
            "errors_jsonl": str(root / "errors.jsonl"),
            "provenance_json": str(root / "provenance.json"),
        },
        "issues": issues,
    }


def format_validation_text(report: dict[str, Any]) -> str:
    counts = report.get("counts") if isinstance(report.get("counts"), dict) else {}
    lines = [
        f"status: {report.get('status')}",
        f"run_dir: {report.get('run_dir')}",
        "counts:",
    ]
    for key in (
        "pages_manifest",
        "pages_completed",
        "models",
        "model_outputs",
        "model_regions",
        "fused_pages",
        "fused_regions",
        "transcripts",
        "performance_rows",
        "page_errors",
        "errors",
        "warnings",
    ):
        if key in counts:
            lines.append(f"  {key}: {counts[key]}")
    issues = report.get("issues") if isinstance(report.get("issues"), list) else []
    if issues:
        lines.append("issues:")
        for issue in issues[:50]:
            location = ""
            if issue.get("page_id"):
                location += f" page={issue['page_id']}"
            if issue.get("model_id"):
                location += f" model={issue['model_id']}"
            if issue.get("path"):
                location += f" path={issue['path']}"
            lines.append(f"  [{issue.get('level')}] {issue.get('code')}: {issue.get('message')}{location}")
        if len(issues) > 50:
            lines.append(f"  ... {len(issues) - 50} more issues")
    return "\n".join(lines) + "\n"
