from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from newsbag.contracts import is_safe_artifact_id


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


def _path_from_summary(
    run_dir: Path,
    summary: dict[str, Any],
    output_key: str,
    fallback: Path,
    issues: list[dict[str, Any]],
) -> Path:
    outputs = summary.get("outputs") if isinstance(summary.get("outputs"), dict) else {}
    raw = outputs.get(output_key)
    if not raw:
        return fallback.resolve()
    candidate = Path(str(raw)).expanduser()
    if not candidate.is_absolute():
        candidate = run_dir / candidate
    candidate = candidate.resolve()
    try:
        candidate.relative_to(run_dir.resolve())
    except ValueError:
        issues.append(
            {
                "level": "error",
                "code": "output_path_outside_run",
                "path": str(candidate),
                "message": f"summary output {output_key} escapes the run directory",
            }
        )
        return fallback.resolve()
    return candidate


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _status_from_issues(issues: list[dict[str, Any]], *, warnings_are_errors: bool = False) -> str:
    error_count = sum(1 for issue in issues if issue.get("level") == "error")
    warning_count = sum(1 for issue in issues if issue.get("level") == "warning")
    if error_count or (warnings_are_errors and warning_count):
        return "error"
    if warning_count:
        return "warning"
    return "ok"


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


def validate_parse_input_manifest(
    manifest_path: Path,
    *,
    require_files: bool = False,
    require_checksums: bool = False,
    verify_checksums: bool = False,
    warnings_are_errors: bool = False,
) -> dict[str, Any]:
    manifest = manifest_path.expanduser().resolve()
    issues: list[dict[str, Any]] = []
    rows = _read_jsonl(manifest, issues)
    seen_page_ids: set[str] = set()
    rows_with_files = 0
    rows_with_checksums = 0
    source_systems: set[str] = set()

    for row in rows:
        page_id = str(row.get("page_id") or "").strip()
        if not page_id:
            _issue(
                issues,
                level="error",
                code="missing_page_id",
                message="manifest row is missing page_id",
                path=manifest,
            )
        elif not is_safe_artifact_id(page_id):
            _issue(
                issues,
                level="error",
                code="invalid_page_id",
                message="page_id must be a portable artifact identifier using letters, digits, dot, underscore, or hyphen",
                path=manifest,
                page_id=page_id,
            )
        elif page_id in seen_page_ids:
            _issue(
                issues,
                level="error",
                code="duplicate_page_id",
                message="page_id appears more than once",
                path=manifest,
                page_id=page_id,
            )
        else:
            seen_page_ids.add(page_id)

        image_path_raw = str(row.get("image_path") or "").strip()
        image_path: Path | None = None
        if not image_path_raw:
            _issue(
                issues,
                level="error",
                code="missing_image_path",
                message="manifest row is missing image_path",
                path=manifest,
                page_id=page_id,
            )
        else:
            image_path = Path(image_path_raw).expanduser()
            if not image_path.is_absolute():
                image_path = (manifest.parent / image_path).resolve()
            if image_path.is_file():
                rows_with_files += 1
            else:
                _issue(
                    issues,
                    level="error" if require_files else "warning",
                    code="missing_image_file",
                    message="image_path does not point to an existing file",
                    path=image_path,
                    page_id=page_id,
                )

        page_number = row.get("page_number")
        if page_number not in (None, ""):
            try:
                int(page_number)
            except (TypeError, ValueError):
                _issue(
                    issues,
                    level="error",
                    code="invalid_page_number",
                    message="page_number must be an integer or null",
                    path=manifest,
                    page_id=page_id,
                )

        checksum = str(row.get("checksum_sha256") or "").strip()
        if checksum:
            rows_with_checksums += 1
            if not (len(checksum) == 64 and all(char in "0123456789abcdefABCDEF" for char in checksum)):
                _issue(
                    issues,
                    level="error",
                    code="invalid_checksum",
                    message="checksum_sha256 must be a 64-character hex digest",
                    path=manifest,
                    page_id=page_id,
                )
            elif verify_checksums and image_path is not None and image_path.is_file():
                actual = _sha256_file(image_path)
                if actual.lower() != checksum.lower():
                    _issue(
                        issues,
                        level="error",
                        code="checksum_mismatch",
                        message="checksum_sha256 does not match image file bytes",
                        path=image_path,
                        page_id=page_id,
                    )
        elif require_checksums:
            _issue(
                issues,
                level="error",
                code="missing_checksum",
                message="manifest row is missing checksum_sha256",
                path=manifest,
                page_id=page_id,
            )

        source = row.get("source")
        if source is not None:
            if not isinstance(source, dict):
                _issue(
                    issues,
                    level="error",
                    code="invalid_source",
                    message="source must be a JSON object when present",
                    path=manifest,
                    page_id=page_id,
                )
            else:
                source_system = str(source.get("source_system") or "").strip()
                if source_system:
                    source_systems.add(source_system)
                else:
                    _issue(
                        issues,
                        level="warning",
                        code="missing_source_system",
                        message="source.source_system is empty",
                        path=manifest,
                        page_id=page_id,
                    )

        metadata = row.get("metadata")
        if metadata is not None and not isinstance(metadata, dict):
            _issue(
                issues,
                level="error",
                code="invalid_metadata",
                message="metadata must be a JSON object when present",
                path=manifest,
                page_id=page_id,
            )

    if not rows:
        _issue(
            issues,
            level="error",
            code="empty_manifest",
            message="parse input manifest contains no rows",
            path=manifest,
        )

    error_count = sum(1 for issue in issues if issue.get("level") == "error")
    warning_count = sum(1 for issue in issues if issue.get("level") == "warning")
    return {
        "contract": "parse-input-manifest-validation-v1",
        "status": _status_from_issues(issues, warnings_are_errors=warnings_are_errors),
        "manifest_path": str(manifest),
        "counts": {
            "rows": len(rows),
            "unique_page_ids": len(seen_page_ids),
            "rows_with_files": rows_with_files,
            "rows_with_checksums": rows_with_checksums,
            "source_systems": len(source_systems),
            "errors": error_count,
            "warnings": warning_count,
        },
        "source_systems": sorted(source_systems),
        "require_files": require_files,
        "require_checksums": require_checksums,
        "verify_checksums": verify_checksums,
        "issues": issues,
    }


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
        elif runtime_status not in {"ok", "skipped"}:
            _issue(
                issues,
                level="error",
                code="invalid_runtime_status",
                message="successful model output runtime status must be ok or skipped",
                path=path,
                page_id=page_id,
                model_id=model_id,
            )
        resource_class = str(runtime.get("resource_class") or "").strip()
        if not resource_class:
            _issue(
                issues,
                level="error",
                code="missing_runtime_resource_class",
                message="model output runtime is missing resource_class",
                path=path,
                page_id=page_id,
                model_id=model_id,
            )
        try:
            runtime_seconds = float(runtime.get("seconds"))
        except (TypeError, ValueError):
            runtime_seconds = math.nan
        if not math.isfinite(runtime_seconds) or runtime_seconds < 0.0:
            _issue(
                issues,
                level="error",
                code="invalid_runtime_seconds",
                message="model output runtime seconds must be nonnegative and finite",
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
            message="fused model_ids do not match successful models from the page plan",
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


def _validate_model_plan(
    rows: list[dict[str, Any]],
    *,
    page_ids: list[str],
    summary_model_ids: list[str],
    profile_name: str,
    path: Path,
    issues: list[dict[str, Any]],
) -> dict[str, list[str]]:
    plan_by_page: dict[str, list[str]] = {}
    model_union: list[str] = []
    known_page_ids = set(page_ids)

    for row in rows:
        page_id = str(row.get("page_id") or "").strip()
        if not page_id:
            _issue(
                issues,
                level="error",
                code="model_plan_missing_page_id",
                message="model plan row is missing page_id",
                path=path,
            )
            continue
        if page_id in plan_by_page:
            _issue(
                issues,
                level="error",
                code="duplicate_model_plan_page",
                message="model plan contains more than one row for the page",
                path=path,
                page_id=page_id,
            )
            continue
        if page_id not in known_page_ids:
            _issue(
                issues,
                level="error",
                code="unknown_model_plan_page",
                message="model plan page_id is absent from the parse input manifest",
                path=path,
                page_id=page_id,
            )
        if row.get("contract") != "parser-model-plan-v1":
            _issue(
                issues,
                level="error",
                code="invalid_model_plan_contract",
                message="model plan row must use contract parser-model-plan-v1",
                path=path,
                page_id=page_id,
            )
        if str(row.get("profile_name") or "") != profile_name:
            _issue(
                issues,
                level="error",
                code="model_plan_profile_mismatch",
                message="model plan profile_name does not match summary profile",
                path=path,
                page_id=page_id,
            )

        complexity = str(row.get("estimated_complexity") or "")
        if complexity not in {"easy", "medium", "hard"}:
            _issue(
                issues,
                level="error",
                code="invalid_page_complexity",
                message="estimated_complexity must be easy, medium, or hard",
                path=path,
                page_id=page_id,
            )
        profile = row.get("profile")
        if not isinstance(profile, dict):
            _issue(
                issues,
                level="error",
                code="missing_page_profile",
                message="model plan row is missing its page profile",
                path=path,
                page_id=page_id,
            )
        elif (
            str(profile.get("page_id") or "") != page_id
            or str(profile.get("estimated_complexity") or "") != complexity
        ):
            _issue(
                issues,
                level="error",
                code="model_plan_profile_mismatch",
                message="embedded page profile does not match the model plan row",
                path=path,
                page_id=page_id,
            )

        models = row.get("models")
        model_ids: list[str] = []
        resource_classes: set[str] = set()
        if not isinstance(models, list) or not models:
            _issue(
                issues,
                level="error",
                code="empty_model_plan",
                message="model plan row must contain at least one model",
                path=path,
                page_id=page_id,
            )
        else:
            for model in models:
                if not isinstance(model, dict):
                    _issue(
                        issues,
                        level="error",
                        code="invalid_planned_model",
                        message="planned model must be an object",
                        path=path,
                        page_id=page_id,
                    )
                    continue
                model_id = str(model.get("model_id") or "").strip()
                family = str(model.get("family") or "").strip()
                resource_class = str(model.get("resource_class") or "").strip()
                if not model_id or not family or not resource_class:
                    _issue(
                        issues,
                        level="error",
                        code="invalid_planned_model",
                        message="planned model requires model_id, family, and resource_class",
                        path=path,
                        page_id=page_id,
                        model_id=model_id,
                    )
                    continue
                if not is_safe_artifact_id(model_id):
                    _issue(
                        issues,
                        level="error",
                        code="invalid_planned_model_id",
                        message="planned model_id is not a portable artifact identifier",
                        path=path,
                        page_id=page_id,
                        model_id=model_id,
                    )
                    continue
                if model_id in model_ids:
                    _issue(
                        issues,
                        level="error",
                        code="duplicate_planned_model",
                        message="model appears more than once in the page plan",
                        path=path,
                        page_id=page_id,
                        model_id=model_id,
                    )
                    continue
                model_ids.append(model_id)
                resource_classes.add(resource_class)
                if model_id not in model_union:
                    model_union.append(model_id)

        declared_resources = row.get("resource_classes")
        if declared_resources != sorted(resource_classes):
            _issue(
                issues,
                level="error",
                code="model_plan_resource_mismatch",
                message="resource_classes do not match the planned models",
                path=path,
                page_id=page_id,
            )
        if not str(row.get("routing_reason") or "").strip():
            _issue(
                issues,
                level="error",
                code="missing_routing_reason",
                message="model plan row is missing routing_reason",
                path=path,
                page_id=page_id,
            )
        plan_by_page[page_id] = model_ids

    for page_id in page_ids:
        if page_id not in plan_by_page:
            _issue(
                issues,
                level="error",
                code="missing_model_plan_page",
                message="parse input page has no model plan row",
                path=path,
                page_id=page_id,
            )
            plan_by_page[page_id] = []

    if model_union != summary_model_ids:
        _issue(
            issues,
            level="error",
            code="summary_model_ids_mismatch",
            message="summary model_ids do not match the ordered union of the page plans",
            path=path,
        )
    return plan_by_page


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

    model_plan_path = _path_from_summary(
        root,
        summary,
        "model_plan",
        root / "manifests" / "model_plan.jsonl",
        issues,
    )
    plan_summary_path = _path_from_summary(
        root,
        summary,
        "plan_summary",
        root / "reports" / "plan_summary.json",
        issues,
    )
    performance_summary_path = _path_from_summary(
        root,
        summary,
        "performance_summary",
        root / "reports" / "performance_summary.json",
        issues,
    )
    manifest_rows = _read_jsonl(root / "manifests" / "parse_input.jsonl", issues)
    model_plan_rows = _read_jsonl(model_plan_path, issues)
    errors_rows = _read_jsonl(root / "errors.jsonl", issues)
    performance_rows = _read_jsonl(root / "reports" / "performance.jsonl", issues)
    performance_json = _read_json(root / "reports" / "performance.json", issues)
    plan_summary_json = _read_json(plan_summary_path, issues)
    performance_summary_json = _read_json(performance_summary_path, issues)
    provenance_json = _read_json(root / "provenance.json", issues)
    input_manifest_validation = _read_json(root / "reports" / "input_manifest_validation.json", issues)

    model_ids = [str(item) for item in summary.get("model_ids", []) if str(item)]
    page_ids = [str(row.get("page_id") or "") for row in manifest_rows if str(row.get("page_id") or "")]
    page_error_ids = {
        str(row.get("page_id") or "")
        for row in errors_rows
        if str(row.get("page_id") or "") and str(row.get("scope") or "page") == "page"
    }
    adapter_error_rows = [row for row in errors_rows if str(row.get("scope") or "") == "adapter"]
    adapter_error_pairs = {
        (str(row.get("page_id") or ""), str(row.get("model_id") or ""))
        for row in adapter_error_rows
        if str(row.get("page_id") or "") and str(row.get("model_id") or "")
    }
    expected_completed_pages = [page_id for page_id in page_ids if page_id not in page_error_ids]
    plan_by_page = _validate_model_plan(
        model_plan_rows,
        page_ids=page_ids,
        summary_model_ids=model_ids,
        profile_name=str(summary.get("profile") or ""),
        path=model_plan_path,
        issues=issues,
    )
    model_outputs_dir = _path_from_summary(
        root,
        summary,
        "model_outputs",
        root / "outputs" / "model_outputs",
        issues,
    )
    fused_pages_dir = _path_from_summary(
        root,
        summary,
        "fused_pages",
        root / "outputs" / "fused_pages",
        issues,
    )
    transcripts_dir = _path_from_summary(
        root,
        summary,
        "transcripts",
        root / "outputs" / "transcripts",
        issues,
    )

    if int(summary.get("page_count") or -1) != len(page_ids):
        _issue(
            issues,
            level="error",
            code="page_count_mismatch",
            message=f"summary page_count={summary.get('page_count')} but manifest has {len(page_ids)} pages",
            path=summary_path,
        )

    performance = summary.get("performance") if isinstance(summary.get("performance"), dict) else {}
    planned_invocations = sum(len(plan_by_page.get(page_id, [])) for page_id in page_ids)
    adapter_performance_rows = [
        row for row in performance_rows if str(row.get("stage") or "") == "model_adapter"
    ]
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
    if performance.get("pages_failed") != len(page_error_ids):
        _issue(
            issues,
            level="error",
            code="pages_failed_mismatch",
            message=f"performance pages_failed={performance.get('pages_failed')} but errors.jsonl has {len(page_error_ids)} failed pages",
            path=summary_path,
        )
    if performance.get("adapter_invocations_planned") != planned_invocations:
        _issue(
            issues,
            level="error",
            code="planned_invocations_mismatch",
            message=(
                f"performance adapter_invocations_planned={performance.get('adapter_invocations_planned')} "
                f"but model plan has {planned_invocations}"
            ),
            path=summary_path,
        )
    if performance.get("adapter_invocations_observed") != len(adapter_performance_rows):
        _issue(
            issues,
            level="error",
            code="observed_invocations_mismatch",
            message=(
                f"performance adapter_invocations_observed={performance.get('adapter_invocations_observed')} "
                f"but raw performance has {len(adapter_performance_rows)} adapter rows"
            ),
            path=summary_path,
        )
    if performance.get("adapter_errors") != len(adapter_error_rows):
        _issue(
            issues,
            level="error",
            code="adapter_error_count_mismatch",
            message=(
                f"performance adapter_errors={performance.get('adapter_errors')} "
                f"but errors.jsonl has {len(adapter_error_rows)} adapter errors"
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
            code="run_has_errors",
            message=f"run reports {len(errors_rows)} execution errors",
            path=root / "errors.jsonl",
        )

    planned_pairs = {
        (page_id, model_id)
        for page_id in page_ids
        for model_id in plan_by_page.get(page_id, [])
    }
    observed_pair_counts = Counter(
        (str(row.get("page_id") or ""), str(row.get("model_id") or ""))
        for row in adapter_performance_rows
    )
    observed_pairs = set(observed_pair_counts)
    missing_pairs = planned_pairs - observed_pairs
    unexpected_pairs = observed_pairs - planned_pairs
    duplicate_pairs = sorted(pair for pair, count in observed_pair_counts.items() if count != 1)
    if missing_pairs:
        _issue(
            issues,
            level="error",
            code="missing_adapter_performance_rows",
            message=f"raw performance is missing {len(missing_pairs)} planned adapter invocations",
            path=root / "reports" / "performance.jsonl",
        )
    if unexpected_pairs:
        _issue(
            issues,
            level="error",
            code="unexpected_adapter_performance_rows",
            message=f"raw performance has {len(unexpected_pairs)} unplanned adapter invocations",
            path=root / "reports" / "performance.jsonl",
        )
    if duplicate_pairs:
        _issue(
            issues,
            level="error",
            code="duplicate_adapter_performance_rows",
            message=f"raw performance repeats {len(duplicate_pairs)} page/model invocations",
            path=root / "reports" / "performance.jsonl",
        )
    raw_adapter_error_pairs = {
        (str(row.get("page_id") or ""), str(row.get("model_id") or ""))
        for row in adapter_performance_rows
        if str(row.get("status") or "") == "error"
    }
    if raw_adapter_error_pairs != adapter_error_pairs:
        _issue(
            issues,
            level="error",
            code="adapter_error_rows_mismatch",
            message="adapter failures differ between errors.jsonl and raw performance",
            path=root / "reports" / "performance.jsonl",
        )
    known_page_ids = set(page_ids)
    for row in performance_rows:
        stage = str(row.get("stage") or "")
        page_id = str(row.get("page_id") or "")
        try:
            seconds = float(row.get("seconds"))
        except (TypeError, ValueError):
            seconds = math.nan
        if not stage or page_id not in known_page_ids or not math.isfinite(seconds) or seconds < 0.0:
            _issue(
                issues,
                level="error",
                code="invalid_performance_row",
                message="performance row requires a known page, stage, and nonnegative finite seconds",
                path=root / "reports" / "performance.jsonl",
                page_id=page_id,
                model_id=str(row.get("model_id") or ""),
            )

    if not isinstance(performance_json, dict):
        _issue(
            issues,
            level="error",
            code="invalid_performance_report",
            message="reports/performance.json must be an object",
            path=root / "reports" / "performance.json",
        )
    elif performance_json != performance:
        _issue(
            issues,
            level="error",
            code="performance_report_mismatch",
            message="reports/performance.json does not match summary performance",
            path=root / "reports" / "performance.json",
        )
    if not isinstance(plan_summary_json, dict):
        _issue(
            issues,
            level="error",
            code="invalid_plan_summary",
            message="reports/plan_summary.json must be an object",
            path=plan_summary_path,
        )
    else:
        if plan_summary_json.get("contract") != "parser-model-plan-summary-v1":
            _issue(
                issues,
                level="error",
                code="invalid_plan_summary_contract",
                message="plan summary must use contract parser-model-plan-summary-v1",
                path=plan_summary_path,
            )
        expected_plan_fields = {
            "profile_name": str(summary.get("profile") or ""),
            "pages_planned": len(page_ids),
            "adapter_invocations_planned": planned_invocations,
            "model_ids": model_ids,
        }
        for key, expected in expected_plan_fields.items():
            if plan_summary_json.get(key) != expected:
                _issue(
                    issues,
                    level="error",
                    code="plan_summary_mismatch",
                    message=f"plan summary {key} does not match the run bundle",
                    path=plan_summary_path,
                )
    if not isinstance(performance_summary_json, dict):
        _issue(
            issues,
            level="error",
            code="invalid_performance_summary",
            message="reports/performance_summary.json must be an object",
            path=performance_summary_path,
        )
    else:
        if performance_summary_json.get("contract") != "parser-performance-summary-v1":
            _issue(
                issues,
                level="error",
                code="invalid_performance_summary_contract",
                message="performance summary must use contract parser-performance-summary-v1",
                path=performance_summary_path,
            )
        if performance_summary_json.get("raw_rows") != len(performance_rows):
            _issue(
                issues,
                level="error",
                code="performance_summary_row_mismatch",
                message="performance summary raw_rows does not match performance.jsonl",
                path=performance_summary_path,
            )
        coverage = (
            performance_summary_json.get("coverage")
            if isinstance(performance_summary_json.get("coverage"), dict)
            else {}
        )
        expected_coverage = {
            "planned_invocations": planned_invocations,
            "observed_invocations": len(observed_pairs),
            "missing_invocations": len(missing_pairs),
            "unexpected_invocations": len(unexpected_pairs),
        }
        for key, expected in expected_coverage.items():
            if coverage.get(key) != expected:
                _issue(
                    issues,
                    level="error",
                    code="performance_coverage_mismatch",
                    message=f"performance summary coverage {key} does not match raw artifacts",
                    path=performance_summary_path,
                )
    if not isinstance(provenance_json, dict):
        _issue(
            issues,
            level="error",
            code="invalid_provenance",
            message="provenance.json must be an object",
            path=root / "provenance.json",
        )
    elif provenance_json != summary.get("provenance"):
        _issue(
            issues,
            level="error",
            code="provenance_mismatch",
            message="provenance.json does not match summary provenance",
            path=root / "provenance.json",
        )
    if not isinstance(input_manifest_validation, dict):
        _issue(
            issues,
            level="error",
            code="invalid_input_manifest_validation",
            message="reports/input_manifest_validation.json must be an object",
            path=root / "reports" / "input_manifest_validation.json",
        )
    elif input_manifest_validation.get("status") not in ("ok", "warning"):
        _issue(
            issues,
            level="error",
            code="input_manifest_validation_failed",
            message=f"input manifest validation status is {input_manifest_validation.get('status')}",
            path=root / "reports" / "input_manifest_validation.json",
        )

    fused_count = 0
    transcript_count = 0
    model_output_count = 0
    model_region_count = 0
    fused_region_count = 0
    runtime_status_counts: dict[str, int] = {}
    expected_model_outputs = 0

    successful_models_by_page: dict[str, list[str]] = {}
    for page_id in page_ids:
        successful_model_ids = [
            model_id
            for model_id in plan_by_page.get(page_id, [])
            if (page_id, model_id) not in adapter_error_pairs
        ]
        successful_models_by_page[page_id] = successful_model_ids
        expected_model_outputs += len(successful_model_ids)
        for model_id in plan_by_page.get(page_id, []):
            model_path = model_outputs_dir / model_id / f"{page_id}.json"
            if (page_id, model_id) in adapter_error_pairs:
                if model_path.exists():
                    model_output_count += 1
                    _issue(
                        issues,
                        level="error",
                        code="failed_adapter_has_model_output",
                        message="failed adapter left a model output artifact",
                        path=model_path,
                        page_id=page_id,
                        model_id=model_id,
                    )
                continue
            model_stats = _validate_model_output(model_path, page_id=page_id, model_id=model_id, issues=issues)
            if model_path.exists():
                model_output_count += 1
            model_region_count += int(model_stats["regions"])
            runtime_status = str(model_stats["status"] or "unknown")
            runtime_status_counts[runtime_status] = runtime_status_counts.get(runtime_status, 0) + 1

    for page_id in expected_completed_pages:
        fused_path = fused_pages_dir / f"{page_id}.json"
        transcript_path = transcripts_dir / f"{page_id}.txt"
        fused_stats = _validate_fused_page(
            fused_path,
            transcript_path=transcript_path,
            page_id=page_id,
            model_ids=successful_models_by_page.get(page_id, []),
            issues=issues,
        )
        if fused_path.exists():
            fused_count += 1
        if fused_stats["has_transcript"]:
            transcript_count += 1
        fused_region_count += int(fused_stats["regions"])

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
            "pages_failed": len(page_error_ids),
            "model_plan_rows": len(model_plan_rows),
            "adapter_invocations_planned": planned_invocations,
            "models": len(model_ids),
            "model_outputs": model_output_count,
            "model_regions": model_region_count,
            "fused_pages": fused_count,
            "fused_regions": fused_region_count,
            "transcripts": transcript_count,
            "performance_rows": len(performance_rows),
            "page_errors": len(page_error_ids),
            "adapter_errors": len(adapter_error_rows),
            "execution_errors": len(errors_rows),
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
            "performance_summary_json": str(performance_summary_path),
            "model_plan_jsonl": str(model_plan_path),
            "plan_summary_json": str(plan_summary_path),
            "errors_jsonl": str(root / "errors.jsonl"),
            "provenance_json": str(root / "provenance.json"),
            "input_manifest_validation_json": str(root / "reports" / "input_manifest_validation.json"),
        },
        "issues": issues,
    }


def format_validation_text(report: dict[str, Any]) -> str:
    counts = report.get("counts") if isinstance(report.get("counts"), dict) else {}
    lines = [
        f"status: {report.get('status')}",
        f"run_dir: {report.get('run_dir') or report.get('manifest_path') or report.get('output_jsonl')}",
        "counts:",
    ]
    preferred_keys = (
        "rows",
        "unique_page_ids",
        "rows_with_files",
        "rows_with_checksums",
        "source_systems",
        "pages_manifest",
        "pages_completed",
        "pages_failed",
        "model_plan_rows",
        "adapter_invocations_planned",
        "models",
        "model_outputs",
        "model_regions",
        "fused_pages",
        "fused_regions",
        "transcripts",
        "performance_rows",
        "page_errors",
        "adapter_errors",
        "execution_errors",
        "errors",
        "warnings",
    )
    for key in preferred_keys:
        if key in counts:
            lines.append(f"  {key}: {counts[key]}")
    for key in sorted(set(counts) - set(preferred_keys)):
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
