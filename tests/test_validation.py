from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from newsbag.bagging import run_bagging_canary
from newsbag.validation import validate_bagging_run, validate_parse_input_manifest


def _write_page(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (320, 480), "white")
    draw = ImageDraw.Draw(image)
    for y in range(40, 420, 28):
        draw.rectangle([30, y, 280, y + 8], fill="black")
    image.save(path)


def _write_manifest(path: Path, image_path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "page_id": "page-validate",
                "image_path": str(image_path),
                "issue_id": "issue-validate",
                "checksum_sha256": _sha256(image_path),
                "source": {
                    "source_system": "fixture",
                    "source_id": "fixture-page-validate",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_validate_parse_input_manifest_accepts_checksum_fixture(tmp_path: Path) -> None:
    image_path = tmp_path / "inputs" / "page-validate.png"
    _write_page(image_path)
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest, image_path)

    report = validate_parse_input_manifest(
        manifest,
        require_files=True,
        require_checksums=True,
        verify_checksums=True,
    )

    assert report["status"] == "ok"
    assert report["counts"]["rows"] == 1
    assert report["counts"]["rows_with_files"] == 1
    assert report["counts"]["rows_with_checksums"] == 1
    assert report["source_systems"] == ["fixture"]
    assert report["issues"] == []


def test_validate_parse_input_manifest_flags_duplicate_page_id(tmp_path: Path) -> None:
    image_path = tmp_path / "inputs" / "page-validate.png"
    _write_page(image_path)
    manifest = tmp_path / "manifest.jsonl"
    row = {
        "page_id": "page-validate",
        "image_path": str(image_path),
        "checksum_sha256": _sha256(image_path),
    }
    manifest.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")

    report = validate_parse_input_manifest(manifest, require_files=True, verify_checksums=True)

    assert report["status"] == "error"
    assert any(issue["code"] == "duplicate_page_id" for issue in report["issues"])


def test_validate_parse_input_manifest_rejects_path_traversal_page_id(tmp_path: Path) -> None:
    image_path = tmp_path / "inputs" / "page-validate.png"
    _write_page(image_path)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"page_id": "../outside", "image_path": str(image_path)}) + "\n",
        encoding="utf-8",
    )

    report = validate_parse_input_manifest(manifest, require_files=True)

    assert report["status"] == "error"
    assert any(issue["code"] == "invalid_page_id" for issue in report["issues"])


def test_validate_parse_input_manifest_cli_writes_report(tmp_path: Path) -> None:
    image_path = tmp_path / "inputs" / "page-validate.png"
    _write_page(image_path)
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest, image_path)
    output_json = tmp_path / "validation.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "newsbag",
            "validate-parse-input-manifest",
            "--manifest",
            str(manifest),
            "--require-files",
            "--require-checksums",
            "--verify-checksums",
            "--output-json",
            str(output_json),
            "--json",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    stdout_report = json.loads(result.stdout)
    disk_report = json.loads(output_json.read_text(encoding="utf-8"))
    assert stdout_report["status"] == "ok"
    assert disk_report["status"] == "ok"
    assert disk_report["counts"]["rows_with_checksums"] == 1


def test_validate_bagging_run_accepts_complete_run(tmp_path: Path) -> None:
    image_path = tmp_path / "inputs" / "page-validate.png"
    _write_page(image_path)
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest, image_path)
    run_dir = tmp_path / "run"
    run_bagging_canary(manifest_path=manifest, run_dir=run_dir, profile_name="baseline", repo_root=Path.cwd())

    report = validate_bagging_run(run_dir)

    assert report["status"] == "ok"
    assert report["counts"]["pages_manifest"] == 1
    assert report["counts"]["pages_completed"] == 1
    assert report["counts"]["model_plan_rows"] == 1
    assert report["counts"]["adapter_invocations_planned"] == 1
    assert report["counts"]["models"] == 1
    assert report["counts"]["model_outputs"] == 1
    assert report["counts"]["fused_pages"] == 1
    assert report["counts"]["transcripts"] == 1
    assert Path(report["paths"]["input_manifest_validation_json"]).exists()
    assert Path(report["paths"]["model_plan_jsonl"]).exists()
    assert Path(report["paths"]["performance_summary_json"]).exists()
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["outputs"]["model_plan"] == "manifests/model_plan.jsonl"
    assert summary["outputs"]["transcripts"] == "outputs/transcripts"
    assert report["issues"] == []


def test_validate_bagging_run_flags_missing_transcript(tmp_path: Path) -> None:
    image_path = tmp_path / "inputs" / "page-validate.png"
    _write_page(image_path)
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest, image_path)
    run_dir = tmp_path / "run"
    run_bagging_canary(manifest_path=manifest, run_dir=run_dir, profile_name="baseline", repo_root=Path.cwd())
    (run_dir / "outputs" / "transcripts" / "page-validate.txt").unlink()

    report = validate_bagging_run(run_dir)

    assert report["status"] == "error"
    assert any(issue["code"] == "missing_transcript_file" for issue in report["issues"])


def test_validate_bagging_run_flags_tampered_model_plan(tmp_path: Path) -> None:
    image_path = tmp_path / "inputs" / "page-validate.png"
    _write_page(image_path)
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest, image_path)
    run_dir = tmp_path / "run"
    run_bagging_canary(manifest_path=manifest, run_dir=run_dir, profile_name="baseline", repo_root=Path.cwd())
    plan_path = run_dir / "manifests" / "model_plan.jsonl"
    plan_row = json.loads(plan_path.read_text(encoding="utf-8"))
    plan_row["models"] = []
    plan_row["resource_classes"] = []
    plan_path.write_text(json.dumps(plan_row) + "\n", encoding="utf-8")

    report = validate_bagging_run(run_dir)

    assert report["status"] == "error"
    assert any(issue["code"] == "empty_model_plan" for issue in report["issues"])
    assert any(issue["code"] == "summary_model_ids_mismatch" for issue in report["issues"])


def test_validate_bagging_run_flags_tampered_performance_summary(tmp_path: Path) -> None:
    image_path = tmp_path / "inputs" / "page-validate.png"
    _write_page(image_path)
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest, image_path)
    run_dir = tmp_path / "run"
    run_bagging_canary(manifest_path=manifest, run_dir=run_dir, profile_name="baseline", repo_root=Path.cwd())
    performance_path = run_dir / "reports" / "performance_summary.json"
    performance = json.loads(performance_path.read_text(encoding="utf-8"))
    performance["raw_rows"] += 1
    performance_path.write_text(json.dumps(performance) + "\n", encoding="utf-8")

    report = validate_bagging_run(run_dir)

    assert report["status"] == "error"
    assert any(issue["code"] == "performance_summary_row_mismatch" for issue in report["issues"])


def test_validate_bagging_run_rejects_output_path_escape(tmp_path: Path) -> None:
    image_path = tmp_path / "inputs" / "page-validate.png"
    _write_page(image_path)
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest, image_path)
    run_dir = tmp_path / "run"
    run_bagging_canary(manifest_path=manifest, run_dir=run_dir, profile_name="baseline", repo_root=Path.cwd())
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["outputs"]["model_plan"] = "/etc/passwd"
    summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")

    report = validate_bagging_run(run_dir)

    assert report["status"] == "error"
    assert any(issue["code"] == "output_path_outside_run" for issue in report["issues"])
