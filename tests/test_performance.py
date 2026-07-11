from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from newsbag.bagging import run_bagging_canary
from newsbag.performance import summarize_bagging_performance, write_bagging_performance_summary


def _write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    image_path = tmp_path / "inputs" / "page-performance.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (360, 540), "white")
    draw = ImageDraw.Draw(image)
    for y in range(42, 490, 30):
        draw.rectangle([30, y, 330, y + 8], fill="black")
    image.save(image_path)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"page_id": "page-performance", "image_path": str(image_path)}) + "\n",
        encoding="utf-8",
    )
    return image_path, manifest


def test_performance_summary_reconciles_plan_and_raw_timings(tmp_path: Path) -> None:
    _, manifest = _write_fixture(tmp_path)
    run_dir = tmp_path / "run"
    run_bagging_canary(
        manifest_path=manifest,
        run_dir=run_dir,
        profile_name="full",
        repo_root=Path.cwd(),
    )

    report = summarize_bagging_performance(run_dir)

    assert report["contract"] == "parser-performance-summary-v1"
    assert report["raw_rows"] == 6
    assert report["coverage"] == {
        "planned_invocations": 3,
        "observed_invocations": 3,
        "successful_invocations": 3,
        "failed_invocations": 0,
        "missing_invocations": 0,
        "unexpected_invocations": 0,
        "coverage_ratio": 1.0,
        "missing_sample": [],
        "unexpected_sample": [],
    }
    assert set(report["by_stage"]) == {"fusion", "model_adapter", "page_profile", "page_total"}
    assert set(report["by_model"]) == {
        "baseline_geometry_v1",
        "column_detector_v1",
        "legal_notice_probe_v1",
    }
    assert report["by_model"]["column_detector_v1"]["family"] == "layout"
    assert report["by_model"]["column_detector_v1"]["resource_class"] == "cpu"
    assert set(report["by_model_family"]) == {"layout", "profile", "text_probe"}
    assert report["by_resource_class"]["cpu"]["count"] == 3
    assert report["by_page_complexity"]["medium"]["count"] == 6
    assert report["adapter_invocations"]["seconds_p95"] >= 0.0
    assert write_bagging_performance_summary(run_dir) == report


def test_summarize_performance_cli_writes_requested_report(tmp_path: Path) -> None:
    _, manifest = _write_fixture(tmp_path)
    run_dir = tmp_path / "run"
    run_bagging_canary(
        manifest_path=manifest,
        run_dir=run_dir,
        profile_name="baseline",
        repo_root=Path.cwd(),
    )
    output_path = tmp_path / "rebuilt-performance.json"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "newsbag",
            "summarize-performance",
            "--run-dir",
            str(run_dir),
            "--output-json",
            str(output_path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    stdout_report = json.loads(completed.stdout)
    disk_report = json.loads(output_path.read_text(encoding="utf-8"))
    assert stdout_report == disk_report
    assert disk_report["coverage"]["coverage_ratio"] == 1.0
