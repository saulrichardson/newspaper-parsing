from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from newsbag.bagging import run_bagging_canary


def _write_page(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (320, 480), "white")
    draw = ImageDraw.Draw(image)
    for y in range(40, 420, 28):
        draw.rectangle([30, y, 280, y + 8], fill="black")
    image.save(path)


def test_bagging_canary_writes_contract_artifacts(tmp_path: Path) -> None:
    image_path = tmp_path / "inputs" / "page-001.png"
    _write_page(image_path)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "page_id": "page-001",
                "image_path": str(image_path),
                "issue_id": "issue-001",
                "page_number": 1,
                "source": {
                    "source_system": "fixture",
                    "source_id": "fixture-page-001",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"

    bundle = run_bagging_canary(manifest_path=manifest, run_dir=run_dir, profile_name="full", repo_root=Path.cwd())

    assert bundle.page_count == 1
    assert bundle.performance["pages_completed"] == 1
    assert (run_dir / "profiles" / "page-001.json").exists()
    assert (run_dir / "outputs" / "fused_pages" / "page-001.json").exists()
    assert (run_dir / "outputs" / "transcripts" / "page-001.txt").exists()
    assert (run_dir / "reports" / "performance.json").exists()
    assert (run_dir / "provenance.json").exists()
    fused = json.loads((run_dir / "outputs" / "fused_pages" / "page-001.json").read_text(encoding="utf-8"))
    assert fused["model_ids"] == ["baseline_geometry_v1", "column_detector_v1", "legal_notice_probe_v1"]
    assert fused["quality"]["region_count"] >= 3

