from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from newsbag.bagging import run_bagging_canary
from newsbag.validation import validate_bagging_run


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
            }
        )
        + "\n",
        encoding="utf-8",
    )


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
    assert report["counts"]["models"] == 1
    assert report["counts"]["model_outputs"] == 1
    assert report["counts"]["fused_pages"] == 1
    assert report["counts"]["transcripts"] == 1
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
