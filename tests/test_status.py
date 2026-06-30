from __future__ import annotations

import json
from pathlib import Path

from newsbag.status import summarize_run


def _write_manifest(run_dir: Path, slugs: list[str]) -> None:
    manifest_dir = run_dir / "manifests"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "images.resolved.txt").write_text(
        "\n".join(f"/input/{slug}.png" for slug in slugs) + "\n",
        encoding="utf-8",
    )


def _write_config(run_dir: Path) -> None:
    (run_dir / "manifests" / "config.resolved.json").write_text(
        json.dumps(
            {
                "paddle_layout_variants": [{"variant_id": "pld_v3"}],
                "paddle_vl15": {"enabled": True, "variant_id": "pvl15"},
                "dell": {"enabled": True, "variant_id": "dell"},
                "mineru": {"enabled": True, "variant_id": "mineru"},
                "fusion": {"recommended_variant": "fused"},
                "transcription": {"enabled": True, "variant": "fused"},
            }
        ),
        encoding="utf-8",
    )


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}", encoding="utf-8")


def test_summarize_run_counts_indexed_outputs_and_missing_slugs(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    slugs = ["page-001", "page-002", "page-003"]
    _write_manifest(run_dir, slugs)
    _write_config(run_dir)

    _touch(
        run_dir
        / "outputs"
        / "sources"
        / "paddle_layout"
        / "pld_v3"
        / "page-001"
        / "layout_boxes.normalized.json"
    )
    _touch(
        run_dir
        / "outputs"
        / "sources"
        / "paddle_layout"
        / "pld_v3"
        / "extra-page"
        / "layout_boxes.normalized.json"
    )
    _touch(
        run_dir
        / "outputs"
        / "sources"
        / "paddle_vl15"
        / "pvl15"
        / "page-002"
        / "parsing_blocks.json"
    )
    _touch(
        run_dir
        / "outputs"
        / "fusion"
        / "fused"
        / "page-003"
        / "fused_boxes.json"
    )
    _touch(run_dir / "outputs" / "transcription" / "fused" / "page-003" / "transcript.txt")

    summary = summarize_run(run_dir, missing_limit=2)
    stages = summary["stages"]

    assert stages["paddle_layout"]["pld_v3"]["done"] == 1
    assert stages["paddle_layout"]["pld_v3"]["total"] == 3
    assert stages["paddle_layout"]["pld_v3"]["missing"] == ["page-002", "page-003"]
    assert stages["paddle_vl15"]["done"] == 1
    assert stages["paddle_vl15"]["missing"] == ["page-001", "page-003"]
    assert stages["fusion"]["done"] == 1
    assert stages["transcription"]["count"]["done"] == 1

