from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from newsbag.bagging import run_bagging_canary
from newsbag.legacy import coerce_bbox_xyxy, legacy_payload_to_regions


def _write_page(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (360, 520), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([40, 50, 310, 120], fill="black")
    draw.rectangle([40, 160, 310, 250], fill="black")
    image.save(path)


def test_legacy_payload_to_regions_normalizes_common_box_shapes() -> None:
    payload = {
        "slug": "page-legacy",
        "variant": "paddle_layout_x",
        "boxes": [
            {
                "source_family": "paddle",
                "source_model": "paddle_layout_x",
                "source_label": "article",
                "norm_label": "text",
                "bbox_xyxy": [10, 20, 110, 80],
                "score": None,
                "reading_order": 2,
                "text": "legacy article text",
            },
            {
                "label": "photo",
                "coordinate": [140, 30, 220, 120],
                "score": 0.83,
            },
            {
                "label": "broken",
                "bbox_xyxy": [20, 20, 10, 10],
            },
        ],
    }

    regions = legacy_payload_to_regions(payload, page_id="page-legacy", model_id="legacy_model")

    assert len(regions) == 2
    assert regions[0]["label"] == "text"
    assert regions[0]["confidence"] == 0.5
    assert regions[0]["text"] == "legacy article text"
    assert regions[0]["metadata"]["legacy_source_family"] == "paddle"
    assert regions[1]["label"] == "image"
    assert regions[1]["confidence"] == 0.83
    assert coerce_bbox_xyxy([[0, 0], [10, 1], [9, 20], [1, 19]]) == [0.0, 0.0, 10.0, 20.0]


def test_legacy_converter_runs_as_command_adapter(tmp_path: Path) -> None:
    image_path = tmp_path / "inputs" / "page-legacy.png"
    _write_page(image_path)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"page_id": "page-legacy", "image_path": str(image_path), "issue_id": "issue-legacy"}) + "\n",
        encoding="utf-8",
    )
    legacy_json = image_path.with_suffix(".legacy.json")
    legacy_json.parent.mkdir(parents=True, exist_ok=True)
    legacy_json.write_text(
        json.dumps(
            {
                "boxes": [
                    {
                        "source_family": "mineru",
                        "source_model": "mineru25",
                        "source_label": "paragraph",
                        "bbox_xyxy": [35, 45, 315, 130],
                        "text": "legacy transcript from imported model",
                    }
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = tmp_path / "bagging-legacy.json"
    script = Path.cwd() / "scripts" / "legacy_layout_to_model_output.py"
    config.write_text(
        json.dumps(
            {
                "include_builtin_adapters": False,
                "command_adapters": [
                    {
                        "model_id": "legacy_mineru_import",
                        "family": "layout",
                        "resource_class": "cpu",
                        "profiles": ["legacy"],
                        "command": [
                            sys.executable,
                            str(script),
                            "--input-json",
                            "{image_dir}/{image_stem}.legacy.json",
                            "--page-id",
                            "{page_id}",
                            "--model-id",
                            "{model_id}",
                            "--source-family",
                            "mineru",
                            "--output-json",
                            "{output_path}",
                        ],
                        "timeout_seconds": 10,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    bundle = run_bagging_canary(
        manifest_path=manifest,
        run_dir=tmp_path / "run",
        profile_name="legacy",
        config_path=config,
        repo_root=Path.cwd(),
    )

    assert bundle.performance["pages_completed"] == 1
    fused = json.loads((tmp_path / "run" / "outputs" / "fused_pages" / "page-legacy.json").read_text())
    assert "legacy transcript from imported model" in fused["transcript"]
    model_output = json.loads(
        (tmp_path / "run" / "outputs" / "model_outputs" / "legacy_mineru_import" / "page-legacy.json").read_text()
    )
    assert model_output["runtime"]["metadata"]["adapter_kind"] == "command"
    assert model_output["runtime"]["metadata"]["source_adapter_kind"] == "legacy_layout_import"
    assert model_output["runtime"]["metadata"]["input_region_count"] == 1
    assert model_output["runtime"]["metadata"]["skipped_region_count"] == 0
    assert model_output["regions"][0]["metadata"]["legacy_source_family"] == "mineru"
