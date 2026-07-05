from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from newsbag.bagging import adapters_for_profile, run_bagging_canary


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


def test_command_adapter_config_runs_subprocess_and_fuses_output(tmp_path: Path) -> None:
    image_path = tmp_path / "inputs" / "page-002.png"
    _write_page(image_path)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "page_id": "page-002",
                "image_path": str(image_path),
                "issue_id": "issue-002",
                "page_number": 2,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    adapter_script = tmp_path / "adapter.py"
    adapter_script.write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "import json",
                "import sys",
                "from pathlib import Path",
                "page_id = sys.argv[1]",
                "model_id = sys.argv[2]",
                "output_path = Path(sys.argv[3])",
                "payload = {",
                "  'page_id': page_id,",
                "  'model_id': model_id,",
                "  'regions': [",
                "    {",
                "      'bbox_xyxy': [20, 30, 220, 90],",
                "      'label': 'text',",
                "      'confidence': 0.91,",
                "      'text': 'command adapter transcript text',",
                "      'reading_order': 1,",
                "      'metadata': {'adapter': 'fixture'},",
                "    }",
                "  ],",
                "  'metadata': {'fixture': True},",
                "}",
                "output_path.write_text(json.dumps(payload) + '\\n', encoding='utf-8')",
            ]
        ),
        encoding="utf-8",
    )
    config = tmp_path / "bagging-config.json"
    config.write_text(
        json.dumps(
            {
                "include_builtin_adapters": False,
                "command_adapters": [
                    {
                        "model_id": "fixture_command_v1",
                        "family": "layout",
                        "resource_class": "cpu",
                        "profiles": ["command"],
                        "command": [
                            sys.executable,
                            str(adapter_script),
                            "{page_id}",
                            "{model_id}",
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
    run_dir = tmp_path / "run"

    bundle = run_bagging_canary(
        manifest_path=manifest,
        run_dir=run_dir,
        profile_name="command",
        config_path=config,
        repo_root=Path.cwd(),
    )

    assert bundle.model_ids == ["fixture_command_v1"]
    assert bundle.performance["pages_completed"] == 1
    fused = json.loads((run_dir / "outputs" / "fused_pages" / "page-002.json").read_text(encoding="utf-8"))
    assert fused["model_ids"] == ["fixture_command_v1"]
    assert "command adapter transcript text" in fused["transcript"]
    model_output = json.loads(
        (run_dir / "outputs" / "model_outputs" / "fixture_command_v1" / "page-002.json").read_text(
            encoding="utf-8"
        )
    )
    assert model_output["runtime"]["metadata"]["adapter_kind"] == "command"
    assert model_output["regions"][0]["metadata"]["adapter"] == "fixture"


def test_configured_profile_without_matching_adapters_fails(tmp_path: Path) -> None:
    config = tmp_path / "bagging-config.json"
    config.write_text(json.dumps({"include_builtin_adapters": False, "command_adapters": []}) + "\n")

    try:
        adapters_for_profile("missing", config_path=config)
    except ValueError as exc:
        assert "matched no adapters" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected missing adapter profile to fail")
