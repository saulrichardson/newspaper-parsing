from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

from newsbag.bagging import run_bagging_canary
from newsbag.legacy_run import build_legacy_bagging_config, discover_legacy_layout_sources


def _write_legacy_source(run_dir: Path, source_root: str, variant: str, slug: str, text: str) -> None:
    path = run_dir / "outputs" / "sources" / source_root / variant / slug / "layout_boxes.normalized.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "boxes": [
                    {
                        "source_family": source_root,
                        "source_model": variant,
                        "source_label": "paragraph",
                        "bbox_xyxy": [10, 20, 180, 90],
                        "score": 0.77,
                        "text": text,
                    }
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_build_legacy_run_config_discovers_variants_and_runs_bagging(tmp_path: Path) -> None:
    legacy_run = tmp_path / "legacy-run"
    _write_legacy_source(legacy_run, "paddle_layout", "pp_doc_v3", "page-a", "paddle imported text")
    _write_legacy_source(legacy_run, "dell", "dell_c0005", "page-a", "dell imported text")

    sources = discover_legacy_layout_sources(legacy_run)
    assert [source.model_id for source in sources] == [
        "legacy_paddle_layout_pp_doc_v3",
        "legacy_dell_dell_c0005",
    ]

    config, summary = build_legacy_bagging_config(legacy_run, profile_name="legacy_import")
    assert summary["source_count"] == 2
    assert len(config["command_adapters"]) == 2
    assert all("--allow-missing" in adapter["command"] for adapter in config["command_adapters"])

    config_path = tmp_path / "legacy-config.json"
    config_path.write_text(json.dumps(config) + "\n", encoding="utf-8")
    image_path = tmp_path / "page-a.png"
    Image.new("RGB", (220, 300), "white").save(image_path)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"page_id": "page-a", "image_path": str(image_path)}) + "\n", encoding="utf-8")

    bundle = run_bagging_canary(
        manifest_path=manifest,
        run_dir=tmp_path / "run",
        profile_name="legacy_import",
        config_path=config_path,
        repo_root=Path.cwd(),
    )

    assert bundle.performance["pages_completed"] == 1
    transcript = (tmp_path / "run" / "outputs" / "transcripts" / "page-a.txt").read_text(encoding="utf-8")
    assert "paddle imported text" in transcript
    assert "dell imported text" in transcript


def test_legacy_import_allow_missing_emits_skipped_output(tmp_path: Path) -> None:
    output_path = tmp_path / "model-output.json"
    missing_path = tmp_path / "missing.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/legacy_layout_to_model_output.py",
            "--input-json",
            str(missing_path),
            "--page-id",
            "page-missing",
            "--model-id",
            "legacy_missing",
            "--output-json",
            str(output_path),
            "--allow-missing",
        ],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["runtime"]["status"] == "skipped"
    assert payload["runtime"]["metadata"]["missing_input_paths"] == [str(missing_path.resolve())]
    assert payload["regions"] == []


def test_build_legacy_run_config_fails_when_no_sources(tmp_path: Path) -> None:
    try:
        build_legacy_bagging_config(tmp_path / "empty-run")
    except ValueError as exc:
        assert "no legacy layout sources found" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected empty legacy run discovery to fail")
