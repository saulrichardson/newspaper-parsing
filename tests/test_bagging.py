from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from newsbag.bagging import adapters_for_profile, plan_bagging, profile_page, run_bagging_canary
from newsbag.contracts import ParseInputPage
from newsbag.validation import validate_bagging_run


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
    assert (run_dir / "reports" / "input_manifest_validation.json").exists()
    assert (run_dir / "provenance.json").exists()
    provenance = json.loads((run_dir / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["input_manifest_validation_status"] == "ok"
    fused = json.loads((run_dir / "outputs" / "fused_pages" / "page-001.json").read_text(encoding="utf-8"))
    assert fused["model_ids"] == ["baseline_geometry_v1", "column_detector_v1", "legal_notice_probe_v1"]
    assert fused["quality"]["region_count"] >= 3


def test_page_profiler_bounds_pixel_work_and_reuses_manifest_checksum(tmp_path: Path) -> None:
    image_path = tmp_path / "inputs" / "large-page.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (2048, 3072), 255).save(image_path)
    checksum = "a" * 64

    profile = profile_page(
        ParseInputPage(
            page_id="large-page",
            image_path=str(image_path),
            checksum_sha256=checksum,
        )
    )

    assert profile.width == 2048
    assert profile.height == 3072
    assert profile.metadata["profile_sample_width"] <= 1024
    assert profile.metadata["profile_sample_height"] <= 1024
    assert profile.metadata["image_sha256"] == checksum
    assert profile.metadata["checksum_source"] == "manifest"


def test_bagging_resolves_relative_image_paths_from_manifest_directory(tmp_path: Path) -> None:
    image_path = tmp_path / "inputs" / "relative-page.png"
    _write_page(image_path)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"page_id": "relative-page", "image_path": "inputs/relative-page.png"}) + "\n",
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"

    bundle = run_bagging_canary(
        manifest_path=manifest,
        run_dir=run_dir,
        profile_name="baseline",
        repo_root=Path.cwd(),
    )

    copied = json.loads((run_dir / "manifests" / "parse_input.jsonl").read_text(encoding="utf-8"))
    assert bundle.performance["pages_completed"] == 1
    assert copied["image_path"] == str(image_path.resolve())


def test_bagging_refuses_to_mix_outputs_in_existing_run_directory(tmp_path: Path) -> None:
    image_path = tmp_path / "inputs" / "page-existing.png"
    _write_page(image_path)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"page_id": "page-existing", "image_path": str(image_path)}) + "\n",
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    run_bagging_canary(
        manifest_path=manifest,
        run_dir=run_dir,
        profile_name="baseline",
        repo_root=Path.cwd(),
    )

    try:
        run_bagging_canary(
            manifest_path=manifest,
            run_dir=run_dir,
            profile_name="baseline",
            repo_root=Path.cwd(),
        )
    except FileExistsError as exc:
        assert "must be empty or absent" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected an existing run directory to fail")


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


def test_config_rejects_model_id_path_traversal(tmp_path: Path) -> None:
    config = tmp_path / "bagging-config.json"
    config.write_text(
        json.dumps(
            {
                "include_builtin_adapters": False,
                "command_adapters": [
                    {
                        "model_id": "../outside",
                        "profiles": ["unsafe"],
                        "command": ["unused"],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    try:
        adapters_for_profile("unsafe", config_path=config)
    except ValueError as exc:
        assert "portable artifact identifier" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected unsafe model_id to fail")


def test_plan_bagging_routes_adapters_by_page_complexity(tmp_path: Path) -> None:
    easy_path = tmp_path / "inputs" / "easy.png"
    hard_path = tmp_path / "inputs" / "hard.png"
    easy_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (320, 320), "white").save(easy_path)
    Image.new("RGB", (320, 640), "black").save(hard_path)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(
            json.dumps({"page_id": page_id, "image_path": str(image_path)})
            for page_id, image_path in (("easy-page", easy_path), ("hard-page", hard_path))
        )
        + "\n",
        encoding="utf-8",
    )
    config = tmp_path / "bagging-config.json"
    config.write_text(
        json.dumps(
            {
                "include_builtin_adapters": False,
                "command_adapters": [
                    {
                        "model_id": "all_pages_v1",
                        "family": "layout",
                        "resource_class": "cpu",
                        "profiles": ["routed"],
                        "complexities": ["easy", "medium", "hard"],
                        "command": ["unused"],
                    },
                    {
                        "model_id": "hard_pages_v1",
                        "family": "ocr",
                        "resource_class": "gpu",
                        "profiles": ["routed"],
                        "complexities": ["hard"],
                        "command": ["unused"],
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    run_dir = tmp_path / "plan"

    summary = plan_bagging(
        manifest_path=manifest,
        run_dir=run_dir,
        profile_name="routed",
        config_path=config,
        repo_root=Path.cwd(),
    )

    rows = [
        json.loads(line)
        for line in (run_dir / "manifests" / "model_plan.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    by_page = {row["page_id"]: row for row in rows}
    assert [model["model_id"] for model in by_page["easy-page"]["models"]] == ["all_pages_v1"]
    assert [model["model_id"] for model in by_page["hard-page"]["models"]] == [
        "all_pages_v1",
        "hard_pages_v1",
    ]
    assert by_page["easy-page"]["estimated_complexity"] == "easy"
    assert by_page["hard-page"]["estimated_complexity"] == "hard"
    assert summary["adapter_invocations_planned"] == 3
    assert summary["resource_class_invocation_counts"] == {"cpu": 2, "gpu": 1}
    assert not (run_dir / "summary.json").exists()


def test_bagging_keeps_successful_outputs_when_one_adapter_fails(tmp_path: Path) -> None:
    image_path = tmp_path / "inputs" / "page-partial.png"
    _write_page(image_path)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"page_id": "page-partial", "image_path": str(image_path)}) + "\n",
        encoding="utf-8",
    )
    good_script = tmp_path / "good_adapter.py"
    good_script.write_text(
        "\n".join(
            [
                "import json, sys",
                "from pathlib import Path",
                "page_id, model_id, output_path = sys.argv[1:4]",
                "payload = {'page_id': page_id, 'model_id': model_id, 'regions': [",
                "  {'bbox_xyxy': [10, 10, 100, 80], 'label': 'text', 'confidence': 0.9, 'text': 'kept'}",
                "]}",
                "Path(output_path).write_text(json.dumps(payload) + '\\n', encoding='utf-8')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    failing_script = tmp_path / "failing_adapter.py"
    failing_script.write_text("raise SystemExit('intentional adapter failure')\n", encoding="utf-8")
    command = ["{python}", "", "{page_id}", "{model_id}", "{output_path}"]
    config = tmp_path / "bagging-config.json"
    config.write_text(
        json.dumps(
            {
                "include_builtin_adapters": False,
                "command_adapters": [
                    {
                        "model_id": "good_v1",
                        "family": "ocr",
                        "resource_class": "cpu",
                        "profiles": ["partial"],
                        "command": command[:1] + [str(good_script)] + command[2:],
                    },
                    {
                        "model_id": "failing_v1",
                        "family": "layout",
                        "resource_class": "gpu",
                        "profiles": ["partial"],
                        "command": command[:1] + [str(failing_script)],
                    },
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
        profile_name="partial",
        config_path=config,
        repo_root=Path.cwd(),
    )

    assert bundle.performance["pages_completed"] == 1
    assert bundle.performance["pages_failed"] == 0
    assert bundle.performance["adapter_errors"] == 1
    fused = json.loads((run_dir / "outputs" / "fused_pages" / "page-partial.json").read_text(encoding="utf-8"))
    assert fused["model_ids"] == ["good_v1"]
    assert fused["transcript"] == "kept"
    errors = [json.loads(line) for line in (run_dir / "errors.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [(row["scope"], row["model_id"]) for row in errors] == [("adapter", "failing_v1")]
    performance = json.loads((run_dir / "reports" / "performance_summary.json").read_text(encoding="utf-8"))
    assert performance["coverage"]["failed_invocations"] == 1
    assert performance["coverage"]["missing_invocations"] == 0
    validation = validate_bagging_run(run_dir)
    issue_codes = {issue["code"] for issue in validation["issues"]}
    assert validation["status"] == "error"
    assert "run_has_errors" in issue_codes
    assert "fused_model_ids_mismatch" not in issue_codes
    assert "model_output_count_mismatch" not in issue_codes
