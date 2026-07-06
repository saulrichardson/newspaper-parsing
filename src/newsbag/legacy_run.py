from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

LEGACY_SOURCE_ROOTS: dict[str, str] = {
    "paddle_layout": "paddle",
    "paddle_vl15": "paddle",
    "dell": "dell",
    "mineru": "mineru",
}


@dataclass(frozen=True)
class LegacyLayoutSource:
    source_root: str
    source_family: str
    variant_id: str
    variant_dir: str
    normalized_json_count: int
    model_id: str


def _safe_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_]+", "_", value.strip())
    token = re.sub(r"_+", "_", token).strip("_").lower()
    return token or "unknown"


def legacy_model_id(source_root: str, variant_id: str) -> str:
    return f"legacy_{_safe_token(source_root)}_{_safe_token(variant_id)}"


def discover_legacy_layout_sources(
    legacy_run_dir: Path,
    *,
    source_roots: list[str] | None = None,
) -> list[LegacyLayoutSource]:
    run_dir = legacy_run_dir.expanduser().resolve()
    selected_roots = source_roots or list(LEGACY_SOURCE_ROOTS)
    unknown = sorted(set(selected_roots) - set(LEGACY_SOURCE_ROOTS))
    if unknown:
        raise ValueError(f"unknown legacy source roots: {unknown}; expected one of {sorted(LEGACY_SOURCE_ROOTS)}")

    sources_root = run_dir / "outputs" / "sources"
    discovered: list[LegacyLayoutSource] = []
    for source_root in selected_roots:
        root = sources_root / source_root
        if not root.exists():
            continue
        for variant_dir in sorted((item for item in root.iterdir() if item.is_dir()), key=lambda path: path.name):
            normalized = sorted(variant_dir.glob("*/layout_boxes.normalized.json"))
            if not normalized:
                continue
            variant_id = variant_dir.name
            discovered.append(
                LegacyLayoutSource(
                    source_root=source_root,
                    source_family=LEGACY_SOURCE_ROOTS[source_root],
                    variant_id=variant_id,
                    variant_dir=str(variant_dir),
                    normalized_json_count=len(normalized),
                    model_id=legacy_model_id(source_root, variant_id),
                )
            )
    return discovered


def build_legacy_bagging_config(
    legacy_run_dir: Path,
    *,
    profile_name: str = "legacy_import",
    source_roots: list[str] | None = None,
    allow_missing: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    run_dir = legacy_run_dir.expanduser().resolve()
    sources = discover_legacy_layout_sources(run_dir, source_roots=source_roots)
    if not sources:
        selected = source_roots or list(LEGACY_SOURCE_ROOTS)
        raise ValueError(
            "no legacy layout sources found under "
            f"{run_dir / 'outputs' / 'sources'} for source roots {selected}; "
            "expected */<variant>/<page_slug>/layout_boxes.normalized.json"
        )
    command_adapters: list[dict[str, Any]] = []
    for source in sources:
        input_pattern = str(Path(source.variant_dir) / "{image_stem}" / "layout_boxes.normalized.json")
        command = [
            "{python}",
            "{repo_root}/scripts/legacy_layout_to_model_output.py",
            "--input-json",
            input_pattern,
            "--page-id",
            "{page_id}",
            "--model-id",
            "{model_id}",
            "--source-family",
            source.source_family,
            "--output-json",
            "{output_path}",
        ]
        if allow_missing:
            command.append("--allow-missing")
        command_adapters.append(
            {
                "model_id": source.model_id,
                "family": "layout",
                "resource_class": "cpu",
                "profiles": [profile_name],
                "timeout_seconds": 120,
                "command": command,
                "metadata": {
                    "legacy_source_root": source.source_root,
                    "legacy_source_family": source.source_family,
                    "legacy_variant_id": source.variant_id,
                    "legacy_variant_dir": source.variant_dir,
                    "legacy_normalized_json_count": source.normalized_json_count,
                    "allow_missing": allow_missing,
                },
            }
        )

    config = {"include_builtin_adapters": False, "command_adapters": command_adapters}
    summary = {
        "legacy_run_dir": str(run_dir),
        "profile": profile_name,
        "allow_missing": allow_missing,
        "source_count": len(sources),
        "sources": [asdict(source) for source in sources],
    }
    return config, summary


def write_legacy_bagging_config(
    *,
    legacy_run_dir: Path,
    output_config: Path,
    profile_name: str = "legacy_import",
    source_roots: list[str] | None = None,
    allow_missing: bool = True,
    output_summary: Path | None = None,
) -> dict[str, Any]:
    config, summary = build_legacy_bagging_config(
        legacy_run_dir,
        profile_name=profile_name,
        source_roots=source_roots,
        allow_missing=allow_missing,
    )
    out = output_config.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if output_summary is not None:
        summary_path = output_summary.expanduser().resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary
