from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from PIL import Image, ImageStat

from newsbag.contracts import (
    FusedPage,
    ModelOutput,
    NormalizedRegion,
    PageModelPlan,
    PageProfile,
    ParseInputPage,
    PlannedModel,
    RunBundle,
    RuntimeInfo,
    is_safe_artifact_id,
    read_json,
    read_parse_input_manifest,
    write_json,
    write_jsonl,
)
from newsbag.performance import write_bagging_performance_summary
from newsbag.validation import validate_parse_input_manifest


PAGE_COMPLEXITIES = ("easy", "medium", "hard")
PROFILE_SAMPLE_MAX_SIDE = 1024


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    family: str
    resource_class: str
    profile_names: tuple[str, ...]
    complexity_names: tuple[str, ...] = PAGE_COMPLEXITIES


@dataclass(frozen=True)
class AdapterContext:
    run_dir: Path
    manifest_path: Path
    repo_root: Path


class ModelAdapter(Protocol):
    spec: ModelSpec

    def run(self, page: ParseInputPage, profile: PageProfile, context: AdapterContext) -> ModelOutput:
        ...


@dataclass(frozen=True)
class CommandAdapterConfig:
    model_id: str
    family: str
    resource_class: str
    profile_names: tuple[str, ...]
    command: tuple[str, ...]
    complexity_names: tuple[str, ...] = PAGE_COMPLEXITIES
    timeout_seconds: float = 600.0
    env: dict[str, str] = field(default_factory=dict)
    cwd: str = ""


@dataclass(frozen=True)
class BaggingConfig:
    include_builtin_adapters: bool = True
    command_adapters: tuple[CommandAdapterConfig, ...] = ()


@dataclass(frozen=True)
class PreparedBaggingPlan:
    pages: list[ParseInputPage]
    profiles: dict[str, PageProfile]
    plan_rows: list[PageModelPlan]
    adapters_by_id: dict[str, ModelAdapter]
    plan_summary: dict[str, Any]
    performance_rows: list[dict[str, object]]
    manifest_validation: dict[str, Any]


MODEL_REGISTRY: tuple[ModelSpec, ...] = (
    ModelSpec("baseline_geometry_v1", "profile", "cpu", ("baseline", "adaptive", "full")),
    ModelSpec("column_detector_v1", "layout", "cpu", ("adaptive", "full")),
    ModelSpec("legal_notice_probe_v1", "text_probe", "cpu", ("full",)),
)


def _git_commit(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def profile_page(page: ParseInputPage) -> PageProfile:
    image_path = Path(page.image_path).expanduser().resolve()
    with Image.open(image_path) as image:
        width, height = image.size
        image.draft("L", (PROFILE_SAMPLE_MAX_SIDE, PROFILE_SAMPLE_MAX_SIDE))
        gray = image.convert("L")
        gray.thumbnail(
            (PROFILE_SAMPLE_MAX_SIDE, PROFILE_SAMPLE_MAX_SIDE),
            resample=Image.Resampling.NEAREST,
        )
        sample_width, sample_height = gray.size
        stat = ImageStat.Stat(gray)
        mean_luma = float(stat.mean[0])
        histogram = gray.histogram()
        dark_pixels = sum(histogram[:96])
        total_pixels = max(1, sample_width * sample_height)
        dark_share = dark_pixels / total_pixels

    if dark_share > 0.22 or height / max(width, 1) > 1.8:
        complexity = "hard"
    elif dark_share > 0.12:
        complexity = "medium"
    else:
        complexity = "easy"

    return PageProfile(
        page_id=page.page_id,
        width=width,
        height=height,
        aspect_ratio=round(width / max(height, 1), 6),
        mean_luma=round(mean_luma, 4),
        dark_pixel_share=round(dark_share, 6),
        estimated_complexity=complexity,
        metadata={
            "image_sha256": page.checksum_sha256 or _sha256(image_path),
            "checksum_source": "manifest" if page.checksum_sha256 else "computed",
            "profile_sample_width": sample_width,
            "profile_sample_height": sample_height,
        },
    )


class BaselineGeometryAdapter:
    spec = MODEL_REGISTRY[0]

    def run(self, page: ParseInputPage, profile: PageProfile, context: AdapterContext) -> ModelOutput:
        started = time.perf_counter()
        margin_x = max(8, int(profile.width * 0.08))
        margin_y = max(8, int(profile.height * 0.08))
        region = NormalizedRegion(
            region_id=f"{page.page_id}:baseline:main",
            bbox_xyxy=[margin_x, margin_y, profile.width - margin_x, profile.height - margin_y],
            label="text",
            confidence=0.55,
            source_model=self.spec.model_id,
            text=f"[baseline text region for {page.page_id}]",
            reading_order=1,
            provenance=[self.spec.model_id],
            metadata={"adapter_family": self.spec.family},
        )
        return ModelOutput(
            page_id=page.page_id,
            model_id=self.spec.model_id,
            regions=[region],
            runtime=RuntimeInfo(
                seconds=round(time.perf_counter() - started, 6),
                resource_class=self.spec.resource_class,
                status="ok",
            ),
            profile=profile,
        )


class ColumnDetectorAdapter:
    spec = MODEL_REGISTRY[1]

    def run(self, page: ParseInputPage, profile: PageProfile, context: AdapterContext) -> ModelOutput:
        started = time.perf_counter()
        gutter = max(8, int(profile.width * 0.03))
        mid = profile.width // 2
        top = max(8, int(profile.height * 0.10))
        bottom = profile.height - max(8, int(profile.height * 0.08))
        regions = [
            NormalizedRegion(
                region_id=f"{page.page_id}:columns:left",
                bbox_xyxy=[max(0, int(profile.width * 0.06)), top, mid - gutter, bottom],
                label="text",
                confidence=0.68,
                source_model=self.spec.model_id,
                text=f"[left column candidate for {page.page_id}]",
                reading_order=1,
                provenance=[self.spec.model_id],
                metadata={"adapter_family": self.spec.family},
            ),
            NormalizedRegion(
                region_id=f"{page.page_id}:columns:right",
                bbox_xyxy=[mid + gutter, top, profile.width - int(profile.width * 0.06), bottom],
                label="text",
                confidence=0.68,
                source_model=self.spec.model_id,
                text=f"[right column candidate for {page.page_id}]",
                reading_order=2,
                provenance=[self.spec.model_id],
                metadata={"adapter_family": self.spec.family},
            ),
        ]
        return ModelOutput(
            page_id=page.page_id,
            model_id=self.spec.model_id,
            regions=regions,
            runtime=RuntimeInfo(
                seconds=round(time.perf_counter() - started, 6),
                resource_class=self.spec.resource_class,
                status="ok",
            ),
            profile=profile,
        )


class LegalNoticeProbeAdapter:
    spec = MODEL_REGISTRY[2]

    def run(self, page: ParseInputPage, profile: PageProfile, context: AdapterContext) -> ModelOutput:
        started = time.perf_counter()
        region = NormalizedRegion(
            region_id=f"{page.page_id}:legal-probe:footer",
            bbox_xyxy=[
                int(profile.width * 0.08),
                int(profile.height * 0.70),
                int(profile.width * 0.92),
                int(profile.height * 0.93),
            ],
            label="legal_notice_candidate",
            confidence=0.42 if profile.estimated_complexity == "easy" else 0.72,
            source_model=self.spec.model_id,
            text=f"[legal notice candidate for {page.page_id}]",
            reading_order=3,
            provenance=[self.spec.model_id],
            metadata={"adapter_family": self.spec.family},
        )
        return ModelOutput(
            page_id=page.page_id,
            model_id=self.spec.model_id,
            regions=[region],
            runtime=RuntimeInfo(
                seconds=round(time.perf_counter() - started, 6),
                resource_class=self.spec.resource_class,
                status="ok",
            ),
            profile=profile,
        )


ADAPTERS: dict[str, ModelAdapter] = {
    adapter.spec.model_id: adapter
    for adapter in (
        BaselineGeometryAdapter(),
        ColumnDetectorAdapter(),
        LegalNoticeProbeAdapter(),
    )
}


def _config_bool(payload: dict[str, Any], key: str, default: bool) -> bool:
    value = payload.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _config_names(value: Any, *, field_name: str) -> tuple[str, ...]:
    raw_items = [value] if isinstance(value, str) else value
    if not isinstance(raw_items, (list, tuple)):
        raise ValueError(f"{field_name} must be a string or list of strings")
    names = tuple(dict.fromkeys(str(item).strip() for item in raw_items if str(item).strip()))
    if not names:
        raise ValueError(f"{field_name} must contain at least one value")
    return names


def load_bagging_config(config_path: Path | None) -> BaggingConfig:
    if config_path is None:
        return BaggingConfig()
    payload = read_json(config_path.expanduser().resolve())
    if not isinstance(payload, dict):
        raise ValueError(f"bagging config must be a JSON object: {config_path}")
    command_adapters: list[CommandAdapterConfig] = []
    for index, raw in enumerate(payload.get("command_adapters") or [], start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"command_adapters[{index}] must be a JSON object")
        model_id = str(raw.get("model_id") or "").strip()
        command_raw = raw.get("command")
        if not isinstance(command_raw, list):
            raise ValueError(f"command_adapters[{index}].command must be a JSON list")
        command = tuple(str(part) for part in command_raw)
        if not model_id:
            raise ValueError(f"command_adapters[{index}] is missing model_id")
        if not is_safe_artifact_id(model_id):
            raise ValueError(
                f"command_adapters[{index}].model_id must be a portable artifact identifier"
            )
        if not command:
            raise ValueError(f"command_adapters[{index}] is missing command")
        profile_names = _config_names(
            raw.get("profiles", raw.get("profile_names", [])),
            field_name=f"command_adapters[{index}].profiles",
        )
        complexity_names = _config_names(
            raw.get("complexities", raw.get("complexity_names", PAGE_COMPLEXITIES)),
            field_name=f"command_adapters[{index}].complexities",
        )
        unknown_complexities = sorted(set(complexity_names) - set(PAGE_COMPLEXITIES))
        if unknown_complexities:
            raise ValueError(
                f"command_adapters[{index}] has unknown complexities {unknown_complexities}; "
                f"expected values from {list(PAGE_COMPLEXITIES)}"
            )
        timeout_seconds = float(raw.get("timeout_seconds") or 600.0)
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
            raise ValueError(f"command_adapters[{index}].timeout_seconds must be positive and finite")
        command_adapters.append(
            CommandAdapterConfig(
                model_id=model_id,
                family=str(raw.get("family") or "external").strip() or "external",
                resource_class=str(raw.get("resource_class") or "cpu").strip() or "cpu",
                profile_names=profile_names,
                command=command,
                complexity_names=complexity_names,
                timeout_seconds=timeout_seconds,
                env={str(k): str(v) for k, v in dict(raw.get("env") or {}).items()},
                cwd=str(raw.get("cwd") or ""),
            )
        )
    return BaggingConfig(
        include_builtin_adapters=_config_bool(payload, "include_builtin_adapters", True),
        command_adapters=tuple(command_adapters),
    )


def _region_from_payload(
    raw: dict[str, Any],
    *,
    page_id: str,
    model_id: str,
    index: int,
) -> NormalizedRegion:
    bbox = raw.get("bbox_xyxy") or raw.get("bbox") or raw.get("box")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError(f"region {index} for {page_id}/{model_id} is missing bbox_xyxy")
    label = str(raw.get("label") or raw.get("norm_label") or "text").strip() or "text"
    confidence_raw = raw.get("confidence", raw.get("score", 0.0))
    confidence = max(0.0, min(1.0, float(confidence_raw)))
    reading_order_raw = raw.get("reading_order")
    region_id = str(raw.get("region_id") or f"{page_id}:{model_id}:{index:04d}")
    provenance_raw = raw.get("provenance") or [model_id]
    provenance = [str(item) for item in provenance_raw] if isinstance(provenance_raw, list) else [str(provenance_raw)]
    metadata = dict(raw.get("metadata") or {})
    return NormalizedRegion(
        region_id=region_id,
        bbox_xyxy=[float(value) for value in bbox],
        label=label,
        confidence=confidence,
        source_model=str(raw.get("source_model") or model_id),
        text=str(raw.get("text") or ""),
        reading_order=int(reading_order_raw) if reading_order_raw not in (None, "") else None,
        provenance=provenance,
        metadata=metadata,
    )


def model_output_from_payload(
    payload: dict[str, Any],
    *,
    page_id: str,
    model_id: str,
    spec: ModelSpec,
    profile: PageProfile,
    elapsed_seconds: float,
    runtime_metadata: dict[str, Any] | None = None,
) -> ModelOutput:
    payload_page_id = str(payload.get("page_id") or page_id)
    payload_model_id = str(payload.get("model_id") or model_id)
    if payload_page_id != page_id:
        raise ValueError(f"adapter {model_id} returned page_id={payload_page_id!r}, expected {page_id!r}")
    if payload_model_id != model_id:
        raise ValueError(f"adapter {model_id} returned model_id={payload_model_id!r}")
    raw_regions = payload.get("regions") or []
    if not isinstance(raw_regions, list):
        raise ValueError(f"adapter {model_id} returned non-list regions")
    runtime_raw = payload.get("runtime") if isinstance(payload.get("runtime"), dict) else {}
    metadata = dict(runtime_raw.get("metadata") or {})
    if runtime_metadata:
        metadata.update(runtime_metadata)
    return ModelOutput(
        page_id=page_id,
        model_id=model_id,
        regions=[
            _region_from_payload(raw, page_id=page_id, model_id=model_id, index=index)
            for index, raw in enumerate(raw_regions, start=1)
            if isinstance(raw, dict)
        ],
        runtime=RuntimeInfo(
            seconds=round(float(runtime_raw.get("seconds", elapsed_seconds) or 0.0), 6),
            resource_class=str(runtime_raw.get("resource_class") or spec.resource_class),
            status=str(runtime_raw.get("status") or "ok"),
            error=str(runtime_raw.get("error") or ""),
            metadata=metadata,
        ),
        profile=profile,
        metadata=dict(payload.get("metadata") or {}),
    )


class CommandAdapter:
    def __init__(self, config: CommandAdapterConfig) -> None:
        self.config = config
        self.spec = ModelSpec(
            model_id=config.model_id,
            family=config.family,
            resource_class=config.resource_class,
            profile_names=config.profile_names,
            complexity_names=config.complexity_names,
        )

    def _format_command(
        self,
        *,
        page: ParseInputPage,
        profile: PageProfile,
        context: AdapterContext,
        output_path: Path,
        profile_path: Path,
    ) -> list[str]:
        image_path = Path(page.image_path).expanduser().resolve()
        replacements = {
            "page_id": page.page_id,
            "image_path": str(image_path),
            "image_dir": str(image_path.parent),
            "image_name": image_path.name,
            "image_stem": image_path.stem,
            "issue_id": page.issue_id,
            "page_number": "" if page.page_number is None else str(page.page_number),
            "manifest_dir": str(context.manifest_path.parent),
            "run_dir": str(context.run_dir),
            "repo_root": str(context.repo_root),
            "output_path": str(output_path),
            "profile_path": str(profile_path),
            "model_id": self.spec.model_id,
            "python": sys.executable,
            "width": str(profile.width),
            "height": str(profile.height),
        }
        return [part.format(**replacements) for part in self.config.command]

    def run(self, page: ParseInputPage, profile: PageProfile, context: AdapterContext) -> ModelOutput:
        started = time.perf_counter()
        output_path = context.run_dir / "work" / "command_adapters" / self.spec.model_id / f"{page.page_id}.json"
        profile_path = context.run_dir / "profiles" / f"{page.page_id}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        command = self._format_command(
            page=page,
            profile=profile,
            context=context,
            output_path=output_path,
            profile_path=profile_path,
        )
        env = os.environ.copy()
        env.update(self.config.env)
        cwd = Path(self.config.cwd).expanduser().resolve() if self.config.cwd else context.repo_root
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            text=True,
            capture_output=True,
            timeout=self.config.timeout_seconds,
            check=False,
        )
        elapsed = time.perf_counter() - started
        if completed.returncode != 0:
            raise RuntimeError(
                f"adapter {self.spec.model_id} failed with exit {completed.returncode}: "
                f"{completed.stderr[-1200:] or completed.stdout[-1200:]}"
            )
        if output_path.exists():
            payload = read_json(output_path)
        else:
            stdout = completed.stdout.strip()
            if not stdout:
                raise RuntimeError(f"adapter {self.spec.model_id} wrote neither {output_path} nor stdout JSON")
            payload = json.loads(stdout)
        if not isinstance(payload, dict):
            raise ValueError(f"adapter {self.spec.model_id} output must be a JSON object")
        return model_output_from_payload(
            payload,
            page_id=page.page_id,
            model_id=self.spec.model_id,
            spec=self.spec,
            profile=profile,
            elapsed_seconds=elapsed,
            runtime_metadata={
                "adapter_kind": "command",
                "command": command,
                "stdout_tail": completed.stdout[-1200:],
                "stderr_tail": completed.stderr[-1200:],
                "output_path": str(output_path),
            },
        )


def adapters_for_profile(profile_name: str, config_path: Path | None = None) -> list[ModelAdapter]:
    config = load_bagging_config(config_path)
    adapters: list[ModelAdapter] = []
    if config.include_builtin_adapters:
        adapters.extend(ADAPTERS[spec.model_id] for spec in MODEL_REGISTRY if profile_name in spec.profile_names)
    adapters.extend(
        CommandAdapter(adapter_config)
        for adapter_config in config.command_adapters
        if profile_name in adapter_config.profile_names
    )
    if not adapters:
        available_profiles = set()
        if config.include_builtin_adapters:
            for spec in MODEL_REGISTRY:
                available_profiles.update(spec.profile_names)
        for adapter_config in config.command_adapters:
            available_profiles.update(adapter_config.profile_names)
        raise ValueError(f"profile {profile_name!r} matched no adapters; available profiles: {sorted(available_profiles)}")
    model_ids = [adapter.spec.model_id for adapter in adapters]
    duplicates = sorted({model_id for model_id in model_ids if model_ids.count(model_id) > 1})
    if duplicates:
        raise ValueError(f"duplicate adapter model_id values: {duplicates}")
    return adapters


def _require_fresh_run_dir(run_dir: Path) -> None:
    if not run_dir.exists():
        return
    if not run_dir.is_dir():
        raise NotADirectoryError(f"run directory path is not a directory: {run_dir}")
    if any(run_dir.iterdir()):
        raise FileExistsError(f"run directory must be empty or absent: {run_dir}")


def _prepare_bagging_plan(
    *,
    manifest_path: Path,
    run_dir: Path,
    profile_name: str,
    config_path: Path | None,
    repo_root: Path,
) -> PreparedBaggingPlan:
    _require_fresh_run_dir(run_dir)
    manifest_validation = validate_parse_input_manifest(
        manifest_path,
        require_files=True,
        verify_checksums=True,
    )
    write_json(run_dir / "reports" / "input_manifest_validation.json", manifest_validation)
    if manifest_validation["status"] == "error":
        report_path = run_dir / "reports" / "input_manifest_validation.json"
        raise ValueError(f"parse input manifest validation failed: {report_path}")

    pages = read_parse_input_manifest(manifest_path)
    if not pages:
        raise ValueError(f"parse input manifest is empty: {manifest_path}")
    adapters = adapters_for_profile(profile_name, config_path=config_path)
    adapters_by_id = {adapter.spec.model_id: adapter for adapter in adapters}
    profiles: dict[str, PageProfile] = {}
    plan_rows: list[PageModelPlan] = []
    profile_performance_rows: list[dict[str, object]] = []

    write_jsonl(run_dir / "manifests" / "parse_input.jsonl", pages)
    for page in pages:
        profile_started = time.perf_counter()
        page_profile = profile_page(page)
        profiles[page.page_id] = page_profile
        write_json(run_dir / "profiles" / f"{page.page_id}.json", page_profile)
        selected = [
            adapter
            for adapter in adapters
            if page_profile.estimated_complexity in adapter.spec.complexity_names
        ]
        if not selected:
            raise ValueError(
                f"page {page.page_id!r} with complexity {page_profile.estimated_complexity!r} "
                f"matched no adapters in profile {profile_name!r}"
            )
        planned_models = [
            PlannedModel(
                model_id=adapter.spec.model_id,
                family=adapter.spec.family,
                resource_class=adapter.spec.resource_class,
            )
            for adapter in selected
        ]
        plan_rows.append(
            PageModelPlan(
                contract="parser-model-plan-v1",
                page_id=page.page_id,
                profile_name=profile_name,
                estimated_complexity=page_profile.estimated_complexity,
                models=planned_models,
                resource_classes=sorted({model.resource_class for model in planned_models}),
                routing_reason=(
                    f"matched profile={profile_name!r} and "
                    f"estimated_complexity={page_profile.estimated_complexity!r}"
                ),
                profile=page_profile,
            )
        )
        profile_performance_rows.append(
            {
                "page_id": page.page_id,
                "stage": "page_profile",
                "model_id": "",
                "family": "profiler",
                "seconds": round(time.perf_counter() - profile_started, 6),
                "status": "ok",
                "resource_class": "cpu",
                "profile_name": profile_name,
                "estimated_complexity": page_profile.estimated_complexity,
            }
        )

    model_counts: dict[str, int] = {}
    resource_counts: dict[str, int] = {}
    complexity_counts: dict[str, int] = {}
    for row in plan_rows:
        complexity_counts[row.estimated_complexity] = complexity_counts.get(row.estimated_complexity, 0) + 1
        for model in row.models:
            model_counts[model.model_id] = model_counts.get(model.model_id, 0) + 1
            resource_counts[model.resource_class] = resource_counts.get(model.resource_class, 0) + 1

    model_ids = [adapter.spec.model_id for adapter in adapters if adapter.spec.model_id in model_counts]
    plan_summary = {
        "contract": "parser-model-plan-summary-v1",
        "profile_name": profile_name,
        "pages_planned": len(plan_rows),
        "adapter_invocations_planned": sum(model_counts.values()),
        "model_ids": model_ids,
        "resource_classes": sorted(resource_counts),
        "complexity_counts": dict(sorted(complexity_counts.items())),
        "resource_class_invocation_counts": dict(sorted(resource_counts.items())),
        "models": {
            model_id: {
                "family": adapters_by_id[model_id].spec.family,
                "resource_class": adapters_by_id[model_id].spec.resource_class,
                "complexities": list(adapters_by_id[model_id].spec.complexity_names),
                "pages_planned": model_counts[model_id],
            }
            for model_id in model_ids
        },
        "paths": {
            "parse_input_manifest": "manifests/parse_input.jsonl",
            "model_plan": "manifests/model_plan.jsonl",
            "profiles": "profiles",
        },
        "provenance": {
            "repo_commit": _git_commit(repo_root),
            "input_manifest": str(manifest_path),
            "bagging_config": str(config_path) if config_path is not None else "",
        },
    }
    write_jsonl(run_dir / "manifests" / "model_plan.jsonl", plan_rows)
    write_json(run_dir / "reports" / "plan_summary.json", plan_summary)
    return PreparedBaggingPlan(
        pages=pages,
        profiles=profiles,
        plan_rows=plan_rows,
        adapters_by_id=adapters_by_id,
        plan_summary=plan_summary,
        performance_rows=profile_performance_rows,
        manifest_validation=manifest_validation,
    )


def plan_bagging(
    *,
    manifest_path: Path,
    run_dir: Path,
    profile_name: str = "adaptive",
    config_path: Path | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    manifest = manifest_path.expanduser().resolve()
    root = run_dir.expanduser().resolve()
    config = config_path.expanduser().resolve() if config_path is not None else None
    prepared = _prepare_bagging_plan(
        manifest_path=manifest,
        run_dir=root,
        profile_name=profile_name,
        config_path=config,
        repo_root=(repo_root or Path.cwd()).expanduser().resolve(),
    )
    return prepared.plan_summary


def fuse_page(page_id: str, outputs: list[ModelOutput]) -> FusedPage:
    regions: list[NormalizedRegion] = []
    model_ids = [output.model_id for output in outputs]
    for output in outputs:
        for region in output.regions:
            if region.label == "text" and region.confidence < 0.50:
                continue
            regions.append(region)

    regions.sort(key=lambda region: (region.reading_order or 9999, region.bbox_xyxy[1], region.bbox_xyxy[0]))
    transcript = "\n".join(region.text for region in regions if region.text)
    disagreement_score = round(max(0, len(regions) - len(outputs)) / max(1, len(regions)), 6)
    return FusedPage(
        page_id=page_id,
        regions=regions,
        transcript=transcript,
        model_ids=model_ids,
        disagreement_score=disagreement_score,
        quality={
            "region_count": len(regions),
            "text_region_count": sum(1 for region in regions if region.label == "text"),
            "legal_notice_candidate_count": sum(1 for region in regions if region.label == "legal_notice_candidate"),
        },
        provenance={"model_ids": model_ids},
    )


def run_bagging_canary(
    *,
    manifest_path: Path,
    run_dir: Path,
    profile_name: str = "adaptive",
    config_path: Path | None = None,
    repo_root: Path | None = None,
) -> RunBundle:
    started = time.perf_counter()
    manifest_path = manifest_path.expanduser().resolve()
    run_dir = run_dir.expanduser().resolve()
    repo_root = (repo_root or Path.cwd()).expanduser().resolve()
    config_path = config_path.expanduser().resolve() if config_path is not None else None
    prepared = _prepare_bagging_plan(
        manifest_path=manifest_path,
        run_dir=run_dir,
        profile_name=profile_name,
        config_path=config_path,
        repo_root=repo_root,
    )
    context = AdapterContext(run_dir=run_dir, manifest_path=manifest_path, repo_root=repo_root)
    errors: list[dict[str, object]] = []
    performance_rows = list(prepared.performance_rows)
    fused_pages: list[FusedPage] = []
    page_by_id = {page.page_id: page for page in prepared.pages}

    for plan_row in prepared.plan_rows:
        page = page_by_id[plan_row.page_id]
        page_profile = prepared.profiles[page.page_id]
        page_started = time.perf_counter()
        outputs: list[ModelOutput] = []
        adapter_error_count = 0
        for planned_model in plan_row.models:
            adapter = prepared.adapters_by_id[planned_model.model_id]
            adapter_started = time.perf_counter()
            try:
                output = adapter.run(page, page_profile, context)
                adapter_elapsed = round(time.perf_counter() - adapter_started, 6)
                runtime_status = output.runtime.status.lower()
                if runtime_status not in {"ok", "skipped"}:
                    raise RuntimeError(
                        output.runtime.error
                        or f"adapter {output.model_id} returned unsupported status={output.runtime.status!r}"
                    )
                write_json(
                    run_dir / "outputs" / "model_outputs" / output.model_id / f"{page.page_id}.json",
                    output,
                )
                outputs.append(output)
                performance_rows.append(
                    {
                        "page_id": page.page_id,
                        "stage": "model_adapter",
                        "model_id": output.model_id,
                        "family": adapter.spec.family,
                        "seconds": adapter_elapsed,
                        "reported_seconds": output.runtime.seconds,
                        "status": output.runtime.status,
                        "resource_class": output.runtime.resource_class,
                        "profile_name": profile_name,
                        "estimated_complexity": page_profile.estimated_complexity,
                    }
                )
            except Exception as exc:  # pragma: no cover - focused failure test exercises this path
                adapter_error_count += 1
                adapter_elapsed = round(time.perf_counter() - adapter_started, 6)
                error_row = {
                    "scope": "adapter",
                    "page_id": page.page_id,
                    "stage": "model_adapter",
                    "model_id": adapter.spec.model_id,
                    "family": adapter.spec.family,
                    "resource_class": adapter.spec.resource_class,
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                errors.append(error_row)
                performance_rows.append(
                    {
                        **error_row,
                        "seconds": adapter_elapsed,
                        "profile_name": profile_name,
                        "estimated_complexity": page_profile.estimated_complexity,
                    }
                )

        page_succeeded = False
        if outputs:
            fusion_started = time.perf_counter()
            try:
                fused = fuse_page(page.page_id, outputs)
                write_json(run_dir / "outputs" / "fused_pages" / f"{page.page_id}.json", fused)
                transcript_path = run_dir / "outputs" / "transcripts" / f"{page.page_id}.txt"
                transcript_path.parent.mkdir(parents=True, exist_ok=True)
                transcript_path.write_text(fused.transcript + "\n", encoding="utf-8")
                review_path = run_dir / "review" / f"{page.page_id}.md"
                review_path.parent.mkdir(parents=True, exist_ok=True)
                review_path.write_text(
                    f"# {page.page_id}\n\n"
                    f"- planned models: {', '.join(model.model_id for model in plan_row.models)}\n"
                    f"- completed models: {', '.join(fused.model_ids)}\n"
                    f"- regions: {len(fused.regions)}\n"
                    f"- disagreement_score: {fused.disagreement_score}\n"
                    f"- adapter_errors: {adapter_error_count}\n",
                    encoding="utf-8",
                )
                fused_pages.append(fused)
                page_succeeded = True
                performance_rows.append(
                    {
                        "page_id": page.page_id,
                        "stage": "fusion",
                        "model_id": "",
                        "family": "fusion",
                        "seconds": round(time.perf_counter() - fusion_started, 6),
                        "status": "ok",
                        "resource_class": "cpu",
                        "profile_name": profile_name,
                        "estimated_complexity": page_profile.estimated_complexity,
                    }
                )
            except Exception as exc:  # pragma: no cover - defensive run-bundle failure handling
                fusion_elapsed = round(time.perf_counter() - fusion_started, 6)
                errors.append(
                    {
                        "scope": "page",
                        "page_id": page.page_id,
                        "stage": "fusion",
                        "model_id": "",
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                performance_rows.append(
                    {
                        "page_id": page.page_id,
                        "stage": "fusion",
                        "model_id": "",
                        "family": "fusion",
                        "seconds": fusion_elapsed,
                        "status": "error",
                        "resource_class": "cpu",
                        "profile_name": profile_name,
                        "estimated_complexity": page_profile.estimated_complexity,
                        "error": str(exc),
                    }
                )
        else:
            errors.append(
                {
                    "scope": "page",
                    "page_id": page.page_id,
                    "stage": "fusion",
                    "model_id": "",
                    "status": "error",
                    "error_type": "NoSuccessfulAdapters",
                    "error": "all planned model adapters failed",
                }
            )

        resource_classes = plan_row.resource_classes
        performance_rows.append(
            {
                "page_id": page.page_id,
                "stage": "page_total",
                "model_id": "",
                "family": "orchestration",
                "seconds": round(time.perf_counter() - page_started, 6),
                "status": "ok" if page_succeeded else "error",
                "resource_class": resource_classes[0] if len(resource_classes) == 1 else "mixed",
                "profile_name": profile_name,
                "estimated_complexity": page_profile.estimated_complexity,
                "adapter_errors": adapter_error_count,
            }
        )

    write_jsonl(run_dir / "reports" / "performance.jsonl", performance_rows)
    write_jsonl(run_dir / "errors.jsonl", errors)
    elapsed = round(time.perf_counter() - started, 6)
    page_error_count = sum(1 for row in errors if row.get("scope") == "page")
    adapter_error_count = sum(1 for row in errors if row.get("scope") == "adapter")
    adapter_rows = [row for row in performance_rows if row.get("stage") == "model_adapter"]
    performance = {
        "seconds_total": elapsed,
        "pages_attempted": len(prepared.pages),
        "pages_completed": len(fused_pages),
        "pages_per_minute": round((len(fused_pages) / elapsed) * 60, 6) if elapsed > 0 else 0,
        "pages_failed": page_error_count,
        "adapter_invocations_planned": prepared.plan_summary["adapter_invocations_planned"],
        "adapter_invocations_observed": len(adapter_rows),
        "adapter_errors": adapter_error_count,
        "errors": len(errors),
    }
    bundle = RunBundle(
        run_id=run_dir.name,
        run_dir=str(run_dir),
        input_manifest=str(manifest_path),
        profile=profile_name,
        page_count=len(prepared.pages),
        model_ids=list(prepared.plan_summary["model_ids"]),
        outputs={
            "parse_input_manifest": "manifests/parse_input.jsonl",
            "model_plan": "manifests/model_plan.jsonl",
            "plan_summary": "reports/plan_summary.json",
            "model_outputs": "outputs/model_outputs",
            "fused_pages": "outputs/fused_pages",
            "transcripts": "outputs/transcripts",
            "review": "review",
            "performance_rows": "reports/performance.jsonl",
            "performance_summary": "reports/performance_summary.json",
        },
        performance=performance,
        provenance={
            "repo_commit": _git_commit(repo_root),
            "contract_version": "parser-bagging-v2",
            "model_plan_contract": "parser-model-plan-v1",
            "performance_contract": "parser-performance-summary-v1",
            "bagging_config": str(config_path) if config_path is not None else "",
            "input_manifest_validation": str(run_dir / "reports" / "input_manifest_validation.json"),
            "input_manifest_validation_status": prepared.manifest_validation["status"],
        },
    )
    write_json(run_dir / "summary.json", bundle)
    write_json(run_dir / "reports" / "performance.json", performance)
    write_json(run_dir / "provenance.json", bundle.provenance)
    write_bagging_performance_summary(run_dir)
    return bundle
