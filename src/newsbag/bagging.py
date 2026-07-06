from __future__ import annotations

import hashlib
import json
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
    PageProfile,
    ParseInputPage,
    RunBundle,
    RuntimeInfo,
    read_json,
    read_parse_input_manifest,
    write_json,
    write_jsonl,
)
from newsbag.validation import validate_parse_input_manifest


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    family: str
    resource_class: str
    profile_names: tuple[str, ...]


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
    timeout_seconds: float = 600.0
    env: dict[str, str] = field(default_factory=dict)
    cwd: str = ""


@dataclass(frozen=True)
class BaggingConfig:
    include_builtin_adapters: bool = True
    command_adapters: tuple[CommandAdapterConfig, ...] = ()


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
        gray = image.convert("L")
        width, height = gray.size
        stat = ImageStat.Stat(gray)
        mean_luma = float(stat.mean[0])
        histogram = gray.histogram()
        dark_pixels = sum(histogram[:96])
        total_pixels = max(1, width * height)
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
        metadata={"image_sha256": _sha256(image_path)},
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
        command = tuple(str(part) for part in (raw.get("command") or []))
        if not model_id:
            raise ValueError(f"command_adapters[{index}] is missing model_id")
        if not command:
            raise ValueError(f"command_adapters[{index}] is missing command")
        profile_names = tuple(str(item).strip() for item in raw.get("profiles", raw.get("profile_names", [])) if str(item).strip())
        if not profile_names:
            raise ValueError(f"command_adapters[{index}] must define at least one profile")
        command_adapters.append(
            CommandAdapterConfig(
                model_id=model_id,
                family=str(raw.get("family") or "external").strip() or "external",
                resource_class=str(raw.get("resource_class") or "cpu").strip() or "cpu",
                profile_names=profile_names,
                command=command,
                timeout_seconds=float(raw.get("timeout_seconds") or 600.0),
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
    repo_root = repo_root or Path.cwd().resolve()
    manifest_validation = validate_parse_input_manifest(manifest_path, require_files=True)
    write_json(run_dir / "reports" / "input_manifest_validation.json", manifest_validation)
    if manifest_validation["status"] == "error":
        report_path = run_dir / "reports" / "input_manifest_validation.json"
        raise ValueError(f"parse input manifest validation failed: {report_path}")
    pages = read_parse_input_manifest(manifest_path)
    if not pages:
        raise ValueError(f"parse input manifest is empty: {manifest_path}")
    adapters = adapters_for_profile(profile_name, config_path=config_path)
    context = AdapterContext(run_dir=run_dir, manifest_path=manifest_path, repo_root=repo_root)
    errors: list[dict[str, object]] = []
    performance_rows: list[dict[str, object]] = []
    fused_pages: list[FusedPage] = []

    write_jsonl(run_dir / "manifests" / "parse_input.jsonl", pages)

    for page in pages:
        page_started = time.perf_counter()
        try:
            page_profile = profile_page(page)
            write_json(run_dir / "profiles" / f"{page.page_id}.json", page_profile)
            outputs: list[ModelOutput] = []
            for adapter in adapters:
                output = adapter.run(page, page_profile, context)
                outputs.append(output)
                write_json(
                    run_dir / "outputs" / "model_outputs" / output.model_id / f"{page.page_id}.json",
                    output,
                )
                performance_rows.append(
                    {
                        "page_id": page.page_id,
                        "stage": "model_adapter",
                        "model_id": output.model_id,
                        "seconds": output.runtime.seconds,
                        "status": output.runtime.status,
                        "resource_class": output.runtime.resource_class,
                    }
                )
            fused = fuse_page(page.page_id, outputs)
            fused_pages.append(fused)
            write_json(run_dir / "outputs" / "fused_pages" / f"{page.page_id}.json", fused)
            transcript_path = run_dir / "outputs" / "transcripts" / f"{page.page_id}.txt"
            transcript_path.parent.mkdir(parents=True, exist_ok=True)
            transcript_path.write_text(fused.transcript + "\n", encoding="utf-8")
            review_path = run_dir / "review" / f"{page.page_id}.md"
            review_path.parent.mkdir(parents=True, exist_ok=True)
            review_path.write_text(
                f"# {page.page_id}\n\n"
                f"- models: {', '.join(fused.model_ids)}\n"
                f"- regions: {len(fused.regions)}\n"
                f"- disagreement_score: {fused.disagreement_score}\n",
                encoding="utf-8",
            )
            performance_rows.append(
                {
                    "page_id": page.page_id,
                    "stage": "page_total",
                    "model_id": "",
                    "seconds": round(time.perf_counter() - page_started, 6),
                    "status": "ok",
                    "resource_class": "cpu",
                }
            )
        except Exception as exc:  # pragma: no cover - exercised by canary failure handling
            errors.append({"page_id": page.page_id, "status": "error", "error": str(exc)})

    write_jsonl(run_dir / "reports" / "performance.jsonl", performance_rows)
    write_jsonl(run_dir / "errors.jsonl", errors)
    elapsed = round(time.perf_counter() - started, 6)
    performance = {
        "seconds_total": elapsed,
        "pages_attempted": len(pages),
        "pages_completed": len(fused_pages),
        "pages_per_minute": round((len(fused_pages) / elapsed) * 60, 6) if elapsed > 0 else 0,
        "errors": len(errors),
    }
    bundle = RunBundle(
        run_id=run_dir.name,
        run_dir=str(run_dir),
        input_manifest=str(manifest_path),
        profile=profile_name,
        page_count=len(pages),
        model_ids=[adapter.spec.model_id for adapter in adapters],
        outputs={
            "model_outputs": str(run_dir / "outputs" / "model_outputs"),
            "fused_pages": str(run_dir / "outputs" / "fused_pages"),
            "transcripts": str(run_dir / "outputs" / "transcripts"),
            "review": str(run_dir / "review"),
        },
        performance=performance,
        provenance={
            "repo_commit": _git_commit(repo_root),
            "contract_version": "parser-bagging-v1",
            "bagging_config": str(config_path.expanduser().resolve()) if config_path is not None else "",
            "input_manifest_validation": str(run_dir / "reports" / "input_manifest_validation.json"),
            "input_manifest_validation_status": manifest_validation["status"],
        },
    )
    write_json(run_dir / "summary.json", bundle)
    write_json(run_dir / "reports" / "performance.json", performance)
    write_json(run_dir / "provenance.json", bundle.provenance)
    return bundle
