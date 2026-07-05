from __future__ import annotations

import hashlib
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from PIL import Image, ImageStat

from newsbag.contracts import (
    FusedPage,
    ModelOutput,
    NormalizedRegion,
    PageProfile,
    ParseInputPage,
    RunBundle,
    RuntimeInfo,
    read_parse_input_manifest,
    write_json,
    write_jsonl,
)


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    family: str
    resource_class: str
    profile_names: tuple[str, ...]


class ModelAdapter(Protocol):
    spec: ModelSpec

    def run(self, page: ParseInputPage, profile: PageProfile) -> ModelOutput:
        ...


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

    def run(self, page: ParseInputPage, profile: PageProfile) -> ModelOutput:
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

    def run(self, page: ParseInputPage, profile: PageProfile) -> ModelOutput:
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

    def run(self, page: ParseInputPage, profile: PageProfile) -> ModelOutput:
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


def adapters_for_profile(profile_name: str) -> list[ModelAdapter]:
    if profile_name not in {"baseline", "adaptive", "full"}:
        raise ValueError("profile must be one of: baseline, adaptive, full")
    return [ADAPTERS[spec.model_id] for spec in MODEL_REGISTRY if profile_name in spec.profile_names]


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
    repo_root: Path | None = None,
) -> RunBundle:
    started = time.perf_counter()
    manifest_path = manifest_path.expanduser().resolve()
    run_dir = run_dir.expanduser().resolve()
    repo_root = repo_root or Path.cwd().resolve()
    pages = read_parse_input_manifest(manifest_path)
    if not pages:
        raise ValueError(f"parse input manifest is empty: {manifest_path}")
    adapters = adapters_for_profile(profile_name)
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
                output = adapter.run(page, page_profile)
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
        },
    )
    write_json(run_dir / "summary.json", bundle)
    write_json(run_dir / "reports" / "performance.json", performance)
    write_json(run_dir / "provenance.json", bundle.provenance)
    return bundle
