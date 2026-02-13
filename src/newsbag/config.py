from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class PaddleLayoutVariant:
    variant_id: str
    model_name: str
    threshold: float = 0.30
    timeout_sec: int = 2700


@dataclass
class PaddleVL15Config:
    enabled: bool = True
    variant_id: str = "pvl15_docparser_v15"
    timeout_sec: int = 3600
    layout_model: str = "PP-DocLayoutV3"
    layout_threshold: float = 0.30


@dataclass
class DellConfig:
    enabled: bool = True
    variant_id: str = "dell_c0005_i010"
    python_bin: str = "python3"
    script_path: str = "src/newsbag/runners/run_dell_layout_only.py"
    model_path: str = ""
    label_map_path: str = ""
    repo_src: str = ""
    provider: str = "auto"
    conf: float = 0.005
    iou: float = 0.10
    imgsz: int = 1280
    require_cuda_provider: bool = False
    min_nonempty_pages: int = 1


@dataclass
class MinerConfig:
    enabled: bool = True
    variant_id: str = "mineru25"
    python_bin: str = "python3"
    script_path: str = "src/newsbag/runners/run_mineru25_layout.py"
    model_id: str = "opendatalab/MinerU2.5-2509-1.2B"
    max_pages: int = 0
    require_cuda: bool = False
    min_nonempty_pages: int = 1


@dataclass
class FusionConfig:
    line_cover_threshold: float = 0.50
    recommended_variant: str = "P4_paddle_union4_plus_dell_plus_mineru"


@dataclass
class ReviewConfig:
    # "all": render per-page PNG boards for every page in the manifest (can be slow on large runs).
    # "top20": render only the pages needed for the Top-20 packs (fast; recommended on Torch).
    mode: str = "all"
    top_k_informative: int = 20
    top_k_miner_delta: int = 20


@dataclass
class TranscriptionConfig:
    enabled: bool = True
    # If empty, use outputs/fusion/summary.json -> recommended_variant.
    variant: str = ""
    labels: List[str] = field(default_factory=lambda: ["text", "title"])
    min_overlap: float = 0.30
    # If empty, pipeline uses top device from device_order (or cpu fallback).
    device: str = ""
    cpu_threads: int = 8
    timeout_sec: int = 3600
    max_pages: int = 0
    resume: bool = True


@dataclass
class PipelineConfig:
    manifest_path: Path
    run_root: Path
    run_name: str = "newsbag_run"
    paddleocr_bin: str = "paddleocr"
    device_order: List[str] = field(default_factory=lambda: ["gpu:0", "cpu"])
    cpu_threads: int = 8
    resume: bool = True
    allow_gpu_to_cpu_fallback: bool = True

    paddle_layout_variants: List[PaddleLayoutVariant] = field(default_factory=list)
    paddle_vl15: PaddleVL15Config = field(default_factory=PaddleVL15Config)
    dell: DellConfig = field(default_factory=DellConfig)
    mineru: MinerConfig = field(default_factory=MinerConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    review: ReviewConfig = field(default_factory=ReviewConfig)
    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)


DEFAULT_PADDLE_LAYOUT_VARIANTS = [
    PaddleLayoutVariant("pld_v2_thr03", "PP-DocLayoutV2", 0.30),
    PaddleLayoutVariant("pld_v3_thr03", "PP-DocLayoutV3", 0.30),
    PaddleLayoutVariant("pld_plusL_thr03", "PP-DocLayout_plus-L", 0.30),
]


def _require_path(raw: str, field_name: str) -> Path:
    p = Path(os.path.expandvars(raw)).expanduser()
    if not str(raw).strip():
        raise ValueError(f"Missing required field: {field_name}")
    return p


def _resolve_optional_path(raw: str, base_dir: Path) -> str:
    if not raw:
        return ""
    expanded = Path(os.path.expandvars(raw)).expanduser()
    if expanded.is_absolute():
        return str(expanded)

    # Many Torch flows generate a run-scoped config under a run directory (e.g. RUN_DIR/manifests).
    # In those cases, paths like "src/newsbag/..." should resolve relative to the repo checkout,
    # not relative to the run directory. We therefore also consider CWD as a likely repo root.
    cwd = Path.cwd().resolve()
    c0 = (cwd / expanded).resolve()
    c1 = (base_dir / expanded).resolve()
    c2 = (base_dir.parent / expanded).resolve()
    path_txt = str(expanded).replace("\\", "/")
    prefer_repo_root = path_txt.startswith(("src/", "scripts/", "torch/", "docs/", "configs/"))
    if prefer_repo_root:
        # Prefer repo-root-ish resolution: CWD first (common in sbatch scripts that `cd` into repo),
        # then base_dir.parent (when config lives in configs/), then base_dir.
        for cand in (c0, c2, c1):
            if cand.exists():
                return str(cand)

        # Don't fabricate a non-existent absolute path (this can later break subprocess calls).
        return str(expanded)

    # For non-repo-root paths, prefer base_dir, then base_dir.parent, then CWD.
    for cand in (c1, c2, c0):
        if cand.exists():
            return str(cand)
    return str(expanded)


def load_config(path: Path) -> PipelineConfig:
    cfg_path = Path(path).resolve()
    payload: Dict[str, Any] = json.loads(cfg_path.read_text(encoding="utf-8"))
    base_dir = cfg_path.parent

    manifest = _require_path(str(payload.get("manifest_path", "")), "manifest_path")
    run_root = _require_path(str(payload.get("run_root", "")), "run_root")
    if not manifest.is_absolute():
        manifest = (base_dir / manifest).resolve()
    if not run_root.is_absolute():
        run_root = (base_dir / run_root).resolve()

    layout_raw = payload.get("paddle_layout_variants") or []
    layout_variants: List[PaddleLayoutVariant] = []
    for row in layout_raw:
        layout_variants.append(
            PaddleLayoutVariant(
                variant_id=row["variant_id"],
                model_name=row["model_name"],
                threshold=float(row.get("threshold", 0.30)),
                timeout_sec=int(row.get("timeout_sec", 2700)),
            )
        )
    if not layout_variants:
        layout_variants = DEFAULT_PADDLE_LAYOUT_VARIANTS

    pvl = payload.get("paddle_vl15", {})
    paddle_vl15 = PaddleVL15Config(
        enabled=bool(pvl.get("enabled", True)),
        variant_id=str(pvl.get("variant_id", "pvl15_docparser_v15")),
        timeout_sec=int(pvl.get("timeout_sec", 3600)),
        layout_model=str(pvl.get("layout_model", "PP-DocLayoutV3")),
        layout_threshold=float(pvl.get("layout_threshold", 0.30)),
    )

    dl = payload.get("dell", {})
    dell = DellConfig(
        enabled=bool(dl.get("enabled", True)),
        variant_id=str(dl.get("variant_id", "dell_c0005_i010")),
        python_bin=str(os.path.expandvars(str(dl.get("python_bin", "python3")))),
        script_path=_resolve_optional_path(
            str(dl.get("script_path", "src/newsbag/runners/run_dell_layout_only.py")),
            base_dir,
        ),
        model_path=_resolve_optional_path(str(dl.get("model_path", "")), base_dir),
        label_map_path=_resolve_optional_path(str(dl.get("label_map_path", "")), base_dir),
        repo_src=_resolve_optional_path(str(dl.get("repo_src", "")), base_dir),
        provider=str(dl.get("provider", "auto")),
        conf=float(dl.get("conf", 0.005)),
        iou=float(dl.get("iou", 0.10)),
        imgsz=int(dl.get("imgsz", 1280)),
        require_cuda_provider=bool(dl.get("require_cuda_provider", False)),
        min_nonempty_pages=int(dl.get("min_nonempty_pages", 1)),
    )

    mn = payload.get("mineru", {})
    mineru = MinerConfig(
        enabled=bool(mn.get("enabled", True)),
        variant_id=str(mn.get("variant_id", "mineru25")),
        python_bin=str(os.path.expandvars(str(mn.get("python_bin", "python3")))),
        script_path=_resolve_optional_path(
            str(mn.get("script_path", "src/newsbag/runners/run_mineru25_layout.py")),
            base_dir,
        ),
        model_id=str(mn.get("model_id", "opendatalab/MinerU2.5-2509-1.2B")),
        max_pages=int(mn.get("max_pages", 0)),
        require_cuda=bool(mn.get("require_cuda", False)),
        min_nonempty_pages=int(mn.get("min_nonempty_pages", 1)),
    )

    fs = payload.get("fusion", {})
    fusion = FusionConfig(
        line_cover_threshold=float(fs.get("line_cover_threshold", 0.50)),
        recommended_variant=str(fs.get("recommended_variant", "P4_paddle_union4_plus_dell_plus_mineru")),
    )

    rv = payload.get("review", {})
    review = ReviewConfig(
        mode=str(rv.get("mode", "all")),
        top_k_informative=int(rv.get("top_k_informative", 20)),
        top_k_miner_delta=int(rv.get("top_k_miner_delta", 20)),
    )

    tr = payload.get("transcription", {})
    labels_raw = tr.get("labels", ["text", "title"])
    labels: List[str]
    if isinstance(labels_raw, list):
        labels = [str(x).strip().lower() for x in labels_raw if str(x).strip()]
    else:
        labels = [
            s.strip().lower()
            for s in str(labels_raw).split(",")
            if s.strip()
        ]
    if not labels:
        labels = ["text", "title"]
    transcription = TranscriptionConfig(
        enabled=bool(tr.get("enabled", True)),
        variant=str(tr.get("variant", "")),
        labels=labels,
        min_overlap=float(tr.get("min_overlap", 0.30)),
        device=str(tr.get("device", "")),
        cpu_threads=int(tr.get("cpu_threads", payload.get("cpu_threads", 8))),
        timeout_sec=int(tr.get("timeout_sec", 3600)),
        max_pages=int(tr.get("max_pages", 0)),
        resume=bool(tr.get("resume", True)),
    )

    return PipelineConfig(
        manifest_path=manifest,
        run_root=run_root,
        run_name=str(payload.get("run_name", "newsbag_run")),
        paddleocr_bin=str(os.path.expandvars(str(payload.get("paddleocr_bin", "paddleocr")))),
        device_order=[str(x) for x in payload.get("device_order", ["gpu:0", "cpu"])],
        cpu_threads=int(payload.get("cpu_threads", 8)),
        resume=bool(payload.get("resume", True)),
        allow_gpu_to_cpu_fallback=bool(payload.get("allow_gpu_to_cpu_fallback", True)),
        paddle_layout_variants=layout_variants,
        paddle_vl15=paddle_vl15,
        dell=dell,
        mineru=mineru,
        fusion=fusion,
        review=review,
        transcription=transcription,
    )


def config_to_jsonable(cfg: PipelineConfig) -> Dict[str, Any]:
    return {
        "manifest_path": str(cfg.manifest_path),
        "run_root": str(cfg.run_root),
        "run_name": cfg.run_name,
        "paddleocr_bin": cfg.paddleocr_bin,
        "device_order": cfg.device_order,
        "cpu_threads": cfg.cpu_threads,
        "resume": cfg.resume,
        "allow_gpu_to_cpu_fallback": cfg.allow_gpu_to_cpu_fallback,
        "paddle_layout_variants": [
            {
                "variant_id": v.variant_id,
                "model_name": v.model_name,
                "threshold": v.threshold,
                "timeout_sec": v.timeout_sec,
            }
            for v in cfg.paddle_layout_variants
        ],
        "paddle_vl15": {
            "enabled": cfg.paddle_vl15.enabled,
            "variant_id": cfg.paddle_vl15.variant_id,
            "timeout_sec": cfg.paddle_vl15.timeout_sec,
            "layout_model": cfg.paddle_vl15.layout_model,
            "layout_threshold": cfg.paddle_vl15.layout_threshold,
        },
        "dell": {
            "enabled": cfg.dell.enabled,
            "variant_id": cfg.dell.variant_id,
            "python_bin": cfg.dell.python_bin,
            "script_path": cfg.dell.script_path,
            "model_path": cfg.dell.model_path,
            "label_map_path": cfg.dell.label_map_path,
            "repo_src": cfg.dell.repo_src,
            "provider": cfg.dell.provider,
            "conf": cfg.dell.conf,
            "iou": cfg.dell.iou,
            "imgsz": cfg.dell.imgsz,
            "require_cuda_provider": cfg.dell.require_cuda_provider,
            "min_nonempty_pages": cfg.dell.min_nonempty_pages,
        },
        "mineru": {
            "enabled": cfg.mineru.enabled,
            "variant_id": cfg.mineru.variant_id,
            "python_bin": cfg.mineru.python_bin,
            "script_path": cfg.mineru.script_path,
            "model_id": cfg.mineru.model_id,
            "max_pages": cfg.mineru.max_pages,
            "require_cuda": cfg.mineru.require_cuda,
            "min_nonempty_pages": cfg.mineru.min_nonempty_pages,
        },
        "fusion": {
            "line_cover_threshold": cfg.fusion.line_cover_threshold,
            "recommended_variant": cfg.fusion.recommended_variant,
        },
        "review": {
            "mode": cfg.review.mode,
            "top_k_informative": cfg.review.top_k_informative,
            "top_k_miner_delta": cfg.review.top_k_miner_delta,
        },
        "transcription": {
            "enabled": cfg.transcription.enabled,
            "variant": cfg.transcription.variant,
            "labels": list(cfg.transcription.labels),
            "min_overlap": cfg.transcription.min_overlap,
            "device": cfg.transcription.device,
            "cpu_threads": cfg.transcription.cpu_threads,
            "timeout_sec": cfg.transcription.timeout_sec,
            "max_pages": cfg.transcription.max_pages,
            "resume": cfg.transcription.resume,
        },
    }
