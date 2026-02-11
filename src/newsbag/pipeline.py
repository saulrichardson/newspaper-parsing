from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from newsbag.config import PipelineConfig, config_to_jsonable
from newsbag.fusion import run_fusion
from newsbag.review import build_review_bundle
from newsbag.runners.external import run_dell, run_mineru
from newsbag.runners.paddle import run_paddle_layout_variants, run_paddle_vl15_docparser
from newsbag.utils.io import ensure_dir, write_json, write_lines


def _read_manifest_images(manifest: Path) -> List[Path]:
    rows: List[Path] = []
    for ln in manifest.read_text(encoding="utf-8").splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        rows.append(Path(os.path.expandvars(s)).expanduser())
    return rows


def _validate_manifest_images(images: List[Path]) -> List[Path]:
    missing = [p for p in images if not p.exists()]
    if missing:
        preview = "\n".join(str(x) for x in missing[:20])
        raise FileNotFoundError(
            f"Manifest contains {len(missing)} missing image paths. First 20:\n{preview}"
        )
    return [p.resolve() for p in images]


def _build_run_dir(cfg: PipelineConfig, run_dir_override: Optional[Path]) -> Path:
    if run_dir_override:
        run_dir = run_dir_override.expanduser().resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = (cfg.run_root / f"{cfg.run_name}_{stamp}").expanduser().resolve()

    ensure_dir(run_dir)
    ensure_dir(run_dir / "manifests")
    ensure_dir(run_dir / "logs")
    ensure_dir(run_dir / "reports")
    ensure_dir(run_dir / "outputs")
    ensure_dir(run_dir / "review")
    return run_dir


def _write_run_metadata(cfg: PipelineConfig, images: List[Path], run_dir: Path) -> Path:
    manifest_copy = run_dir / "manifests" / "images.resolved.txt"
    write_lines(manifest_copy, [str(p) for p in images])

    cfg_json = config_to_jsonable(cfg)
    cfg_json["manifest_path"] = str(manifest_copy)
    cfg_json["resolved_at"] = datetime.now().isoformat(timespec="seconds")
    write_json(run_dir / "manifests" / "config.resolved.json", cfg_json)
    return manifest_copy


ALLOWED_STAGES = {
    "paddle_layout",
    "paddle_vl15",
    "dell",
    "mineru",
    "fusion",
    "review",
}


def run_pipeline(
    cfg: PipelineConfig,
    run_dir_override: Optional[Path] = None,
    stages: Optional[set[str]] = None,
) -> Path:
    images = _validate_manifest_images(_read_manifest_images(cfg.manifest_path))
    if not images:
        raise ValueError(f"No image paths found in manifest: {cfg.manifest_path}")

    run_dir = _build_run_dir(cfg, run_dir_override)
    resolved_manifest = _write_run_metadata(cfg, images, run_dir)

    stages_to_run = set(ALLOWED_STAGES) if stages is None else set(stages)
    unknown = sorted(stages_to_run - ALLOWED_STAGES)
    if unknown:
        raise ValueError(f"Unknown stages: {unknown}. Allowed: {sorted(ALLOWED_STAGES)}")

    if "paddle_layout" in stages_to_run:
        run_paddle_layout_variants(cfg, images, run_dir)
    if "paddle_vl15" in stages_to_run and cfg.paddle_vl15.enabled:
        run_paddle_vl15_docparser(cfg, cfg.paddle_vl15, images, run_dir)
    if "dell" in stages_to_run and cfg.dell.enabled:
        run_dell(cfg.dell, resolved_manifest, run_dir, resume=cfg.resume)
    if "mineru" in stages_to_run and cfg.mineru.enabled:
        run_mineru(cfg.mineru, resolved_manifest, run_dir, resume=cfg.resume)

    fusion_root: Optional[Path] = None
    if "fusion" in stages_to_run:
        fusion_root = run_fusion(
            images=images,
            run_dir=run_dir,
            paddle_layout_variant_ids=[x.variant_id for x in cfg.paddle_layout_variants],
            paddle_vl15_variant_id=cfg.paddle_vl15.variant_id,
            dell_variant_id=cfg.dell.variant_id,
            mineru_variant_id=cfg.mineru.variant_id,
            line_cover_threshold=cfg.fusion.line_cover_threshold,
            preferred_recommended_variant=cfg.fusion.recommended_variant,
        )
    elif "review" in stages_to_run:
        candidate = run_dir / "outputs" / "fusion"
        if (candidate / "summary.json").exists():
            fusion_root = candidate
        else:
            raise FileNotFoundError(
                "review stage requested but fusion outputs not found. "
                "Run with '--stages fusion,review' or ensure outputs/fusion/summary.json exists."
            )

    if "review" in stages_to_run and fusion_root is not None:
        build_review_bundle(
            images=images,
            run_dir=run_dir,
            paddle_layout_variant_ids=[x.variant_id for x in cfg.paddle_layout_variants],
            paddle_vl15_variant_id=cfg.paddle_vl15.variant_id,
            dell_variant_id=cfg.dell.variant_id,
            mineru_variant_id=cfg.mineru.variant_id,
            fusion_root=fusion_root,
            mode=cfg.review.mode,
            top_k_informative=cfg.review.top_k_informative,
            top_k_miner_delta=cfg.review.top_k_miner_delta,
        )

    latest_link = cfg.run_root.expanduser().resolve() / "latest"
    if latest_link.exists() or latest_link.is_symlink():
        if latest_link.is_symlink() or latest_link.is_file():
            latest_link.unlink()
    latest_link.symlink_to(run_dir, target_is_directory=True)

    return run_dir
