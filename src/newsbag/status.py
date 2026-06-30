from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class StageCount:
    done: int
    total: int
    missing: Tuple[str, ...] = ()
    note: str = ""

    @property
    def pct(self) -> float:
        if self.total <= 0:
            return 0.0
        return 100.0 * (self.done / self.total)


def _read_lines(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _manifest_slugs(manifest_path: Path) -> List[str]:
    slugs: List[str] = []
    for ln in _read_lines(manifest_path):
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        p = Path(os.path.expandvars(s)).expanduser()
        slugs.append(p.stem)
    return slugs


def _find_manifest(run_dir: Path) -> Path:
    cands = [
        run_dir / "manifests" / "images.resolved.txt",
        run_dir / "manifests" / "images.input.txt",
        run_dir / "manifests" / "images.txt",
    ]
    for c in cands:
        if c.exists():
            return c
    raise FileNotFoundError(
        "Could not find a manifest under run_dir/manifests/. "
        "Expected one of: images.resolved.txt, images.input.txt, images.txt"
    )


def _load_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _find_run_config(run_dir: Path) -> Tuple[Path, Dict[str, Any]]:
    cands = [
        run_dir / "manifests" / "config.resolved.json",
        run_dir / "manifests" / "config.input.json",
    ]
    for c in cands:
        if c.exists():
            return c, _load_json_if_exists(c)
    return Path(""), {}


def _count_expected_paths(
    slugs: Iterable[str],
    mk_path,
    missing_limit: int = 0,
) -> StageCount:
    done = 0
    missing: List[str] = []
    total = 0
    for slug in slugs:
        total += 1
        try:
            p = mk_path(slug)
        except Exception:
            p = None
        if p is not None and Path(p).exists():
            done += 1
        else:
            if missing_limit > 0 and len(missing) < missing_limit:
                missing.append(str(slug))
    return StageCount(done=done, total=total, missing=tuple(missing))


def _count_indexed_slug_outputs(
    slugs: Iterable[str],
    root: Path,
    required_relative_path: Path | str,
    missing_limit: int = 0,
    note: str = "indexed output directory",
) -> StageCount:
    expected = tuple(str(slug) for slug in slugs)
    expected_set = set(expected)
    required_relative_path = Path(required_relative_path)
    completed: set[str] = set()

    if root.exists():
        for child in root.iterdir():
            if child.name not in expected_set or not child.is_dir():
                continue
            if (child / required_relative_path).exists():
                completed.add(child.name)

    missing: List[str] = []
    if missing_limit > 0:
        for slug in expected:
            if slug not in completed:
                missing.append(slug)
                if len(missing) >= missing_limit:
                    break

    return StageCount(
        done=len(completed),
        total=len(expected),
        missing=tuple(missing),
        note=note,
    )


def _infer_paddle_layout_variant_ids(cfg: Dict[str, Any], run_dir: Path) -> List[str]:
    rows = cfg.get("paddle_layout_variants") or []
    vids = [str(r.get("variant_id") or "").strip() for r in rows if isinstance(r, dict)]
    vids = [v for v in vids if v]
    if vids:
        return vids
    # Fallback: inspect output directory if config is missing.
    root = run_dir / "outputs" / "sources" / "paddle_layout"
    if not root.exists():
        return []
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


def summarize_run(
    run_dir: Path,
    missing_limit: int = 0,
) -> Dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    manifest_path = _find_manifest(run_dir)
    slugs = _manifest_slugs(manifest_path)
    total_pages = len(slugs)
    cfg_path, cfg = _find_run_config(run_dir)

    paddle_layout_variant_ids = _infer_paddle_layout_variant_ids(cfg, run_dir)
    pvl_cfg = cfg.get("paddle_vl15") or {}
    pvl_enabled = bool(pvl_cfg.get("enabled", True))
    pvl_variant = str(pvl_cfg.get("variant_id") or "").strip()

    dell_cfg = cfg.get("dell") or {}
    dell_enabled = bool(dell_cfg.get("enabled", True))
    dell_variant = str(dell_cfg.get("variant_id") or "").strip()

    miner_cfg = cfg.get("mineru") or {}
    miner_enabled = bool(miner_cfg.get("enabled", True))
    miner_variant = str(miner_cfg.get("variant_id") or "").strip()

    fusion_cfg = cfg.get("fusion") or {}
    fusion_variant = str(fusion_cfg.get("recommended_variant") or "").strip()

    review_cfg = cfg.get("review") or {}
    review_mode = str(review_cfg.get("mode") or "").strip() or "all"

    tr_cfg = cfg.get("transcription") or {}
    tr_enabled = bool(tr_cfg.get("enabled", True))
    tr_variant = str(tr_cfg.get("variant") or "").strip()

    fusion_summary_path = run_dir / "outputs" / "fusion" / "summary.json"
    fusion_summary = _load_json_if_exists(fusion_summary_path) if fusion_summary_path.exists() else {}
    fusion_rec = str(fusion_summary.get("recommended_variant") or "").strip()
    resolved_fusion_variant = fusion_variant or fusion_rec
    resolved_tr_variant = tr_variant or resolved_fusion_variant

    stages: Dict[str, Any] = {}

    # Paddle layout (3 variants typically).
    paddle_counts: Dict[str, Any] = {}
    for vid in paddle_layout_variant_ids:
        paddle_counts[vid] = _count_indexed_slug_outputs(
            slugs,
            run_dir / "outputs" / "sources" / "paddle_layout" / vid,
            "layout_boxes.normalized.json",
            missing_limit=missing_limit,
        )
    stages["paddle_layout"] = paddle_counts

    # Paddle VL1.5 doc_parser.
    if not pvl_enabled:
        stages["paddle_vl15"] = {"disabled": True}
    else:
        if not pvl_variant:
            stages["paddle_vl15"] = {"error": "enabled but variant_id missing"}
        else:
            stages["paddle_vl15"] = _count_indexed_slug_outputs(
                slugs,
                run_dir / "outputs" / "sources" / "paddle_vl15" / pvl_variant,
                "parsing_blocks.json",
                missing_limit=missing_limit,
            )

    # Dell + MinerU.
    if not dell_enabled:
        stages["dell"] = {"disabled": True}
    else:
        if not dell_variant:
            stages["dell"] = {"error": "enabled but variant_id missing"}
        else:
            stages["dell"] = _count_indexed_slug_outputs(
                slugs,
                run_dir / "outputs" / "sources" / "dell" / dell_variant,
                "layout_boxes.normalized.json",
                missing_limit=missing_limit,
            )

    if not miner_enabled:
        stages["mineru"] = {"disabled": True}
    else:
        if not miner_variant:
            stages["mineru"] = {"error": "enabled but variant_id missing"}
        else:
            stages["mineru"] = _count_indexed_slug_outputs(
                slugs,
                run_dir / "outputs" / "sources" / "mineru" / miner_variant,
                "layout_boxes.normalized.json",
                missing_limit=missing_limit,
            )

    # Fusion.
    if fusion_summary_path.exists():
        fused_pages = set((fusion_summary.get("pages") or {}).keys())
        missing = []
        if missing_limit > 0:
            missing = [s for s in slugs if s not in fused_pages][:missing_limit]
        stages["fusion"] = StageCount(done=len(fused_pages), total=total_pages, missing=tuple(missing), note="summary.json")
    else:
        if resolved_fusion_variant:
            stages["fusion"] = _count_indexed_slug_outputs(
                slugs,
                run_dir / "outputs" / "fusion" / resolved_fusion_variant,
                "fused_boxes.json",
                missing_limit=missing_limit,
            )
        else:
            stages["fusion"] = {"pending": True, "note": "summary.json not found and no recommended_variant resolved"}

    # Review.
    review_inform = run_dir / "review" / "top20_informative" / "ranking_top20.tsv"
    review_miner = run_dir / "review" / "top20_miner_delta" / "ranking_top20.tsv"
    board_count = 0
    pages_root = run_dir / "review" / "pages"
    if pages_root.exists():
        board_count = len(list(pages_root.glob("*/06_board.png")))
    stages["review"] = {
        "mode": review_mode,
        "has_top20_informative": review_inform.exists(),
        "has_top20_miner_delta": review_miner.exists(),
        "rendered_boards": board_count,
        "expected_boards_in_all_mode": total_pages if review_mode == "all" else None,
    }

    # Transcription.
    if not tr_enabled:
        stages["transcription"] = {"disabled": True}
    else:
        if not resolved_tr_variant:
            stages["transcription"] = {"pending": True, "note": "enabled but transcription variant could not be resolved"}
        else:
            out_root = run_dir / "outputs" / "transcription" / resolved_tr_variant
            report = out_root / "transcription_report.tsv"
            if report.exists():
                # Prefer report-driven status because it encodes missing_image/missing_fused/no_ocr_boxes too.
                import csv  # local import keeps module lightweight

                ok = 0
                seen = 0
                missing: List[str] = []
                with report.open("r", encoding="utf-8", newline="") as f:
                    rr = csv.DictReader(f, delimiter="\t")
                    for row in rr:
                        seen += 1
                        st = str(row.get("status") or "").strip().lower()
                        slug = str(row.get("slug") or "").strip()
                        if st in ("ok", "resume"):
                            ok += 1
                        elif missing_limit > 0 and slug and len(missing) < missing_limit:
                            missing.append(slug)
                stages["transcription"] = {
                    "variant": resolved_tr_variant,
                    "report": str(report),
                    "count": StageCount(done=ok, total=max(total_pages, seen), missing=tuple(missing), note="transcription_report.tsv"),
                }
            else:
                stages["transcription"] = {
                    "variant": resolved_tr_variant,
                    "count": _count_indexed_slug_outputs(
                        slugs,
                        out_root,
                        "transcript.txt",
                        missing_limit=missing_limit,
                    ),
                }

    def _to_jsonable(obj: Any) -> Any:
        if isinstance(obj, StageCount):
            return {
                "done": obj.done,
                "total": obj.total,
                "pct": round(obj.pct, 2),
                "missing": list(obj.missing),
                "note": obj.note,
            }
        if isinstance(obj, dict):
            return {k: _to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_jsonable(v) for v in obj]
        return obj

    out: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "manifest": str(manifest_path),
        "config": str(cfg_path) if str(cfg_path) else "",
        "total_pages": total_pages,
        "resolved_fusion_variant": resolved_fusion_variant,
        "resolved_transcription_variant": resolved_tr_variant,
        "stages": _to_jsonable(stages),
    }
    return out


def format_summary_text(summary: Dict[str, Any]) -> str:
    def fmt_count(c: Dict[str, Any]) -> str:
        done = int(c.get("done", 0))
        total = int(c.get("total", 0))
        pct = float(c.get("pct", 0.0))
        return f"{done}/{total} ({pct:.1f}%)"

    lines: List[str] = []
    lines.append(f"run_dir: {summary.get('run_dir','')}")
    lines.append(f"manifest: {summary.get('manifest','')} (pages={summary.get('total_pages', 0)})")
    if summary.get("config"):
        lines.append(f"config: {summary.get('config')}")
    if summary.get("resolved_fusion_variant"):
        lines.append(f"fusion.recommended_variant: {summary.get('resolved_fusion_variant')}")
    if summary.get("resolved_transcription_variant"):
        lines.append(f"transcription.variant: {summary.get('resolved_transcription_variant')}")

    stages = summary.get("stages") or {}

    pl = stages.get("paddle_layout") or {}
    if isinstance(pl, dict) and pl:
        lines.append("")
        lines.append("paddle_layout:")
        for vid in sorted(pl.keys()):
            c = pl[vid]
            if isinstance(c, dict) and "done" in c:
                lines.append(f"  - {vid}: {fmt_count(c)}")
            else:
                lines.append(f"  - {vid}: (no data)")

    pv = stages.get("paddle_vl15")
    if pv is not None:
        lines.append("")
        if isinstance(pv, dict) and pv.get("disabled"):
            lines.append("paddle_vl15: disabled")
        elif isinstance(pv, dict) and "done" in pv:
            lines.append(f"paddle_vl15: {fmt_count(pv)}")
        else:
            lines.append(f"paddle_vl15: {pv}")

    for key in ("dell", "mineru", "fusion"):
        st = stages.get(key)
        if st is None:
            continue
        lines.append("")
        if isinstance(st, dict) and st.get("disabled"):
            lines.append(f"{key}: disabled")
        elif isinstance(st, dict) and "done" in st:
            lines.append(f"{key}: {fmt_count(st)}")
        else:
            lines.append(f"{key}: {st}")

    rv = stages.get("review") or {}
    lines.append("")
    if isinstance(rv, dict):
        lines.append(
            "review: "
            + f"mode={rv.get('mode')} "
            + f"top20_informative={bool(rv.get('has_top20_informative'))} "
            + f"top20_miner_delta={bool(rv.get('has_top20_miner_delta'))} "
            + f"rendered_boards={rv.get('rendered_boards')}"
        )
    else:
        lines.append(f"review: {rv}")

    tr = stages.get("transcription")
    lines.append("")
    if isinstance(tr, dict) and tr.get("disabled"):
        lines.append("transcription: disabled")
    elif isinstance(tr, dict) and isinstance(tr.get("count"), dict) and "done" in tr["count"]:
        lines.append(f"transcription: {fmt_count(tr['count'])} (variant={tr.get('variant')})")
    elif isinstance(tr, dict) and "done" in tr:
        lines.append(f"transcription: {fmt_count(tr)}")
    else:
        lines.append(f"transcription: {tr}")

    return "\n".join(lines) + "\n"
