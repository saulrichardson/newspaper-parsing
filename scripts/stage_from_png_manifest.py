#!/usr/bin/env python3
"""
Stage a subset of layouts/PNGs from a png_manifest.json.

Reads a manifest with shape:
  {
    "entries": [
      {"slug": "<png stem>", "png_path": "...", "layout_path": "...", ...},
      ...
    ]
  }

Writes an output directory containing:
  - slugs.txt: unique slugs from the manifest
  - layouts.txt: existing local layout JSON paths (one per line)
  - missing_layouts.txt: slugs lacking local layouts
  - missing_pngs.txt: slugs lacking local PNGs
  - summary.json: counts and provenance

This is purely a staging step: it does not call any models.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from os.path import expanduser
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage manifest slugs into @file lists for sync/batch.")
    p.add_argument(
        "--png-manifest",
        required=True,
        help="Path to png_manifest.json (must contain entries[].slug)",
    )
    p.add_argument(
        "--layout-root",
        default="newspaper-parsing-local/data/unique_outputs_dedup",
        help="Root directory containing <slug>.json layout files",
    )
    p.add_argument(
        "--png-root",
        default="newspaper-parsing-local/data/unique_png",
        help="Root directory containing PNG files (nested OK)",
    )
    p.add_argument(
        "--out-dir",
        default="newspaper-parsing-local/data/staged_manifest",
        help="Directory to write staging outputs",
    )
    p.add_argument(
        "--require-layouts",
        action="store_true",
        help="Fail if any manifest slug lacks a local layout",
    )
    p.add_argument(
        "--require-pngs",
        action="store_true",
        help="Fail if any manifest slug lacks a local PNG",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    manifest_path = Path(expanduser(args.png_manifest))
    if not manifest_path.is_file():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    data = json.loads(manifest_path.read_text())
    entries = data.get("entries")
    if not isinstance(entries, list):
        raise SystemExit("Manifest must be a dict with an 'entries' list")

    slugs = [e.get("slug") for e in entries if isinstance(e, dict)]
    slugs = [s for s in slugs if isinstance(s, str) and s.strip()]
    if not slugs:
        raise SystemExit("No slugs found in manifest entries")

    counts = Counter(slugs)
    unique_slugs = sorted(counts.keys())

    layout_root = Path(expanduser(args.layout_root))
    png_root = Path(expanduser(args.png_root))
    out_dir = Path(expanduser(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build local PNG stem set once.
    if not png_root.is_dir():
        raise SystemExit(f"PNG root not found: {png_root}")
    local_png_stems = {p.stem for p in png_root.rglob("*.png")}

    layouts: list[str] = []
    missing_layouts: list[str] = []
    missing_pngs: list[str] = []

    for slug in unique_slugs:
        layout_path = layout_root / f"{slug}.json"
        if layout_path.is_file():
            layouts.append(str(layout_path))
        else:
            missing_layouts.append(slug)
        if slug not in local_png_stems:
            missing_pngs.append(slug)

    (out_dir / "slugs.txt").write_text("\n".join(unique_slugs))
    (out_dir / "layouts.txt").write_text("\n".join(layouts))
    (out_dir / "missing_layouts.txt").write_text("\n".join(missing_layouts))
    (out_dir / "missing_pngs.txt").write_text("\n".join(missing_pngs))

    summary = {
        "manifest_path": str(manifest_path),
        "total_slugs": len(slugs),
        "unique_slugs": len(unique_slugs),
        "duplicate_slugs": sum(1 for c in counts.values() if c > 1),
        "layouts_found": len(layouts),
        "layouts_missing": len(missing_layouts),
        "pngs_missing": len(missing_pngs),
        "layout_root": str(layout_root),
        "png_root": str(png_root),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))

    if args.require_layouts and missing_layouts:
        raise SystemExit(f"{len(missing_layouts)} slugs lack layouts; see {out_dir/'missing_layouts.txt'}")
    if args.require_pngs and missing_pngs:
        raise SystemExit(f"{len(missing_pngs)} slugs lack PNGs; see {out_dir/'missing_pngs.txt'}")


if __name__ == "__main__":
    main()

