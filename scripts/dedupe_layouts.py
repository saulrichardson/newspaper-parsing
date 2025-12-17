#!/usr/bin/env python3
"""
Deduplicate layout JSON outputs against PNG images using strong content hashes.

Logic:
1) Hash every PNG in `newspaper-parsing-local/data/unique_png` (SHA256).
2) Map each JSON output to an image_id (tail token after final "_") and then to
   the PNG hash. JSONs without a matching PNG become orphans.
3) For each hash group, pick the best JSON by priority:
      a) filenames starting with "newspaper-downloads_dedupe-webp_unique_png_"
      b) filenames containing "downloads_zoning-missing_shard"
      c) everything else
   Ties break by newer mtime, then larger size.
4) Copy the chosen JSON into `newspaper-parsing-local/data/unique_outputs_dedup`
   named "<canonical_image_id>.json", where the canonical id is the lexicographically
   smallest PNG name sharing the same hash. Originals stay untouched.
5) Write a manifest with full provenance to
   `newspaper-parsing-local/data/unique_outputs_dedup_manifest.json`, plus
   orphan lists for traceability.

Run from repo root:
    python scripts/dedupe_layouts.py
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
PNG_DIR = ROOT / "newspaper-parsing-local" / "data" / "unique_png"
JSON_DIR = ROOT / "newspaper-parsing-local" / "data" / "unique_outputs"
OUT_DIR = ROOT / "newspaper-parsing-local" / "data" / "unique_outputs_dedup"
MANIFEST_PATH = ROOT / "newspaper-parsing-local" / "data" / "unique_outputs_dedup_manifest.json"
ORPHAN_JSON_PATH = ROOT / "newspaper-parsing-local" / "data" / "unique_outputs_orphan_json.json"
PNG_NO_JSON_PATH = ROOT / "newspaper-parsing-local" / "data" / "unique_outputs_png_without_json.json"


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return hex SHA256 for a file without loading it all into memory."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def priority(stem: str) -> int:
    """Lower number = higher priority."""
    if stem.startswith("newspaper-downloads_dedupe-webp_unique_png_"):
        return 0
    if "downloads_zoning-missing_shard" in stem:
        return 1
    return 2


def build_png_maps() -> Tuple[Dict[str, List[Dict]], Dict[str, str]]:
    """Hash PNGs and build maps: hash -> list of png records, and id -> hash."""
    hash_to_pngs: Dict[str, List[Dict]] = defaultdict(list)
    id_to_hash: Dict[str, str] = {}
    png_paths = list(PNG_DIR.rglob("*.png"))
    for i, path in enumerate(png_paths, 1):
        img_id = path.stem
        file_hash = sha256_file(path)
        hash_to_pngs[file_hash].append({"id": img_id, "path": str(path)})
        # If same id maps to multiple hashes (shouldn't), keep first but note collision.
        if img_id in id_to_hash and id_to_hash[img_id] != file_hash:
            # Collision unlikely; log and keep first to stay deterministic.
            print(f"⚠️  Name collision for {img_id}: {id_to_hash[img_id]} vs {file_hash}; keeping first.")
        else:
            id_to_hash[img_id] = file_hash
        if i % 2000 == 0:
            print(f"Hashed {i}/{len(png_paths)} PNGs...")
    print(f"Finished hashing {len(png_paths)} PNGs; unique hashes: {len(hash_to_pngs)}")
    return hash_to_pngs, id_to_hash


def gather_jsons(id_to_hash: Dict[str, str]):
    """Group JSONs by PNG hash; return grouped, orphan list."""
    grouped: Dict[str, List[Path]] = defaultdict(list)
    orphans: List[str] = []
    json_paths = list(JSON_DIR.rglob("*.json"))
    for i, path in enumerate(json_paths, 1):
        stem = path.stem
        img_id = stem.split("_")[-1]
        file_hash = id_to_hash.get(img_id)
        if file_hash:
            grouped[file_hash].append(path)
        else:
            orphans.append(str(path))
        if i % 5000 == 0:
            print(f"Scanned {i}/{len(json_paths)} JSONs...")
    print(f"Grouped JSONs for {len(grouped)} PNG hashes; orphans: {len(orphans)}")
    return grouped, orphans


def pick_best(paths: List[Path]) -> Path:
    """Choose best JSON path by priority, then mtime desc, then size desc."""
    return sorted(
        paths,
        key=lambda p: (
            priority(p.stem),
            -p.stat().st_mtime,
            -p.stat().st_size,
        ),
    )[0]


def main() -> None:
    if not PNG_DIR.exists():
        raise SystemExit(f"PNG directory missing: {PNG_DIR}")
    if not JSON_DIR.exists():
        raise SystemExit(f"JSON directory missing: {JSON_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    hash_to_pngs, id_to_hash = build_png_maps()
    grouped, orphans = gather_jsons(id_to_hash)

    manifest = []
    kept_count = 0
    for file_hash, json_list in grouped.items():
        best = pick_best(json_list)
        # Choose canonical image id: smallest PNG name sharing this hash.
        png_records = hash_to_pngs.get(file_hash, [])
        if not png_records:
            # Shouldn't happen; skip safely.
            continue
        canonical_id = sorted(r["id"] for r in png_records)[0]
        dest = OUT_DIR / f"{canonical_id}.json"
        os.makedirs(dest.parent, exist_ok=True)
        # Copy with metadata.
        dest.write_bytes(best.read_bytes())
        os.utime(dest, (best.stat().st_atime, best.stat().st_mtime))

        dropped = [str(p) for p in json_list if p != best]
        manifest.append(
            {
                "hash": file_hash,
                "canonical_image_id": canonical_id,
                "png_paths": [r["path"] for r in png_records],
                "kept_json": str(best),
                "copied_to": str(dest),
                "kept_reason": "priority->mtime->size",
                "dropped_jsons": dropped,
            }
        )
        kept_count += 1
        if kept_count % 5000 == 0:
            print(f"Deduped {kept_count} groups...")

    # PNGs with no JSON.
    png_without_json = []
    for h, png_list in hash_to_pngs.items():
        if h not in grouped:
            png_without_json.extend([p["path"] for p in png_list])

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    ORPHAN_JSON_PATH.write_text(json.dumps(orphans, indent=2))
    PNG_NO_JSON_PATH.write_text(json.dumps(png_without_json, indent=2))

    print(f"Done. Kept {kept_count} JSONs into {OUT_DIR}.")
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Orphan JSON list: {ORPHAN_JSON_PATH}")
    print(f"PNGs lacking JSON: {PNG_NO_JSON_PATH}")


if __name__ == "__main__":
    main()
