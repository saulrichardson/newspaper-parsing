#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a manifest from image files.")
    p.add_argument("--input", required=True, help="Directory containing newspaper scans.")
    p.add_argument("--output", required=True, help="Output manifest file path.")
    p.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan subdirectories.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.input).expanduser().resolve()
    out = Path(args.output).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Input directory not found: {root}")

    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
    if args.recursive:
        files = [p for p in root.rglob("*") if p.suffix.lower() in exts and p.is_file()]
    else:
        files = [p for p in root.iterdir() if p.suffix.lower() in exts and p.is_file()]
    files = sorted(files, key=lambda p: p.name.lower())

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(str(p) for p in files) + ("\n" if files else ""), encoding="utf-8")
    print(f"Wrote {len(files)} entries to {out}")


if __name__ == "__main__":
    main()
