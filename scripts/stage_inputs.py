#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Allow direct script execution from a fresh checkout without requiring
# `pip install -e .` on the login node.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from newsbag.input_staging import stage_inputs_to_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Stage pipeline inputs and write an image manifest from mixed sources "
            "(single image, directory, manifest file, tar/zip archive)."
        )
    )
    p.add_argument(
        "--input",
        action="append",
        required=True,
        help=(
            "Input path(s). May be repeated. Each value may be an image file, "
            "directory, manifest text file, or archive (.tar/.tar.gz/.tgz/.zip). "
            "Comma-separated lists are also accepted."
        ),
    )
    p.add_argument("--output", required=True, help="Output manifest path.")
    p.add_argument("--staging-dir", required=True, help="Staging directory for extracted archives and summary.")
    p.add_argument("--recursive", action="store_true", help="Recursively scan directory inputs.")
    p.add_argument("--max-pages", type=int, default=0, help="Optional cap on manifest entries.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    report = stage_inputs_to_manifest(
        inputs=args.input,
        output_manifest=Path(args.output),
        staging_dir=Path(args.staging_dir),
        recursive=args.recursive,
        max_pages=int(args.max_pages or 0),
    )
    print(
        f"Staged {report.image_count} images -> {report.manifest_path} "
        f"(archives={report.archive_count}, manifests={report.manifest_input_count}, "
        f"dirs={report.directory_input_count}, files={report.file_input_count})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
