#!/usr/bin/env python3
"""Build a parser-bagging command-adapter config from a legacy newsbag run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from newsbag.legacy_run import LEGACY_SOURCE_ROOTS, write_legacy_bagging_config
except ModuleNotFoundError:  # pragma: no cover - convenience for direct source-tree use
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from newsbag.legacy_run import LEGACY_SOURCE_ROOTS, write_legacy_bagging_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-run-dir", type=Path, required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, default=None)
    parser.add_argument("--profile", default="legacy_import")
    parser.add_argument(
        "--source-root",
        action="append",
        choices=sorted(LEGACY_SOURCE_ROOTS),
        help="Limit discovery to one source root. May be passed more than once.",
    )
    parser.add_argument(
        "--strict-missing",
        action="store_true",
        help="Do not add --allow-missing to generated adapters.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = write_legacy_bagging_config(
        legacy_run_dir=args.legacy_run_dir,
        output_config=args.output_config,
        output_summary=args.output_summary,
        profile_name=str(args.profile),
        source_roots=args.source_root,
        allow_missing=not bool(args.strict_missing),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
