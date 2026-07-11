from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from newsbag import __version__
from newsbag.config import load_config
from newsbag.bagging import plan_bagging, run_bagging_canary
from newsbag.legacy_run import LEGACY_SOURCE_ROOTS, write_legacy_bagging_config
from newsbag.pipeline import run_pipeline
from newsbag.performance import write_bagging_performance_summary
from newsbag.status import format_summary_text, summarize_run
from newsbag.validation import (
    format_validation_text,
    validate_bagging_run,
    validate_parse_input_manifest,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="newsbag",
        description="Manifest-driven newspaper layout, OCR model bagging, fusion, and transcripts.",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    runp = sub.add_parser("run", help="Run the full end-to-end layout + transcription pipeline.")
    runp.add_argument("--config", required=True, help="Path to pipeline JSON config.")
    runp.add_argument(
        "--run-dir",
        default="",
        help="Optional explicit output run directory. Default: run_root/run_name_TIMESTAMP",
    )
    runp.add_argument(
        "--stages",
        default="",
        help=(
            "Comma-separated stages to run. Default is all stages. "
            "Allowed: paddle_layout,paddle_vl15,dell,mineru,fusion,review,transcription"
        ),
    )

    statp = sub.add_parser("status", help="Summarize progress of an existing run directory.")
    statp.add_argument(
        "--run-dir",
        default="",
        help=(
            "Run directory to inspect. If omitted, uses $RUN_DIR if set; otherwise errors. "
            "Example: /scratch/$USER/paddleocr_vl15/runs/layout_bagging_YYYYMMDD_HHMMSS"
        ),
    )
    statp.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of human-readable text.",
    )
    statp.add_argument(
        "--missing",
        type=int,
        default=0,
        help="If >0, include up to N missing slugs per stage.",
    )

    bagp = sub.add_parser("bagging-canary", help="Run the manifest-driven parser-bagging canary.")
    bagp.add_argument("--manifest", required=True, help="Parse input JSONL manifest.")
    bagp.add_argument("--run-dir", required=True, help="Output run directory.")
    bagp.add_argument("--config", default="", help="Optional parser-bagging adapter config JSON.")
    bagp.add_argument(
        "--profile",
        default="adaptive",
        help="Model bag profile.",
    )

    planp = sub.add_parser("plan-bagging", help="Profile pages and write a model plan without running adapters.")
    planp.add_argument("--manifest", required=True, help="Parse input JSONL manifest.")
    planp.add_argument("--run-dir", required=True, help="Output run-planning directory.")
    planp.add_argument("--config", default="", help="Optional parser-bagging adapter config JSON.")
    planp.add_argument("--profile", default="adaptive", help="Model bag profile.")

    perfp = sub.add_parser("summarize-performance", help="Rebuild a run's aggregate performance report.")
    perfp.add_argument("--run-dir", required=True, help="Parser run directory.")
    perfp.add_argument(
        "--output-json",
        default="",
        help="Optional output path. Default: RUN_DIR/reports/performance_summary.json",
    )

    manp = sub.add_parser("validate-parse-input-manifest", help="Validate a parser parse-input/source-artifact manifest.")
    manp.add_argument("--manifest", required=True, help="Parse input JSONL manifest.")
    manp.add_argument("--json", action="store_true", help="Emit JSON instead of human-readable text.")
    manp.add_argument("--output-json", default="", help="Optional path to write the validation report JSON.")
    manp.add_argument("--require-files", action="store_true", help="Fail when image_path files are missing.")
    manp.add_argument("--require-checksums", action="store_true", help="Fail when checksum_sha256 is empty.")
    manp.add_argument("--verify-checksums", action="store_true", help="Compare checksum_sha256 to image file bytes.")
    manp.add_argument("--strict", action="store_true", help="Return nonzero for warnings as well as errors.")

    legp = sub.add_parser(
        "legacy-run-config",
        help="Build a parser-bagging command-adapter config from a legacy newsbag run directory.",
    )
    legp.add_argument("--legacy-run-dir", required=True, help="Legacy newsbag run directory.")
    legp.add_argument("--output-config", required=True, help="Output parser-bagging adapter config JSON.")
    legp.add_argument("--output-summary", default="", help="Optional discovery summary JSON path.")
    legp.add_argument("--profile", default="legacy_import", help="Profile name to attach to generated adapters.")
    legp.add_argument(
        "--source-root",
        action="append",
        choices=sorted(LEGACY_SOURCE_ROOTS),
        help="Limit discovery to one source root. May be passed more than once.",
    )
    legp.add_argument(
        "--strict-missing",
        action="store_true",
        help="Do not add --allow-missing to generated adapters.",
    )

    valp = sub.add_parser("validate-run", help="Validate a parser-bagging run bundle.")
    valp.add_argument("--run-dir", required=True, help="Run directory to validate.")
    valp.add_argument("--json", action="store_true", help="Emit JSON instead of human-readable text.")
    valp.add_argument("--output-json", default="", help="Optional path to write the validation report JSON.")
    valp.add_argument(
        "--strict",
        action="store_true",
        help="Return nonzero for warnings as well as errors.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "status":
        run_dir_txt = str(args.run_dir).strip() or os.environ.get("RUN_DIR", "").strip()
        if not run_dir_txt:
            raise SystemExit("ERROR: --run-dir is required (or set RUN_DIR=...).")
        run_dir = Path(run_dir_txt).expanduser().resolve()
        summary = summarize_run(run_dir, missing_limit=int(args.missing))
        if bool(args.json):
            print(json.dumps(summary, indent=2, sort_keys=True))
        else:
            print(format_summary_text(summary), end="")
        return 0

    if args.command == "bagging-canary":
        bundle = run_bagging_canary(
            manifest_path=Path(args.manifest),
            run_dir=Path(args.run_dir),
            profile_name=str(args.profile),
            config_path=Path(args.config) if str(args.config).strip() else None,
        )
        print(json.dumps(bundle.performance | {"run_dir": bundle.run_dir}, indent=2, sort_keys=True))
        return 1 if int(bundle.performance.get("errors", 0) or 0) else 0

    if args.command == "plan-bagging":
        summary = plan_bagging(
            manifest_path=Path(args.manifest),
            run_dir=Path(args.run_dir),
            profile_name=str(args.profile),
            config_path=Path(args.config) if str(args.config).strip() else None,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    if args.command == "summarize-performance":
        report = write_bagging_performance_summary(
            Path(args.run_dir),
            Path(args.output_json) if str(args.output_json).strip() else None,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    if args.command == "validate-parse-input-manifest":
        report = validate_parse_input_manifest(
            Path(args.manifest),
            require_files=bool(args.require_files),
            require_checksums=bool(args.require_checksums),
            verify_checksums=bool(args.verify_checksums),
            warnings_are_errors=bool(args.strict),
        )
        if str(args.output_json).strip():
            output_path = Path(args.output_json).expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if bool(args.json):
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print(format_validation_text(report), end="")
        if report["status"] == "error" or (bool(args.strict) and report["status"] != "ok"):
            return 1
        return 0

    if args.command == "legacy-run-config":
        summary = write_legacy_bagging_config(
            legacy_run_dir=Path(args.legacy_run_dir),
            output_config=Path(args.output_config),
            output_summary=Path(args.output_summary) if str(args.output_summary).strip() else None,
            profile_name=str(args.profile),
            source_roots=args.source_root,
            allow_missing=not bool(args.strict_missing),
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    if args.command == "validate-run":
        report = validate_bagging_run(Path(args.run_dir))
        if str(args.output_json).strip():
            output_path = Path(args.output_json).expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if bool(args.json):
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print(format_validation_text(report), end="")
        if report["status"] == "error" or (bool(args.strict) and report["status"] != "ok"):
            return 1
        return 0

    if args.command != "run":
        raise ValueError(f"Unsupported command: {args.command}")

    cfg = load_config(Path(args.config))
    run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else None
    stages = None
    if str(args.stages).strip():
        stages = {s.strip().lower() for s in str(args.stages).split(",") if s.strip()}
    out = run_pipeline(cfg, run_dir_override=run_dir, stages=stages)
    print(str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
