from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from newsbag.config import load_config
from newsbag.bagging import run_bagging_canary
from newsbag.pipeline import run_pipeline
from newsbag.status import format_summary_text, summarize_run


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="newsbag",
        description="Newspaper layout bagging pipeline (Paddle4 + Dell + MinerU).",
    )
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
