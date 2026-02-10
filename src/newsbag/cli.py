from __future__ import annotations

import argparse
import sys
from pathlib import Path

from newsbag.config import load_config
from newsbag.pipeline import run_pipeline


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="newsbag",
        description="Newspaper layout bagging pipeline (Paddle4 + Dell + MinerU).",
    )
    sub = p.add_subparsers(dest="command", required=True)

    runp = sub.add_parser("run", help="Run the full end-to-end layout pipeline.")
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
            "Allowed: paddle_layout,paddle_vl15,dell,mineru,fusion,review"
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
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
