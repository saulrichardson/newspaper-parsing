#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from newsbag.transcription import run_transcription


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run Paddle OCR per page and attach OCR lines to fused layout boxes "
            "(text/title) for transcript generation."
        )
    )
    p.add_argument("--run-dir", required=True, help="Pipeline run directory containing outputs/fusion.")
    p.add_argument(
        "--variant",
        default="",
        help="Fusion variant to transcribe (default: summary.recommended_variant).",
    )
    p.add_argument(
        "--paddleocr-bin",
        default="paddleocr",
        help="Path to paddleocr CLI binary.",
    )
    p.add_argument("--device", default="gpu:0", help="Paddle device for OCR (e.g. gpu:0 or cpu).")
    p.add_argument("--cpu-threads", type=int, default=8, help="CPU threads for paddleocr.")
    p.add_argument("--timeout-sec", type=int, default=3600, help="Per-page OCR timeout.")
    p.add_argument(
        "--min-overlap",
        type=float,
        default=0.30,
        help="Minimum line-area overlap ratio required to assign OCR line to a fused box.",
    )
    p.add_argument(
        "--labels",
        default="text,title",
        help="Comma-separated fused labels to include for transcript regions.",
    )
    p.add_argument("--max-pages", type=int, default=0, help="Optional page cap for smoke runs.")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip OCR if per-page ocr_lines.json already exists.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_root = run_transcription(
        run_dir=Path(args.run_dir).expanduser().resolve(),
        paddleocr_bin=args.paddleocr_bin,
        variant=args.variant.strip(),
        labels=[x.strip().lower() for x in str(args.labels).split(",") if x.strip()],
        min_overlap=args.min_overlap,
        device=args.device,
        cpu_threads=args.cpu_threads,
        timeout_sec=args.timeout_sec,
        max_pages=args.max_pages,
        resume=args.resume,
    )
    print(str(out_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
