#!/usr/bin/env python3
"""Tiny command-backed adapter used by local/Torch parser-bagging smokes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page-id", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    margin_x = max(12, int(args.width * 0.10))
    margin_y = max(12, int(args.height * 0.12))
    output = {
        "page_id": args.page_id,
        "model_id": args.model_id,
        "regions": [
            {
                "bbox_xyxy": [
                    margin_x,
                    margin_y,
                    args.width - margin_x,
                    min(args.height - margin_y, margin_y + max(40, int(args.height * 0.20))),
                ],
                "label": "text",
                "confidence": 0.88,
                "text": f"fixture command adapter text for {args.page_id}",
                "reading_order": 1,
                "metadata": {"adapter_kind": "fixture_command"},
            }
        ],
        "metadata": {"fixture": True},
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
