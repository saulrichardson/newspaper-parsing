#!/usr/bin/env python3
"""Convert legacy normalized layout JSON into the parser-bagging ModelOutput contract."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

try:
    from newsbag.legacy import (
        build_legacy_model_output,
        extract_legacy_box_records,
        legacy_payload_to_regions,
        read_legacy_json,
    )
except ModuleNotFoundError:  # pragma: no cover - convenience for direct source-tree use
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from newsbag.legacy import (
        build_legacy_model_output,
        extract_legacy_box_records,
        legacy_payload_to_regions,
        read_legacy_json,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, action="append", required=True)
    parser.add_argument("--page-id", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--source-family", default="")
    parser.add_argument("--resource-class", default="cpu")
    parser.add_argument("--default-confidence", type=float, default=0.5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    regions: list[dict] = []
    input_paths: list[str] = []
    input_region_count = 0
    for input_json in args.input_json:
        input_path = input_json.expanduser().resolve()
        payload = read_legacy_json(input_path)
        input_paths.append(str(input_path))
        records = extract_legacy_box_records(payload, page_id=str(args.page_id))
        input_region_count += len(records)
        regions.extend(
            legacy_payload_to_regions(
                records,
                page_id=str(args.page_id),
                model_id=str(args.model_id),
                source_family=str(args.source_family),
                default_confidence=float(args.default_confidence),
                start_index=len(regions) + 1,
            )
        )

    output = build_legacy_model_output(
        page_id=str(args.page_id),
        model_id=str(args.model_id),
        regions=regions,
        source_family=str(args.source_family),
        resource_class=str(args.resource_class),
        input_paths=input_paths,
        started=started,
        default_confidence=float(args.default_confidence),
        input_region_count=input_region_count,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
