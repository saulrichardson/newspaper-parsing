#!/usr/bin/env python3
"""
One-command runner:
  Gemini OCR (*.vlm.json) -> zoning classifier (openai:gpt-5-nano)

This script does three things:
  1) Scans a Gemini VLM output directory for per-page *.vlm.json files.
  2) Keeps ONLY pages that:
       - were produced by a Gemini OCR run (page.model starts with "gemini:")
       - contain at least one box with status=="ok" and a non-empty transcript
     It writes those page JSON paths into an @file list (pages_to_classify.txt).
  3) Starts/reuses a local agent-gateway (unless --gateway-url is provided),
     then runs scripts/run_zoning_classifier.py using openai:gpt-5-nano.

Fail-fast philosophy:
  - We only stage pages with real OCR text.
  - The classifier itself enforces page.model == gemini:* and will refuse empty input.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage Gemini OCR pages that have transcripts, then classify them with openai:gpt-5-nano."
    )
    p.add_argument(
        "--vlm-dir",
        default=None,
        help='Directory containing Gemini per-page "*.vlm.json" files (searched recursively).',
    )
    p.add_argument(
        "--output-dir",
        default="newspaper-parsing-local/data/zoning_labels_openai_gpt5nano",
        help="Where to write per-page classifier outputs + staging artifacts",
    )
    p.add_argument(
        "--prompt-path",
        default="prompts/zoning_ocr_classifier_prompt_text.txt",
        help="Prompt file path",
    )
    p.add_argument(
        "--port",
        type=int,
        default=8011,
        help="Port to start gateway on (if auto-starting)",
    )
    p.add_argument(
        "--gateway-url",
        default=None,
        help="If provided, use this existing gateway instead of starting one",
    )
    p.add_argument("--max-concurrency", type=int, default=4, help="Pages to classify concurrently")
    p.add_argument("--timeout", type=float, default=180.0, help="Gateway timeout seconds per call")
    p.add_argument("--max-retries", type=int, default=2, help="Retries per page on parse/transport error")
    p.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="If set, only stage/classify up to this many pages (useful for smoke tests).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only stage pages_to_classify.txt + summary.json; do not start gateway or call the classifier.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Do not pass --skip-existing to the classifier (will rewrite existing per-page outputs).",
    )
    return p.parse_args()


def _gateway_health_ok(url: str) -> bool:
    health_url = url.rstrip("/") + "/healthz"
    try:
        with urllib.request.urlopen(health_url, timeout=1.0) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def wait_for_gateway(url: str, timeout_s: float = 30.0, proc: subprocess.Popen | None = None) -> None:
    deadline = time.time() + timeout_s
    health_url = url.rstrip("/") + "/healthz"
    last_err: Exception | None = None
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError("Gateway process exited before becoming healthy.")
        try:
            with urllib.request.urlopen(health_url, timeout=2) as r:
                if r.status == 200:
                    return
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(0.5)
    raise RuntimeError(f"Gateway did not become healthy at {health_url}: {last_err}")


def _find_default_vlm_dir(repo_root: Path) -> Path:
    data_root = repo_root / "newspaper-parsing-local" / "data"
    if not data_root.is_dir():
        raise SystemExit(f"Expected local data directory not found: {data_root}")

    candidates: list[Path] = []
    for d in sorted(data_root.glob("vlm_out*gemini*")):
        if not d.is_dir():
            continue
        if any(d.rglob("*.vlm.json")):
            candidates.append(d)

    if not candidates:
        raise SystemExit(
            f"No Gemini VLM output dirs found under {data_root}.\n"
            "Pass --vlm-dir pointing at a directory containing *.vlm.json files."
        )
    if len(candidates) > 1:
        msg = "\n".join(f"  - {c}" for c in candidates)
        raise SystemExit(f"Multiple Gemini VLM dirs found; pass --vlm-dir explicitly:\n{msg}")
    return candidates[0]


def _page_has_ok_text(data: dict) -> bool:
    boxes = data.get("boxes")
    if not isinstance(boxes, list) or not boxes:
        return False
    for b in boxes:
        if not isinstance(b, dict):
            continue
        if b.get("status") != "ok":
            continue
        t = b.get("transcript") or ""
        if isinstance(t, str) and t.strip():
            return True
    return False


def stage_pages(*, vlm_dir: Path, out_dir: Path, max_pages: int | None) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pages_list_path = out_dir / "pages_to_classify.txt"
    summary_path = out_dir / "pages_to_classify.summary.json"

    all_paths = sorted(vlm_dir.rglob("*.vlm.json"))
    kept: list[str] = []

    skipped_not_gemini = 0
    skipped_no_text = 0
    skipped_parse_error = 0

    for idx, path in enumerate(all_paths, start=1):
        try:
            data = json.loads(path.read_text())
        except Exception:
            skipped_parse_error += 1
            continue

        model = (data.get("model") or "").strip()
        if not model.lower().startswith("gemini:"):
            skipped_not_gemini += 1
            continue

        if not _page_has_ok_text(data):
            skipped_no_text += 1
            continue

        kept.append(str(path.resolve()))
        if max_pages is not None and len(kept) >= int(max_pages):
            break

        if idx % 5000 == 0:
            print(f"Scanned {idx}/{len(all_paths)} pages; kept {len(kept)}…")

    pages_list_path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")

    summary = {
        "vlm_dir": str(vlm_dir),
        "total_vlm_jsons_seen": len(all_paths),
        "pages_staged": len(kept),
        "skipped_not_gemini": skipped_not_gemini,
        "skipped_no_text": skipped_no_text,
        "skipped_parse_error": skipped_parse_error,
        "pages_list_path": str(pages_list_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    if not kept:
        raise SystemExit(
            "No pages were staged for classification. This usually means:\n"
            "  - wrong --vlm-dir (not Gemini outputs), or\n"
            "  - pages contain no status==ok transcripts.\n"
            f"See: {summary_path}"
        )

    return pages_list_path, summary_path


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    vlm_dir = Path(args.vlm_dir).expanduser() if args.vlm_dir else _find_default_vlm_dir(repo_root)
    if not vlm_dir.is_dir():
        raise SystemExit(f"--vlm-dir not found or not a directory: {vlm_dir}")

    out_dir = (repo_root / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    pages_list_path, _summary_path = stage_pages(vlm_dir=vlm_dir, out_dir=out_dir, max_pages=args.max_pages)

    if args.dry_run:
        print("Dry run complete; not starting gateway or calling classifier.")
        return

    gateway_proc: subprocess.Popen | None = None
    gateway_url = args.gateway_url

    if gateway_url is None:
        port = int(args.port)
        gateway_url = f"http://127.0.0.1:{port}"

        if _gateway_health_ok(gateway_url):
            print(f"Gateway already running at {gateway_url}; reusing it.")
        else:
            cmd = [
                sys.executable,
                "-m",
                "uvicorn",
                "gateway.app:create_app",
                "--factory",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--log-level",
                "error",
            ]
            gateway_proc = subprocess.Popen(cmd, cwd=str(repo_root))
            try:
                wait_for_gateway(gateway_url, timeout_s=45.0, proc=gateway_proc)
            except Exception:
                gateway_proc.terminate()
                gateway_proc.wait(timeout=5)
                raise

    try:
        run_cmd = [
            sys.executable,
            "scripts/run_zoning_classifier.py",
            "--pages",
            f"@{pages_list_path}",
            "--output-dir",
            str(out_dir),
            "--model",
            "openai:gpt-5-nano",
            "--gateway-url",
            gateway_url,
            "--prompt-path",
            args.prompt_path,
            "--max-concurrency",
            str(args.max_concurrency),
            "--timeout",
            str(args.timeout),
            "--max-retries",
            str(args.max_retries),
        ]
        if not args.overwrite:
            run_cmd.append("--skip-existing")

        subprocess.run(run_cmd, check=True, cwd=str(repo_root))
    finally:
        if gateway_proc is not None:
            gateway_proc.terminate()
            try:
                gateway_proc.wait(timeout=5)
            except Exception:  # noqa: BLE001
                gateway_proc.kill()


if __name__ == "__main__":
    main()

