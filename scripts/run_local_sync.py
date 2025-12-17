#!/usr/bin/env python3
"""
One-command local synchronous runner.

If --gateway-url is not supplied, this script will:
  1) start a local agent-gateway via uvicorn on --port
  2) wait for /healthz
  3) run scripts/run_vlm_gateway.py against your layouts
  4) shut the gateway down

This is a convenience wrapper so you don't need two terminals.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run local sync VLM with optional auto-start gateway.")
    p.add_argument("--layouts", required=True, help='Glob or "@file" listing layout JSON paths')
    p.add_argument("--output-dir", required=True, help="Where to write VLM JSON outputs")
    p.add_argument("--model", required=True, help="Model name, e.g., gemini:gemini-2.5-flash")
    p.add_argument("--png-root", default=None, help="Optional directory to find PNGs by basename")
    p.add_argument("--max-concurrency", type=int, default=4, help="Total box requests in flight")
    p.add_argument("--page-concurrency", type=int, default=1, help="Pages in flight")
    p.add_argument("--timeout", type=float, default=60.0, help="Gateway timeout seconds per call")
    p.add_argument("--max-retries", type=int, default=2, help="Retries per bbox on parse/transport error")
    p.add_argument("--save-crops-dir", default=None, help="Optional directory to save bbox crops")
    p.add_argument("--max-crop-megapixels", type=float, default=3.0, help="Crop megapixel guard (0 disables)")
    p.add_argument("--max-crop-dim", type=int, default=2048, help="Crop longest-edge guard (0 disables)")
    p.add_argument("--skip-existing", action="store_true", help="Skip layouts with existing outputs")
    p.add_argument("--skip-bad-layouts", action="store_true", help="Skip layouts that fail to load/crop")
    p.add_argument("--manifest-path", default=None, help="Override manifest.jsonl path")
    p.add_argument(
        "--gateway-url",
        default=None,
        help="If provided, use this existing gateway instead of starting one",
    )
    p.add_argument("--port", type=int, default=8000, help="Port to start gateway on (if auto-starting)")
    return p.parse_args()


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


def _gateway_health_ok(url: str) -> bool:
    health_url = url.rstrip("/") + "/healthz"
    try:
        with urllib.request.urlopen(health_url, timeout=1.0) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def main() -> None:
    args = parse_args()

    gateway_proc: subprocess.Popen | None = None
    gateway_url = args.gateway_url

    if gateway_url is None:
        # Auto-start gateway (unless one is already running on this port).
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
            gateway_proc = subprocess.Popen(cmd, cwd=str(Path(__file__).resolve().parents[1]))
            try:
                wait_for_gateway(gateway_url, timeout_s=45.0, proc=gateway_proc)
            except Exception:
                gateway_proc.terminate()
                gateway_proc.wait(timeout=5)
                raise

    try:
        run_cmd = [
            sys.executable,
            "scripts/run_vlm_gateway.py",
            "--layouts",
            args.layouts,
            "--output-dir",
            args.output_dir,
            "--model",
            args.model,
            "--gateway-url",
            gateway_url,
            "--max-concurrency",
            str(args.max_concurrency),
            "--page-concurrency",
            str(args.page_concurrency),
            "--timeout",
            str(args.timeout),
            "--max-retries",
            str(args.max_retries),
            "--max-crop-megapixels",
            str(args.max_crop_megapixels),
            "--max-crop-dim",
            str(args.max_crop_dim),
        ]
        if args.png_root:
            run_cmd += ["--png-root", args.png_root]
        if args.save_crops_dir:
            run_cmd += ["--save-crops-dir", args.save_crops_dir]
        if args.skip_existing:
            run_cmd.append("--skip-existing")
        if args.skip_bad_layouts:
            run_cmd.append("--skip-bad-layouts")
        if args.manifest_path:
            run_cmd += ["--manifest-path", args.manifest_path]

        subprocess.run(run_cmd, check=True, cwd=str(Path(__file__).resolve().parents[1]))
    finally:
        if gateway_proc is not None:
            gateway_proc.terminate()
            try:
                gateway_proc.wait(timeout=5)
            except Exception:  # noqa: BLE001
                gateway_proc.kill()


if __name__ == "__main__":
    main()
