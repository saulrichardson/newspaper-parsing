#!/usr/bin/env python3
"""
One-command page QA runner with optional auto-start gateway.

If --gateway-url is not supplied, this script will:
  1) start a local agent-gateway via uvicorn on --port (or reuse if already running)
  2) wait for /healthz
  3) run scripts/run_page_qa.py against your per-page *.vlm.json files
  4) shut the gateway down (only if it started it)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run local page QA with optional auto-start gateway.")
    p.add_argument("--pages", required=True, help='Glob or "@file" listing per-page *.vlm.json paths')
    p.add_argument("--output-dir", required=True, help="Where to write per-page QA JSON outputs")
    p.add_argument("--model", required=True, help="QA model, e.g. openai:gpt-5-nano")
    p.add_argument("--max-concurrency", type=int, default=4, help="Pages to QA concurrently")
    p.add_argument("--timeout", type=float, default=120.0, help="Gateway timeout seconds per QA call")
    p.add_argument("--max-retries", type=int, default=2, help="Retries per page on parse/transport error")
    p.add_argument("--skip-existing", action="store_true", help="Skip pages with existing QA outputs")
    p.add_argument("--manifest-path", default=None, help="Override manifest.jsonl path")
    p.add_argument("--allow-test-providers", action="store_true", help="Allow non-real providers for debugging")
    p.add_argument(
        "--gateway-url",
        default=None,
        help="If provided, use this existing gateway instead of starting one",
    )
    p.add_argument("--port", type=int, default=8000, help="Port to start gateway on (if auto-starting)")
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


def main() -> None:
    args = parse_args()

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
            "scripts/run_page_qa.py",
            "--pages",
            args.pages,
            "--output-dir",
            args.output_dir,
            "--model",
            args.model,
            "--gateway-url",
            gateway_url,
            "--max-concurrency",
            str(args.max_concurrency),
            "--timeout",
            str(args.timeout),
            "--max-retries",
            str(args.max_retries),
        ]
        if args.skip_existing:
            run_cmd.append("--skip-existing")
        if args.manifest_path:
            run_cmd += ["--manifest-path", args.manifest_path]
        if args.allow_test_providers:
            run_cmd.append("--allow-test-providers")

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

