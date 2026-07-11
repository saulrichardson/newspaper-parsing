#!/usr/bin/env python3
from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_TRACKED_BYTES = 10 * 1024 * 1024
FORBIDDEN_PARTS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
FORBIDDEN_NAMES = {".DS_Store", ".env", "credentials.json", "id_rsa", "id_ed25519"}
FORBIDDEN_SUFFIXES = {".key", ".p12", ".pem", ".pid", ".pyc"}
SECRET_PATTERNS = {
    "private key": re.compile(rb"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY"),
    "AWS access key": re.compile(rb"AKIA[0-9A-Z]{16}"),
    "Google API key": re.compile(rb"AIza[0-9A-Za-z_-]{35}"),
    "GitHub token": re.compile(rb"gh[pousr]_[0-9A-Za-z]{30,}"),
    "OpenAI-style key": re.compile(rb"sk-[0-9A-Za-z_-]{20,}"),
    "Slack token": re.compile(rb"xox[baprs]-[0-9A-Za-z-]{10,}"),
    "local macOS path": re.compile(rb"/" + rb"Users/[A-Za-z0-9._-]+/"),
}


def _publishable_files() -> list[Path]:
    completed = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return [ROOT / raw.decode("utf-8") for raw in completed.stdout.split(b"\0") if raw]


def main() -> int:
    findings: list[str] = []
    publishable_files = _publishable_files()
    for path in publishable_files:
        relative = path.relative_to(ROOT)
        if any(part in FORBIDDEN_PARTS for part in relative.parts):
            findings.append(f"runtime/cache path is tracked: {relative}")
        if path.name in FORBIDDEN_NAMES or path.suffix.lower() in FORBIDDEN_SUFFIXES:
            findings.append(f"sensitive/runtime filename is tracked: {relative}")
        if not path.is_file():
            continue
        size = path.stat().st_size
        if size > MAX_TRACKED_BYTES:
            findings.append(f"tracked file exceeds {MAX_TRACKED_BYTES} bytes: {relative} ({size})")
            continue
        payload = path.read_bytes()
        if b"\0" in payload:
            continue
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(payload):
                findings.append(f"possible {label} in tracked file: {relative}")

    if findings:
        print("public-safety scan failed:")
        for finding in findings:
            print(f"- {finding}")
        return 1
    print(f"public-safety scan passed: {len(publishable_files)} publishable files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
