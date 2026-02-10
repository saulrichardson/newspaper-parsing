from __future__ import annotations

import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ProcResult:
    cmd: List[str]
    rc: int
    seconds: int
    log_path: Path


def run_cmd(
    cmd: List[str],
    log_path: Path,
    timeout_sec: int,
    env: Optional[Dict[str, str]] = None,
) -> ProcResult:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    merged = os.environ.copy()
    if env:
        merged.update(env)

    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as logf:
        logf.write("$ " + " ".join(shlex.quote(x) for x in cmd) + "\n")
        logf.flush()
        p = subprocess.run(
            cmd,
            stdout=logf,
            stderr=subprocess.STDOUT,
            env=merged,
            timeout=timeout_sec,
            check=False,
        )
        rc = int(p.returncode)
    dt = int(time.time() - t0)
    return ProcResult(cmd=cmd, rc=rc, seconds=dt, log_path=log_path)
