from __future__ import annotations

import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def now_utc_iso() -> str:
    fixed = os.environ.get("BABAPPA_FIXED_TIMESTAMP_UTC")
    if fixed:
        return fixed
    return datetime.now(tz=timezone.utc).isoformat()


def get_git_commit(cwd: str | Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(cwd), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except Exception:
        return None


def get_system_metadata(cwd: str | Path) -> dict[str, object]:
    return {
        "build_timestamp_utc": now_utc_iso(),
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "git_commit": get_git_commit(cwd),
    }
