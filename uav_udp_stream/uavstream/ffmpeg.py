"""Locate the ffmpeg / ffprobe binaries.

Checks, in order: the FFMPEG env var, the system PATH, and the default winget
install location (the PATH entry winget adds only applies to new shells)."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def _winget_candidates(name: str) -> list[Path]:
    base = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
    if not base.exists():
        return []
    return list(base.glob(f"Gyan.FFmpeg*/**/bin/{name}"))


def find(name: str = "ffmpeg") -> str:
    """Absolute path to ffmpeg/ffprobe, or raise with a clear message."""
    exe = name if name.endswith(".exe") or os.name != "nt" else name + ".exe"

    env = os.environ.get(name.upper())
    if env and Path(env).exists():
        return env

    found = shutil.which(exe) or shutil.which(name)
    if found:
        return found

    for cand in _winget_candidates(exe):
        return str(cand)

    raise FileNotFoundError(
        f"could not find {name}. Install it with:  winget install Gyan.FFmpeg\n"
        f"or set the {name.upper()} environment variable to its full path."
    )
