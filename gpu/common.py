"""Shared GPU-stage helpers.

Import-safe on a CPU-only machine: torch is only imported inside functions.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


class GpuError(RuntimeError):
    pass


def pick_device(requested: str = "auto") -> str:
    """Resolve ``auto|cuda|cpu`` to a concrete torch device string."""
    if requested == "cpu":
        return "cpu"

    try:
        import torch
    except ImportError as exc:
        raise GpuError(
            "torch is not installed in this environment. This module must run inside "
            "its stage venv (see core/envs.py) or a Colab runtime with the stage's "
            "requirements installed."
        ) from exc

    if torch.cuda.is_available():
        return "cuda"
    if requested == "cuda":
        raise GpuError(
            "runtime.device=cuda was requested but torch.cuda.is_available() is False.\n"
            "  Check `nvidia-smi`, and that the installed torch build has CUDA support."
        )
    return "cpu"


def free_vram_mb() -> int | None:
    """Free VRAM on the default device, in MiB. None if there is no GPU."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
        if out.returncode != 0:
            return None
        return int(out.stdout.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def require_vram(needed_mb: int, *, stage: str, hint: str = "") -> None:
    """Fail fast and legibly instead of dying in a CUDA OOM mid-render."""
    free = free_vram_mb()
    if free is None:
        return  # No GPU visible to nvidia-smi; pick_device already handled that.
    if free >= needed_mb:
        return

    detail = ""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        )
        if out.returncode == 0 and out.stdout.strip():
            detail = "\n  Processes currently holding VRAM:\n" + "\n".join(
                f"    {line}" for line in out.stdout.strip().splitlines()
            )
    except (OSError, subprocess.SubprocessError):
        pass

    raise GpuError(
        f"Not enough free VRAM for the {stage} stage: need ~{needed_mb} MiB, "
        f"{free} MiB free.{detail}\n"
        f"  {hint}".rstrip()
    )


def set_seed(seed: int | None) -> None:
    if seed is None or seed < 0:
        return
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def weights_dir(sub: str = "") -> Path:
    """Cache location for downloaded weights, overridable for Drive-backed caching."""
    root = Path(os.environ.get("PRESENTER_WEIGHTS_DIR") or (repo_root() / "weights"))
    d = root / sub if sub else root
    d.mkdir(parents=True, exist_ok=True)
    return d


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def report_device(stage: str, device: str) -> None:
    line = f"[{stage}] device={device}"
    if device == "cuda":
        try:
            import torch

            name = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_memory // (1024**2)
            free = free_vram_mb()
            line += f" ({name}, {total} MiB total"
            if free is not None:
                line += f", {free} MiB free"
            line += ")"
        except Exception:  # noqa: BLE001 - diagnostics must never break the run
            pass
    print(line, flush=True)
