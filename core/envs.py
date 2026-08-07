"""Resolve and invoke the isolated per-stage Python environments.

Why this exists: the GPU stages have mutually exclusive dependency pins
(chatterbox-tts wants torch 2.6.0 / transformers 5.2.0, LatentSync wants torch
2.5.1 / transformers 4.48.0), so they cannot share an interpreter. Each lives in
its own venv under ``.venvs/`` and the CPU orchestrator drives it by subprocess.

Resolution order for a stage's interpreter:

1. ``PRESENTER_<STAGE>_PYTHON`` environment variable, if set.
2. ``.venvs/<stage>/bin/python``, if it exists.
3. The current interpreter — the Colab case, where there is a single environment
   and the stage's requirements were installed directly into it.

Because of (3) the same ``gpu/`` module works as an importable function inside a
Colab cell and as a subprocess target here, with no code changes.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from core.config import REPO_ROOT

STAGES = ("voice", "lipsync", "captions")

#: Import name that proves a stage's dependencies are present.
STAGE_PROBE = {
    "voice": "chatterbox",
    "lipsync": "diffusers",
    "captions": "whisperx",
}


class StageEnvError(RuntimeError):
    """Raised when a stage environment is missing or unusable."""


def env_dir(stage: str) -> Path:
    return REPO_ROOT / ".venvs" / stage


def _explicit_override(stage: str) -> Path | None:
    raw = os.environ.get(f"PRESENTER_{stage.upper()}_PYTHON")
    if not raw:
        return None
    p = Path(raw).expanduser()
    if not p.is_file():
        raise StageEnvError(
            f"PRESENTER_{stage.upper()}_PYTHON points at a missing interpreter: {p}"
        )
    return p


def _current_interpreter_has(stage: str) -> bool:
    """True if the running interpreter can already import the stage's deps."""
    probe = STAGE_PROBE.get(stage)
    if not probe:
        return False
    import importlib.util

    try:
        return importlib.util.find_spec(probe) is not None
    except (ImportError, ValueError):
        return False


def resolve_python(stage: str, *, required: bool = True) -> Path | None:
    """Return the interpreter that can run ``stage``."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}")

    override = _explicit_override(stage)
    if override is not None:
        return override

    candidate = env_dir(stage) / "bin" / "python"
    if candidate.is_file():
        return candidate

    if _current_interpreter_has(stage):
        return Path(sys.executable)

    if not required:
        return None

    raise StageEnvError(
        f"No environment found for the '{stage}' stage.\n"
        f"  Expected: {candidate}\n"
        f"  Create it with:  ./scripts/setup_envs.sh {stage}\n"
        f"  Or point PRESENTER_{stage.upper()}_PYTHON at a suitable interpreter.\n"
        f"  (On Colab, instead `pip install -r requirements/{stage}.txt` into the "
        f"runtime and this stage will use the current interpreter.)"
    )


def stage_available(stage: str) -> bool:
    try:
        return resolve_python(stage, required=False) is not None
    except StageEnvError:
        return False


def run_stage(
    stage: str,
    module: str,
    args: list[str],
    *,
    cwd: Path | None = None,
    extra_env: dict[str, str] | None = None,
    log_path: Path | None = None,
) -> None:
    """Run ``python -m <module> <args>`` inside the stage's environment.

    Output is streamed to this process's stdout and, if ``log_path`` is given,
    tee'd to that file so a failed batch item leaves a diagnosable trace.
    """
    python = resolve_python(stage)

    env = os.environ.copy()
    # The stage modules live in this repo, which is not installed into the
    # stage venvs — put the repo root on their import path explicitly.
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(REPO_ROOT), existing) if p)
    env.setdefault("PYTHONUNBUFFERED", "1")
    # Keep HF/torch caches out of $HOME so Colab runs can point them at Drive.
    env.setdefault("HF_HOME", str(REPO_ROOT / "weights" / "hf"))
    env.setdefault("TORCH_HOME", str(REPO_ROOT / "weights" / "torch"))
    if extra_env:
        env.update(extra_env)

    cmd = [str(python), "-m", module, *args]
    print(f"[{stage}] $ {' '.join(cmd)}", flush=True)

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)

    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        lines.append(line)
        if len(lines) > 4000:  # bound memory on very chatty stages
            del lines[:1000]

    code = proc.wait()

    if log_path is not None:
        log_path.write_text("".join(lines), encoding="utf-8")

    if code != 0:
        tail = "".join(lines[-40:])
        raise StageEnvError(
            f"stage '{stage}' failed with exit code {code}\n"
            f"  command: {' '.join(cmd)}\n"
            f"  last output:\n{tail}"
        )


def describe() -> list[tuple[str, str, str]]:
    """Return ``(stage, status, detail)`` rows for the doctor command."""
    rows = []
    for stage in STAGES:
        try:
            python = resolve_python(stage, required=False)
        except StageEnvError as exc:
            rows.append((stage, "ERROR", str(exc).splitlines()[0]))
            continue
        if python is None:
            rows.append((stage, "MISSING", f"create with ./scripts/setup_envs.sh {stage}"))
        else:
            rows.append((stage, "OK", str(python)))
    return rows


def require_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise StageEnvError(
            "ffmpeg not found on PATH. Install it (apt install ffmpeg) — every stage "
            "shells out to it for muxing and re-encoding."
        )
    return exe
