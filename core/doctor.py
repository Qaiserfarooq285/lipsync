"""Environment diagnostics: ``python -m core.doctor``.

Confirms ``core/`` imports and runs with no GPU, reports which stage
environments are ready, and does a lightweight license/consent audit.
"""

from __future__ import annotations

import shutil
import subprocess
import sys

from core import envs
from core.config import REPO_ROOT
from core.consent import consent_status

#: Repos/weights this project must never use (see docs/licenses.md).
_BARRED_NAME_FRAGMENTS = ("wav2lip", "f5-tts", "f5_tts", "f5tts")


def check_gpu() -> tuple[bool, str]:
    if not shutil.which("nvidia-smi"):
        return False, "nvidia-smi not found (fine if you're only using Colab)"
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        )
    except subprocess.SubprocessError:
        return False, "nvidia-smi present but did not respond"
    if out.returncode != 0 or not out.stdout.strip():
        return False, "nvidia-smi present but reported no GPU"
    return True, out.stdout.strip()


def audit_licenses() -> int:
    print("License audit — full record in docs/licenses.md.\n")
    problems = 0

    ls_root = REPO_ROOT / "vendor" / "LatentSync"
    print(f"  vendor/LatentSync: {'present' if ls_root.is_dir() else 'NOT vendored (run ./scripts/vendor_latentsync.sh)'}")

    for stage_dir in sorted((REPO_ROOT / ".venvs").glob("*")) if (REPO_ROOT / ".venvs").is_dir() else []:
        site_dirs = list(stage_dir.glob("lib/python*/site-packages"))
        if not site_dirs:
            continue
        installed = {p.name.lower() for p in site_dirs[0].iterdir()}
        for fragment in _BARRED_NAME_FRAGMENTS:
            if any(fragment in name for name in installed):
                print(f"  BARRED component detected in .venvs/{stage_dir.name}: matches {fragment!r}")
                problems += 1

    print(
        "\n  Face detector default: mediapipe (Apache 2.0, commercial-safe).\n"
        "  InsightFace is available as an opt-in (video.face_detector=insightface) for\n"
        "  comparison only — its pretrained weights are non-commercial. Never enable it\n"
        "  for real, distributed commercial output."
    )
    print("\n  Review docs/licenses.md 'Open issues to resolve before commercial go-live'.")
    return 1 if problems else 0


def main() -> int:
    print("=== presenter-video environment check ===\n")

    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"Repo root: {REPO_ROOT}\n")

    has_gpu, gpu_info = check_gpu()
    print(f"Local GPU: {'yes' if has_gpu else 'no'} — {gpu_info}\n")
    print(f"ffmpeg: {'found' if shutil.which('ffmpeg') else 'MISSING (apt install ffmpeg)'}\n")

    print("Stage environments:")
    for stage, status, detail in envs.describe():
        print(f"  {stage:10s} {status:8s} {detail}")

    print()
    granted, problems = consent_status()
    print(f"Consent: {'GRANTED' if granted else 'NOT GRANTED (real media blocked; placeholders only)'}")
    for p in problems:
        print(f"  - {p}")

    print()
    return audit_licenses()


if __name__ == "__main__":
    sys.exit(main())
