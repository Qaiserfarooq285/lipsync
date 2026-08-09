"""Assess a source clip before you spend GPU time on it.

Reports the things that actually predict output quality, in priority order:
how much of the frame the face fills, then resolution, framerate, and whether
there is usable audio to clone a voice from.

Face size dominates. LatentSync warps the detected face into a 420x560 crop and
generates at 512x512, so a face occupying ~13% of frame height means the model is
inventing most of what it renders and then discarding much of it when compositing
back. No render setting recovers detail the camera never captured.

CPU-only: the optional face measurement is delegated to the lipsync stage
environment via subprocess, so this module stays importable anywhere.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from core.config import REPO_ROOT
from core.envs import resolve_python
from iolib import media

#: Face height as a fraction of frame height, and what to expect from each band.
_BANDS = (
    (0.30, "excellent", "Face fills the frame well - the model has real detail to work with."),
    (0.22, "good", "Comfortable. Output should hold up at full resolution."),
    (0.15, "workable", "Usable, but the mouth will be softer than the rest of the frame."),
    (0.00, "poor", "Face is small; expect blurred teeth and lips regardless of settings."),
)

_PROBE_SNIPPET = r"""
import json, sys
import cv2
from gpu.patches.mediapipe_face_detector import MediaPipeFaceDetector

path = sys.argv[1]
cap = cv2.VideoCapture(path)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
# Sample a few frames spread through the clip rather than trusting frame 0,
# which is often a fade-in or a turned head.
targets = [int(total * f) for f in (0.15, 0.35, 0.55, 0.75)] if total else [0]

det = MediaPipeFaceDetector(device="cuda")
results = []
for idx in targets:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok:
        continue
    h, w = frame.shape[:2]
    bbox, _ = det(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if bbox is None:
        results.append({"frame": idx, "found": False})
        continue
    fw, fh = bbox[2] - bbox[0], bbox[3] - bbox[1]
    results.append({
        "frame": idx, "found": True,
        "face_w": int(fw), "face_h": int(fh),
        "frac": fh / h if h else 0.0,
    })
det.close()
cap.release()
print("RESULT_JSON:" + json.dumps(results))
"""


def measure_face(path: Path) -> list[dict] | None:
    """Sample frames and measure face size. None if the lipsync env is missing."""
    try:
        python = resolve_python("lipsync", required=False)
    except Exception:  # noqa: BLE001 - diagnostics must not hard-fail
        return None
    if python is None:
        return None

    import os

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(REPO_ROOT), env.get("PYTHONPATH", "")) if p
    )
    try:
        proc = subprocess.run(
            [str(python), "-c", _PROBE_SNIPPET, str(path)],
            cwd=str(REPO_ROOT), env=env,
            capture_output=True, text=True, timeout=300,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            return json.loads(line[len("RESULT_JSON:"):])
    return None


def _band(frac: float) -> tuple[str, str]:
    for threshold, label, advice in _BANDS:
        if frac >= threshold:
            return label, advice
    return "poor", ""


def report(path: Path) -> int:
    path = Path(path)
    if not path.is_file():
        print(f"error: not found: {path}", file=sys.stderr)
        return 1

    try:
        info = media.probe_video(path)
    except media.MediaError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"\n=== {path.name} ===\n")
    print(f"  resolution   {info.width}x{info.height}")
    print(f"  duration     {info.duration:.2f}s")
    print(f"  framerate    {info.fps:.2f} fps")
    print(f"  audio track  {'yes' if info.has_audio else 'NO - cannot extract a voice sample'}")

    faces = measure_face(path)
    print()
    if faces is None:
        print("  face size    (skipped - lipsync env not installed)")
        print("               run ./scripts/setup_envs.sh lipsync to enable this check")
        return 0

    found = [f for f in faces if f.get("found")]
    if not found:
        print("  face size    NO FACE DETECTED in any sampled frame")
        print("               the lip-sync stage will fail on this clip")
        return 1

    fracs = [f["frac"] for f in found]
    avg = sum(fracs) / len(fracs)
    label, advice = _band(avg)
    sample = found[0]

    print(f"  face size    {sample['face_w']}x{sample['face_h']} px  "
          f"({avg * 100:.1f}% of frame height)")
    print(f"  verdict      {label.upper()} - {advice}")
    if len(found) < len(faces):
        print(f"               note: face missing in {len(faces) - len(found)} of "
              f"{len(faces)} sampled frames - check for turned-away shots")

    if avg < 0.22:
        gain = 0.25 / avg if avg else 0
        print()
        print("  to improve   Crop tighter to head-and-shoulders before running, if the")
        print(f"               framing allows - roughly {gain:.1f}x closer would reach the")
        print("               comfortable band. Cropping a high-resolution source keeps real")
        print("               pixels; upscaling a small face does not add any.")

    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Assess a source clip's suitability for lip-sync before rendering."
    )
    ap.add_argument("clips", nargs="+", help="one or more video files")
    args = ap.parse_args(argv)

    worst = 0
    for clip in args.clips:
        worst = max(worst, report(Path(clip)))
    print()
    return worst


if __name__ == "__main__":
    sys.exit(main())
