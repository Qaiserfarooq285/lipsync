"""Optional post-process: lift the dark voids LatentSync leaves inside the mouth.

Why this exists. LatentSync regenerates the whole mouth interior every frame and
has no reliable teeth prior at this face scale, so it renders holes of near-black
where teeth and tongue should be. Measured on a 6s slice, real footage sits at
~23% near-black pixels inside the inner-lip polygon; LatentSync's output sits at
32-42%. Crucially this is *engine-intrinsic*, not something this pipeline causes:
the same +9 to +11 point penalty reproduces on ByteDance's own demo footage with
native audio. Guidance scale does not help - raising it deepens the voids while
it widens the lips.

Because the voids come and go frame to frame, they read as blinking. That is the
part a viewer actually notices, and it is what this stage targets.

What it does, deliberately conservatively:
  * lifts only pixels darker than ``threshold``, and only inside the inner-lip
    polygon, on a curve that leaves midtones untouched - so genuine lip shadow
    and the dark line between closed lips survive;
  * feathers the mask so nothing shows a hard edge;
  * temporally smooths the *lift amount* (not the pixels), so the correction
    itself cannot introduce a new flicker.

This is cosmetic repair of a model failure, not a fix for the model. It cannot
invent teeth that were never generated. Run it, look at the result, and keep it
only if it genuinely reads better - the eye is the arbiter here, not the metric.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

#: MediaPipe FaceMesh inner-lip ring. The interior of this polygon is the region
#: LatentSync synthesises and the only place this stage is allowed to touch.
INNER_LIP = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415,
             310, 311, 312, 13, 82, 81, 80, 191]


def _lift_curve(gray, threshold: int, strength: float):
    """Amount to add per pixel: strongest at black, zero at ``threshold``."""
    import numpy as np

    deficit = np.clip((threshold - gray.astype(np.float32)) / max(threshold, 1), 0.0, 1.0)
    # Squared falloff keeps the correction off midtones; only true voids move much.
    return strength * (deficit ** 2)


def run(
    src: Path | str,
    dst: Path | str,
    *,
    threshold: int = 55,
    strength: float = 42.0,
    feather: int = 9,
    temporal: float = 0.6,
    audio_from: Path | str | None = None,
) -> Path:
    """Lift dark voids inside the mouth. Returns the written path."""
    import cv2
    import mediapipe as mp
    import numpy as np

    src, dst = Path(src), Path(dst)
    if not src.is_file():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tmp = dst.with_name(dst.stem + "_silent.mp4")
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=False,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)

    prev_gain = None
    touched = 0
    total = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        total += 1
        res = mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            writer.write(frame)
            continue

        lm = res.multi_face_landmarks[0].landmark
        pts = np.array([[int(lm[i].x * w), int(lm[i].y * h)] for i in INNER_LIP], np.int32)
        mask = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(mask, cv2.convexHull(pts), 255)
        if mask.sum() == 0:
            writer.write(frame)
            continue

        k = feather | 1  # blur kernel must be odd
        soft = cv2.GaussianBlur(mask, (k, k), 0).astype(np.float32) / 255.0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gain = _lift_curve(gray, threshold, strength)

        # Smooth the correction across time so the repair never flickers itself.
        if prev_gain is not None and temporal > 0:
            gain = temporal * prev_gain + (1.0 - temporal) * gain
        prev_gain = gain

        add = (gain * soft)[:, :, None]
        out = np.clip(frame.astype(np.float32) + add, 0, 255).astype(np.uint8)
        writer.write(out)
        touched += 1

    cap.release()
    writer.release()
    mesh.close()

    # cv2's mp4v output is a re-encode away from anything deliverable, and it
    # drops audio entirely - remux to h264 and restore the track.
    audio_src = Path(audio_from) if audio_from else src
    has_audio = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(audio_src)],
        capture_output=True, text=True,
    ).stdout.strip() != ""

    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(tmp)]
    if has_audio:
        cmd += ["-i", str(audio_src), "-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-b:a", "192k"]
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p", str(dst)]
    subprocess.run(cmd, check=True)
    tmp.unlink(missing_ok=True)

    print(f"[enhance] {touched}/{total} frames adjusted -> {dst}", flush=True)
    return dst


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Lift dark voids inside the generated mouth.")
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=int, default=55,
                    help="only pixels darker than this are lifted")
    ap.add_argument("--strength", type=float, default=42.0,
                    help="maximum luminance added, at pure black")
    ap.add_argument("--feather", type=int, default=9)
    ap.add_argument("--temporal", type=float, default=0.6,
                    help="0 = no temporal smoothing of the correction, 0.9 = heavy")
    ap.add_argument("--audio-from", default=None)
    args = ap.parse_args(argv)

    run(args.src, args.out, threshold=args.threshold, strength=args.strength,
        feather=args.feather, temporal=args.temporal, audio_from=args.audio_from)
    return 0


if __name__ == "__main__":
    sys.exit(main())
