"""Measure lip-sync quality on two axes that fail independently.

Kept in the repo rather than a scratch directory because it is what turned a
guessing game into a decidable one, and every future quality question about this
pipeline should be answered with it rather than by eye alone.

  aperture   how far the lips travel. Catches under-articulation - lips that
             never fully close on M/B/P, or never fully open on vowels.
  dark void  whether the mouth interior renders teeth and tongue or a black
             hole. LatentSync regenerates the whole interior every frame, and
             when it fails it leaves near-black patches that flicker in and out.
             Geometry metrics are completely blind to this, which is how a
             change once measured as an improvement looked worse on screen.

Always compare against the *source* clip measured the same way: the numbers are
only meaningful as a delta against what the real footage scores. Never compare
across resolutions - downscale to a common size first, since interpolation
lightens dark pixels and will invent an improvement that is not there.

Usage:
    python scripts/lipsync_metrics.py source.mp4 candidate_a.mp4 candidate_b.mp4

Requires mediapipe + opencv, so run it with the lipsync stage interpreter:
    .venvs/lipsync/bin/python scripts/lipsync_metrics.py ...
"""

from __future__ import annotations

import argparse
import sys

UP, LO, CHIN, FORE = 13, 14, 152, 10
INNER_LIP = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415,
             310, 311, 312, 13, 82, 81, 80, 191]

#: Below this normalised aperture the mouth is treated as closed - there is no
#: interior to judge, and including those frames would dilute the void metric.
OPEN_THRESHOLD = 0.025
#: Grey level below which a pixel counts as a void rather than shadow.
DARK_LEVEL = 40


def analyse(path: str, max_frames: int = 2000) -> dict:
    import cv2
    import mediapipe as mp
    import numpy as np

    mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=False,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(path)

    apertures, dark_fracs, lums = [], [], []
    n = 0
    while n < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        h, w = frame.shape[:2]
        res = mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            continue
        lm = res.multi_face_landmarks[0].landmark
        face_h = abs(lm[CHIN].y - lm[FORE].y) * h
        if face_h <= 0:
            continue
        aperture = abs(lm[LO].y - lm[UP].y) * h / face_h
        apertures.append(aperture)
        if aperture < OPEN_THRESHOLD:
            continue

        pts = np.array([[int(lm[i].x * w), int(lm[i].y * h)] for i in INNER_LIP], np.int32)
        mask = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(mask, cv2.convexHull(pts), 255)
        mask = cv2.erode(mask, np.ones((5, 5), np.uint8))  # stay off the lip edge
        sel = mask > 0
        if sel.sum() < 40:
            continue
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[sel]
        dark_fracs.append(float((grey < DARK_LEVEL).mean()))
        lums.append(float(grey.mean()))

    cap.release()
    mesh.close()

    if not apertures:
        return {"path": path, "error": "no face detected"}

    a = np.array(apertures)
    d = np.array(dark_fracs) if dark_fracs else np.array([0.0])
    return {
        "path": path,
        "frames": len(apertures),
        "open_frames": len(dark_fracs),
        "aperture_p05": round(float(np.percentile(a, 5)), 4),
        "aperture_p95": round(float(np.percentile(a, 95)), 4),
        "aperture_range": round(float(np.percentile(a, 95) - np.percentile(a, 5)), 4),
        "near_black_pct": round(float(d.mean() * 100), 2),
        "interior_lum": round(float(np.mean(lums)), 1) if lums else None,
        # How much the void flickers between frames - the "blinking" a viewer sees.
        "void_flicker": round(float(np.abs(np.diff(d)).mean() * 100), 2) if len(d) > 1 else None,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("videos", nargs="+", help="first is treated as the reference/source")
    args = ap.parse_args(argv)

    results = [analyse(v) for v in args.videos]
    ref = results[0]

    hdr = f"{'clip':30s} {'near-black':>11s} {'interior':>9s} {'ap-range':>9s} {'flicker':>8s}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        if "error" in r:
            print(f"{r['path'][-30:]:30s}  {r['error']}")
            continue
        print(f"{r['path'][-30:]:30s} {r['near_black_pct']:10.2f}% {r['interior_lum']:9.1f} "
              f"{r['aperture_range']:9.4f} {r['void_flicker']:8.2f}")

    if len(results) > 1 and "error" not in ref:
        print(f"\ndeltas against {ref['path'][-30:]} (source):")
        for r in results[1:]:
            if "error" in r:
                continue
            print(f"  {r['path'][-30:]:30s} near-black {r['near_black_pct'] - ref['near_black_pct']:+6.2f} pts"
                  f"   aperture {(r['aperture_range'] / ref['aperture_range'] - 1) * 100:+6.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
