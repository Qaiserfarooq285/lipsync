"""Measure constant audio/video lag: does the mouth move at the right *time*?

This is the temporal axis, and it is independent of the spatial one. A perfectly
aligned face crop can still open the mouth several frames late, and a badly
aligned crop can be perfectly timed. `scripts/lipsync_metrics.py` measures
appearance; this measures timing.

Method: correlate per-frame mouth aperture against the speech envelope at a
range of frame offsets, and report the offset with the highest correlation.
A non-zero peak means a constant lag - the mouth is consistently early or late,
which reads as "out of sync" even when every shape is individually plausible.

STATUS: this estimator is NOT sensitive enough to be trusted on its own, and it
says so at runtime rather than guessing.

Validated against ground truth - real footage carrying its own audio, in sync by
construction - it reported +400ms on demo1 and +167ms on demo3, where the answer
is ~0. Peak correlations were 0.10 and 0.09, i.e. the argmax was picking noise.
Speech loudness and mouth openness are only loosely coupled (/m/ is loud and
closed, /f/ is quiet and open), so weak correlation is the expected outcome, not
a bug in this particular implementation.

It is kept because the refusal is itself useful: it establishes that no cheap
landmark-vs-envelope measure will settle a sync question here, so nobody has to
rediscover that. When a real number is needed, use SyncNet - LatentSync already
ships latentsync_syncnet.pt, which is trained for exactly this and scores
audio-visual offset directly.

Interpreting a result that does clear the floor: at 25fps one frame is 40ms, and
desync becomes perceptible around 45-125ms depending on direction.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

import cv2
import mediapipe as mp
import numpy as np

UP, LO, CHIN, FORE = 13, 14, 152, 10


def aperture_series(path: str) -> tuple[np.ndarray, float]:
    mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=False,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    vals = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        res = mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            vals.append(np.nan)
            continue
        lm = res.multi_face_landmarks[0].landmark
        fh = abs(lm[CHIN].y - lm[FORE].y) * h
        vals.append(abs(lm[LO].y - lm[UP].y) * h / fh if fh > 0 else np.nan)
    cap.release()
    mesh.close()
    a = np.array(vals, dtype=float)
    # Short detection gaps would otherwise punch holes in the correlation.
    if np.isnan(a).any():
        idx = np.arange(len(a))
        good = ~np.isnan(a)
        if good.sum() > 1:
            a = np.interp(idx, idx[good], a[good])
    return a, fps


def envelope(path: str, n_frames: int, fps: float) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", "16000",
         "-f", "s16le", "-"], capture_output=True).stdout
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if a.size == 0:
        return np.zeros(n_frames)
    per = int(16000 / fps)
    out = [float(np.sqrt(np.mean(a[i * per:(i + 1) * per] ** 2)))
           if a[i * per:(i + 1) * per].size else 0.0 for i in range(n_frames)]
    return np.array(out)


def measure(video: str, audio: str | None = None, max_lag: int = 10) -> dict:
    ap, fps = aperture_series(video)
    env = envelope(audio or video, len(ap), fps)
    n = min(len(ap), len(env))
    ap, env = ap[:n], env[:n]

    # Correlate rate-of-change, not level: the mouth opening is an event, and
    # speech onsets are events, whereas absolute aperture and absolute loudness
    # have no reason to track each other.
    da = np.abs(np.diff(ap))
    de = np.abs(np.diff(env))
    if da.std() < 1e-9 or de.std() < 1e-9:
        return {"error": "no variation to correlate"}

    results = []
    for lag in range(-max_lag, max_lag + 1):
        # positive lag = video shifted later = video currently EARLY
        shifted = np.roll(da, lag)
        m = min(len(shifted), len(de))
        r = float(np.corrcoef(shifted[:m], de[:m])[0, 1])
        results.append((lag, r))

    best_lag, best_r = max(results, key=lambda t: t[1])
    zero_r = dict(results)[0]

    # Refuse to report a lag the data cannot support.
    #
    # Validated against ground truth - real footage carrying its own audio, which
    # is in sync by construction - this estimator returned +400ms on demo1 and
    # +167ms on demo3, with peak correlations of 0.10 and 0.09. Those should have
    # been ~0. At that correlation the argmax is picking noise, so the peak
    # location is not evidence of anything.
    #
    # The floor below is the observed false-positive level, not a tuned
    # threshold: anything at or under it is indistinguishable from the failures
    # above. Speech loudness and mouth openness are only loosely coupled (/m/ is
    # loud and closed, /f/ is quiet and open), so weak correlation is expected
    # and this estimator is simply not sensitive enough on its own.
    #
    # For a real measurement use SyncNet - LatentSync ships latentsync_syncnet.pt
    # for exactly this - rather than trusting a stronger-looking peak here.
    noise_floor = 0.25
    spread = best_r - zero_r
    reliable = best_r >= noise_floor and spread >= 0.10

    return {
        "video": video,
        "fps": round(fps, 3),
        "frames": n,
        "best_lag_frames": best_lag if reliable else None,
        "best_lag_ms": round(best_lag * 1000.0 / fps, 1) if reliable else None,
        "corr_at_best": round(best_r, 4),
        "corr_at_zero": round(zero_r, 4),
        "reliable": reliable,
        "curve": [(l, round(r, 4)) for l, r in results],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Measure constant A/V lag in a lip-synced video.")
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--audio", default=None, help="audio file, if not the video's own track")
    ap.add_argument("--max-lag", type=int, default=10)
    ap.add_argument("--curve", action="store_true", help="print the full correlation curve")
    args = ap.parse_args(argv)

    for v in args.videos:
        r = measure(v, args.audio, args.max_lag)
        if "error" in r:
            print(f"{v}: {r['error']}")
            continue
        print(f"{v}")
        print(f"  frames {r['frames']} @ {r['fps']}fps")
        if not r["reliable"]:
            print(f"  NO RELIABLE LAG ESTIMATE (peak corr {r['corr_at_best']} "
                  f"below the {0.25} noise floor)")
            print("  This estimator returned +400ms and +167ms on footage that is in")
            print("  sync by construction, so a weak peak here means nothing. Use")
            print("  SyncNet for a real measurement.")
        else:
            lag = r["best_lag_frames"]
            direction = ("in sync" if lag == 0 else
                         f"video {'EARLY' if lag > 0 else 'LATE'} by {abs(r['best_lag_ms']):.0f}ms")
            print(f"  best lag {lag:+d} frames ({r['best_lag_ms']:+.0f} ms) -> {direction}")
            print(f"  corr at best {r['corr_at_best']}, at zero {r['corr_at_zero']}")
        if args.curve:
            for lag, rr in r["curve"]:
                bar = "#" * max(0, int(rr * 100))
                print(f"    {lag:+3d}  {rr:+.4f}  {bar}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
