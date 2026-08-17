"""Decide whether an upload is worth rendering, before spending GPU time on it.

This is the commercially important half of hosting. A render costs ~20 minutes of
exclusive GPU; a bad upload costs the same 20 minutes and then produces something
the user blames the product for. Measuring the source first turns that into a
six-second answer.

What it measures, in the order the evidence says matters:

  mouth darkness   how much of the mouth interior is already near-black in the
                   *source*. This is the single best predictor of the defect
                   users actually complain about. LatentSync regenerates the
                   whole interior every frame; when the source gives it no teeth
                   or tongue to learn from, it fills the gap with a flickering
                   black hole.
  face size        face height as a fraction of frame height. The model warps the
                   face into a fixed crop, so a small face means it is inventing
                   most of what it renders. Fixed at the shoot; no setting
                   recovers it.
  detection rate   how often a face is found at all. Turned heads, cutaways and
                   hands across the face all break the affine alignment.

Calibration, stated plainly because it matters: the darkness thresholds rest on
**three** measured clips, not a curve.

    source near-black   engine penalty   outcome
    5.91%               +3.29 pts        excellent, no visible void
    6.43%               +11.37 pts       acceptable, no visible void
    23.11%              +12.68 pts       rejected by the client on sight

So the boundary between "fine" and "complaint" sits somewhere between 6% and 23%,
and everything below is an interpolation. These numbers should be re-fitted once
real uploads accumulate - see `record_outcome` for the intended feedback path.
Treat a WARN as "probably fine, tell the user why it might not be", never as a
guarantee.

CPU-orchestrated: the measurement runs in the lipsync stage environment via
subprocess, so this module imports no torch and no mediapipe and stays usable
from a web worker.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from core.config import REPO_ROOT
from core.envs import resolve_python
from iolib import media

ACCEPT, WARN, REJECT = "accept", "warn", "reject"
_ORDER = {ACCEPT: 0, WARN: 1, REJECT: 2}

#: Near-black percentage inside the inner-lip polygon, measured on the source.
#: 12 is the midpoint of the evidence gap above, pulled down toward the known-good
#: end because the failure is expensive and the recovery is not (re-shoot with a
#: fill light) - a false WARN costs a sentence of explanation, a false ACCEPT
#: costs a render and a complaint.
DARK_WARN, DARK_REJECT = 12.0, 22.0

#: Face height as a fraction of frame height, in *landmark* units (forehead to
#: chin), which run smaller than a detector's bounding box - core.check_footage
#: reports the box convention, so its numbers read roughly 1.1x these.
#:
#: Advisory only: face size never triggers a REJECT on its own. On the three
#: clips with known outcomes it was **anti**-correlated with quality - the
#: largest face (16.8% by detector) produced the render the client rejected,
#: and the best render in the project came from a 13.2% face. Within the range
#: actually observed, lighting swamped it entirely.
#:
#: That is not a claim that face size does not matter; the reasoning behind
#: check_footage's bands - a small face means fewer real pixels to work with -
#: is sound, and no clip here was small enough to test it. It is a statement
#: that this project has no evidence for a rejection boundary, so it does not
#: assert one. TINY is set far below anything measured, where the face is too
#: small for the crop to contain meaningful detail at all.
#: WARN sits *below* every clip measured (11.7%, 11.8%, 17.3%), all of which
#: rendered without a face-size complaint. Warning above that would fire on the
#: best clip in the project and teach users to ignore the gate.
FACE_WARN, FACE_TINY = 0.10, 0.06

#: Fraction of sampled frames in which a face was found.
DETECT_WARN, DETECT_REJECT = 0.90, 0.60

#: Consecutive frames to analyse, starting 10% into the clip. 200 is eight
#: seconds at 25fps - long enough to cross several phrases and catch a turned
#: head, short enough that a three-minute upload does not cost three minutes of
#: CPU. Consecutive rather than scattered because the measurement runs FaceMesh
#: in tracking mode; see the note in _PROBE.
SAMPLE_FRAMES = 200

_PROBE = r"""
import json, sys
import cv2, numpy as np
import mediapipe as mp

# Same landmark set and thresholds as scripts/lipsync_metrics.py - the gate must
# speak the same units as the tool used to judge finished renders, or a source
# score and an output score cannot be compared.
UP, LO, CHIN, FORE = 13, 14, 152, 10
INNER_LIP = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415,
             310, 311, 312, 13, 82, 81, 80, 191]
OPEN_THRESHOLD = 0.025
DARK_LEVEL = 40

path, max_frames = sys.argv[1], int(sys.argv[2])
cap = cv2.VideoCapture(path)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

# A contiguous window read in order, not frames scattered across the clip.
#
# static_image_mode=False keeps FaceMesh in tracking mode, and that is not a
# performance detail - it changes the answer. Measured on the same file, static
# mode scored a well-lit clip at 0.33% near-black where tracking mode scored it
# 6.43%, because per-frame detection places the inner-lip polygon slightly
# tighter and misses the dark sliver at the lip line. A badly-lit clip scored
# 22.96% vs 23.11%, i.e. barely moved. So static mode compresses exactly the
# range these thresholds are calibrated across, and tracking mode is what
# scripts/lipsync_metrics.py uses to judge finished renders. Both tools must
# report the same units or a source score cannot be compared to an output score.
#
# Tracking needs consecutive frames, so cost is bounded by taking a window
# rather than by striding.
start = int(total * 0.10) if total > max_frames else 0
if start:
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1, refine_landmarks=False,
    min_detection_confidence=0.5, min_tracking_confidence=0.5)

found, fracs, darks, lums, apertures = 0, [], [], [], []
sampled = 0
while sampled < max_frames:
    ok, frame = cap.read()
    if not ok:
        break
    sampled += 1
    h, w = frame.shape[:2]
    res = mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:
        continue
    found += 1
    lm = res.multi_face_landmarks[0].landmark
    face_h = abs(lm[CHIN].y - lm[FORE].y) * h
    if face_h <= 0:
        continue
    # Raw landmark span, forehead to chin. Deliberately not rescaled to a
    # detector's bounding-box convention: a fudge factor fitted to one clip
    # produced a false REJECT on the best-performing clip in the project.
    # The bands in this module are stated in these units instead.
    fracs.append(face_h / h)
    aperture = abs(lm[LO].y - lm[UP].y) * h / face_h
    apertures.append(aperture)
    if aperture < OPEN_THRESHOLD:
        continue
    pts = np.array([[int(lm[i].x * w), int(lm[i].y * h)] for i in INNER_LIP], np.int32)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(pts), 255)
    mask = cv2.erode(mask, np.ones((5, 5), np.uint8))
    sel = mask > 0
    if sel.sum() < 40:
        continue
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[sel]
    darks.append(float((grey < DARK_LEVEL).mean()))
    lums.append(float(grey.mean()))

cap.release()
mesh.close()

print("RESULT_JSON:" + json.dumps({
    "sampled": sampled,
    "found": found,
    "open_frames": len(darks),
    "face_frac": float(np.mean(fracs)) if fracs else None,
    "near_black_pct": float(np.mean(darks) * 100) if darks else None,
    "interior_lum": float(np.mean(lums)) if lums else None,
    "aperture_p95": float(np.percentile(apertures, 95)) if apertures else None,
}))
"""


@dataclass
class GateResult:
    path: str
    verdict: str = ACCEPT
    reasons: list[str] = field(default_factory=list)
    advice: list[str] = field(default_factory=list)
    measured: dict = field(default_factory=dict)

    def fail(self, verdict: str, reason: str, advice: str | None = None) -> None:
        self.verdict = max(self.verdict, verdict, key=lambda v: _ORDER[v])
        self.reasons.append(reason)
        if advice:
            self.advice.append(advice)

    @property
    def ok(self) -> bool:
        return self.verdict != REJECT


def _measure(path: Path, samples: int = SAMPLE_FRAMES) -> dict | None:
    """Run the mediapipe probe in the lipsync env. None if unavailable."""
    try:
        python = resolve_python("lipsync", required=False)
    except Exception:  # noqa: BLE001 - the gate must degrade, not crash
        return None
    if python is None:
        return None

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(REPO_ROOT), env.get("PYTHONPATH", "")) if p
    )
    try:
        proc = subprocess.run(
            [str(python), "-c", _PROBE, str(path), str(samples)],
            cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=600,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            return json.loads(line[len("RESULT_JSON:"):])
    return None


def evaluate(clip: str | Path, *, samples: int = SAMPLE_FRAMES) -> GateResult:
    """Score a source clip and return accept / warn / reject with reasons."""
    clip = Path(clip)
    result = GateResult(path=str(clip))

    if not clip.is_file():
        result.fail(REJECT, f"file not found: {clip}")
        return result

    try:
        info = media.probe_video(clip)
    except media.MediaError as exc:
        result.fail(REJECT, f"not a decodable video: {exc}")
        return result

    result.measured["container"] = {
        "width": info.width, "height": info.height,
        "fps": round(info.fps, 3), "duration": round(info.duration, 2),
        "has_audio": info.has_audio,
    }

    if info.duration < 3.0:
        result.fail(
            REJECT, f"clip is only {info.duration:.1f}s",
            "Supply at least a few seconds; the clip is looped to cover the script, "
            "and a very short loop reads as an obvious repeat.",
        )
    if not info.has_audio:
        result.fail(
            WARN, "no audio track",
            "A separate voice reference will be required, since none can be "
            "extracted from this clip.",
        )

    m = _measure(clip, samples)
    if m is None:
        result.fail(
            WARN, "quality probe unavailable (lipsync env not installed)",
            "Run ./scripts/setup_envs.sh lipsync to enable mouth-lighting and "
            "face-size scoring. Format checks ran; quality checks did not.",
        )
        return result

    result.measured["quality"] = {
        k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()
    }

    sampled, found = m.get("sampled") or 0, m.get("found") or 0
    if sampled == 0:
        result.fail(REJECT, "could not read any frames")
        return result

    detect_rate = found / sampled
    result.measured["quality"]["detect_rate"] = round(detect_rate, 3)

    if detect_rate < DETECT_REJECT:
        result.fail(
            REJECT,
            f"a face was found in only {detect_rate * 100:.0f}% of sampled frames",
            "The lip-sync stage needs a clearly visible, forward-facing face "
            "throughout. Trim to a section where the presenter faces the camera.",
        )
    elif detect_rate < DETECT_WARN:
        result.fail(
            WARN,
            f"a face was missing in {(1 - detect_rate) * 100:.0f}% of sampled frames",
            "Expect artefacts wherever the head turns away or a hand crosses the face.",
        )

    frac = m.get("face_frac")
    if frac is not None:
        if frac < FACE_TINY:
            result.fail(
                REJECT,
                f"face fills only {frac * 100:.0f}% of frame height",
                "The face carries too little real detail to render lips at all. "
                "Frame tighter at the camera - cropping or upscaling afterwards "
                "does not help, because the model crops to the face by landmark.",
            )
        elif frac < FACE_WARN:
            result.fail(
                WARN,
                f"face fills {frac * 100:.0f}% of frame height",
                "On the smaller side, so the mouth may render softer than the rest "
                "of the frame. Note this did not predict quality on the clips "
                "measured so far - lighting mattered far more.",
            )

    dark = m.get("near_black_pct")
    if dark is None:
        result.fail(
            WARN, "the mouth never opened far enough to measure its interior",
            "Scoring assumed the worst case. If the presenter barely speaks in this "
            "clip, supply one where they do.",
        )
    elif dark >= DARK_REJECT:
        result.fail(
            REJECT,
            f"mouth interior is {dark:.1f}% near-black in the source",
            "This is the strongest predictor of the black-gap-between-the-lips "
            "artefact, and it is a lighting problem, not a settings problem. Add a "
            "fill light that actually reaches inside the mouth and re-shoot; every "
            "render setting tested moved this less than the lighting did.",
        )
    elif dark >= DARK_WARN:
        result.fail(
            WARN,
            f"mouth interior is {dark:.1f}% near-black in the source",
            "Renderable, but expect some darkening between the lips. More frontal "
            "fill light would reduce it.",
        )

    if not result.reasons:
        result.reasons.append(
            f"mouth {dark:.1f}% near-black, face {frac * 100:.0f}% of frame, "
            f"detected in {detect_rate * 100:.0f}% of frames"
        )
    return result


def record_outcome(path: Path, clip: str, measured: dict, accepted: bool, rating: str) -> None:
    """Append a rendered clip's source score and human verdict to a JSONL file.

    The thresholds above are interpolated from three clips. They only stop being
    guesses if real outcomes are logged against them, so the intended loop is:
    gate scores the source, a human rates the delivered video, and the pairs
    accumulate here until the boundary can be fitted rather than chosen.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "clip": clip, "measured": measured,
            "accepted": accepted, "rating": rating,
        }) + "\n")


def _print(result: GateResult) -> None:
    banner = {ACCEPT: "ACCEPT", WARN: "ACCEPT WITH WARNINGS", REJECT: "REJECT"}
    print(f"\n=== {Path(result.path).name} ===\n")
    c = result.measured.get("container")
    if c:
        print(f"  {c['width']}x{c['height']}  {c['fps']:.2f}fps  {c['duration']:.1f}s  "
              f"audio {'yes' if c['has_audio'] else 'no'}")
    q = result.measured.get("quality")
    if q:
        dark = q.get("near_black_pct")
        frac = q.get("face_frac")
        print(f"  mouth interior   {dark:.1f}% near-black" if dark is not None
              else "  mouth interior   not measurable")
        print(f"  face size        {frac * 100:.0f}% of frame height" if frac is not None
              else "  face size        not measurable")
        print(f"  face detected    {q.get('detect_rate', 0) * 100:.0f}% of sampled frames")
    print(f"\n  VERDICT: {banner[result.verdict]}\n")
    for r in result.reasons:
        print(f"    - {r}")
    if result.advice:
        print()
        for a in result.advice:
            print(f"    > {a}")
    print()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Score a source clip and decide whether it is worth rendering."
    )
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--samples", type=int, default=SAMPLE_FRAMES,
                    help="consecutive frames to analyse")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a report")
    args = ap.parse_args(argv)

    results = [evaluate(c, samples=args.samples) for c in args.clips]

    if args.json:
        print(json.dumps([asdict(r) for r in results], indent=2))
    else:
        for r in results:
            _print(r)

    # Non-zero only on outright rejection, so callers can gate a batch on it
    # while still letting warned clips through.
    return 1 if any(r.verdict == REJECT for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
