"""Normalise an arbitrary uploaded clip into the one shape the pipeline expects.

Batch work on curated footage can get away with feeding sources straight to the
model, because a human picked them and they happened to agree. Hosted work
cannot: uploads arrive at 24/25/30/60fps, variable frame rate, interlaced,
rotated by metadata rather than by pixels, and at wildly different audio levels.
Every one of those either silently degrades output or silently changes timing.

The rule here is that **the pipeline should never see variance it did not ask
for**. Normalise once, at ingest, and record what changed. Downstream stages then
get to assume 25fps CFR progressive yuv420p, which is what they already assume
today - the difference is that after this module the assumption is true.

Specifically, this exists because:

* LatentSync's ``read_video`` hardcodes ``ffmpeg -r 25`` and does it *silently*.
  A 30fps upload loses 5 frames a second with nothing in the log. Doing the
  conversion here makes it visible, makes it lossless-ish (one re-encode instead
  of two), and keeps frame indices meaningful when comparing input to output.
* Variable-frame-rate phone footage has no single ``fps`` to report. ``ffprobe``
  averages it, ``extend_video`` then builds a CFR timeline from that average, and
  the two disagree - which shows up as drift late in a long clip.
* Rotation is usually stored as a display-matrix flag, not baked into pixels.
  FFmpeg honours it; ``cv2.VideoCapture`` does not. ``core.check_footage`` reads
  frames with OpenCV, so a portrait phone upload gets scored sideways and its
  face measured against the wrong axis. Baking rotation makes every consumer
  agree about which way is up.

CPU-only and dependency-light: ffmpeg/ffprobe via :mod:`iolib.media`, nothing else.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from iolib import media

#: The canonical shape. 25fps because that is what LatentSync's audio feature
#: chunking assumes (`feature2chunks(fps=video_fps)` with video_fps=25); changing
#: it means changing the model's notion of how audio maps to frames, which is not
#: a knob to turn casually.
TARGET_FPS = 25.0
TARGET_PIX_FMT = "yuv420p"

#: Long-edge cap for the *driving* clip. The model crops to the face and works at
#: 512px regardless, so pixels beyond this buy nothing and cost decode time and
#: RAM - upstream decodes the whole clip into memory before trimming it.
MAX_LONG_EDGE = 1920

#: Loudness target for the cloned-voice reference, in LUFS. Chatterbox takes its
#: timbre *and* level cue from the reference sample, so unnormalised references
#: produce outputs that differ in loudness from one job to the next.
REFERENCE_LUFS = -20.0

#: Loudness target for delivered speech. -16 LUFS is the common web/streaming
#: figure and leaves headroom for a caption sting or bed music later.
DELIVERY_LUFS = -16.0


@dataclass
class VideoVariance:
    """Everything about an upload that differs from the canonical shape."""

    path: Path
    width: int = 0
    height: int = 0
    fps: float = 0.0
    r_fps: float = 0.0
    duration: float = 0.0
    pix_fmt: str = ""
    field_order: str = ""
    rotation: float = 0.0
    has_audio: bool = False
    audio_channels: int = 0
    audio_rate: int = 0
    findings: list[str] = field(default_factory=list)

    @property
    def is_vfr(self) -> bool:
        """Variable frame rate: nominal and average rates disagree materially."""
        if self.fps <= 0 or self.r_fps <= 0:
            return False
        return abs(self.r_fps - self.fps) / max(self.r_fps, 1e-6) > 0.02

    @property
    def is_interlaced(self) -> bool:
        return self.field_order not in ("", "progressive", "unknown")

    @property
    def needs_work(self) -> bool:
        return bool(self.findings)


def _probe_variance(path: Path) -> VideoVariance:
    data = media.ffprobe_json(path)
    streams = data.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if v is None:
        raise media.MediaError(f"no video stream in {path}")

    def _rate(value: str | None, default: float = 0.0) -> float:
        num, _, den = (value or "").partition("/")
        try:
            return float(num) / float(den) if float(den) else default
        except (ValueError, ZeroDivisionError):
            return default

    # Rotation lives in one of two places depending on ffmpeg version and
    # container: a display-matrix side-data entry, or a stream tag.
    rotation = 0.0
    for sd in v.get("side_data_list") or []:
        if "rotation" in sd:
            try:
                rotation = float(sd["rotation"])
            except (TypeError, ValueError):
                pass
    if not rotation:
        try:
            rotation = float((v.get("tags") or {}).get("rotate") or 0.0)
        except (TypeError, ValueError):
            rotation = 0.0

    info = VideoVariance(
        path=path,
        width=int(v.get("width") or 0),
        height=int(v.get("height") or 0),
        fps=_rate(v.get("avg_frame_rate")),
        r_fps=_rate(v.get("r_frame_rate")),
        duration=float(data.get("format", {}).get("duration") or 0.0),
        pix_fmt=str(v.get("pix_fmt") or ""),
        field_order=str(v.get("field_order") or ""),
        rotation=rotation,
        has_audio=a is not None,
        audio_channels=int((a or {}).get("channels") or 0),
        audio_rate=int((a or {}).get("sample_rate") or 0),
    )

    f = info.findings
    if abs(info.fps - TARGET_FPS) > 0.01:
        direction = "dropping" if info.fps > TARGET_FPS else "duplicating"
        f.append(
            f"{info.fps:.3f}fps -> {TARGET_FPS:.0f}fps ({direction} frames; "
            "LatentSync would do this silently anyway)"
        )
    if info.is_vfr:
        f.append(
            f"variable frame rate (nominal {info.r_fps:.3f}, average {info.fps:.3f}) "
            "- forcing CFR so the audio/frame mapping stays constant"
        )
    if info.is_interlaced:
        f.append(f"interlaced ({info.field_order}) - deinterlacing")
    if info.rotation:
        f.append(
            f"rotation flag {info.rotation:g}deg not baked into pixels - OpenCV "
            "consumers would see this sideways; baking it in"
        )
    if info.pix_fmt != TARGET_PIX_FMT:
        f.append(f"pixel format {info.pix_fmt} -> {TARGET_PIX_FMT}")
    if max(info.width, info.height) > MAX_LONG_EDGE:
        f.append(
            f"{info.width}x{info.height} exceeds the {MAX_LONG_EDGE}px long-edge cap "
            "- downscaling (the model crops to the face at 512px regardless)"
        )
    if not info.has_audio:
        f.append("no audio track - a voice reference must be supplied separately")

    return info


def probe(path: str | Path) -> VideoVariance:
    """Report how an upload differs from the canonical shape, without changing it."""
    path = Path(path)
    if not path.is_file():
        raise media.MediaError(f"not found: {path}")
    return _probe_variance(path)


def normalise_video(
    src: str | Path,
    dst: str | Path,
    *,
    target_fps: float = TARGET_FPS,
    max_long_edge: int = MAX_LONG_EDGE,
    keep_audio: bool = False,
) -> VideoVariance:
    """Rewrite ``src`` into the canonical shape at ``dst``. Returns what changed.

    Always re-encodes, even when the source already conforms. That is deliberate:
    a stream copy would preserve whatever container-level timing oddity prompted
    the check, and a conforming clip costs one cheap pass. The caller gets a file
    it can make hard assumptions about either way.
    """
    src, dst = Path(src), Path(dst)
    info = probe(src)
    dst.parent.mkdir(parents=True, exist_ok=True)

    filters: list[str] = []
    if info.is_interlaced:
        # Send-field mode would double the rate; we want one frame per field pair.
        filters.append("yadif=mode=0")
    if max_long_edge and max(info.width, info.height) > max_long_edge:
        # Scale the long edge, whichever it is, and keep dimensions even for H.264.
        filters.append(
            f"scale='if(gt(iw,ih),{max_long_edge},-2)':'if(gt(iw,ih),-2,{max_long_edge})'"
            ":flags=lanczos"
        )
    filters.append("setsar=1")
    # fps= rather than -r: as a filter it resamples on the timeline explicitly,
    # which is what makes a VFR source come out genuinely constant.
    filters.append(f"fps={target_fps:.6f}")

    # Rotation is handled by ffmpeg's default autorotate, which applies the
    # display matrix during decode - so re-encoding bakes it into pixels. The
    # flag is not passed explicitly: `-autorotate` takes an argument in ffmpeg 6
    # and silently consumes the following `-i`. The `rotate=0` metadata below is
    # what stops players from applying the rotation a second time.
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-i", str(src),
        "-vf", ",".join(filters),
        "-c:v", "libx264", "-preset", "medium", "-crf", "17",
        "-pix_fmt", TARGET_PIX_FMT,
        # Drop rotation metadata: it is in the pixels now, and leaving it would
        # make players rotate a second time.
        "-metadata:s:v:0", "rotate=0",
        "-vsync", "cfr",
    ]
    cmd += ["-c:a", "aac", "-b:a", "192k"] if (keep_audio and info.has_audio) else ["-an"]
    cmd.append(str(dst))
    media.run(cmd)

    out = probe(dst)
    if abs(out.fps - target_fps) > 0.01:
        raise media.MediaError(
            f"normalisation did not reach {target_fps}fps: {dst} is {out.fps:.3f}fps"
        )
    return info


def _loudness_stats(path: Path) -> dict:
    """Measure integrated loudness with ffmpeg's loudnorm in analysis mode."""
    proc = subprocess.run(
        ["ffmpeg", "-v", "info", "-i", str(path), "-af",
         "loudnorm=print_format=json", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    # loudnorm prints its JSON to stderr, after the usual banner.
    match = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", proc.stderr, re.DOTALL)
    if not match:
        raise media.MediaError(f"could not measure loudness of {path}")
    return json.loads(match.group(0))


def normalise_audio(
    src: str | Path,
    dst: str | Path,
    *,
    target_lufs: float = REFERENCE_LUFS,
    sample_rate: int = 24000,
    mono: bool = True,
) -> dict:
    """Bring an audio file to a known loudness, rate and channel count.

    Two-pass loudnorm: measure, then correct. The single-pass form is a dynamic
    compressor and will pump on speech, which changes the delivery a voice clone
    is supposed to be copying.
    """
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    stats = _loudness_stats(src)

    measured = (
        f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11"
        f":measured_I={stats['input_i']}:measured_TP={stats['input_tp']}"
        f":measured_LRA={stats['input_lra']}:measured_thresh={stats['input_thresh']}"
        f":offset={stats.get('target_offset', 0)}:linear=true:print_format=summary"
    )
    media.run([
        "ffmpeg", "-y", "-v", "error", "-i", str(src),
        "-af", measured,
        "-ac", "1" if mono else "2",
        "-ar", str(sample_rate),
        "-c:a", "pcm_s16le",
        str(dst),
    ])
    return {
        "input_lufs": float(stats["input_i"]),
        "target_lufs": target_lufs,
        "gain_db": round(target_lufs - float(stats["input_i"]), 2),
    }


def pick_voice_reference(
    src: str | Path,
    dst: str | Path,
    *,
    seconds: float = 12.0,
    target_lufs: float = REFERENCE_LUFS,
) -> dict:
    """Extract the best window of speech from a clip to clone the voice from.

    Taking the first N seconds is the obvious approach and a bad one: uploads
    routinely open on room tone, a slate, or a breath. This scans for the window
    with the highest sustained energy and the fewest silent gaps, which in
    practice lands on continuous speech.

    The window is deliberately short. Chatterbox conditions on a reference sample,
    and a longer one is not a better one - it just increases the chance of
    catching a cough, an interruption, or a level change partway through.
    """
    src, dst = Path(src), Path(dst)
    duration = media.probe_duration(src)
    if duration <= seconds:
        stats = normalise_audio(src, dst, target_lufs=target_lufs)
        return {"start": 0.0, "seconds": round(duration, 2), **stats}

    # One decode of the whole track at low rate, then score windows offline -
    # far cheaper than asking ffmpeg for each candidate separately.
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(src), "-vn", "-ac", "1",
         "-ar", "8000", "-f", "s16le", "-"],
        capture_output=True,
    ).stdout
    if not raw:
        raise media.MediaError(f"no decodable audio in {src}")

    import array
    import math

    samples = array.array("h")
    samples.frombytes(raw[: len(raw) - (len(raw) % 2)])
    rate = 8000
    hop = rate // 10  # 100ms buckets
    buckets = []
    for i in range(0, len(samples) - hop, hop):
        chunk = samples[i:i + hop]
        buckets.append(math.sqrt(sum(s * s for s in chunk) / len(chunk)) / 32768.0)
    if not buckets:
        raise media.MediaError(f"audio too short to score in {src}")

    # "Silent" relative to this clip rather than an absolute threshold, so the
    # scoring works on both a quiet lav recording and a loud handheld one.
    peak = max(buckets)
    floor = peak * 0.08
    win = int(seconds * 10)
    best_start, best_score = 0, -1.0
    for i in range(0, len(buckets) - win + 1):
        w = buckets[i:i + win]
        voiced = sum(1 for b in w if b > floor) / win
        # Energy carries the "is this speech" signal; the voiced fraction is
        # weighted heavily because a window broken by pauses is a worse
        # reference than a quieter but continuous one.
        score = (sum(w) / win) * (voiced ** 2)
        if score > best_score:
            best_start, best_score = i, score

    start = best_start / 10.0
    clip = dst.parent / f".{dst.stem}_window.wav"
    dst.parent.mkdir(parents=True, exist_ok=True)
    media.run([
        "ffmpeg", "-y", "-v", "error",
        "-ss", f"{start:.2f}", "-i", str(src), "-t", f"{seconds:.2f}",
        "-vn", "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", str(clip),
    ])
    stats = normalise_audio(clip, dst, target_lufs=target_lufs)
    clip.unlink(missing_ok=True)
    return {"start": round(start, 2), "seconds": seconds, **stats}


def speech_rate(audio: str | Path, script: str) -> float:
    """Words per minute of a synthesised take, for pacing decisions."""
    words = len([w for w in re.split(r"\s+", script.strip()) if w])
    dur = media.probe_duration(Path(audio))
    return (words / dur * 60.0) if dur > 0 else 0.0


#: Comfortable presenter pace. Conversational English runs 140-160wpm; scripted
#: presentation sits at the lower end because the audience has no turn-taking
#: cues to help them follow.
WPM_BAND = (125.0, 165.0)

#: Hard limit on automatic tempo correction. Stretching is not free: on this
#: project's own footage, atempo=0.75 cost 6.6 points of near-black inside the
#: mouth, because it shifts where phonemes land relative to frames the model has
#: already committed to. Anything past this should be fixed by rewriting the
#: script, not by stretching harder.
MAX_AUTO_STRETCH = 0.12


def suggest_speed(audio: str | Path, script: str) -> dict:
    """Recommend a tempo factor from measured pace, rather than a hand-set constant.

    Hand-tuning ``voice.speed`` per clip does not survive contact with hosting -
    nobody is going to eyeball 200 uploads. This derives a factor from what the
    take actually measured, and refuses to stretch far enough to hurt the render.
    """
    wpm = speech_rate(audio, script)
    lo, hi = WPM_BAND
    if wpm <= 0:
        return {"wpm": 0.0, "speed": 1.0, "reason": "could not measure"}
    if lo <= wpm <= hi:
        return {"wpm": round(wpm, 1), "speed": 1.0, "reason": "already in band"}

    target = hi if wpm > hi else lo
    factor = target / wpm
    clamped = max(1.0 - MAX_AUTO_STRETCH, min(1.0 + MAX_AUTO_STRETCH, factor))
    reason = (
        f"{wpm:.0f}wpm is outside {lo:.0f}-{hi:.0f}; "
        f"{'slowing' if factor < 1 else 'speeding'} to reach {target:.0f}wpm"
    )
    if abs(clamped - factor) > 1e-3:
        reason += (
            f" (wanted {factor:.3f}, capped at {clamped:.3f} - stretching further "
            "degrades the mouth render; shorten or lengthen the script instead)"
        )
    return {"wpm": round(wpm, 1), "speed": round(clamped, 3), "reason": reason}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Normalise an uploaded clip into the shape the pipeline expects."
    )
    ap.add_argument("clip")
    ap.add_argument("--out", help="write the normalised video here")
    ap.add_argument("--voice-out", help="extract and normalise a voice reference here")
    ap.add_argument("--voice-seconds", type=float, default=12.0)
    ap.add_argument("--check", action="store_true", help="report variance and exit")
    args = ap.parse_args(argv)

    try:
        info = probe(args.clip)
    except media.MediaError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"\n=== {Path(args.clip).name} ===\n")
    print(f"  {info.width}x{info.height}  {info.fps:.3f}fps  {info.duration:.2f}s  {info.pix_fmt}")
    print(f"  audio: {'%dch @ %dHz' % (info.audio_channels, info.audio_rate) if info.has_audio else 'none'}")
    print()
    if not info.needs_work:
        print("  already canonical - nothing to normalise")
    else:
        print("  will normalise:")
        for note in info.findings:
            print(f"    - {note}")
    print()

    if args.check:
        return 0

    if args.out:
        normalise_video(args.clip, args.out)
        after = probe(args.out)
        print(f"  video -> {args.out}")
        print(f"          {after.width}x{after.height} {after.fps:.3f}fps {after.duration:.2f}s\n")

    if args.voice_out:
        if not info.has_audio:
            print("error: clip has no audio to extract a voice reference from", file=sys.stderr)
            return 1
        stats = pick_voice_reference(args.clip, args.voice_out, seconds=args.voice_seconds)
        print(f"  voice -> {args.voice_out}")
        print(f"          {stats['seconds']:.1f}s from {stats['start']:.1f}s, "
              f"{stats['input_lufs']:.1f} -> {stats['target_lufs']:.1f} LUFS "
              f"({stats['gain_db']:+.1f}dB)\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
