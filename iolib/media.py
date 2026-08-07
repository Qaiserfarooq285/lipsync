"""FFmpeg-backed media probing and manipulation.

Pure subprocess calls — no torch, no GPU. FFmpeg is invoked as an external binary
(never linked), which keeps its GPL build out of this project's license surface.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

EXTEND_MODES = ("loop", "pingpong", "freeze")


class MediaError(RuntimeError):
    pass


def _exe(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise MediaError(f"{name} not found on PATH; install ffmpeg")
    return path


def run(cmd: list[str], *, quiet: bool = True) -> str:
    """Run a command, raising :class:`MediaError` with stderr on failure."""
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise MediaError(
            f"command failed ({proc.returncode}): {' '.join(cmd[:6])}...\n"
            f"{proc.stderr[-2000:]}"
        )
    return proc.stdout


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: float
    duration: float
    n_frames: int
    has_audio: bool


def ffprobe_json(path: Path) -> dict:
    out = run(
        [
            _exe("ffprobe"), "-v", "error",
            "-print_format", "json",
            "-show_format", "-show_streams",
            str(path),
        ]
    )
    return json.loads(out)


def probe_video(path: Path) -> VideoInfo:
    path = Path(path)
    if not path.is_file():
        raise MediaError(f"video not found: {path}")
    data = ffprobe_json(path)
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise MediaError(f"no video stream in {path}")

    num, _, den = (video.get("avg_frame_rate") or "0/1").partition("/")
    try:
        fps = float(num) / float(den) if float(den) else 0.0
    except (ValueError, ZeroDivisionError):
        fps = 0.0
    if fps <= 0:
        num, _, den = (video.get("r_frame_rate") or "25/1").partition("/")
        fps = float(num) / float(den) if float(den) else 25.0

    duration = float(data.get("format", {}).get("duration") or video.get("duration") or 0.0)
    n_frames = int(video.get("nb_frames") or 0) or int(round(duration * fps))

    return VideoInfo(
        path=path,
        width=int(video.get("width") or 0),
        height=int(video.get("height") or 0),
        fps=fps,
        duration=duration,
        n_frames=n_frames,
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
    )


def probe_duration(path: Path) -> float:
    path = Path(path)
    if not path.is_file():
        raise MediaError(f"media not found: {path}")
    data = ffprobe_json(path)
    dur = data.get("format", {}).get("duration")
    if dur is None:
        for stream in data.get("streams", []):
            if stream.get("duration"):
                dur = stream["duration"]
                break
    if dur is None:
        raise MediaError(f"could not determine duration of {path}")
    return float(dur)


def extend_video(
    src: Path,
    dst: Path,
    target_seconds: float,
    *,
    mode: str = "pingpong",
    fps: float | None = None,
) -> Path:
    """Extend ``src`` so it lasts at least ``target_seconds``.

    LatentSync only repaints the mouth inside footage it is given, so the output
    can never be longer than the base clip. This makes the base clip long enough.

    Modes:
      ``loop``     concatenate the clip head-to-tail. Cheapest, but produces a
                   visible jump wherever the last frame meets the first.
      ``pingpong`` concatenate forward+reversed passes so the seam is continuous.
                   Default: for a talking-head clip this reads as natural motion.
      ``freeze``   hold the final frame. Cheapest of all, but the presenter goes
                   visibly still — only sensible for very short overruns.
    """
    src, dst = Path(src), Path(dst)
    if mode not in EXTEND_MODES:
        raise MediaError(f"extend_video mode must be one of {EXTEND_MODES}, got {mode!r}")

    info = probe_video(src)
    fps = fps or info.fps or 25.0
    dst.parent.mkdir(parents=True, exist_ok=True)

    # A little headroom so rounding never leaves us a frame short.
    target = target_seconds + (2.0 / fps)

    if info.duration >= target:
        if dst.resolve() != src.resolve():
            shutil.copy2(src, dst)
        return dst

    ff = _exe("ffmpeg")
    common_out = [
        "-an",                      # audio comes from the cloned voice, not the clip
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "17",
        "-pix_fmt", "yuv420p",
        "-r", f"{fps:.6f}",
        "-t", f"{target:.3f}",
    ]

    if mode == "freeze":
        pad = target - info.duration
        run([
            ff, "-y", "-v", "error",
            "-i", str(src),
            "-vf", f"tpad=stop_mode=clone:stop_duration={pad + 1:.3f}",
            *common_out, str(dst),
        ])
        return dst

    if mode == "loop":
        loops = math.ceil(target / max(info.duration, 1e-6))
        run([
            ff, "-y", "-v", "error",
            "-stream_loop", str(loops),
            "-i", str(src),
            *common_out, str(dst),
        ])
        return dst

    # pingpong: forward + reverse is one cycle of 2x duration.
    cycles = math.ceil(target / max(info.duration * 2.0, 1e-6))
    run([
        ff, "-y", "-v", "error",
        "-i", str(src),
        "-filter_complex",
        (
            "[0:v]split=2[fwd][tmp];"
            "[tmp]reverse[rev];"
            "[fwd][rev]concat=n=2:v=1:a=0[pp];"
            f"[pp]loop=loop={cycles}:size=32767:start=0,setpts=N/FRAME_RATE/TB[out]"
        ),
        "-map", "[out]",
        *common_out, str(dst),
    ])
    return dst


def mux(video: Path, audio: Path, dst: Path, *, copy_video: bool = True) -> Path:
    """Combine a video stream with an audio track."""
    video, audio, dst = Path(video), Path(audio), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    vcodec = ["-c:v", "copy"] if copy_video else [
        "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p"
    ]
    run([
        _exe("ffmpeg"), "-y", "-v", "error",
        "-i", str(video), "-i", str(audio),
        "-map", "0:v:0", "-map", "1:a:0",
        *vcodec,
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        str(dst),
    ])
    return dst


def burn_subtitles(video: Path, srt: Path, dst: Path) -> Path:
    """Hard-burn an SRT into the video."""
    video, srt, dst = Path(video), Path(srt), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    # ffmpeg's subtitles filter needs escaping for : and \ inside the path.
    escaped = str(srt).replace("\\", "\\\\").replace(":", r"\:").replace("'", r"\'")
    style = (
        "FontName=DejaVu Sans,FontSize=18,PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H90000000,BorderStyle=3,Outline=1,Shadow=0,MarginV=28"
    )
    run([
        _exe("ffmpeg"), "-y", "-v", "error",
        "-i", str(video),
        "-vf", f"subtitles='{escaped}':force_style='{style}'",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        str(dst),
    ])
    return dst


def to_hd(src: Path, dst: Path, *, height: int = 1080, fps: float | None = None) -> Path:
    """Upscale/pad to an HD frame with even dimensions, preserving aspect ratio."""
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    vf = (
        f"scale=-2:{height}:flags=lanczos,"
        f"pad=ceil(iw/2)*2:ceil(ih/2)*2"
    )
    cmd = [
        _exe("ffmpeg"), "-y", "-v", "error",
        "-i", str(src),
        "-vf", vf,
        "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
    ]
    if fps:
        cmd += ["-r", f"{fps:.6f}"]
    cmd += ["-c:a", "copy", str(dst)]
    run(cmd)
    return dst


def ensure_wav(src: Path, dst: Path, *, sample_rate: int = 16000, mono: bool = True) -> Path:
    """Transcode any audio into plain PCM WAV at a known rate."""
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    run([
        _exe("ffmpeg"), "-y", "-v", "error",
        "-i", str(src),
        "-vn",
        "-ac", "1" if mono else "2",
        "-ar", str(sample_rate),
        "-c:a", "pcm_s16le",
        str(dst),
    ])
    return dst


def is_playable(path: Path) -> bool:
    """Cheap validity check: does it decode and report a positive duration?"""
    try:
        info = probe_video(Path(path))
    except MediaError:
        return False
    return info.n_frames > 0 and info.duration > 0 and info.width > 0
