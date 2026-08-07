import shutil
import subprocess

import pytest

from iolib import media

FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None

pytestmark = pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg/ffprobe not on PATH")


def _make_clip(path, seconds=1.0, fps=10, size="64x64"):
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", f"testsrc=size={size}:rate={fps}:duration={seconds}",
            "-pix_fmt", "yuv420p", str(path),
        ],
        check=True,
    )


def test_probe_video(tmp_path):
    clip = tmp_path / "clip.mp4"
    _make_clip(clip, seconds=2.0, fps=10)
    info = media.probe_video(clip)
    assert info.width == 64 and info.height == 64
    assert 1.8 <= info.duration <= 2.2
    assert info.fps == pytest.approx(10, abs=0.5)


def test_probe_video_missing_file(tmp_path):
    with pytest.raises(media.MediaError):
        media.probe_video(tmp_path / "nope.mp4")


@pytest.mark.parametrize("mode", ["loop", "pingpong", "freeze"])
def test_extend_video_reaches_target_duration(tmp_path, mode):
    clip = tmp_path / "clip.mp4"
    _make_clip(clip, seconds=1.0, fps=10)
    out = media.extend_video(clip, tmp_path / f"ext_{mode}.mp4", target_seconds=3.5, mode=mode)
    info = media.probe_video(out)
    assert info.duration >= 3.4


def test_extend_video_noop_when_already_long_enough(tmp_path):
    clip = tmp_path / "clip.mp4"
    _make_clip(clip, seconds=3.0, fps=10)
    out = media.extend_video(clip, tmp_path / "same.mp4", target_seconds=1.0, mode="pingpong")
    info = media.probe_video(out)
    assert info.duration >= 2.9


def test_extend_video_rejects_bad_mode(tmp_path):
    clip = tmp_path / "clip.mp4"
    _make_clip(clip, seconds=1.0, fps=10)
    with pytest.raises(media.MediaError):
        media.extend_video(clip, tmp_path / "out.mp4", 2.0, mode="bogus")


def test_is_playable(tmp_path):
    clip = tmp_path / "clip.mp4"
    _make_clip(clip, seconds=1.0, fps=10)
    assert media.is_playable(clip)
    assert not media.is_playable(tmp_path / "missing.mp4")
