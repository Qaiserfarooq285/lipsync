import shutil
import subprocess

import pytest

from core import ingest

FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None

pytestmark = pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg/ffprobe not on PATH")


def _make_clip(path, *, seconds=2.0, fps=30, size="128x128", extra=()):
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", f"testsrc=size={size}:rate={fps}:duration={seconds}",
            *extra, "-pix_fmt", "yuv420p", str(path),
        ],
        check=True,
    )


def _make_tone_clip(path, *, seconds=2.0, fps=25, volume=1.0):
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", f"testsrc=size=128x128:rate={fps}:duration={seconds}",
            "-f", "lavfi", "-i", f"sine=frequency=220:duration={seconds}",
            "-filter:a", f"volume={volume}",
            "-pix_fmt", "yuv420p", "-shortest", str(path),
        ],
        check=True,
    )


def test_probe_flags_non_25fps(tmp_path):
    clip = tmp_path / "c.mp4"
    _make_clip(clip, fps=30)
    info = ingest.probe(clip)
    assert info.needs_work
    assert any("25fps" in f for f in info.findings)


def test_probe_clean_25fps_clip_needs_no_video_work(tmp_path):
    clip = tmp_path / "c.mp4"
    _make_tone_clip(clip, fps=25)
    info = ingest.probe(clip)
    # Audio present and 25fps, so nothing about the video should be flagged.
    assert not info.needs_work, info.findings


def test_probe_flags_missing_audio(tmp_path):
    clip = tmp_path / "c.mp4"
    _make_clip(clip, fps=25)
    info = ingest.probe(clip)
    assert any("no audio" in f for f in info.findings)


@pytest.mark.parametrize("src_fps", [24, 30, 60])
def test_normalise_video_lands_on_25fps(tmp_path, src_fps):
    src, dst = tmp_path / "in.mp4", tmp_path / "out.mp4"
    _make_clip(src, fps=src_fps, seconds=2.0)
    ingest.normalise_video(src, dst)
    out = ingest.probe(dst)
    assert out.fps == pytest.approx(25.0, abs=0.01)
    assert out.pix_fmt == ingest.TARGET_PIX_FMT
    # Duration must survive the rate change; only the frame count changes.
    assert out.duration == pytest.approx(2.0, abs=0.15)


def test_normalise_video_downscales_past_the_cap(tmp_path):
    src, dst = tmp_path / "in.mp4", tmp_path / "out.mp4"
    _make_clip(src, size="3840x2160", fps=25, seconds=1.0)
    ingest.normalise_video(src, dst)
    out = ingest.probe(dst)
    assert max(out.width, out.height) == ingest.MAX_LONG_EDGE


def test_normalise_video_keeps_portrait_orientation(tmp_path):
    """Downscaling must cap the long edge without transposing the frame."""
    src, dst = tmp_path / "in.mp4", tmp_path / "out.mp4"
    _make_clip(src, size="2160x3840", fps=25, seconds=1.0)
    ingest.normalise_video(src, dst)
    out = ingest.probe(dst)
    assert out.height > out.width, "portrait clip came back landscape"
    assert out.height == ingest.MAX_LONG_EDGE


def test_normalise_video_strips_audio_by_default(tmp_path):
    src, dst = tmp_path / "in.mp4", tmp_path / "out.mp4"
    _make_tone_clip(src)
    ingest.normalise_video(src, dst)
    assert not ingest.probe(dst).has_audio


def test_normalise_audio_hits_the_loudness_target(tmp_path):
    src, dst = tmp_path / "in.mp4", tmp_path / "ref.wav"
    _make_tone_clip(src, seconds=6.0, volume=0.05)
    stats = ingest.normalise_audio(src, dst, target_lufs=-20.0)
    assert stats["gain_db"] > 0, "a quiet source should be turned up"
    after = ingest._loudness_stats(dst)
    assert float(after["input_i"]) == pytest.approx(-20.0, abs=1.5)


def test_pick_voice_reference_finds_the_loud_window(tmp_path):
    """The reference window should land on speech, not on the leading silence."""
    src, dst = tmp_path / "in.wav", tmp_path / "ref.wav"
    # 6s of near-silence, then 6s of tone.
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono:d=6",
         "-f", "lavfi", "-i", "sine=frequency=220:duration=6:sample_rate=24000",
         "-filter_complex", "[0:a][1:a]concat=n=2:v=0:a=1[o]", "-map", "[o]",
         "-c:a", "pcm_s16le", str(src)],
        check=True,
    )
    result = ingest.pick_voice_reference(src, dst, seconds=4.0)
    assert result["start"] >= 5.5, f"picked the silent head at {result['start']}s"


def test_pick_voice_reference_handles_clip_shorter_than_window(tmp_path):
    src, dst = tmp_path / "in.mp4", tmp_path / "ref.wav"
    _make_tone_clip(src, seconds=3.0)
    result = ingest.pick_voice_reference(src, dst, seconds=12.0)
    assert result["start"] == 0.0
    assert dst.is_file()


def test_speech_rate_measures_words_per_minute(tmp_path):
    audio = tmp_path / "a.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
         "-i", "sine=frequency=220:duration=30:sample_rate=24000",
         "-c:a", "pcm_s16le", str(audio)],
        check=True,
    )
    # 75 words over 30s is 150wpm.
    script = " ".join("word" for _ in range(75))
    assert ingest.speech_rate(audio, script) == pytest.approx(150.0, abs=1.0)


def test_suggest_speed_clamps_extreme_corrections(monkeypatch):
    """A very fast take must not be stretched past the safe limit."""
    monkeypatch.setattr(ingest, "speech_rate", lambda a, s: 270.0)
    result = ingest.suggest_speed("unused.wav", "unused")
    assert result["speed"] == pytest.approx(1.0 - ingest.MAX_AUTO_STRETCH, abs=1e-3)
    assert "capped" in result["reason"]


def test_suggest_speed_reports_in_band_takes_unchanged(monkeypatch):
    monkeypatch.setattr(ingest, "speech_rate", lambda a, s: 145.0)
    result = ingest.suggest_speed("unused.wav", "unused")
    assert result["speed"] == 1.0
    assert result["reason"] == "already in band"
