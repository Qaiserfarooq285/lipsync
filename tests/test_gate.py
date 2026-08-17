import shutil
import subprocess

import pytest

from core import gate

FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
needs_ffmpeg = pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="ffmpeg/ffprobe not on PATH")


def _clip(path, *, seconds=10.0, audio=True, fps=25):
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i", f"testsrc=size=128x128:rate={fps}:duration={seconds}",
    ]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=220:duration={seconds}"]
    cmd += ["-pix_fmt", "yuv420p", "-shortest", str(path)]
    subprocess.run(cmd, check=True)
    return path


def _good():
    """A measurement that should pass every threshold."""
    return {
        "sampled": 200, "found": 200, "open_frames": 150,
        "face_frac": 0.20, "near_black_pct": 5.0,
        "interior_lum": 80.0, "aperture_p95": 0.09,
    }


@pytest.fixture
def stub(monkeypatch):
    """Replace the mediapipe probe so verdict logic is testable without a GPU env."""
    def _apply(**overrides):
        data = _good() | overrides
        monkeypatch.setattr(gate, "_measure", lambda *a, **k: data)
        return data
    return _apply


# -- verdict escalation ---------------------------------------------------

def test_verdict_takes_the_worst_of_several():
    r = gate.GateResult(path="x")
    r.fail(gate.WARN, "a")
    r.fail(gate.REJECT, "b")
    r.fail(gate.WARN, "c")
    assert r.verdict == gate.REJECT
    assert not r.ok


def test_a_warn_never_downgrades_a_reject():
    r = gate.GateResult(path="x")
    r.fail(gate.REJECT, "b")
    r.fail(gate.WARN, "a")
    assert r.verdict == gate.REJECT


def test_clean_result_accepts():
    r = gate.GateResult(path="x")
    assert r.verdict == gate.ACCEPT and r.ok


# -- container checks -----------------------------------------------------

def test_missing_file_is_rejected(tmp_path):
    assert gate.evaluate(tmp_path / "nope.mp4").verdict == gate.REJECT


@needs_ffmpeg
def test_non_video_is_rejected(tmp_path):
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"not a video")
    assert gate.evaluate(junk).verdict == gate.REJECT


@needs_ffmpeg
def test_very_short_clip_is_rejected(tmp_path, stub):
    stub()
    clip = _clip(tmp_path / "c.mp4", seconds=1.0)
    result = gate.evaluate(clip)
    assert result.verdict == gate.REJECT
    assert any("only 1" in r for r in result.reasons)


@needs_ffmpeg
def test_missing_audio_warns_but_does_not_reject(tmp_path, stub):
    stub()
    clip = _clip(tmp_path / "c.mp4", audio=False)
    result = gate.evaluate(clip)
    assert result.verdict == gate.WARN
    assert any("no audio" in r for r in result.reasons)


# -- quality thresholds ---------------------------------------------------

@needs_ffmpeg
def test_well_lit_clip_is_accepted(tmp_path, stub):
    stub()
    assert gate.evaluate(_clip(tmp_path / "c.mp4")).verdict == gate.ACCEPT


@needs_ffmpeg
def test_dark_mouth_is_rejected(tmp_path, stub):
    stub(near_black_pct=gate.DARK_REJECT + 1)
    result = gate.evaluate(_clip(tmp_path / "c.mp4"))
    assert result.verdict == gate.REJECT
    assert any("near-black" in r for r in result.reasons)
    assert any("lighting problem" in a for a in result.advice)


@needs_ffmpeg
def test_borderline_mouth_darkness_warns(tmp_path, stub):
    stub(near_black_pct=(gate.DARK_WARN + gate.DARK_REJECT) / 2)
    assert gate.evaluate(_clip(tmp_path / "c.mp4")).verdict == gate.WARN


@needs_ffmpeg
def test_the_three_calibration_clips_land_where_measured(tmp_path, stub):
    """demo1 5.91 and demo3 6.43 must pass; the rejected clip's 23.11 must not."""
    for value in (5.91, 6.43):
        stub(near_black_pct=value)
        assert gate.evaluate(_clip(tmp_path / "c.mp4")).verdict == gate.ACCEPT, value
    stub(near_black_pct=23.11)
    assert gate.evaluate(_clip(tmp_path / "c.mp4")).verdict == gate.REJECT


@needs_ffmpeg
def test_face_size_alone_never_rejects(tmp_path, stub):
    """Face size is advisory: it was anti-correlated with quality when measured."""
    stub(face_frac=gate.FACE_WARN - 0.01)
    assert gate.evaluate(_clip(tmp_path / "c.mp4")).verdict == gate.WARN


@needs_ffmpeg
def test_the_best_measured_clip_is_not_warned_about_face_size(tmp_path, stub):
    """demo1 measured 0.117 and produced the project's best render."""
    stub(face_frac=0.117)
    result = gate.evaluate(_clip(tmp_path / "c.mp4"))
    assert not any("frame height" in r for r in result.reasons)


@needs_ffmpeg
def test_a_truly_tiny_face_is_rejected(tmp_path, stub):
    stub(face_frac=gate.FACE_TINY - 0.01)
    assert gate.evaluate(_clip(tmp_path / "c.mp4")).verdict == gate.REJECT


@needs_ffmpeg
def test_frequent_face_loss_is_rejected(tmp_path, stub):
    stub(sampled=200, found=int(200 * (gate.DETECT_REJECT - 0.1)))
    result = gate.evaluate(_clip(tmp_path / "c.mp4"))
    assert result.verdict == gate.REJECT


@needs_ffmpeg
def test_occasional_face_loss_warns(tmp_path, stub):
    stub(sampled=200, found=int(200 * (gate.DETECT_WARN - 0.05)))
    assert gate.evaluate(_clip(tmp_path / "c.mp4")).verdict == gate.WARN


@needs_ffmpeg
def test_unmeasurable_mouth_warns_rather_than_passing_silently(tmp_path, stub):
    stub(near_black_pct=None, open_frames=0)
    result = gate.evaluate(_clip(tmp_path / "c.mp4"))
    assert result.verdict == gate.WARN
    assert any("never opened" in r for r in result.reasons)


@needs_ffmpeg
def test_unreadable_frames_are_rejected(tmp_path, stub):
    stub(sampled=0, found=0)
    assert gate.evaluate(_clip(tmp_path / "c.mp4")).verdict == gate.REJECT


# -- degradation ----------------------------------------------------------

@needs_ffmpeg
def test_missing_probe_env_warns_instead_of_crashing(tmp_path, monkeypatch):
    """Without the lipsync env the gate must degrade, not fail the upload."""
    monkeypatch.setattr(gate, "_measure", lambda *a, **k: None)
    result = gate.evaluate(_clip(tmp_path / "c.mp4"))
    assert result.verdict == gate.WARN
    assert any("probe unavailable" in r for r in result.reasons)


def test_record_outcome_appends_jsonl(tmp_path):
    import json

    log = tmp_path / "outcomes.jsonl"
    gate.record_outcome(log, "a.mp4", {"near_black_pct": 5.0}, True, "good")
    gate.record_outcome(log, "b.mp4", {"near_black_pct": 25.0}, False, "bad")
    rows = [json.loads(l) for l in log.read_text().splitlines()]
    assert [r["clip"] for r in rows] == ["a.mp4", "b.mp4"]
    assert rows[1]["accepted"] is False
