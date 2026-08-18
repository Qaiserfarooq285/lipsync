"""core.pipeline.run_job orchestration, exercised with every stage stubbed out.

No real subprocess ever runs here - envs.run_stage is monkeypatched per test -
so these check the *orchestration logic* (what runs, what's skipped, what a
stage failure does to the rest of the job), not the stages themselves.
"""

from __future__ import annotations

import pytest

from core import envs
from core.config import config_from_dict


@pytest.fixture
def cfg(tmp_path):
    clip = tmp_path / "presenter.mp4"
    ref = tmp_path / "ref.wav"
    clip.write_bytes(b"x")
    ref.write_bytes(b"x")
    return config_from_dict({
        "job": {"name": "t", "output_dir": str(tmp_path / "out")},
        "script": {"mode": "text", "text": "hello world"},
        "voice": {"reference_audio": str(ref)},
        "video": {"presenter_clip": str(clip)},
        "captions": {"enabled": True},
        "runtime": {"work_dir": str(tmp_path / "work"), "gate": "off"},
    })


def _stub_stage_success(monkeypatch, *, fail_captions=False):
    """Replace envs.run_stage with something that writes the artifact it was
    asked to produce, so the pipeline's own file-existence checks are satisfied
    without any model actually running."""
    def fake_run_stage(stage, module, args, **kwargs):
        out = None
        for flag in ("--out", "--srt-out"):
            if flag in args:
                out = args[args.index(flag) + 1]
        if stage == "captions" and fail_captions:
            raise envs.StageEnvError("No environment found for the 'captions' stage.")
        if out:
            from pathlib import Path
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).write_bytes(b"x")
    monkeypatch.setattr(envs, "run_stage", fake_run_stage)


def _stub_assembly(monkeypatch, captured: dict):
    """Assembly needs real ffmpeg; replace it and record what it was called with."""
    from iolib import assembly

    def fake_finalize(video, audio, dst, **kwargs):
        from pathlib import Path
        captured["srt"] = kwargs.get("srt")
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        Path(dst).write_bytes(b"x")
        return dst

    monkeypatch.setattr(assembly, "finalize", fake_finalize)
    monkeypatch.setattr(assembly, "write_manifest", lambda *a, **k: None)


def test_a_captions_failure_does_not_abort_the_job(cfg, monkeypatch):
    """The bug this guards: a missing captions env used to take the whole job
    down, discarding a lip-sync render that had already finished."""
    _stub_stage_success(monkeypatch, fail_captions=True)
    captured = {}
    _stub_assembly(monkeypatch, captured)

    from core.pipeline import run_job

    out = run_job(cfg)  # must not raise
    assert out.is_file()
    assert captured["srt"] is None, "assembly should not be told a subtitle file exists"


def test_captions_succeeding_still_reaches_assembly(cfg, monkeypatch):
    _stub_stage_success(monkeypatch, fail_captions=False)
    captured = {}
    _stub_assembly(monkeypatch, captured)

    from core.pipeline import run_job

    run_job(cfg)
    assert captured["srt"] is not None


def test_a_lipsync_failure_still_aborts_the_job(cfg, monkeypatch):
    """Unlike captions, the lip-sync render is not optional - its failure must
    propagate, or assembly would run on a video that was never produced."""
    def fake_run_stage(stage, module, args, **kwargs):
        if stage == "lipsync":
            raise envs.StageEnvError("boom")
        out = None
        for flag in ("--out",):
            if flag in args:
                out = args[args.index(flag) + 1]
        if out:
            from pathlib import Path
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).write_bytes(b"x")
    monkeypatch.setattr(envs, "run_stage", fake_run_stage)

    from core.pipeline import PipelineError, run_job

    with pytest.raises(envs.StageEnvError):
        run_job(cfg)


def test_captions_disabled_never_calls_the_captions_stage(cfg, monkeypatch):
    cfg.raw["captions"]["enabled"] = False
    calls = []

    def fake_run_stage(stage, module, args, **kwargs):
        calls.append(stage)
        out = args[args.index("--out") + 1] if "--out" in args else None
        if out:
            from pathlib import Path
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).write_bytes(b"x")

    monkeypatch.setattr(envs, "run_stage", fake_run_stage)
    _stub_assembly(monkeypatch, {})

    from core.pipeline import run_job

    run_job(cfg)
    assert "captions" not in calls
