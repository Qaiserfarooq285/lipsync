import json

import pytest

from core import consent
from web import jobs as jobstore
from web.jobs import JobStore, _friendly_error, _ProgressTracker, choose_resolution


# -- progress tracking ----------------------------------------------------

def _feed(tracker, lines):
    out = []
    for line in lines:
        u = tracker.feed(line)
        if u and "percent" in u:
            out.append(u["percent"])
    return out


def test_stage_lines_move_the_bar_to_each_stage_floor():
    t = _ProgressTracker(captions=False)
    pcts = _feed(t, [
        "[pipeline] [j] resolving script\n",
        "[pipeline] [j] synthesising voice\n",
        "[pipeline] [j] running lip-sync\n",
        "[pipeline] [j] assembling final output\n",
    ])
    assert pcts == [
        JobStore._floor("script"), JobStore._floor("voice"),
        JobStore._floor("lipsync"), JobStore._floor("assembly"),
    ]


def test_cached_stages_still_advance_the_bar():
    """'up to date, skipping' must not leave the bar stuck at the previous stage."""
    t = _ProgressTracker(captions=False)
    pcts = _feed(t, [
        "[pipeline] [j] script up to date, skipping\n",
        "[pipeline] [j] voice up to date, skipping\n",
        "[pipeline] [j] lip-sync up to date, skipping\n",
    ])
    assert pcts == [
        JobStore._floor("script"), JobStore._floor("voice"), JobStore._floor("lipsync"),
    ]


def test_nested_tqdm_bars_are_told_apart():
    """The inner denoise loop resets constantly; only the outer loop is progress."""
    t = _ProgressTracker(captions=False)
    t.feed("[pipeline] [j] running lip-sync\n")
    lines = []
    for i in range(1, 101):                      # frame prep
        lines.append(f" {i}/100 [00:01<00:00]\n")
    for chunk in range(1, 11):                   # outer chunks
        lines.append(f" {chunk}/10 [00:05<00:00]\n")
        for step in range(1, 31):                # inner denoise steps
            lines.append(f" {step}/30 [00:00<00:00]\n")
    pcts = _feed(t, lines)

    assert pcts == sorted(pcts), "progress went backwards"
    floor, span = JobStore._floor("lipsync"), JobStore._span("lipsync")
    assert max(pcts) <= floor + span + 0.01
    assert max(pcts) > floor + span * 0.9, "outer loop never drove the bar near the top"


def test_inner_loop_alone_never_drives_the_bar_backwards():
    t = _ProgressTracker(captions=False)
    t.feed("[pipeline] [j] running lip-sync\n")
    lines = [f" {i}/100 [00:01<00:00]\n" for i in range(1, 101)]
    lines += [" 1/10 [00:05<00:00]\n"]
    lines += [f" {s}/30 [00:00<00:00]\n" for s in range(1, 31)]
    lines += [f" {s}/30 [00:00<00:00]\n" for s in range(1, 31)]   # inner resets
    pcts = _feed(t, lines)
    assert pcts == sorted(pcts)


def test_progress_never_exceeds_one_hundred():
    t = _ProgressTracker(captions=False)
    t.feed("[pipeline] [j] running lip-sync\n")
    lines = [f" {i}/10 [00:01<00:00]\n" for i in range(1, 11)]
    lines += [" 5/5 [00:01<00:00]\n", " 3/3 [00:01<00:00]\n"]
    lines += [f" {i}/10 [00:01<00:00]\n" for i in range(1, 40)]   # overshoot
    for p in _feed(t, lines):
        assert 0 <= p <= 100


def test_stage_change_resets_the_message_rotation():
    t = _ProgressTracker(captions=False)
    first = t.feed("[pipeline] [j] synthesising voice\n")
    second = t.feed("[pipeline] [j] running lip-sync\n")
    assert first["message"] == jobstore.STAGE_CHATTER["voice"][0]
    assert second["message"] == jobstore.STAGE_CHATTER["lipsync"][0]


def test_unrelated_lines_do_not_move_the_bar():
    t = _ProgressTracker(captions=False)
    t.feed("[pipeline] [j] running lip-sync\n")
    assert _feed(t, ["loading checkpoint\n", "[lipsync] dtype=torch.float16\n"]) == []


# -- error translation ----------------------------------------------------

@pytest.mark.parametrize("needle,expected", [
    ("torch.cuda.OutOfMemoryError: CUDA out of memory. Tried...", "GPU ran out of memory"),
    ("core.consent.ConsentError: refusing", "Consent was not confirmed"),
    ("RuntimeError: no face detected in clip", "No face could be found"),
])
def test_known_failures_get_a_plain_explanation(needle, expected):
    assert expected in _friendly_error(needle)


def test_unknown_failure_is_surfaced_not_swallowed():
    """An unexplained error the user can paste to us beats a reassuring lie."""
    msg = _friendly_error("Traceback...\nValueError: something very specific broke\n")
    assert "something very specific broke" in msg


def test_empty_log_still_returns_something():
    assert _friendly_error("") != ""


# -- job naming -----------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("Dr Ajay", "dr_ajay"),
    ("  Tariq  ", "tariq"),
    ("A/B'; rm -rf /", "a_b_rm_rf"),
    ("!!!", "presenter"),
    ("", "presenter"),
])
def test_presenter_names_become_shell_safe_job_names(raw, expected):
    from web.server import _safe_name
    assert _safe_name(raw) == expected


def test_long_names_are_truncated():
    from web.server import _safe_name
    assert len(_safe_name("x" * 200)) <= 40


# -- per-job consent ------------------------------------------------------

def _record(tmp_path, **over):
    fields = {
        "presenter_name": "Dr Ajay", "attested_by": "web-ui",
        "scope": "demo", "voice_cloning": True, "likeness_synthesis": True,
    }
    fields.update(over)
    path = tmp_path / "consent.json"
    consent.write_job_consent(path, **fields)
    return path


def test_a_complete_record_grants_consent(tmp_path):
    granted, problems = consent.job_consent_status(_record(tmp_path))
    assert granted, problems


def test_missing_record_is_refused(tmp_path):
    granted, problems = consent.job_consent_status(tmp_path / "nope.json")
    assert not granted and "not found" in problems[0]


@pytest.mark.parametrize("field", ["voice_cloning", "likeness_synthesis"])
def test_a_declined_permission_is_refused(tmp_path, field):
    granted, problems = consent.job_consent_status(_record(tmp_path, **{field: False}))
    assert not granted
    assert any(field in p for p in problems)


def test_an_empty_presenter_name_is_refused(tmp_path):
    granted, _ = consent.job_consent_status(_record(tmp_path, presenter_name="   "))
    assert not granted


def test_a_placeholder_value_is_refused(tmp_path):
    granted, problems = consent.job_consent_status(_record(tmp_path, presenter_name="<name>"))
    assert not granted
    assert any("unfilled" in p for p in problems)


def test_malformed_json_is_refused_not_crashed(tmp_path):
    path = tmp_path / "consent.json"
    path.write_text("{not json", encoding="utf-8")
    granted, problems = consent.job_consent_status(path)
    assert not granted and "unreadable" in problems[0]


def test_a_json_array_is_refused(tmp_path):
    path = tmp_path / "consent.json"
    path.write_text("[]", encoding="utf-8")
    assert not consent.job_consent_status(path)[0]


def test_truthy_but_not_true_does_not_grant_permission(tmp_path):
    """'yes' and 1 must not stand in for a real affirmation."""
    path = tmp_path / "consent.json"
    path.write_text(json.dumps({
        "presenter_name": "A", "attested_by": "b", "granted_at": "now",
        "scope": "s", "voice_cloning": "yes", "likeness_synthesis": 1,
    }), encoding="utf-8")
    assert not consent.job_consent_status(path)[0]


def test_a_job_consent_record_overrides_the_global_file(tmp_path, monkeypatch):
    """A per-job record must not fall back to CONSENT.md, in either direction."""
    from core.config import config_from_dict

    monkeypatch.setattr(consent, "consent_status", lambda: (True, []))
    clip = tmp_path / "c.mp4"
    clip.write_bytes(b"x")
    cfg = config_from_dict({
        "job": {"name": "j", "consent_record": str(_record(tmp_path, voice_cloning=False))},
        "video": {"presenter_clip": str(clip)},
        "voice": {"reference_audio": str(clip)},
    })
    # Global consent is granted, but this job's own record declines - it must lose.
    with pytest.raises(consent.ConsentError):
        consent.assert_allowed(cfg)


def test_a_valid_job_record_is_enough_without_the_global_file(tmp_path, monkeypatch):
    from core.config import config_from_dict

    monkeypatch.setattr(consent, "consent_status", lambda: (False, ["not granted"]))
    clip = tmp_path / "c.mp4"
    clip.write_bytes(b"x")
    cfg = config_from_dict({
        "job": {"name": "j", "consent_record": str(_record(tmp_path))},
        "video": {"presenter_clip": str(clip)},
        "voice": {"reference_audio": str(clip)},
    })
    consent.assert_allowed(cfg)  # must not raise


# -- resolution auto-selection --------------------------------------------
#
# This is the fix for a real incident: a job rendered on the softer 256px
# profile because of a manually-set environment variable that survived a
# server restart nobody remembered was carrying it. choose_resolution()
# replaces that with a decision made fresh from actual VRAM at render time.

def test_plenty_of_free_vram_picks_full_quality(monkeypatch):
    from gpu import common
    monkeypatch.setattr(common, "free_vram_mb", lambda: 11000)
    resolution, note = choose_resolution()
    assert resolution == 512
    assert "512" in note


def test_tight_vram_drops_to_the_smaller_profile(monkeypatch):
    from gpu import common
    monkeypatch.setattr(common, "free_vram_mb", lambda: 5000)
    resolution, note = choose_resolution()
    assert resolution == 256
    assert "reduced quality" in note


def test_the_boundary_itself_still_gets_full_quality(monkeypatch):
    """Exactly the threshold must round in the model's favour, not against it."""
    from gpu import common
    monkeypatch.setattr(common, "free_vram_mb", lambda: jobstore._VRAM_NEEDED[512])
    resolution, _ = choose_resolution()
    assert resolution == 512


def test_very_little_vram_still_returns_a_usable_answer_not_a_crash(monkeypatch):
    from gpu import common
    monkeypatch.setattr(common, "free_vram_mb", lambda: 500)
    resolution, note = choose_resolution()
    assert resolution == 256
    assert note


def test_unreadable_vram_falls_back_to_the_default_rather_than_guessing_low(monkeypatch):
    """nvidia-smi being unavailable is not evidence the GPU is under pressure."""
    from gpu import common
    monkeypatch.setattr(common, "free_vram_mb", lambda: None)
    resolution, _ = choose_resolution()
    assert resolution == 512


# -- job store ------------------------------------------------------------

def test_created_jobs_get_their_own_directory(tmp_path):
    store = JobStore(root=tmp_path)
    a = store.create("alice", captions=False)
    b = store.create("bob", captions=True)
    assert a.id != b.id
    assert a.dir().is_dir() and b.dir().is_dir()
    assert store.get(a.id) is a


def test_public_payload_hides_internal_paths(tmp_path):
    store = JobStore(root=tmp_path)
    job = store.create("alice", captions=False)
    job.output = "/internal/path.mp4"
    payload = job.public()
    assert "output" not in payload
    assert payload["video_url"] is None       # not finished yet


def test_finished_job_exposes_a_video_url(tmp_path):
    store = JobStore(root=tmp_path)
    job = store.create("alice", captions=False)
    job.status = jobstore.Status.DONE
    assert job.public()["video_url"] == f"/api/jobs/{job.id}/video"


def test_stage_weights_cover_the_whole_bar():
    assert sum(w for _, w in jobstore.STAGE_WEIGHTS) == pytest.approx(1.0)


def test_every_stage_has_something_to_say():
    for stage, _ in jobstore.STAGE_WEIGHTS:
        assert jobstore.STAGE_CHATTER.get(stage), f"{stage} has no status messages"
