from core.state import JobState, fingerprint


def test_fingerprint_stable_and_order_independent():
    a = fingerprint({"x": 1, "y": 2})
    b = fingerprint({"y": 2, "x": 1})
    assert a == b


def test_fingerprint_changes_with_content():
    assert fingerprint({"x": 1}) != fingerprint({"x": 2})


def test_job_state_fresh_requires_matching_fingerprint_and_artifact(tmp_path):
    state_path = tmp_path / "state.json"
    artifact = tmp_path / "out.txt"

    state = JobState(state_path)
    assert not state.is_fresh("voice", "abc", artifact)

    artifact.write_text("data")
    state.mark_done("voice", "abc")
    assert state.is_fresh("voice", "abc", artifact)

    assert not state.is_fresh("voice", "different-fp", artifact)

    artifact.unlink()
    assert not state.is_fresh("voice", "abc", artifact)


def test_job_state_persists_across_instances(tmp_path):
    state_path = tmp_path / "state.json"
    artifact = tmp_path / "out.txt"
    artifact.write_text("data")

    JobState(state_path).mark_done("script", "fp1")
    reloaded = JobState(state_path)
    assert reloaded.is_fresh("script", "fp1", None)


def test_job_state_clear(tmp_path):
    state_path = tmp_path / "state.json"
    state = JobState(state_path)
    state.mark_done("script", "fp1")
    state.clear("script")
    assert not state.is_fresh("script", "fp1", None)
