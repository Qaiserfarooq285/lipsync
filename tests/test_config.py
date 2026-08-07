from pathlib import Path

import pytest

from core.config import REPO_ROOT, config_from_dict, load_config, resolve_path


def test_resolve_path_relative_and_absolute():
    assert resolve_path("assets/samples/x.wav") == REPO_ROOT / "assets/samples/x.wav"
    assert resolve_path("/abs/path.wav") == Path("/abs/path.wav")
    assert resolve_path(None) is None
    assert resolve_path("") is None


def test_defaults_merge_and_job_name_from_filename(tmp_path):
    cfg_file = tmp_path / "my_job.yaml"
    cfg_file.write_text("script:\n  mode: text\n  text: hi\n", encoding="utf-8")
    cfg = load_config(cfg_file)
    assert cfg.name == "my_job"
    assert cfg.voice["engine"] == "chatterbox"          # default preserved
    assert cfg.script["text"] == "hi"                    # override applied


def test_validate_catches_missing_assets(tmp_path):
    cfg = config_from_dict({
        "job": {"name": "t"},
        "script": {"mode": "text", "text": "hello"},
        "voice": {"reference_audio": "nope/missing.wav"},
        "video": {"presenter_clip": "nope/missing.mp4"},
    })
    problems = cfg.validate()
    assert any("reference_audio" in p for p in problems)
    assert any("presenter_clip" in p for p in problems)


def test_validate_empty_text_script():
    cfg = config_from_dict({
        "script": {"mode": "text", "text": "   "},
        "voice": {"reference_audio": "assets/samples/placeholder_voice.wav"},
        "video": {"presenter_clip": "assets/samples/placeholder_presenter.mp4"},
    })
    problems = cfg.validate()
    assert any("script.text is empty" in p for p in problems)


def test_selftest_config_is_valid():
    cfg = load_config(REPO_ROOT / "configs" / "selftest.yaml")
    assert cfg.validate() == []


def test_artifact_paths_are_deterministic(tmp_path):
    cfg = config_from_dict({"job": {"name": "job1", "output_dir": str(tmp_path / "out")}})
    assert cfg.artifact("final").name == "job1.mp4"
    assert cfg.artifact("final").parent == cfg.output_dir
    with pytest.raises(KeyError):
        cfg.artifact("nonsense")
