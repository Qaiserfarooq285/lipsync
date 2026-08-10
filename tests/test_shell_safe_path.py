"""LatentSync interpolates the clip path into an ffmpeg command and runs it with
shell=True, unquoted (vendor/LatentSync/latentsync/utils/util.py read_video), so
a path containing a space is split into separate arguments. Real client footage
routinely has spaces in the filename, so this is not an edge case."""

import subprocess
from pathlib import Path

import pytest

from gpu.lipsync_latentsync import _SHELL_UNSAFE, _shell_safe_clip


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not a real video")
    return path


def test_clean_path_is_passed_through_untouched(tmp_path):
    src = _touch(tmp_path / "clip.mp4")
    assert _shell_safe_clip(src, tmp_path / "scratch") == src


@pytest.mark.parametrize("name", [
    "dr ajay avtar 5 c50.mp4",
    "clip(final).mp4",
    "clip's take.mp4",
    "clip;rm.mp4",
])
def test_unsafe_names_are_staged_under_a_safe_one(tmp_path, name):
    src = _touch(tmp_path / name)
    scratch = tmp_path / "scratch"
    staged = _shell_safe_clip(src, scratch)

    assert staged != src
    assert not (set(str(staged)) & _SHELL_UNSAFE)
    assert staged.read_bytes() == b"not a real video"


def test_staged_path_survives_an_unquoted_shell_command(tmp_path):
    """The whole point: the staged path must not break when dropped into a
    shell command without quoting, the way upstream does it."""
    src = _touch(tmp_path / "dr ajay avtar 5 c50.mp4")
    staged = _shell_safe_clip(src, tmp_path / "scratch")

    proc = subprocess.run(f"cat {staged}", shell=True, capture_output=True)
    assert proc.returncode == 0
    assert proc.stdout == b"not a real video"


def test_resolving_the_staged_path_would_undo_the_fix(tmp_path):
    """Regression guard. The staged path is a symlink, so .resolve() follows it
    back to the original spacey name - which is exactly the bug that made the
    first version of this fix fail in a real run."""
    src = _touch(tmp_path / "dr ajay avtar 5 c50.mp4")
    staged = _shell_safe_clip(src, tmp_path / "scratch")

    assert not (set(str(staged)) & _SHELL_UNSAFE)
    if staged.is_symlink():
        assert set(str(staged.resolve())) & _SHELL_UNSAFE


def test_returns_absolute_paths_even_when_given_relative_ones(tmp_path, monkeypatch):
    """gpu/latentsync_runner.py chdir()s into the vendored checkout before
    ffmpeg ever sees this path, so anything relative silently stops resolving.
    A relative --work-dir on the CLI produced exactly that failure in a real run.
    """
    monkeypatch.chdir(tmp_path)
    _touch(tmp_path / "clips" / "dr ajay avtar 5 c50.mp4")

    staged = _shell_safe_clip(
        Path("clips/dr ajay avtar 5 c50.mp4"), Path("work/scratch")
    )
    assert staged.is_absolute()
    assert not (set(str(staged)) & _SHELL_UNSAFE)

    # And it must still be readable once the process has moved elsewhere.
    monkeypatch.chdir(tmp_path.parent)
    assert staged.read_bytes() == b"not a real video"


def test_clean_relative_path_is_made_absolute_too(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _touch(tmp_path / "clip.mp4")
    staged = _shell_safe_clip(Path("clip.mp4"), Path("scratch"))
    assert staged.is_absolute()


def test_restaging_is_idempotent(tmp_path):
    src = _touch(tmp_path / "a b.mp4")
    scratch = tmp_path / "scratch"
    first = _shell_safe_clip(src, scratch)
    second = _shell_safe_clip(src, scratch)
    assert first == second
    assert second.read_bytes() == b"not a real video"
