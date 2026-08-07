"""Durable-storage sync.

On Colab this mounts Google Drive; anywhere else ``drive_dir`` is just a
directory on a filesystem that outlives the run (an external disk, a NAS mount,
or nothing at all). Either way the pipeline copies finished artifacts there so an
ephemeral runtime dying mid-batch never loses completed work.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

DEFAULT_COLAB_MOUNT = Path("/content/drive")


def in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
    except ImportError:
        return False
    return True


def mount_drive(mountpoint: Path = DEFAULT_COLAB_MOUNT, *, force: bool = False) -> Path | None:
    """Mount Google Drive if running on Colab. Returns the MyDrive path or None.

    Safe to call on a non-Colab machine — it is a no-op that returns None.
    """
    if not in_colab():
        return None

    my_drive = mountpoint / "MyDrive"
    if my_drive.is_dir() and not force:
        return my_drive

    from google.colab import drive as colab_drive  # type: ignore[import-not-found]

    colab_drive.mount(str(mountpoint), force_remount=force)
    return my_drive if my_drive.is_dir() else None


def resolve_drive_dir(configured: Path | None) -> Path | None:
    """Pick the durable output directory.

    Order: explicit config value, then ``PRESENTER_DRIVE_DIR``, then — on Colab
    only — a default folder inside MyDrive. Returns None when there is nowhere
    durable to sync to, which is a normal, non-fatal state.
    """
    if configured is not None:
        return Path(configured)

    env = os.environ.get("PRESENTER_DRIVE_DIR")
    if env:
        return Path(env).expanduser()

    if in_colab():
        my_drive = mount_drive()
        if my_drive is not None:
            return my_drive / "presenter-video"

    return None


def sync_out(paths: list[Path], drive_dir: Path | None, *, subdir: str = "") -> list[Path]:
    """Copy artifacts to durable storage. Returns the destination paths written."""
    if drive_dir is None:
        return []

    dest_root = Path(drive_dir) / subdir if subdir else Path(drive_dir)
    dest_root.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for src in paths:
        src = Path(src)
        if not src.exists():
            continue
        dest = dest_root / src.name
        # Copy to a temp name then replace, so an interrupted copy never leaves a
        # truncated file that a resumed run would mistake for finished work.
        tmp = dest.with_suffix(dest.suffix + ".partial")
        shutil.copy2(src, tmp)
        os.replace(tmp, dest)
        written.append(dest)
    return written


def cache_dir_for_weights(drive_dir: Path | None) -> Path | None:
    """Where to cache downloaded model weights so a re-provisioned runtime reuses them."""
    if drive_dir is None:
        return None
    d = Path(drive_dir) / "weights"
    d.mkdir(parents=True, exist_ok=True)
    return d
