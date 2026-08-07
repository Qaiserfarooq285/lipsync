"""Final output assembly and the per-job manifest."""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from iolib import media


def _git_rev(repo: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def finalize(
    lipsync_video: Path,
    voice_audio: Path,
    dst: Path,
    *,
    target_height: int | None = 1080,
    srt: Path | None = None,
    burn: bool = False,
    work_dir: Path | None = None,
) -> Path:
    """Produce the delivered MP4 from the lip-synced render.

    LatentSync already muxes the driving audio into its output, but it re-encodes
    at the model's working resolution. This upscales to HD, re-attaches the
    full-quality audio, and optionally burns in subtitles.
    """
    lipsync_video, voice_audio, dst = Path(lipsync_video), Path(voice_audio), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(work_dir) if work_dir else dst.parent
    scratch.mkdir(parents=True, exist_ok=True)

    stage = lipsync_video
    if target_height:
        info = media.probe_video(lipsync_video)
        if info.height < target_height:
            stage = media.to_hd(stage, scratch / "hd.mp4", height=target_height)

    # Re-attach the cloned voice at full quality rather than trusting the
    # model's internal remux.
    muxed = media.mux(stage, voice_audio, scratch / "muxed.mp4", copy_video=True)

    if srt is not None and burn and Path(srt).is_file():
        media.burn_subtitles(muxed, Path(srt), dst)
    else:
        if dst.exists():
            dst.unlink()
        muxed.replace(dst)

    return dst


def write_manifest(
    path: Path,
    *,
    cfg: Any,
    artifacts: dict[str, Path | None],
    extra: dict[str, Any] | None = None,
) -> Path:
    """Record what was produced, from what, and with which settings."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    final = artifacts.get("final")
    info = None
    if final and Path(final).is_file():
        try:
            v = media.probe_video(Path(final))
            info = {
                "width": v.width, "height": v.height,
                "fps": round(v.fps, 3), "duration_s": round(v.duration, 3),
                "n_frames": v.n_frames, "has_audio": v.has_audio,
            }
        except media.MediaError:
            info = None

    from core.config import REPO_ROOT

    doc = {
        "job": cfg.name,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "revisions": {
            "repo": _git_rev(REPO_ROOT),
            "latentsync": _git_rev(REPO_ROOT / "vendor" / "LatentSync"),
        },
        "inputs": {
            "presenter_clip": str(cfg.presenter_clip),
            "reference_audio": str(cfg.reference_audio),
        },
        "settings": {
            "voice": cfg.voice,
            "video": cfg.video,
            "captions": cfg.captions,
        },
        "artifacts": {k: (str(v) if v else None) for k, v in artifacts.items()},
        "output": info,
    }
    if extra:
        doc.update(extra)

    path.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    return path
