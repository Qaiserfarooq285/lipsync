"""Lip-sync stage: presenter clip + cloned-voice WAV -> lip-synced MP4.

Engine: bytedance/LatentSync (Apache 2.0 code and weights). Runs in the
``lipsync`` stage environment (Python 3.10, torch 2.5.1) — see requirements/lipsync.txt
and core/envs.py for why it's isolated from the voice stage.

Defaults to **LatentSync 1.5** (256px, ~8 GB VRAM minimum), not 1.6 (512px, ~18 GB
minimum) — this project targets a 12 GB card, and 1.6 does not fit regardless of
what else is using it. Both versions share the same code; only the checkpoint and
the U-Net config's resolution differ (see vendor/LatentSync/docs/changelog_v1.6.md).

Face detection defaults to the MediaPipe-based detector in
``gpu/patches/mediapipe_face_detector.py`` rather than LatentSync's stock
InsightFace one, because InsightFace's pretrained weights are non-commercial
research-only (see docs/licenses.md). Set ``video.face_detector: insightface`` to
opt back into upstream's detector for comparison — never for commercial output.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gpu.common import GpuError, free_vram_mb, pick_device, report_device, require_vram

REPO_ROOT = Path(__file__).resolve().parent.parent
LATENTSYNC_ROOT = REPO_ROOT / "vendor" / "LatentSync"

#: HF repo + filenames per version. Same filenames, different repo/weights.
_CHECKPOINT_REPOS = {
    "1.5": "ByteDance/LatentSync-1.5",
    "1.6": "ByteDance/LatentSync-1.6",
}
_CHECKPOINT_FILES = ["latentsync_unet.pt", "whisper/tiny.pt"]

#: (unet_config, resolution) per version — LatentSync's own pairing.
_VERSION_PROFILE = {
    "1.5": ("configs/unet/stage2.yaml", 256),
    "1.6": ("configs/unet/stage2_512.yaml", 512),
}

#: Minimum free VRAM to attempt each version, per upstream's stated minimums,
#: with headroom since "minimum" in the README means "just barely fits".
_VRAM_MIN_MB = {"1.5": 8500, "1.6": 19000}


def resolve_version(resolution: int) -> str:
    if resolution >= 512:
        return "1.6"
    return "1.5"


def ensure_checkpoints(version: str, *, weights_dir: Path | None = None) -> Path:
    """Download the LatentSync checkpoint pair if not already cached.

    Cached under ``weights/latentsync/<version>/`` (or Drive, via
    ``PRESENTER_WEIGHTS_DIR``) rather than inside ``vendor/`` so re-vendoring
    LatentSync never re-triggers a multi-GB download.
    """
    from huggingface_hub import hf_hub_download

    if version not in _CHECKPOINT_REPOS:
        raise GpuError(f"unknown LatentSync version {version!r}; expected 1.5 or 1.6")

    base = Path(weights_dir) if weights_dir else (REPO_ROOT / "weights")
    ckpt_dir = base / "latentsync" / version
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    repo_id = _CHECKPOINT_REPOS[version]
    for filename in _CHECKPOINT_FILES:
        dest = ckpt_dir / filename
        if dest.is_file():
            continue
        print(f"[lipsync] downloading {repo_id}/{filename} -> {dest}", flush=True)
        hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(ckpt_dir))

    # LatentSync's inference code hardcodes "checkpoints/..." relative to its own
    # CWD (see gpu/latentsync_runner.py) — symlink our cache in rather than copy.
    linked = LATENTSYNC_ROOT / "checkpoints"
    if linked.is_symlink() or linked.is_dir():
        if linked.is_symlink() and linked.resolve() == ckpt_dir.resolve():
            return ckpt_dir
        if linked.is_dir() and not linked.is_symlink():
            raise GpuError(
                f"{linked} exists and is a real directory, not our managed symlink. "
                "Remove it or point weights.dir elsewhere."
            )
    if linked.is_symlink():
        linked.unlink()
    linked.symlink_to(ckpt_dir, target_is_directory=True)

    return ckpt_dir


def run(
    presenter_clip: Path | str,
    voice_audio: Path | str,
    out_path: Path | str,
    *,
    device: str = "auto",
    resolution: int = 256,
    inference_steps: int = 20,
    guidance_scale: float = 1.5,
    seed: int = 1247,
    enable_deepcache: bool = True,
    extend_mode: str = "pingpong",
    face_detector: str = "mediapipe",
    work_dir: Path | None = None,
    weights_dir: Path | None = None,
    skip_vram_check: bool = False,
) -> Path:
    """Produce a lip-synced MP4. Loops/extends the clip if the audio is longer."""
    from iolib import media

    presenter_clip = Path(presenter_clip)
    voice_audio = Path(voice_audio)
    out_path = Path(out_path)
    if not presenter_clip.is_file():
        raise GpuError(f"presenter clip not found: {presenter_clip}")
    if not voice_audio.is_file():
        raise GpuError(f"voice audio not found: {voice_audio}")
    if not LATENTSYNC_ROOT.is_dir():
        raise GpuError(
            f"LatentSync not vendored at {LATENTSYNC_ROOT}. Run "
            "./scripts/vendor_latentsync.sh first."
        )

    device = pick_device(device)
    if device != "cuda":
        raise GpuError(
            "LatentSync's face alignment and diffusion UNet are CUDA-only — there is "
            "no CPU path. Run this stage on a machine with a GPU, or on Colab."
        )
    report_device("lipsync", device)

    version = resolve_version(resolution)
    unet_config_rel, config_resolution = _VERSION_PROFILE[version]
    if resolution != config_resolution:
        print(
            f"[lipsync] video.resolution={resolution} snapped to {config_resolution} "
            f"(LatentSync {version}'s native resolution)", flush=True,
        )

    if skip_vram_check:
        print("[lipsync] VRAM preflight skipped (--skip-vram-check)", flush=True)
    else:
        version_hint = (
            "Lower video.resolution to 256 to use the smaller LatentSync 1.5 profile."
            if version == "1.6" else
            "Free up VRAM (check `nvidia-smi` for other processes), or pass "
            "--skip-vram-check / skip_vram_check=True to try anyway — upstream's "
            "stated minimum has headroom built in and may be conservative for your card."
        )
        require_vram(
            _VRAM_MIN_MB[version],
            stage="lipsync",
            hint=f"LatentSync {version} needs ~{_VRAM_MIN_MB[version] / 1024:.0f} GB free. {version_hint}",
        )

    scratch = Path(work_dir) if work_dir else out_path.parent / "_lipsync_scratch"
    scratch.mkdir(parents=True, exist_ok=True)

    ckpt_dir = ensure_checkpoints(version, weights_dir=weights_dir)
    unet_ckpt = ckpt_dir / "latentsync_unet.pt"
    if not unet_ckpt.is_file():
        raise GpuError(f"checkpoint missing after download attempt: {unet_ckpt}")

    audio_duration = media.probe_duration(voice_audio)
    video_info = media.probe_video(presenter_clip)
    print(
        f"[lipsync] audio={audio_duration:.2f}s clip={video_info.duration:.2f}s "
        f"({video_info.width}x{video_info.height}@{video_info.fps:.2f}fps)", flush=True,
    )

    driving_clip = presenter_clip
    if video_info.duration < audio_duration:
        driving_clip = media.extend_video(
            presenter_clip, scratch / "extended_clip.mp4", audio_duration,
            mode=extend_mode, fps=video_info.fps,
        )
        print(f"[lipsync] extended clip ({extend_mode}) -> {driving_clip}", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    from gpu import latentsync_runner

    runner_args = [
        "--latentsync-root", str(LATENTSYNC_ROOT),
        "--unet-config-path", unet_config_rel,
        "--inference-ckpt-path", str(unet_ckpt.resolve()),
        "--video-path", str(driving_clip.resolve()),
        "--audio-path", str(voice_audio.resolve()),
        "--video-out-path", str(out_path.resolve()),
        "--inference-steps", str(inference_steps),
        "--guidance-scale", str(guidance_scale),
        "--temp-dir", str((scratch / "latentsync_temp").resolve()),
        "--seed", str(seed),
        "--face-detector", face_detector,
    ]
    if enable_deepcache:
        runner_args.append("--enable-deepcache")

    latentsync_runner.main(runner_args)

    if not out_path.is_file():
        raise GpuError(f"LatentSync reported success but {out_path} is missing")

    return out_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="LatentSync lip-sync stage.")
    ap.add_argument("--presenter-clip", required=False)
    ap.add_argument("--voice-audio", required=False)
    ap.add_argument("--out", required=False)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--inference-steps", type=int, default=20)
    ap.add_argument("--guidance-scale", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=1247)
    ap.add_argument("--no-deepcache", action="store_true")
    ap.add_argument("--extend-mode", default="pingpong", choices=["loop", "pingpong", "freeze"])
    ap.add_argument("--face-detector", default="mediapipe", choices=["mediapipe", "insightface"])
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--weights-dir", default=None)
    ap.add_argument(
        "--skip-vram-check", action="store_true",
        help="skip the free-VRAM preflight and let CUDA OOM speak for itself if it doesn't fit",
    )
    ap.add_argument(
        "--download-only", action="store_true",
        help="just fetch checkpoints for --resolution's LatentSync version and exit",
    )
    args = ap.parse_args(argv)

    if args.download_only:
        version = resolve_version(args.resolution)
        ensure_checkpoints(version, weights_dir=Path(args.weights_dir) if args.weights_dir else None)
        print(f"[lipsync] checkpoints ready for LatentSync {version}")
        return 0

    missing = [
        f"--{name}" for name, val in
        (("presenter-clip", args.presenter_clip), ("voice-audio", args.voice_audio), ("out", args.out))
        if not val
    ]
    if missing:
        ap.error(f"the following arguments are required: {', '.join(missing)}")

    run(
        args.presenter_clip,
        args.voice_audio,
        args.out,
        device=args.device,
        resolution=args.resolution,
        inference_steps=args.inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        enable_deepcache=not args.no_deepcache,
        extend_mode=args.extend_mode,
        face_detector=args.face_detector,
        work_dir=Path(args.work_dir) if args.work_dir else None,
        weights_dir=Path(args.weights_dir) if args.weights_dir else None,
        skip_vram_check=args.skip_vram_check,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
