"""Run MuseTalk inference without the MMLab stack.

MuseTalk is a candidate replacement for LatentSync on this project because it
fails differently: it is a single-pass VAE latent inpainting model rather than
iterative diffusion, it works at 256x256 (which is close to this footage's
native ~262px face, so nothing is upscaled and then thrown away), and it runs
in roughly real time instead of ~50 minutes per clip. Its weights are
creativeml-openrail-m, which permits commercial use - see docs/licenses.md.

Upstream's ``musetalk/utils/preprocessing.py`` imports mmpose at module scope
for one thing: 68 face landmarks. That would drag in mmengine + mmcv 2.0.1 +
mmdet + mmpose, version-locked to torch 2.0.1 and needing a compiled mmcv - a
fragile ~3 GB install for landmarks we already compute with MediaPipe. This
runner registers ``gpu/patches/musetalk_landmarks.py`` in ``sys.modules`` under
the vendored module's name *before* anything imports it, so the mmpose import
never executes and no vendored file is edited.

The upshot is that MuseTalk runs entirely inside the existing ``lipsync`` stage
environment - no new venv, no extra multi-GB download.
"""

from __future__ import annotations

import argparse
import os
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MUSETALK_ROOT = REPO_ROOT / "vendor" / "MuseTalk"


def _install_landmark_patch() -> None:
    """Pre-register a MediaPipe-backed preprocessing module.

    Must run before ``musetalk.utils.preprocessing`` is imported for real,
    because that module calls ``init_model`` on dwpose at import time - it fails
    at import, not at first use.
    """
    from gpu.patches import musetalk_landmarks

    shim = types.ModuleType("musetalk.utils.preprocessing")
    shim.get_landmark_and_bbox = musetalk_landmarks.get_landmark_and_bbox
    shim.read_imgs = musetalk_landmarks.read_imgs
    shim.coord_placeholder = musetalk_landmarks.COORD_PLACEHOLDER
    sys.modules["musetalk.utils.preprocessing"] = shim
    print("[musetalk] landmarks: mediapipe (mmpose/dwpose bypassed)", flush=True)


def _install_low_memory_loader() -> None:
    """Load the UNet without a fp32 round-trip through VRAM.

    Upstream's ``musetalk/models/unet.py`` calls ``torch.load(model_path)`` with
    no ``map_location``, so on a CUDA box the 3.2 GB checkpoint lands directly in
    VRAM, the model is then materialised there too, and only afterwards is it
    halved. Peak is therefore ~6.4 GB just to end up holding ~1.6 GB - which
    fails outright on a 12 GB card that is sharing with anything else.

    Loading on CPU, converting, and moving the finished fp16 model across peaks
    at roughly a quarter of that and is numerically identical.
    """
    import json

    import torch
    from diffusers import UNet2DConditionModel

    import musetalk.models.unet as unet_mod

    def __init__(self, unet_config, model_path, use_float16=False, device=None):
        with open(unet_config, "r") as fh:
            cfg = json.load(fh)
        self.model = UNet2DConditionModel(**cfg)
        self.pe = unet_mod.PositionalEncoding(d_model=384)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        weights = torch.load(model_path, map_location="cpu")
        self.model.load_state_dict(weights)
        del weights
        # Halve before the transfer, not after, so VRAM never sees fp32.
        self.model = self.model.half().to(self.device)

    unet_mod.UNet.__init__ = __init__
    print("[musetalk] low-memory UNet load (fp16 on CPU, then transfer)", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="MuseTalk inference (vendored, MMLab-free).")
    ap.add_argument("--video-path", required=True)
    ap.add_argument("--audio-path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bbox-shift", type=int, default=0,
                    help="shifts the crop's upper bound; negative moves it up")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--version", default="v15", choices=["v1", "v15"])
    args = ap.parse_args(argv)

    if not MUSETALK_ROOT.is_dir():
        raise SystemExit(f"MuseTalk not vendored at {MUSETALK_ROOT}")

    out_path = Path(os.path.abspath(args.out))
    video_path = os.path.abspath(args.video_path)
    audio_path = os.path.abspath(args.audio_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Upstream resolves "models/..." relative to the CWD, as LatentSync does.
    os.chdir(MUSETALK_ROOT)
    sys.path.insert(0, str(MUSETALK_ROOT))
    _install_landmark_patch()

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("MuseTalk inference needs CUDA")

    # Written as a config file because upstream's entrypoint is config-driven.
    cfg_dir = MUSETALK_ROOT / "configs" / "inference"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / "_generated.yaml"
    cfg_path.write_text(
        "task_0:\n"
        f' video_path: "{video_path}"\n'
        f' audio_path: "{audio_path}"\n'
        f" bbox_shift: {args.bbox_shift}\n",
        encoding="utf-8",
    )

    result_dir = MUSETALK_ROOT / "results" / "generated"
    result_dir.mkdir(parents=True, exist_ok=True)

    _install_low_memory_loader()

    from scripts.inference import main as musetalk_main

    ns = argparse.Namespace(
        inference_config=str(cfg_path),
        result_dir=str(result_dir),
        unet_model_path="models/musetalkV15/unet.pth",
        unet_config="models/musetalkV15/musetalk.json",
        vae_type="sd-vae",
        whisper_dir="models/whisper",
        version=args.version,
        bbox_shift=args.bbox_shift,
        batch_size=args.batch_size,
        fps=args.fps,
        gpu_id=0,
        use_float16=True,
        output_vid_name=None,
        use_saved_coord=False,
        saved_coord=False,
        audio_padding_length_left=2,
        audio_padding_length_right=2,
        extra_margin=10,
        parsing_mode="jaw",
        left_cheek_width=90,
        right_cheek_width=90,
        ffmpeg_path="",
    )
    musetalk_main(ns)

    produced = sorted(result_dir.rglob("*.mp4"), key=lambda p: p.stat().st_mtime)
    if not produced:
        raise SystemExit(f"MuseTalk produced no video under {result_dir}")
    newest = produced[-1]
    out_path.write_bytes(newest.read_bytes())
    print(f"[musetalk] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
