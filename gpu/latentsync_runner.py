"""In-process LatentSync inference, adapted from ``vendor/LatentSync/scripts/inference.py``.

Run this with the working directory set to the LatentSync checkout — upstream
resolves ``configs`` and ``checkpoints/...`` relative to the CWD.

We use our own entrypoint rather than upstream's so we can swap the face detector
before the pipeline imports it. LatentSync's stock detector is InsightFace, whose
pretrained models are licensed for non-commercial research only; this project is
commercial, so the default here is a MediaPipe-based detector (Apache 2.0).

Upstream is Apache 2.0 (c) 2024 Bytedance Ltd.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _install_detector(kind: str) -> None:
    """Choose which face detector LatentSync's ImageProcessor will construct."""
    if kind == "insightface":
        print(
            "[lipsync] WARNING: using the InsightFace detector. Its pretrained models "
            "are licensed for NON-COMMERCIAL research use only. Do not ship commercial "
            "output produced this way — see docs/licenses.md.",
            file=sys.stderr, flush=True,
        )
        return

    if kind != "mediapipe":
        raise ValueError(f"unknown face detector: {kind!r}")

    # Import the module first so the attribute we patch is the one ImageProcessor
    # will look up, then replace it before latentsync.utils.image_processor is
    # imported (it does `from .face_detector import FaceDetector`).
    import latentsync.utils.face_detector as fd_mod

    from gpu.patches.mediapipe_face_detector import MediaPipeFaceDetector

    fd_mod.FaceDetector = MediaPipeFaceDetector  # type: ignore[misc]
    print("[lipsync] face detector: mediapipe (Apache 2.0)", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="LatentSync inference (vendored).")
    ap.add_argument("--latentsync-root", required=True)
    ap.add_argument("--unet-config-path", required=True)
    ap.add_argument("--inference-ckpt-path", required=True)
    ap.add_argument("--video-path", required=True)
    ap.add_argument("--audio-path", required=True)
    ap.add_argument("--video-out-path", required=True)
    ap.add_argument("--inference-steps", type=int, default=20)
    ap.add_argument("--guidance-scale", type=float, default=1.5)
    ap.add_argument("--temp-dir", default="temp")
    ap.add_argument("--seed", type=int, default=1247)
    ap.add_argument("--enable-deepcache", action="store_true")
    ap.add_argument(
        "--low-vram", action="store_true",
        help="enable VAE/attention slicing so the 512px profile fits on a smaller card",
    )
    ap.add_argument("--face-detector", default="mediapipe",
                    choices=["mediapipe", "insightface"])
    args = ap.parse_args(argv)

    root = Path(args.latentsync_root).resolve()
    if not (root / "latentsync").is_dir():
        raise SystemExit(f"not a LatentSync checkout: {root}")

    # Upstream reads "configs" and "checkpoints/whisper/tiny.pt" relative to CWD.
    os.chdir(root)
    sys.path.insert(0, str(root))

    _install_detector(args.face_detector)

    import torch
    from accelerate.utils import set_seed
    from DeepCache import DeepCacheSDHelper
    from diffusers import AutoencoderKL, DDIMScheduler
    from omegaconf import OmegaConf

    from latentsync.models.unet import UNet3DConditionModel
    from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
    from latentsync.whisper.audio2feature import Audio2Feature

    config = OmegaConf.load(args.unet_config_path)

    if not torch.cuda.is_available():
        raise SystemExit(
            "LatentSync requires CUDA — its pipeline and face detection are GPU-only.\n"
            "  Run this stage on a GPU machine or a Colab T4 (see notebooks/)."
        )

    # fp16 needs compute capability > 7 (Ampere and later qualify).
    is_fp16_supported = torch.cuda.get_device_capability()[0] > 7
    dtype = torch.float16 if is_fp16_supported else torch.float32
    print(f"[lipsync] dtype={dtype}", flush=True)

    scheduler = DDIMScheduler.from_pretrained("configs")

    if config.model.cross_attention_dim == 768:
        whisper_model_path = "checkpoints/whisper/small.pt"
    elif config.model.cross_attention_dim == 384:
        whisper_model_path = "checkpoints/whisper/tiny.pt"
    else:
        raise SystemExit("cross_attention_dim must be 768 or 384")

    audio_encoder = Audio2Feature(
        model_path=whisper_model_path,
        device="cuda",
        num_frames=config.data.num_frames,
        audio_feat_length=config.data.audio_feat_length,
    )

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=dtype)
    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0

    unet, _ = UNet3DConditionModel.from_pretrained(
        OmegaConf.to_container(config.model),
        args.inference_ckpt_path,
        device="cpu",
    )
    unet = unet.to(dtype=dtype)

    pipeline = LipsyncPipeline(
        vae=vae, audio_encoder=audio_encoder, unet=unet, scheduler=scheduler
    ).to("cuda")

    if args.low_vram:
        # Upstream's stated VRAM minimums assume none of these are on. Together
        # they are what makes the 512px profile fit on a 12 GB card at all.
        #
        # VAE slicing: decode_latents() decodes a whole 16-frame chunk in one
        # call; at 512px that single decode is the largest transient allocation
        # in the run. Slicing decodes frame-by-frame instead.
        pipeline.enable_vae_slicing()
        # Attention slicing is a UNet-side win, but LatentSync's UNet3DConditionModel
        # is a custom model that may not implement the diffusers hook - treat it
        # as best-effort rather than letting an AttributeError kill the run.
        try:
            pipeline.enable_attention_slicing(1)
            print("[lipsync] low-vram: vae slicing + attention slicing", flush=True)
        except (AttributeError, NotImplementedError, ValueError) as exc:
            print(f"[lipsync] low-vram: vae slicing only (attention slicing unavailable: {exc})", flush=True)

    if args.enable_deepcache:
        helper = DeepCacheSDHelper(pipe=pipeline)
        helper.set_params(cache_interval=3, cache_branch_id=0)
        helper.enable()

    if args.seed != -1:
        set_seed(args.seed)
    else:
        torch.seed()
    print(f"[lipsync] seed={torch.initial_seed()}", flush=True)

    pipeline(
        video_path=args.video_path,
        audio_path=args.audio_path,
        video_out_path=args.video_out_path,
        num_frames=config.data.num_frames,
        num_inference_steps=args.inference_steps,
        guidance_scale=args.guidance_scale,
        weight_dtype=dtype,
        width=config.data.resolution,
        height=config.data.resolution,
        mask_image_path=config.data.mask_image_path,
        temp_dir=args.temp_dir,
    )
    print(f"[lipsync] wrote {args.video_out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
