"""Voice-clone stage: script text + reference sample -> cloned-voice WAV.

Engine: resemble-ai/chatterbox (MIT code, MIT weights). Zero-shot — it needs only
a few seconds of clean reference audio, no per-speaker training.

Runs in the ``voice`` stage environment (torch 2.6.0). Import-safe on CPU: all
heavy imports are inside the functions.

Chatterbox applies Resemble's Perth audio watermark to its output by default. That
is left ON deliberately: it is inaudible, and provenance marking is the right
default for synthetic speech (and increasingly a legal expectation).
"""

from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

from gpu.common import GpuError, pick_device, report_device, require_vram, set_seed

#: Silence inserted between synthesised chunks, in seconds. Long enough to read
#: as a sentence boundary, short enough not to sound like a dropout.
CHUNK_GAP_S = 0.28

#: Chatterbox in fp32 needs roughly this much VRAM with headroom for the vocoder.
VRAM_NEEDED_MB = 3000


def _supported_kwargs(fn, candidates: dict) -> dict:
    """Pass only the generation kwargs this Chatterbox version actually accepts."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return {}
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return candidates
    return {k: v for k, v in candidates.items() if k in params}


def run(
    text: str,
    reference_audio: Path | str,
    out_path: Path | str,
    *,
    device: str = "auto",
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    temperature: float = 0.8,
    seed: int = 0,
    max_chars_per_chunk: int = 280,
) -> Path:
    """Synthesise ``text`` in the voice of ``reference_audio``; return the WAV path.

    Long scripts are synthesised in sentence-aligned chunks and concatenated —
    Chatterbox's quality degrades on very long single inputs.
    """
    import torch
    import torchaudio

    from chatterbox.tts import ChatterboxTTS

    from core.scriptgen import chunk_for_tts

    reference_audio = Path(reference_audio)
    out_path = Path(out_path)
    if not reference_audio.is_file():
        raise GpuError(f"reference audio not found: {reference_audio}")
    text = (text or "").strip()
    if not text:
        raise GpuError("refusing to synthesise empty text")

    device = pick_device(device)
    report_device("voice", device)
    if device == "cuda":
        require_vram(
            VRAM_NEEDED_MB,
            stage="voice",
            hint="Free VRAM, or set runtime.device=cpu (slow but works).",
        )
    set_seed(seed)

    print(f"[voice] loading chatterbox onto {device}", flush=True)
    model = ChatterboxTTS.from_pretrained(device=device)

    chunks = chunk_for_tts(text, max_chars=max_chars_per_chunk)
    print(f"[voice] synthesising {len(chunks)} chunk(s), {len(text)} chars", flush=True)

    gap = torch.zeros(1, int(model.sr * CHUNK_GAP_S))
    pieces: list[torch.Tensor] = []

    for i, chunk in enumerate(chunks, 1):
        kwargs = _supported_kwargs(model.generate, {
            "audio_prompt_path": str(reference_audio),
            "exaggeration": exaggeration,
            "cfg_weight": cfg_weight,
            "temperature": temperature,
        })
        print(f"[voice]   chunk {i}/{len(chunks)} ({len(chunk)} chars)", flush=True)
        wav = model.generate(chunk, **kwargs)

        wav = wav.detach().to("cpu").float()
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        elif wav.ndim > 2:
            wav = wav.reshape(1, -1)
        if wav.shape[0] > 1:          # collapse any stereo output to mono
            wav = wav.mean(dim=0, keepdim=True)

        pieces.append(wav)
        if i < len(chunks):
            pieces.append(gap)

    audio = torch.cat(pieces, dim=1)

    # Guard against clipping introduced by concatenation.
    peak = audio.abs().max().item()
    if peak > 0.99:
        audio = audio * (0.98 / peak)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out_path), audio, model.sr)

    seconds = audio.shape[1] / model.sr
    print(f"[voice] wrote {out_path} ({seconds:.2f}s @ {model.sr} Hz)", flush=True)

    if device == "cuda":
        del model
        torch.cuda.empty_cache()

    return out_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Chatterbox voice cloning.")
    ap.add_argument("--text-file", required=True, help="UTF-8 file holding the script")
    ap.add_argument("--reference-audio", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--exaggeration", type=float, default=0.5)
    ap.add_argument("--cfg-weight", type=float, default=0.5)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-chars-per-chunk", type=int, default=280)
    args = ap.parse_args(argv)

    text = Path(args.text_file).read_text(encoding="utf-8")
    run(
        text,
        args.reference_audio,
        args.out,
        device=args.device,
        exaggeration=args.exaggeration,
        cfg_weight=args.cfg_weight,
        temperature=args.temperature,
        seed=args.seed,
        max_chars_per_chunk=args.max_chars_per_chunk,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
