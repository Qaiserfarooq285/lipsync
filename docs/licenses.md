# Component Licenses & Commercial Status

Every component below was checked for **both** its code license and its **weights**
license. A component is only admissible if the *weights* permit commercial use.

Do not add a dependency to this project without adding a row here first.

Last reviewed: 2026-08-07.

## In use (Phase 1)

| Stage | Component | Code license | Weights license | Commercial? | Notes |
|---|---|---|---|---|---|
| Lip-sync | [bytedance/LatentSync](https://github.com/bytedance/LatentSync) | Apache 2.0 | Apache 2.0 (HF: `ByteDance/LatentSync-1.5`) | ✅ Yes | Phase 1 video engine. Edits mouth region of supplied footage. |
| Lip-sync dep | [Stable Diffusion VAE](https://huggingface.co/stabilityai/sd-vae-ft-mse) | — | MIT | ✅ Yes | Pulled in as part of LatentSync's UNet stack. |
| Lip-sync dep | [Whisper (tiny)](https://github.com/openai/whisper) | MIT | MIT | ✅ Yes | LatentSync uses a Whisper audio encoder for phoneme features. |
| Face detect | [InsightFace / SCRFD](https://github.com/deepinsight/insightface) | MIT (code) | **Non-commercial (see note)** | ⚠️ Conditional | LatentSync's affine-transform step uses `face_alignment`/SCRFD. InsightFace's *pretrained models* are stated as research-only. See "Open issues" below. |
| Face detect (alt) | [face-alignment (1adrianb)](https://github.com/1adrianb/face-alignment) | BSD 3-Clause | BSD 3-Clause | ✅ Yes | Preferred detector path; avoids the InsightFace weights question. |
| Voice clone | [resemble-ai/chatterbox](https://github.com/resemble-ai/chatterbox) | MIT | MIT (HF: `ResembleAI/chatterbox`) | ✅ Yes | Zero-shot voice cloning from a short reference sample. |
| Voice dep | [S3Tokenizer / CosyVoice-derived tokenizer](https://github.com/resemble-ai/chatterbox) | MIT | MIT | ✅ Yes | Bundled within Chatterbox's released weights. |
| Watermark | [resemble-ai/perth](https://github.com/resemble-ai/perth) | Apache 2.0 | Apache 2.0 | ✅ Yes | Chatterbox applies an imperceptible audio watermark by default. Kept on. |
| Captions (opt) | [m-bain/whisperX](https://github.com/m-bain/whisperX) | BSD 4-Clause* | MIT (Whisper), MIT (faster-whisper/CTranslate2) | ✅ Yes | Optional stage, default off. *See note on BSD-4 below. |
| Align (opt) | [wav2vec2 base (torchaudio)](https://pytorch.org/audio) | BSD 2-Clause | MIT / Apache 2.0 depending on checkpoint | ✅ Yes | WhisperX word-alignment model. English default is MIT. |
| Script gen | [ollama/ollama](https://github.com/ollama/ollama) | MIT | model-dependent | ✅ Yes | Optional. Qwen2.5 = Apache 2.0; Llama 3.x = Meta Community License (has conditions — see below). |
| Media I/O | [FFmpeg](https://ffmpeg.org) | LGPL 2.1+ (GPL if built with `--enable-gpl`) | n/a | ✅ Yes | Ubuntu's `ffmpeg` is a GPL build. We only *invoke the binary*, never link it, so our code is unaffected. |
| Runtime | [PyTorch](https://github.com/pytorch/pytorch) | BSD 3-Clause | n/a | ✅ Yes | |

## Available but not enabled

| Stage | Component | Code license | Weights license | Commercial? | Notes |
|---|---|---|---|---|---|
| Face restore | [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) | BSD 3-Clause | BSD 3-Clause | ✅ Yes | **Preferred** sharpener if one is ever needed. Clean license. |
| Face restore | [GFPGAN](https://github.com/TencentARC/GFPGAN) | Apache 2.0 (own code) | Apache 2.0, **but** contains StyleGAN2-derived parts (NVIDIA source-available license) | ⚠️ Conditional | Optional only. LatentSync output is already reasonably sharp. Prefer Real-ESRGAN. Not enabled by default. |
| Phase 2 video | [MeiGen-AI/InfiniteTalk](https://github.com/MeiGen-AI/InfiniteTalk) | Apache 2.0 | Apache 2.0 | ✅ Yes | Needs far more VRAM than available here. See Phase 2 notes in README. |
| Phase 2 base | [Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1) | Apache 2.0 | Apache 2.0 | ✅ Yes | Base model for InfiniteTalk. ~65–80 GB VRAM at 720p full quality. |

## Explicitly barred

| Component | Reason |
|---|---|
| [Rudrabha/Wav2Lip](https://github.com/Rudrabha/Wav2Lip) | Weights and code are **research/non-commercial only**. Never use. |
| [SWivid/F5-TTS](https://github.com/SWivid/F5-TTS) **weights** | Released **CC-BY-NC 4.0** (non-commercial). The code is MIT, but the weights are the blocker. Never use the released checkpoints. |
| XTTS-v2 (Coqui) | Coqui Public Model License is non-commercial. |
| Any model whose weights license is unstated | Treat "unspecified" as "not cleared". |

## Open issues to resolve before commercial go-live

These do not block placeholder self-testing, but a lawyer should sign them off before
real, distributed commercial output.

1. **InsightFace pretrained models.** InsightFace's code is MIT, but the project states
   its pretrained models are for non-commercial research. LatentSync's data pipeline
   can pull SCRFD/`buffalo_l` for face detection. **Mitigation implemented:** the
   wrapper prefers the BSD-licensed `face-alignment` detector and logs a loud warning
   if an InsightFace model is loaded. Verify with `python -m core.pipeline --audit-licenses`.
2. **whisperX BSD 4-Clause.** The original BSD-4 "advertising clause" requires
   attribution in advertising materials. It is not copyleft and does not affect our
   code, but if captions ship in a product, include the attribution notice. Optional
   stage, default off.
3. **FFmpeg GPL build.** Ubuntu's stock `ffmpeg` binary is GPL. We shell out to it as a
   separate process (no linking), which does not impose GPL on this project. If you
   ever bundle FFmpeg into a distributed binary, ship an LGPL build instead.
4. **Ollama model choice.** Qwen2.5 (Apache 2.0) is the safe default. Llama 3.x carries
   the Meta Llama Community License, which adds naming/attribution conditions and a
   700M-MAU threshold. Script generation is optional and CPU-side; output text is not
   a derivative work concern for most uses, but pick Qwen to keep it simple.
5. **Voice cloning consent.** Licensing is necessary but not sufficient. Cloning a real
   person's voice additionally requires that person's consent — see `CONSENT.md` — and
   in some jurisdictions (e.g. Tennessee's ELVIS Act, EU AI Act transparency duties)
   carries statutory obligations including disclosure that media is AI-generated.
6. **Synthetic media disclosure.** The EU AI Act Article 50 requires deepfake content to
   be labelled. Consider burning a disclosure into distributed output. Chatterbox's
   Perth audio watermark is left enabled to support provenance.

## Evaluated and rejected

| Component | Weights license | Commercial? | Outcome |
|---|---|---|---|
| [MuseTalk v1.5](https://github.com/TMElyralab/MuseTalk) | creativeml-openrail-m ("available for any purpose, even commercially") | ⚠️ Yes, with OpenRAIL use-restrictions | Tested as a lighter alternative to LatentSync. **Rejected on quality** — see below. |
| [DWPose](https://github.com/IDEA-Research/DWPose) | Apache 2.0 | ✅ Yes | Only needed by MuseTalk; bypassed via MediaPipe. Not installed. |

MuseTalk was evaluated because it is architecturally different (single-pass VAE
inpainting rather than iterative diffusion), roughly 5x faster, and works at a
resolution closer to this project's native face size. On the same 6s slice of
real client footage it measured materially worse than LatentSync on the defect
that motivated the trial:

    source (real)   23.11% near-black inside the mouth, interior luminance 83.5
    LatentSync 1.5  43.09%                                                55.1
    MuseTalk v1.5   52.37%                                                42.1

Visual inspection agreed: the whole lower face is smeared, with beard detail
lost. The cause is structural — MuseTalk squashes a tall crop (roughly 259x414px
here) into a square 256x256 input, so this footage is both downscaled and
aspect-distorted, where LatentSync works at 512.

Caveat on the comparison: our trial substituted MediaPipe for MuseTalk's
mmpose/dwpose landmarks (gpu/patches/musetalk_landmarks.py) to avoid a fragile
~3 GB MMLab install. The crop geometry was checked against upstream's formula
and the nose anchor tuned, but a run with genuine dwpose landmarks could differ.
The resolution disadvantage is independent of that and is the dominant factor.

Note also that creativeml-openrail-m is not a plain permissive licence: it
permits commercial use but attaches use-based restrictions. That would need
reading in full before shipping anything built on it.

## Engine selection for this presenter (settled)

Every alternative below was rendered on the same 6s slice of the real footage and
measured with `scripts/lipsync_metrics.py`, then judged by eye. The eye overruled
the metric twice, so both are recorded.

| Option | near-black | interior lum | verdict |
|---|---|---|---|
| source (real footage) | 23.11% | 83.5 | the target |
| **LatentSync 1.6 @512, guidance 1.5** | 43.09% | 55.1 | **chosen** |
| LatentSync 1.5 @256 | 40.72% | 65.0 | rejected - softness destroys lip structure |
| MuseTalk v1.5 @256 | 52.37% | 42.1 | rejected - smears the whole lower face |

Also tried and refuted, so they need not be retried: raising `guidance_scale`
(more lip travel but deeper voids, 33% -> 42%), inference steps and seed (noise),
selecting a different take (33 windows across all three clips span only
20.4-34.1% near-black), 2x source upscale (the apparent gain was lanczos
lightening dark pixels), and a mouth-void lift post-process (turns the black
hole grey rather than into teeth; kept as opt-in in `gpu/enhance_mouth.py`).

The residual gap is the footage, not the engine. This presenter's mouth interior
measures 23% near-black where LatentSync's own demo subject measures 10%, and
the engine's penalty on its own demo clip is only +1.33 points. Lighting that
reaches inside the mouth, and tighter framing, would move output quality further
than any remaining software lever. Use `python -m core.check_footage <clip>` to
score takes before committing to one.

## Lighting is the dominant factor (controlled test)

Run to settle whether the residual dark-mouth artefact is the engine or the
footage. Identical engine, settings, voice and script; only the source clip
differs. LatentSync 1.5 @256 was used because 512 did not fit the free VRAM at
the time - what matters is that both clips used the same configuration.

| clip | source near-black | output near-black | engine penalty | output flicker |
|---|---|---|---|---|
| Ajay (mouth interior in shadow) | 23.11% | 35.79% | **+12.68 pts** | 15.15 |
| LatentSync demo1 (well lit) | 5.91% | **9.20%** | **+3.29 pts** | **4.78** |

The well-lit clip takes roughly a quarter of the damage and ends with almost no
void. Lighting also drives the other two complaints: the well-lit output keeps
far more lip travel (-10.4% against its source, versus -30.8% for Ajay) and
flickers about three times less.

So the pipeline is not the limiting factor for this presenter, and no further
parameter work will close the gap. Fill light that actually reaches inside the
mouth is worth more than every software lever tested combined.

Note: LatentSync's demo1 subject was used here as a lighting reference for an
internal diagnostic only. It is ByteDance's openly-licensed demo asset, and that
render is not a deliverable.
