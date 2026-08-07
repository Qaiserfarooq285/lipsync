# Presenter Video Pipeline

Turns a written script into a finished HD talking-head video of a chosen presenter,
in that presenter's cloned voice, with no manual editing per video and batch support.
Free / open-source, commercially-licensed components only. See `CLAUDE.md` for the
full build brief and `docs/licenses.md` for the per-component license audit.

```
script text -> Chatterbox (voice clone) -> LatentSync (lip-sync) -> optional WhisperX captions -> HD MP4
```

## Status

Phase 1 (LatentSync + Chatterbox). Self-tested end to end on the openly-licensed
placeholders in `assets/samples/` — **no real presenter media has been used.**
`CONSENT.md` gates that; see below.

## This machine vs. the original brief

`CLAUDE.md` was written assuming a Mac M2 with all GPU work on Colab. This box is
Linux x86_64 with a local **NVIDIA RTX A2000 (12 GB)**, so GPU stages run locally by
default; the Colab notebooks in `notebooks/` remain as a portable fallback and work
unmodified. 12 GB is below a free Colab T4's 16 GB, which is why the lip-sync stage
defaults to **LatentSync 1.5** (256px, ~8 GB VRAM) rather than 1.6 (512px, ~18 GB —
does not fit either GPU).

## Environments

The GPU stages have mutually incompatible pinned dependencies (Chatterbox wants
`torch==2.6.0`/`transformers==5.2.0`; LatentSync wants `torch==2.5.1`/`transformers==4.48.0`;
LatentSync also needs Python 3.10 specifically because `mediapipe==0.10.11` has no
Python 3.12 wheel). Each stage gets its own venv under `.venvs/`, driven by the CPU
orchestrator via subprocess (`core/envs.py`). On Colab, which has a single runtime,
each notebook instead installs one stage's requirements directly into that runtime.

```
.venv/          orchestrator (core/, iolib/) — CPU only, Python 3.12
.venvs/voice/    Chatterbox — Python 3.12, torch 2.6.0
.venvs/lipsync/  LatentSync — Python 3.10, torch 2.5.1
.venvs/captions/ WhisperX — Python 3.12, optional
```

## Setup

```bash
./scripts/setup_envs.sh              # orchestrator + voice + lipsync (~8 GB of downloads)
./scripts/setup_envs.sh captions     # add the optional captions env
./scripts/make_placeholder_assets.sh # (re)populate assets/samples/ if needed
python -m core.doctor                # verify everything is wired up
```

`setup_envs.sh` uses `uv` to provision Python 3.10 and 3.12 interpreters if they
aren't already on the system, vendors LatentSync at a pinned commit into
`vendor/LatentSync/`, and installs each stage's `requirements/*.txt` into its own venv.

First run of the lip-sync stage downloads the LatentSync 1.5 checkpoint (~2 GB) from
Hugging Face into `weights/latentsync/1.5/` (or fetch it up front):

```bash
PYTHONPATH=. .venvs/lipsync/bin/python -m gpu.lipsync_latentsync --download-only --resolution 256
```

## Commands

```bash
# Self-test on placeholders — no consent needed, proves the pipeline end to end.
python -m core.pipeline --config configs/selftest.yaml

# A real job (after filling in configs/example_job.yaml and CONSENT.md).
python -m core.pipeline --config configs/<job>.yaml

# Batch — resumable, keeps going if one job fails.
python -m core.queue --config configs/batch.yaml

# Environment / license / consent check.
python -m core.doctor
```

Each job is resumable at the stage level (`core/state.py`): re-running after a crash
skips script/voice/lip-sync/caption steps whose inputs haven't changed. Pass
`--force` (optionally with specific stage names) to re-run regardless.

## Consent gate

`CONSENT.md` blocks any job pointed at real presenter media until it's filled in and
its status reads `STATUS: GRANTED`. Everything under `assets/samples/` is exempt —
it holds only openly-licensed placeholders (see `assets/samples/README.md`), never a
real, identifiable person. `core/consent.py` enforces this before any job runs;
`python -m core.doctor` reports current status.

## Licensing

Every component's *weights* license — not just its code license — was checked for
commercial permission before inclusion; see `docs/licenses.md` for the full table and
open issues. Two decisions worth knowing up front:

- **Face detector defaults to MediaPipe, not InsightFace.** LatentSync's stock face
  detector uses InsightFace's `buffalo_l` pretrained models, which are licensed for
  non-commercial research only. `gpu/patches/mediapipe_face_detector.py` is a
  drop-in, commercially-clean replacement (Apache 2.0) used by default; InsightFace
  remains available as an explicit opt-in (`video.face_detector: insightface`) for
  comparison only — never for real commercial output.
- **LatentSync 1.5, not 1.6, by default.** Same code, different checkpoint and
  resolution; 1.6 needs more VRAM than either the local card or a free Colab T4
  provides. Set `video.resolution: 512` to opt into 1.6 on a bigger GPU.

## Repo layout

See `CLAUDE.md` for the full rationale. One deliberate deviation: the I/O helper
package is `iolib/`, not `io/` — a top-level package named `io` is unimportable
because CPython binds `sys.modules['io']` to the standard library during interpreter
startup, before any user code runs.

## Tests

```bash
.venv/bin/pytest tests/ -v
```

Covers config loading/validation, the consent gate, script normalisation/chunking,
media probing/extension (via ffmpeg), and resumable job state. GPU stages aren't
unit-tested (no CI GPU); the self-test config is the end-to-end check.
