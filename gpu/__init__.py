"""GPU stages.

Every module here must be importable on a machine with no GPU and no heavy
dependencies installed: all torch/CUDA imports happen *inside* functions, never at
module scope, and nothing calls ``.cuda()`` at import time.

Each stage exposes both:
  * ``run(...)`` — a plain Python function, callable directly from a Colab cell.
  * ``main(argv)`` — a CLI, so the CPU orchestrator can invoke it by subprocess in
    the stage's own isolated environment (see ``core/envs.py``).
"""

__all__ = ["common", "voice_chatterbox", "lipsync_latentsync", "captions_whisperx"]
