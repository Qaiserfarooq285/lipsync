# Placeholder self-test assets

These are **not real presenter media**. They exist so the pipeline can be built and
self-tested end to end before `CONSENT.md` is filled in and real assets arrive.

- `placeholder_presenter.mp4` — copied from `vendor/LatentSync/assets/demo1_video.mp4`
- `placeholder_voice.wav` — copied from `vendor/LatentSync/assets/demo1_audio.wav`

**Provenance & license:** both files are bundled by ByteDance in the
[LatentSync](https://github.com/bytedance/LatentSync) repository (Apache License 2.0,
covering the whole repository including `assets/`) as the canonical demo/test input
for their own `inference.sh`. They are used here for the same purpose: proving the
pipeline produces a valid HD video end to end, without touching a real presenter's
likeness. They are not this project's presenter and are never used for anything beyond
plumbing self-tests.

This directory is exempt from the `CONSENT.md` gate (see `core/consent.py`) precisely
because nothing in it is this project's own presenter — swap in the real presenter's
clip and voice sample only after `CONSENT.md` reads `STATUS: GRANTED`.

Regenerate with `./scripts/make_placeholder_assets.sh` if these are ever removed.
