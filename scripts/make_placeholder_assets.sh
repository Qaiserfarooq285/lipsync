#!/usr/bin/env bash
# Populate assets/samples/ with openly-licensed self-test placeholders.
#
# Source: vendor/LatentSync/assets/demo1_{video,audio} — ByteDance's own demo/test
# fixture, Apache 2.0, bundled specifically for exercising LatentSync's inference
# path. Using it here means the self-test never touches a real, identifiable
# person's likeness. See assets/samples/README.md.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SRC="vendor/LatentSync/assets"
DEST="assets/samples"

if [ ! -f "$SRC/demo1_video.mp4" ] || [ ! -f "$SRC/demo1_audio.wav" ]; then
  echo "error: $SRC/demo1_video.mp4 / demo1_audio.wav not found." >&2
  echo "Run ./scripts/vendor_latentsync.sh first." >&2
  exit 1
fi

mkdir -p "$DEST"
cp "$SRC/demo1_video.mp4" "$DEST/placeholder_presenter.mp4"
cp "$SRC/demo1_audio.wav" "$DEST/placeholder_voice.wav"

echo "Wrote $DEST/placeholder_presenter.mp4 and $DEST/placeholder_voice.wav"
