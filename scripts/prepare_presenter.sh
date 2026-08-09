#!/usr/bin/env bash
# Extract a clean voice reference from a presenter clip, for voice cloning.
#
#   ./scripts/prepare_presenter.sh assets/presenter/clip_01.mp4
#   ./scripts/prepare_presenter.sh assets/presenter/clip_01.mp4 --start 00:00:12 --duration 12
#
# Chatterbox is zero-shot: 10-15 seconds of clean, continuous speech clones a
# voice as well as a minute does. What matters is that the excerpt has no music,
# no second speaker, and no long silences - so pick the window deliberately
# rather than taking whatever is at the head of the clip.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SRC=""; START="00:00:00"; DURATION="12"; OUT="assets/presenter/voice_sample.wav"

while [ $# -gt 0 ]; do
  case "$1" in
    --start)    START="$2"; shift 2 ;;
    --duration) DURATION="$2"; shift 2 ;;
    --out)      OUT="$2"; shift 2 ;;
    -h|--help)  sed -n '2,12p' "$0"; exit 0 ;;
    *)          SRC="$1"; shift ;;
  esac
done

if [ -z "$SRC" ]; then
  echo "usage: $0 <clip> [--start HH:MM:SS] [--duration SECONDS] [--out PATH]" >&2
  exit 1
fi
if [ ! -f "$SRC" ]; then
  echo "error: not found: $SRC" >&2
  exit 1
fi

if ! ffprobe -v error -select_streams a:0 -show_entries stream=codec_type -of csv=p=0 "$SRC" | grep -q audio; then
  echo "error: $SRC has no audio track - pick a clip with the presenter speaking." >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT")"

# 24 kHz mono PCM: Chatterbox's native rate, so nothing is resampled twice.
ffmpeg -y -v error -ss "$START" -t "$DURATION" -i "$SRC" \
  -vn -ac 1 -ar 24000 -c:a pcm_s16le "$OUT"

MEAN="$(ffmpeg -hide_banner -nostats -i "$OUT" -af volumedetect -f null /dev/null 2>&1 \
        | grep mean_volume | sed 's/.*mean_volume: //')"

echo "Wrote $OUT  (${DURATION}s from ${START})"
echo "Level: ${MEAN:-unknown}"
echo
echo "Listen to it before cloning. If you hear music, a second voice, or long"
echo "silence, re-run with a different --start."
