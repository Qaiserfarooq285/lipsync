"""Captions stage (optional, default off): WhisperX transcription + alignment.

Two jobs in one pass:
  1. Produce word-aligned captions (SRT, and optionally burned into the video).
  2. Verify the synthesised audio actually says what the script asked for — a
     cheap sanity check that the voice stage didn't mumble, truncate, or drift.

Engine: m-bain/whisperX (BSD 4-Clause code; Whisper + faster-whisper/CTranslate2
weights are MIT — see docs/licenses.md). Runs in the ``captions`` stage
environment. Import-safe on CPU: heavy imports are inside functions.
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from gpu.common import GpuError, pick_device, report_device


@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class TranscriptionResult:
    text: str
    words: list[Word] = field(default_factory=list)
    language: str = "en"


def _format_srt_timestamp(seconds: float) -> str:
    seconds = max(0.0, seconds)
    ms = round(seconds * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def words_to_srt(words: list[Word], *, max_words_per_cue: int = 10, max_cue_seconds: float = 4.0) -> str:
    """Group words into readable subtitle cues."""
    if not words:
        return ""

    cues: list[list[Word]] = []
    current: list[Word] = []
    for w in words:
        if current and (
            len(current) >= max_words_per_cue
            or (w.end - current[0].start) > max_cue_seconds
            or re.search(r"[.!?]$", current[-1].text)
        ):
            cues.append(current)
            current = []
        current.append(w)
    if current:
        cues.append(current)

    lines = []
    for i, cue in enumerate(cues, 1):
        start, end = cue[0].start, cue[-1].end
        text = " ".join(w.text for w in cue)
        lines.append(f"{i}\n{_format_srt_timestamp(start)} --> {_format_srt_timestamp(end)}\n{text}\n")
    return "\n".join(lines)


def _normalize_for_compare(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    return text.split()


def verify_against_script(transcript: str, script: str) -> dict:
    """Word-level similarity between what was said and what was scripted.

    Not a strict equality check — TTS engines legitimately vary punctuation
    handling and occasionally drop a filler word. This flags material drift
    (skipped sentences, truncation, garbled output), not cosmetic differences.
    """
    a = _normalize_for_compare(script)
    b = _normalize_for_compare(transcript)
    ratio = difflib.SequenceMatcher(None, a, b).ratio() if a and b else 0.0

    sm = difflib.SequenceMatcher(None, a, b)
    missing = []
    for tag, i1, i2, _, _ in sm.get_opcodes():
        if tag in ("delete", "replace"):
            missing.extend(a[i1:i2])

    return {
        "similarity": round(ratio, 4),
        "script_words": len(a),
        "transcript_words": len(b),
        "missing_words": missing[:50],
        "likely_ok": ratio >= 0.75,
    }


def transcribe(
    audio_path: Path | str,
    *,
    device: str = "auto",
    model_size: str = "small",
    language: str = "en",
) -> TranscriptionResult:
    import whisperx

    audio_path = Path(audio_path)
    if not audio_path.is_file():
        raise GpuError(f"audio not found: {audio_path}")

    device = pick_device(device)
    report_device("captions", device)
    compute_type = "float16" if device == "cuda" else "int8"

    print(f"[captions] loading whisperx '{model_size}' ({compute_type})", flush=True)
    model = whisperx.load_model(model_size, device, compute_type=compute_type, language=language)

    audio = whisperx.load_audio(str(audio_path))
    result = model.transcribe(audio, language=language, batch_size=16)
    full_text = " ".join(seg["text"].strip() for seg in result["segments"]).strip()

    try:
        align_model, meta = whisperx.load_align_model(language_code=result["language"], device=device)
        aligned = whisperx.align(result["segments"], align_model, meta, audio, device, return_char_alignments=False)
        words = [
            Word(text=w["word"].strip(), start=float(w.get("start", 0.0)), end=float(w.get("end", 0.0)))
            for seg in aligned["segments"]
            for w in seg.get("words", [])
            if w.get("start") is not None and w.get("end") is not None
        ]
    except Exception as exc:  # noqa: BLE001 - alignment is best-effort
        print(f"[captions] word alignment unavailable ({exc}); falling back to segment timing", flush=True)
        words = [
            Word(text=seg["text"].strip(), start=float(seg["start"]), end=float(seg["end"]))
            for seg in result["segments"]
        ]

    if device == "cuda":
        import torch

        del model
        torch.cuda.empty_cache()

    return TranscriptionResult(text=full_text, words=words, language=result.get("language", language))


def run(
    audio_path: Path | str,
    srt_out: Path | str,
    *,
    device: str = "auto",
    model_size: str = "small",
    language: str = "en",
    script_text: str | None = None,
) -> dict:
    """Transcribe, write an SRT, and (if ``script_text`` given) verify drift."""
    srt_out = Path(srt_out)
    result = transcribe(audio_path, device=device, model_size=model_size, language=language)

    srt_out.parent.mkdir(parents=True, exist_ok=True)
    srt_out.write_text(words_to_srt(result.words), encoding="utf-8")
    print(f"[captions] wrote {srt_out} ({len(result.words)} words)", flush=True)

    report = {"srt": str(srt_out), "transcript": result.text, "language": result.language}
    if script_text is not None:
        verification = verify_against_script(result.text, script_text)
        report["verification"] = verification
        status = "OK" if verification["likely_ok"] else "DRIFT DETECTED"
        print(
            f"[captions] script match: {status} (similarity={verification['similarity']})",
            flush=True,
        )
        if not verification["likely_ok"]:
            print(
                f"[captions]   missing/garbled words (sample): {verification['missing_words'][:15]}",
                flush=True,
            )
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="WhisperX captions + script verification.")
    ap.add_argument("--audio", required=True)
    ap.add_argument("--srt-out", required=True)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--model-size", default="small")
    ap.add_argument("--language", default="en")
    ap.add_argument("--script-file", default=None, help="verify transcript against this script")
    ap.add_argument("--report-out", default=None, help="write the JSON report here")
    args = ap.parse_args(argv)

    script_text = Path(args.script_file).read_text(encoding="utf-8") if args.script_file else None
    report = run(
        args.audio, args.srt_out,
        device=args.device, model_size=args.model_size, language=args.language,
        script_text=script_text,
    )

    if args.report_out:
        import json

        Path(args.report_out).write_text(json.dumps(report, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    sys.exit(main())
