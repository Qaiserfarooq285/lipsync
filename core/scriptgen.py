"""Script acquisition and normalisation. CPU-only, runs anywhere.

Three modes:

``text``      the script is inline in the job config (the default).
``file``      the script is a UTF-8 text file on disk.
``generate``  draft one with a local Ollama model or a free hosted API.

Whatever the source, the text is normalised for speech before it reaches the TTS
stage: markdown stripped, smart punctuation folded to ASCII, whitespace collapsed.
A voice model reads "**bold**" literally if you let it.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import textwrap
import urllib.error
import urllib.request
from pathlib import Path

from core.config import JobConfig, load_config

MAX_SCRIPT_CHARS = 20000


class ScriptError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------

_SMART = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"',
    "–": "-", "—": " - ", "…": "...",
    " ": " ", "​": "",
}


def normalize_for_speech(text: str) -> str:
    """Turn arbitrary written text into something a TTS model reads cleanly."""
    if not text:
        return ""

    for bad, good in _SMART.items():
        text = text.replace(bad, good)

    # Markdown that would otherwise be spoken aloud.
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)   # headings
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)             # code fences
    text = re.sub(r"`([^`]*)`", r"\1", text)                            # inline code
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)                   # images
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)                # links -> label
    text = re.sub(r"(\*\*|__)(.+?)\1", r"\2", text, flags=re.DOTALL)    # bold
    text = re.sub(r"(?<!\w)([*_])(?!\s)(.+?)(?<!\s)\1(?!\w)", r"\2", text, flags=re.DOTALL)
    text = re.sub(r"^\s{0,3}>\s?", "", text, flags=re.MULTILINE)        # blockquotes
    text = re.sub(r"^\s{0,3}[-*+]\s+", "", text, flags=re.MULTILINE)    # bullets
    text = re.sub(r"^\s{0,3}\d+[.)]\s+", "", text, flags=re.MULTILINE)  # numbered
    text = re.sub(r"^\s*[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)   # rules

    # Stage directions the writer left in — never spoken.
    text = re.sub(r"\[(?:pause|beat|smile|laughs?|music|sfx)[^\]]*\]", " ", text, flags=re.I)
    text = re.sub(r"\((?:pause|beat|smile|laughs?)\)", " ", text, flags=re.I)

    # Collapse whitespace but keep paragraph boundaries as sentence breaks.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    text = re.sub(r"\n+", " ", text)
    return text.strip()


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def chunk_for_tts(text: str, max_chars: int = 280) -> list[str]:
    """Split into TTS-sized chunks on sentence boundaries.

    Chatterbox degrades on very long inputs, so long scripts are synthesised in
    pieces and concatenated. Splitting on sentences keeps prosody natural; a
    single sentence longer than ``max_chars`` is split on commas, then hard-wrapped.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    current = ""
    for sentence in _SENT_SPLIT.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            parts, buf = [], ""
            for piece in re.split(r"(?<=,)\s+", sentence):
                if len(buf) + len(piece) + 1 <= max_chars:
                    buf = f"{buf} {piece}".strip()
                else:
                    if buf:
                        parts.append(buf)
                    buf = piece if len(piece) <= max_chars else ""
                    if not buf:
                        parts.extend(textwrap.wrap(piece, max_chars))
            if buf:
                parts.append(buf)
            chunks.extend(parts)
            continue

        if len(current) + len(sentence) + 1 <= max_chars:
            current = f"{current} {sentence}".strip()
        else:
            if current:
                chunks.append(current)
            current = sentence

    if current:
        chunks.append(current)
    return [c for c in chunks if c.strip()]


def estimate_duration(text: str, wpm: int = 150) -> float:
    """Rough spoken length in seconds. Used to pre-extend the presenter clip."""
    words = len(text.split())
    return (words / max(wpm, 1)) * 60.0


# --------------------------------------------------------------------------
# Generation backends (all optional, all free)
# --------------------------------------------------------------------------

PROMPT = """You are writing a short script to be read aloud by a single presenter \
on camera. Write {words} words (plus or minus 15) about: {topic}

Rules:
- Plain spoken prose only. No markdown, no headings, no bullet points.
- No stage directions, speaker labels, or narration about the video.
- Short, declarative sentences that are easy to say out loud.
- Style: {style}
- Output only the script text, nothing else."""


def _generate_ollama(topic: str, model: str, words: int, style: str, temperature: float) -> str:
    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    payload = json.dumps({
        "model": model,
        "prompt": PROMPT.format(topic=topic, words=words, style=style),
        "stream": False,
        "options": {"temperature": temperature},
    }).encode()
    req = urllib.request.Request(
        f"{host}/api/generate", data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            return json.loads(resp.read())["response"]
    except urllib.error.URLError as exc:
        raise ScriptError(
            f"Could not reach Ollama at {host} ({exc}).\n"
            f"  Start it with `ollama serve` and `ollama pull {model}`, or switch the "
            "job to script.mode=text."
        ) from exc


def _generate_openai_compatible(
    topic: str, model: str, words: int, style: str, temperature: float,
    *, base_url: str, api_key_env: str, label: str,
) -> str:
    key = os.environ.get(api_key_env)
    if not key:
        raise ScriptError(
            f"{label} selected but ${api_key_env} is not set. "
            f"Both offer a free tier; put the key in .env or switch to script.mode=text."
        )
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": PROMPT.format(topic=topic, words=words, style=style)}],
        "temperature": temperature,
    }).encode()
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            body = json.loads(resp.read())
        return body["choices"][0]["message"]["content"]
    except (urllib.error.URLError, KeyError, IndexError) as exc:
        raise ScriptError(f"{label} request failed: {exc}") from exc


def generate(spec: dict) -> str:
    topic = spec.get("topic")
    if not topic:
        raise ScriptError("script.generate.topic is required for script.mode=generate")

    backend = (spec.get("backend") or "ollama").lower()
    model = spec.get("model") or "qwen2.5:3b"
    words = int(spec.get("target_words") or 90)
    style = spec.get("style") or "clear, friendly, spoken-word"
    temp = float(spec.get("temperature", 0.7))

    if backend == "ollama":
        return _generate_ollama(topic, model, words, style, temp)
    if backend == "groq":
        return _generate_openai_compatible(
            topic, model, words, style, temp,
            base_url="https://api.groq.com/openai/v1",
            api_key_env="GROQ_API_KEY", label="Groq",
        )
    if backend == "cerebras":
        return _generate_openai_compatible(
            topic, model, words, style, temp,
            base_url="https://api.cerebras.ai/v1",
            api_key_env="CEREBRAS_API_KEY", label="Cerebras",
        )
    raise ScriptError(f"unknown script.generate.backend: {backend!r}")


def ollama_available(model: str | None = None) -> bool:
    """True if an Ollama daemon is reachable (and has ``model``, if given)."""
    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=5) as resp:
            tags = json.loads(resp.read())
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return False
    if model is None:
        return True
    names = {m.get("name", "") for m in tags.get("models", [])}
    return any(n == model or n.startswith(f"{model.split(':')[0]}:") for n in names)


def pull_ollama_model(model: str) -> bool:
    """Best-effort `ollama pull`. Returns True on success."""
    try:
        proc = subprocess.run(["ollama", "pull", model], timeout=3600)
        return proc.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def resolve_script(cfg: JobConfig) -> str:
    """Produce the final, speech-ready script text for a job."""
    spec = cfg.script
    mode = spec.get("mode", "text")

    if mode == "text":
        raw = str(spec.get("text") or "")
    elif mode == "file":
        path = cfg.script_file
        if path is None or not path.is_file():
            raise ScriptError(f"script.file not found: {path}")
        raw = path.read_text(encoding="utf-8")
    elif mode == "generate":
        raw = generate(spec.get("generate") or {})
    else:
        raise ScriptError(f"unknown script.mode: {mode!r}")

    text = normalize_for_speech(raw)
    if not text:
        raise ScriptError(f"script resolved to empty text (mode={mode})")
    if len(text) > MAX_SCRIPT_CHARS:
        raise ScriptError(
            f"script is {len(text)} chars, over the {MAX_SCRIPT_CHARS} limit. "
            "Split it into several jobs — LatentSync renders in real-time-ish and a "
            "very long script will take hours."
        )
    return text


def write_script(cfg: JobConfig, text: str) -> Path:
    dst = cfg.artifact("script")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text + "\n", encoding="utf-8")
    return dst


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Resolve a job's script to normalised text.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--print", action="store_true", help="print to stdout instead of writing")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    text = resolve_script(cfg)
    if args.print:
        print(text)
    else:
        path = write_script(cfg, text)
        est = estimate_duration(text)
        print(f"wrote {path} ({len(text.split())} words, ~{est:.1f}s spoken)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
