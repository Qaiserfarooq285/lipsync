"""Config loading and path resolution.

Every path in a job config is interpreted relative to the repo root unless it is
already absolute, so configs stay portable between this machine, Colab and Drive.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULTS: dict[str, Any] = {
    "job": {
        "name": "job",
        "output_dir": "outputs",
    },
    "script": {
        "mode": "text",  # text | file | generate
        "text": "",
        "file": None,
        "generate": {
            "topic": None,
            "backend": "ollama",  # ollama | groq | cerebras
            "model": "qwen2.5:3b",
            "target_words": 90,
            "style": "clear, friendly, spoken-word",
            "temperature": 0.7,
        },
    },
    "voice": {
        "engine": "chatterbox",
        "reference_audio": "assets/samples/placeholder_voice.wav",
        "exaggeration": 0.5,
        "cfg_weight": 0.5,
        "temperature": 0.8,
        "seed": 0,
        "max_chars_per_chunk": 280,
    },
    "video": {
        "engine": "latentsync",
        "presenter_clip": "assets/samples/placeholder_presenter.mp4",
        "inference_steps": 30,
        "guidance_scale": 1.5,
        "resolution": 256,
        "seed": 1247,
        "extend_mode": "pingpong",  # loop | pingpong | freeze
        # DeepCache reuses cached UNet features across nearby denoising steps to
        # speed up inference — upstream's own default. In testing it visibly
        # dampened frame-to-frame mouth-shape variation (repetitive shapes not
        # tracking the audio); off by default here trades speed for lip accuracy.
        "enable_deepcache": False,
    },
    "captions": {
        "enabled": False,
        "mode": "srt",  # srt | burn
        "model": "small",
        "language": "en",
        "verify_against_script": True,
    },
    "runtime": {
        "device": "auto",  # auto | cuda | cpu
        "work_dir": "work",
        "drive_dir": None,
        "keep_intermediates": True,
        "dry_run": False,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def resolve_path(value: str | os.PathLike | None) -> Path | None:
    """Resolve a config path against the repo root. ``None`` passes through."""
    if value is None or value == "":
        return None
    p = Path(os.path.expandvars(str(value))).expanduser()
    return p if p.is_absolute() else (REPO_ROOT / p)


@dataclass
class JobConfig:
    """A single video job, fully resolved."""

    raw: dict[str, Any]
    source_path: Path | None = None
    _resolved: dict[str, Any] = field(default_factory=dict, repr=False)

    # -- section accessors -------------------------------------------------
    @property
    def job(self) -> dict[str, Any]:
        return self.raw["job"]

    @property
    def script(self) -> dict[str, Any]:
        return self.raw["script"]

    @property
    def voice(self) -> dict[str, Any]:
        return self.raw["voice"]

    @property
    def video(self) -> dict[str, Any]:
        return self.raw["video"]

    @property
    def captions(self) -> dict[str, Any]:
        return self.raw["captions"]

    @property
    def runtime(self) -> dict[str, Any]:
        return self.raw["runtime"]

    @property
    def name(self) -> str:
        return str(self.job["name"])

    # -- derived paths -----------------------------------------------------
    @property
    def output_dir(self) -> Path:
        return resolve_path(self.job["output_dir"]) / self.name

    @property
    def work_dir(self) -> Path:
        return resolve_path(self.runtime["work_dir"]) / self.name

    @property
    def drive_dir(self) -> Path | None:
        return resolve_path(self.runtime.get("drive_dir"))

    @property
    def reference_audio(self) -> Path:
        return resolve_path(self.voice["reference_audio"])

    @property
    def presenter_clip(self) -> Path:
        return resolve_path(self.video["presenter_clip"])

    @property
    def script_file(self) -> Path | None:
        return resolve_path(self.script.get("file"))

    # -- artifact paths ----------------------------------------------------
    def artifact(self, kind: str) -> Path:
        """Canonical location for each stage's output."""
        table = {
            "script": self.work_dir / "script.txt",
            "voice": self.work_dir / "voice.wav",
            "video": self.work_dir / "lipsync.mp4",
            "srt": self.output_dir / f"{self.name}.srt",
            "final": self.output_dir / f"{self.name}.mp4",
            "manifest": self.output_dir / "manifest.json",
            "state": self.work_dir / "state.json",
        }
        if kind not in table:
            raise KeyError(f"unknown artifact kind: {kind!r}")
        return table[kind]

    def validate(self) -> list[str]:
        """Return a list of human-readable problems; empty means OK."""
        problems: list[str] = []

        mode = self.script.get("mode")
        if mode not in {"text", "file", "generate"}:
            problems.append(f"script.mode must be text|file|generate, got {mode!r}")
        if mode == "text" and not str(self.script.get("text") or "").strip():
            problems.append("script.mode is 'text' but script.text is empty")
        if mode == "file":
            sf = self.script_file
            if sf is None:
                problems.append("script.mode is 'file' but script.file is unset")
            elif not sf.is_file():
                problems.append(f"script.file not found: {sf}")
        if mode == "generate" and not (self.script.get("generate") or {}).get("topic"):
            problems.append("script.mode is 'generate' but script.generate.topic is unset")

        if not self.reference_audio.is_file():
            problems.append(f"voice.reference_audio not found: {self.reference_audio}")
        if not self.presenter_clip.is_file():
            problems.append(f"video.presenter_clip not found: {self.presenter_clip}")

        if self.video.get("extend_mode") not in {"loop", "pingpong", "freeze"}:
            problems.append("video.extend_mode must be loop|pingpong|freeze")
        if self.captions.get("mode") not in {"srt", "burn"}:
            problems.append("captions.mode must be srt|burn")
        if self.runtime.get("device") not in {"auto", "cuda", "cpu"}:
            problems.append("runtime.device must be auto|cuda|cpu")

        return problems


def load_config(path: str | os.PathLike) -> JobConfig:
    """Load a YAML job config, merged over DEFAULTS."""
    p = Path(path)
    if not p.is_absolute():
        p = (REPO_ROOT / p) if not p.exists() else p.resolve()
    if not p.is_file():
        raise FileNotFoundError(f"config not found: {p}")
    with open(p, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config must be a YAML mapping: {p}")
    merged = _deep_merge(DEFAULTS, data)
    merged.setdefault("job", {}).setdefault("name", p.stem)
    if merged["job"].get("name") in (None, "", "job"):
        merged["job"]["name"] = p.stem
    return JobConfig(raw=merged, source_path=p)


def config_from_dict(data: dict[str, Any]) -> JobConfig:
    """Build a JobConfig from an in-memory mapping (used by the batch queue)."""
    return JobConfig(raw=_deep_merge(DEFAULTS, data))
