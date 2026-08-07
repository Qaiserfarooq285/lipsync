"""Per-job resumable state.

Each stage's relevant inputs are fingerprinted (a short hash of the config
values and upstream fingerprints that affect it) and recorded here. Re-running
a job — after a crash, an interrupted Colab session, or just to tweak one late
setting — skips any stage whose fingerprint still matches and whose output file
still exists, and re-runs everything downstream of the first change.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def fingerprint(data: Any) -> str:
    blob = json.dumps(data, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


class JobState:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._data: dict = {}
        if self.path.is_file():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self._data = {}
        self._data.setdefault("stages", {})

    def is_fresh(self, stage: str, fp: str, artifact: Path | None) -> bool:
        entry = self._data["stages"].get(stage)
        if entry is None or entry.get("fingerprint") != fp:
            return False
        if artifact is not None and not Path(artifact).is_file():
            return False
        return True

    def mark_done(self, stage: str, fp: str, **extra: Any) -> None:
        self._data["stages"][stage] = {"fingerprint": fp, **extra}
        self._save()

    def clear(self, stage: str | None = None) -> None:
        if stage is None:
            self._data["stages"] = {}
        else:
            self._data["stages"].pop(stage, None)
        self._save()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
