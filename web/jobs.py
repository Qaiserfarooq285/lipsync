"""Job store and the single worker that owns the GPU.

One worker thread, one job at a time. That is not a simplification to be
revisited later - it is required twice over:

  * 12 GB of VRAM fits exactly one 512px render.
  * LatentSync's ``read_video`` hardcodes ``temp_dir = "temp"`` relative to the
    working directory and ``rmtree``s it on entry (vendor/LatentSync/latentsync/
    utils/util.py). Every render shares that path, so a second concurrent job
    deletes the first job's decoded frames mid-read. Until that is patched,
    parallelism here corrupts output rather than speeding anything up.

Progress comes from parsing the pipeline's own stdout rather than from a timer,
so the bar reflects real work. Stage boundaries are taken from the ``[pipeline]``
lines; within the lip-sync stage, which dominates the runtime, position is read
off upstream's tqdm output.
"""

from __future__ import annotations

import json
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import yaml

from core.config import REPO_ROOT

WEB_ROOT = REPO_ROOT / "work" / "web"


class Status(str, Enum):
    QUEUED = "queued"
    CHECKING = "checking"
    REJECTED = "rejected"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


#: Fraction of the total bar each stage is worth. Weights are rough measurements
#: from real runs, not guesses: lip-sync dominates so completely that giving the
#: other stages more would make the bar sit still for most of the wait.
STAGE_WEIGHTS = (
    ("checking", 0.04),
    ("script", 0.01),
    ("voice", 0.10),
    ("lipsync", 0.78),
    ("captions", 0.03),
    ("assembly", 0.04),
)

#: What to tell the user while each stage runs. The pipeline is slow enough that
#: a static label reads as a hang, so each stage carries several lines that cycle.
STAGE_CHATTER = {
    "checking": [
        "Looking at your footage...",
        "Checking how well the mouth is lit...",
        "Measuring the face...",
    ],
    "script": ["Reading your transcript...", "Getting the words in order..."],
    "voice": [
        "Cloning the voice...",
        "Teaching it your script...",
        "Claude is cooking...",
        "Getting the delivery right...",
    ],
    "lipsync": [
        "Claude is cooking...",
        "Painting the mouth, frame by frame...",
        "This is the slow part - hang on...",
        "Matching lips to every syllable...",
        "Still cooking, this one takes a minute...",
        "Almost there...",
    ],
    "captions": ["Writing the subtitles...", "Lining up the words..."],
    "assembly": ["Putting it together...", "Checking audio and video line up...", "Almost done..."],
}


@dataclass
class Job:
    id: str
    name: str
    status: Status = Status.QUEUED
    stage: str = "queued"
    percent: float = 0.0
    message: str = "Waiting in the queue..."
    error: str | None = None
    gate: dict | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    finished_at: str | None = None
    captions: bool = False
    output: str | None = None
    #: Whether a voice reference was obtained, from the clip's own audio or an
    #: uploaded sample. The driving clip is stripped of audio before the gate
    #: sees it, so the gate cannot work this out for itself.
    has_voice: bool = True
    #: What choose_resolution() actually picked for this job, and why - set
    #: right before rendering starts, so a quality complaint is self-diagnosing
    #: from the job record instead of requiring a log dive.
    resolution: int | None = None
    resolution_note: str | None = None
    #: Wall-clock seconds from submission to completion, set once at the end.
    elapsed_seconds: float | None = None

    def public(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        # The output path is an internal detail; the browser gets a URL.
        d.pop("output", None)
        d["video_url"] = f"/api/jobs/{self.id}/video" if self.status is Status.DONE else None
        d["srt_url"] = (
            f"/api/jobs/{self.id}/captions"
            if self.status is Status.DONE and self.captions and self._srt().is_file()
            else None
        )
        return d

    def dir(self) -> Path:
        return WEB_ROOT / self.id

    def _srt(self) -> Path:
        return self.dir() / "out" / self.name / f"{self.name}.srt"


class JobStore:
    """In-memory job registry with a single background worker.

    Deliberately not a database. Jobs are ephemeral render tasks whose artifacts
    live on disk; persisting their status across a restart would imply the queue
    survives one, which it does not - an interrupted render has to be re-run
    regardless.
    """

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root else WEB_ROOT
        self.root.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._queue: queue.Queue[str] = queue.Queue()
        self._worker = threading.Thread(target=self._run_forever, daemon=True)
        self._worker.start()

    # -- registry ---------------------------------------------------------
    def create(self, name: str, *, captions: bool) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], name=name, captions=captions)
        job.dir().mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def all(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def submit(self, job_id: str) -> None:
        self._queue.put(job_id)
        self._update(job_id, message=self._queue_message())

    def _queue_message(self) -> str:
        waiting = self._queue.qsize()
        if waiting <= 1:
            return "Starting..."
        return f"Waiting in the queue - {waiting - 1} job(s) ahead of you..."

    def _update(self, job_id: str, **fields) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            for k, v in fields.items():
                setattr(job, k, v)

    # -- worker -----------------------------------------------------------
    def _run_forever(self) -> None:
        while True:
            job_id = self._queue.get()
            try:
                self._run_one(job_id)
            except Exception as exc:  # noqa: BLE001 - a bad job must not kill the worker
                self._update(
                    job_id, status=Status.FAILED, error=str(exc),
                    message="Something went wrong.",
                    finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                )
            finally:
                self._queue.task_done()

    def _run_one(self, job_id: str) -> None:
        job = self.get(job_id)
        if job is None:
            return

        # -- 1. score the footage before committing the GPU ----------------
        self._update(job_id, status=Status.CHECKING, stage="checking", percent=1.0,
                     message="Looking at your footage...")

        from core import gate

        clip = job.dir() / "presenter.mp4"
        # has_voice, not the clip's own audio: normalise_video strips the track
        # because the delivered speech comes from the clone, not the footage.
        result = gate.evaluate(clip, has_audio=job.has_voice)
        self._update(job_id, gate={
            "verdict": result.verdict,
            "reasons": result.reasons,
            "advice": result.advice,
            "measured": result.measured.get("quality", {}),
        })

        if result.verdict == gate.REJECT:
            self._update(
                job_id, status=Status.REJECTED, stage="checking", percent=0.0,
                message="This clip will not produce a good result.",
                finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            )
            return

        # -- 2. render ------------------------------------------------------
        self._update(job_id, status=Status.RUNNING, stage="script",
                     percent=self._floor("script"), message="Reading your transcript...")
        self._render(job)

        final = job.dir() / "out" / job.name / f"{job.name}.mp4"
        if not final.is_file():
            raise RuntimeError("the pipeline finished but produced no video")

        # Captions are optional and the pipeline delivers the video regardless
        # of whether they succeeded (core.pipeline treats a captions failure as
        # non-fatal) - so a missing .srt here means "that one add-on didn't
        # work," not "something is wrong," and the message should say so rather
        # than claim an unqualified "Done."
        message = "Done."
        if job.captions and not job._srt().is_file():
            message = "Done - the video is ready, but subtitles could not be generated."

        finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        elapsed = (
            datetime.fromisoformat(finished_at) - datetime.fromisoformat(job.created_at)
        ).total_seconds()

        self._update(
            job_id, status=Status.DONE, stage="done", percent=100.0,
            message=message, output=str(final),
            finished_at=finished_at, elapsed_seconds=round(elapsed, 1),
        )

    # -- progress ---------------------------------------------------------
    @staticmethod
    def _floor(stage: str) -> float:
        total = 0.0
        for name, weight in STAGE_WEIGHTS:
            if name == stage:
                break
            total += weight
        return round(total * 100, 1)

    @staticmethod
    def _span(stage: str) -> float:
        return dict(STAGE_WEIGHTS).get(stage, 0.0) * 100

    def _render(self, job: Job) -> None:
        """Run the pipeline as a subprocess, translating its output into progress."""
        config_path = job.dir() / "job.yaml"

        # Decided now, not at upload time: VRAM on a shared card can change
        # between when a job is submitted and when its turn comes.
        resolution, note = choose_resolution()
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        cfg.setdefault("video", {})["resolution"] = resolution
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        self._update(job.id, resolution=resolution, resolution_note=note)
        if resolution != 512:
            # Not just a log line: this is the fact a "why is it blurry" question
            # needs, so it goes where the browser can already see it.
            self._update(job.id, message=f"Note: {note}")

        cmd = [sys.executable, "-m", "core.pipeline", "--config", str(config_path)]
        proc = subprocess.Popen(
            cmd, cwd=str(REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )

        log = (job.dir() / "pipeline.log").open("w", encoding="utf-8")
        tracker = _ProgressTracker(job.captions)
        tail: list[str] = []
        try:
            assert proc.stdout is not None
            for raw in proc.stdout:
                log.write(raw)
                log.flush()
                tail.append(raw)
                if len(tail) > 400:
                    del tail[:200]
                update = tracker.feed(raw)
                if update:
                    self._update(job.id, **update)
        finally:
            log.close()

        if proc.wait() != 0:
            raise RuntimeError(_friendly_error("".join(tail)))


#: tqdm writes `123/456 [` regardless of bar styling; this catches the counts
#: without depending on the bar characters or the rate suffix.
_TQDM_RE = re.compile(r"(\d+)/(\d+) \[")
_STAGE_RE = re.compile(r"^\[pipeline\] \[[^\]]+\] (.+)$")

#: Maps a pipeline log line to the stage it announces.
_STAGE_MARKERS = (
    ("resolving script", "script"),
    ("synthesising voice", "voice"),
    ("running lip-sync", "lipsync"),
    ("generating captions", "captions"),
    ("assembling final output", "assembly"),
)


class _ProgressTracker:
    """Turns pipeline stdout into (percent, message) updates.

    The lip-sync stage is ~78% of the wall clock and emits two nested tqdm bars:
    an outer one over audio chunks and an inner one over denoising steps. Only
    the outer bar is real progress - the inner one resets constantly. They are
    told apart by structure rather than by hardcoded totals: the outer bar is
    whichever one was running when a *different* bar started underneath it.
    """

    def __init__(self, captions: bool) -> None:
        self.stage = "script"
        self.captions = captions
        self._chatter_at = 0.0
        self._chatter_i = 0
        self._bar_totals: list[int] = []
        self._outer_total: int | None = None
        self._inner_total: int | None = None
        self._percent = JobStore._floor("script")

    def feed(self, line: str) -> dict | None:
        stage_change = self._stage_from(line)
        if stage_change:
            self.stage = stage_change
            self._outer_total = self._inner_total = None
            self._bar_totals.clear()
            self._percent = JobStore._floor(stage_change)
            # Restart the rotation so the new stage opens on its own first line
            # rather than mid-way through the previous stage's list.
            self._chatter_i, self._chatter_at = 0, 0.0
            return {"stage": stage_change, "percent": self._percent,
                    "message": self._next_chatter(force=True)}

        if self.stage == "lipsync":
            update = self._lipsync_progress(line)
            if update:
                return update

        # Even with nothing to report, rotate the message so a long silent
        # stretch does not look like a stall.
        rotated = self._next_chatter()
        return {"message": rotated} if rotated else None

    def _stage_from(self, line: str) -> str | None:
        m = _STAGE_RE.match(line.strip())
        if not m:
            return None
        text = m.group(1)
        for marker, stage in _STAGE_MARKERS:
            if text.startswith(marker):
                return stage
        # "up to date, skipping" - the stage is cached, so jump past it.
        for marker, stage in (("script up", "script"), ("voice up", "voice"),
                              ("lip-sync up", "lipsync"), ("captions up", "captions")):
            if text.startswith(marker):
                return stage
        return None

    def _lipsync_progress(self, line: str) -> dict | None:
        m = _TQDM_RE.search(line)
        if not m:
            return None
        current, total = int(m.group(1)), int(m.group(2))

        # Learn the nesting: the first total seen is the frame-preparation pass,
        # then an outer chunk bar, then an inner denoise bar beneath it.
        if total not in self._bar_totals:
            self._bar_totals.append(total)

        if len(self._bar_totals) >= 3:
            # totals[0] = frame prep, [1] = outer chunk loop, [2] = inner steps
            self._outer_total = self._bar_totals[1]
            self._inner_total = self._bar_totals[2]

        floor, span = JobStore._floor("lipsync"), JobStore._span("lipsync")

        if self._outer_total is not None and total == self._outer_total:
            frac = min(current / max(self._outer_total, 1), 1.0)
            # Frame prep already consumed the first slice of the stage.
            self._percent = round(floor + span * (0.15 + 0.85 * frac), 1)
        elif len(self._bar_totals) == 1:
            frac = min(current / max(total, 1), 1.0)
            self._percent = round(floor + span * 0.15 * frac, 1)
        else:
            return None

        return {"percent": self._percent, "stage": "lipsync",
                "message": self._next_chatter(force=True)}

    #: Seconds a status line stays on screen. tqdm emits dozens of lines a
    #: second, so without this the text would strobe.
    CHATTER_PERIOD = 6.0

    def _next_chatter(self, *, force: bool = False) -> str | None:
        now = time.monotonic()
        lines = STAGE_CHATTER.get(self.stage) or ["Working..."]
        if now - self._chatter_at >= self.CHATTER_PERIOD:
            msg = lines[self._chatter_i % len(lines)]
            self._chatter_i += 1
            self._chatter_at = now
            return msg
        # Forced callers (a stage change, or a real progress step) still need a
        # message to attach to their update, but must not advance the rotation.
        if force:
            return lines[max(self._chatter_i - 1, 0) % len(lines)]
        return None


def _friendly_error(log_tail: str) -> str:
    """Turn a stage traceback into something a non-engineer can act on.

    Falls through to the raw tail rather than swallowing an unrecognised
    failure - an unexplained error the user can paste to us beats a reassuring
    message that hides it.
    """
    patterns = (
        # The preflight check, which fires before any work starts. Distinct from
        # a genuine OOM: nothing is wrong with the upload, the card is simply
        # occupied - usually by something else on the machine entirely.
        ("Not enough free VRAM",
         "The GPU is busy right now, so there was not enough memory to start. "
         "Wait for the current work to finish and try again."),
        ("CUDA out of memory", "The GPU ran out of memory. Try a shorter clip or a shorter script."),
        ("ConsentError", "Consent was not confirmed for this upload."),
        ("no face detected", "No face could be found in that video."),
        ("NO FACE DETECTED", "No face could be found in that video."),
        ("consent record", "Consent was not confirmed for this upload."),
    )
    for needle, friendly in patterns:
        if needle in log_tail:
            return friendly
    lines = [l for l in log_tail.strip().splitlines() if l.strip()]
    return lines[-1][:400] if lines else "The render failed with no output."


#: VRAM each render profile needs, mirrored from gpu/lipsync_latentsync.py's own
#: preflight (``_VRAM_MIN_MB``). Not imported directly: that module lives in the
#: isolated ``lipsync`` venv (torch, diffusers, ...) and this one deliberately
#: does not - importing across that boundary would defeat the isolation the
#: rest of this project relies on. Keep the two in sync by hand.
_VRAM_NEEDED = {512: 9500, 256: 4800}


def choose_resolution() -> tuple[int, str]:
    """Pick a render resolution from currently free VRAM, not a fixed setting.

    This replaces an earlier ``LIPSYNC_WEB_RESOLUTION`` environment variable
    that had to be set by hand and then survived silently across server
    restarts - which is exactly what happened once: a render got forced onto
    the softer 256px profile long after the VRAM pressure that justified it
    was gone, and nothing in the job or the UI said so.
    512 (LatentSync 1.6) is this project's settled quality bar - a measured,
    visible loss of lip-structure detail at 256 (LatentSync 1.5) was already
    tested and rejected - so it is only ever dropped to 256 when 512 will not
    fit *right now*, decided fresh for each job at the moment it is about to
    render, and the reason is recorded on the job rather than assumed.
    """
    from gpu.common import free_vram_mb

    free = free_vram_mb()
    if free is None:
        return 512, "GPU memory could not be checked; using the default quality (512)."
    if free >= _VRAM_NEEDED[512]:
        return 512, f"{free} MiB free - rendering at full quality (512)."
    if free >= _VRAM_NEEDED[256]:
        return 256, (
            f"only {free} MiB free (need ~{_VRAM_NEEDED[512]} MiB for full quality) - "
            "rendering at reduced quality (256); lip detail will be softer than usual."
        )
    return 256, (
        f"only {free} MiB free - attempting reduced quality (256) anyway; it may still fail."
    )


def write_job_config(job: Job, *, script: str, captions: bool, presenter_clip: Path,
                     reference_audio: Path, consent_record: Path) -> Path:
    """Write the YAML the pipeline consumes for this upload.

    Resolution is deliberately absent here: it is decided in :func:`_render`,
    right before the render starts, from VRAM conditions at that moment rather
    than at upload time - see :func:`choose_resolution`.
    """
    out_dir = job.dir() / "out"
    cfg = {
        "job": {
            "name": job.name,
            "output_dir": str(out_dir),
            "consent_record": str(consent_record),
        },
        "script": {"mode": "text", "text": script},
        "voice": {"reference_audio": str(reference_audio)},
        "video": {"presenter_clip": str(presenter_clip)},
        "captions": {"enabled": bool(captions), "mode": "srt"},
        "runtime": {
            "work_dir": str(job.dir() / "work"),
            # The gate already ran before this job was queued, and its verdict is
            # shown to the user. Re-running it inside the pipeline would cost
            # another ten seconds to reach the same answer.
            "gate": "off",
        },
    }
    path = job.dir() / "job.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def cleanup(job: Job) -> None:
    """Remove a job's working files, keeping the delivered video."""
    work = job.dir() / "work"
    if work.is_dir():
        shutil.rmtree(work, ignore_errors=True)
