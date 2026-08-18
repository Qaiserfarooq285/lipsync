"""HTTP API and static host for the browser GUI.

Run it:
    .venvs/web/bin/python -m web.server            # http://127.0.0.1:8000

Scope note: this binds to localhost and has no authentication, no rate limiting
and no per-user isolation. That is appropriate for the "test locally, then move
to cloud" plan and inappropriate for anything reachable from the internet - see
``docs/hosting.md`` for what has to change first.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from core import consent, ingest
from core.config import REPO_ROOT
from iolib import media
from web import jobs as jobstore

STATIC = Path(__file__).parent / "static"

#: Uploads are transcoded before anything reads them, so this bounds decode work
#: rather than disk. Ten minutes of 4K is well past anything this pipeline should
#: be handed - the render is per-second of *script*, and the clip only needs to
#: cover it.
MAX_UPLOAD_MB = 500
MAX_SCRIPT_CHARS = 5000

app = FastAPI(title="Presenter Video Pipeline")
store = jobstore.JobStore()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC / "index.html").read_text(encoding="utf-8")


@app.post("/api/jobs")
async def create_job(
    video: UploadFile = File(...),
    script: str = Form(...),
    captions: bool = Form(False),
    presenter_name: str = Form(...),
    consent_confirmed: bool = Form(False),
    voice_sample: UploadFile | None = File(None),
) -> dict:
    script = (script or "").strip()
    if not script:
        raise HTTPException(400, "Please paste the transcript you want spoken.")
    if len(script) > MAX_SCRIPT_CHARS:
        raise HTTPException(400, f"Transcript is too long (limit {MAX_SCRIPT_CHARS} characters).")
    if not (presenter_name or "").strip():
        raise HTTPException(400, "Please enter the name of the person in the video.")
    if not consent_confirmed:
        raise HTTPException(
            400,
            "This tool clones a real person's voice and likeness. Confirm you have "
            "their permission before continuing.",
        )

    name = _safe_name(presenter_name)
    job = store.create(name=name, captions=bool(captions))
    job_dir = job.dir()

    raw = job_dir / "upload"
    try:
        size = _save_upload(video, raw)
        if size > MAX_UPLOAD_MB * 1024 * 1024:
            raise HTTPException(413, f"Video is larger than {MAX_UPLOAD_MB} MB.")

        # Normalise before anything else looks at it, so the gate scores the same
        # pixels the model will see - including rotation, which OpenCV ignores.
        clip = job_dir / "presenter.mp4"
        try:
            ingest.normalise_video(raw, clip)
        except media.MediaError as exc:
            raise HTTPException(400, f"That file could not be read as a video: {exc}")

        reference = job_dir / "voice_reference.wav"
        if voice_sample is not None and voice_sample.filename:
            sample_raw = job_dir / "voice_upload"
            _save_upload(voice_sample, sample_raw)
            try:
                ingest.normalise_audio(sample_raw, reference)
            except media.MediaError as exc:
                raise HTTPException(400, f"That voice sample could not be read: {exc}")
        else:
            try:
                ingest.pick_voice_reference(raw, reference)
            except media.MediaError:
                raise HTTPException(
                    400,
                    "That video has no usable audio, so the voice cannot be cloned "
                    "from it. Upload a separate voice sample as well.",
                )

        record = consent.write_job_consent(
            job_dir / "consent.json",
            presenter_name=presenter_name.strip(),
            attested_by="web-ui",
            scope="Generated presenter video produced through the local web UI.",
            voice_cloning=True,
            likeness_synthesis=True,
            source_note=f"uploaded file: {video.filename}",
        )

        jobstore.write_job_config(
            job, script=script, captions=bool(captions),
            presenter_clip=clip, reference_audio=reference, consent_record=record,
        )
    except HTTPException:
        shutil.rmtree(job_dir, ignore_errors=True)
        with store._lock:  # noqa: SLF001 - same module family, no public remove yet
            store._jobs.pop(job.id, None)
        raise

    raw.unlink(missing_ok=True)
    store.submit(job.id)
    return job.public()


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, "No such job.")
    return job.public()


@app.get("/api/jobs")
def list_jobs() -> list[dict]:
    return [j.public() for j in store.all()]


@app.get("/api/jobs/{job_id}/video")
def job_video(job_id: str) -> FileResponse:
    job = store.get(job_id)
    if job is None or job.status is not jobstore.Status.DONE or not job.output:
        raise HTTPException(404, "No finished video for that job.")
    return FileResponse(job.output, media_type="video/mp4",
                        filename=f"{job.name}.mp4")


@app.get("/api/jobs/{job_id}/captions")
def job_captions(job_id: str) -> FileResponse:
    job = store.get(job_id)
    if job is None or job.status is not jobstore.Status.DONE:
        raise HTTPException(404, "No finished job.")
    srt = job.dir() / "out" / job.name / f"{job.name}.srt"
    if not srt.is_file():
        raise HTTPException(404, "This job has no subtitle file.")
    return FileResponse(srt, media_type="text/plain", filename=f"{job.name}.srt")


@app.post("/api/check")
async def check_clip(video: UploadFile = File(...)) -> dict:
    """Score a clip without queueing a render, so users can test before committing."""
    from core import gate

    tmp = REPO_ROOT / "work" / "web" / "_check"
    tmp.mkdir(parents=True, exist_ok=True)
    raw = tmp / "clip"
    try:
        _save_upload(video, raw)
        normalised = tmp / "clip.mp4"
        # Normalise first so the gate scores the pixels the model will see -
        # including rotation, which OpenCV ignores - but take the audio fact
        # from the original, since normalising strips the track by design.
        original_has_audio = ingest.probe(raw).has_audio
        ingest.normalise_video(raw, normalised)
        result = gate.evaluate(normalised, has_audio=original_has_audio)
        return {
            "verdict": result.verdict,
            "reasons": result.reasons,
            "advice": result.advice,
            "measured": result.measured.get("quality", {}),
        }
    except media.MediaError as exc:
        raise HTTPException(400, f"That file could not be read as a video: {exc}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _safe_name(presenter: str) -> str:
    """A filesystem- and shell-safe job name derived from the presenter's name.

    Upstream builds ffmpeg command lines with shell=True, so a name that reaches
    a path must not carry spaces or quotes - this repo already had to work around
    exactly that for a client filename.
    """
    cleaned = "".join(c if c.isalnum() else "_" for c in presenter.strip().lower())
    cleaned = "_".join(filter(None, cleaned.split("_")))[:40]
    return cleaned or "presenter"


def _save_upload(upload: UploadFile, dst: Path) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    with dst.open("wb") as fh:
        while chunk := upload.file.read(1024 * 1024):
            size += len(chunk)
            fh.write(chunk)
    return size


def main() -> int:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
