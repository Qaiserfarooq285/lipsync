# Hosting

What exists, and what has to change before this faces anyone but you.

Last reviewed: 2026-08-17.

## What runs today

```bash
./scripts/setup_envs.sh web
.venvs/web/bin/python -m web.server        # http://127.0.0.1:8000
```

Upload a clip and a transcript, get a video. The browser page carries the
shooting guidance, a captions toggle, a consent confirmation, an optional
pre-flight footage check, and a progress bar driven by the pipeline's real
output rather than a timer.

| Piece | File |
|---|---|
| API + static host | `web/server.py` |
| Job registry, worker, progress parsing | `web/jobs.py` |
| Browser UI (self-contained, no CDN) | `web/static/index.html` |
| Source-quality scoring | `core/gate.py` |
| Upload normalisation | `core/ingest.py` |

### Request flow

```
upload ─► normalise (fps/rotation/loudness)  ~2s
       ─► extract voice reference             ~4s
       ─► write per-job consent record
       ─► queue
            └─► score the footage             ~10s   ◄── reject here, not after
                └─► render                    minutes
                    └─► assemble + A/V check
```

The score runs *before* the GPU is committed. That ordering is the point: the
defect that draws complaints is a black gap between the lips, it is invisible
until a render finishes, and no setting fixes it afterwards. Ten seconds of CPU
answers it in advance.

## Blockers, in the order they will bite

### 1. Concurrent renders corrupt each other

Hard blocker for more than one GPU worker.

```python
# vendor/LatentSync/latentsync/utils/util.py:48
temp_dir = "temp"
if os.path.exists(temp_dir):
    shutil.rmtree(temp_dir)
```

Relative to the working directory, and `gpu/latentsync_runner.py` `chdir`s into
the vendor checkout — so every render shares `vendor/LatentSync/temp` and wipes
it on entry. A second concurrent job deletes the first's decoded frames
mid-read. The `--temp-dir` this repo already passes goes to a *different*
function and never reaches this one.

`web/jobs.py` therefore runs exactly one job at a time. That is also what 12 GB
of VRAM allows, so today the two limits coincide — but on a bigger card the
temp directory is what stops you scaling, not memory. Fix it by patching
`read_video` to take a caller-supplied directory before adding a second worker.

### 2. No authentication, no isolation, no limits

`web/server.py` binds to `127.0.0.1` and assumes a trusted operator. Before it
is reachable from anywhere else it needs, at minimum: authentication, per-user
job namespacing, a disk quota, and rate limiting. Uploads are currently capped
only by `MAX_UPLOAD_MB`, and finished videos are served to anyone who knows the
job id — the ids are random, which is obscurity, not access control.

### 3. Jobs do not survive a restart

`JobStore` is in-memory by design: an interrupted render has to re-run anyway,
so persisting status would promise a durability the queue does not have. A
multi-user service wants the opposite — a real queue (Redis, or Postgres with
`SKIP LOCKED`) so a deploy does not silently drop in-flight work.

### 4. Consent is per-job now, but not per-account

`core/consent.py` accepts a per-job JSON record (`job.consent_record`), which is
what made web uploads possible at all — `CONSENT.md` is a single global
declaration for one presenter, and falling back to it would let one person's
permission authorise a different person's upload. The record is written from a
checkbox, which is an attestation, not proof. A commercial service needs the
signed document referenced in `docs/licenses.md` issue 5, retained per presenter
and revocable.

### 5. Pacing is still unsolved

Chatterbox delivers 246–269 wpm against a comfortable presenter band of 125–165.
`atempo` cannot close that gap: reaching 165 needs a 0.61 factor, and stretching
past about 0.88 damages the render (0.75 cost 6.6 points of mouth darkness when
measured). `core/ingest.suggest_speed` derives and caps a factor, so the output
is no longer hand-tuned, but it is capped precisely because the real fix is
elsewhere: inserting pauses between sentences, which adds duration without
touching speech rate. Not yet built.

## Moving to a rented GPU

Nothing in the pipeline assumes local paths — configs resolve against the repo
root and stage envs are provisioned by script — so the move is mostly packaging:

1. Bake the three stage venvs and `weights/` into an image. Provisioning at boot
   costs several GB of downloads per cold start.
2. Fix blocker 1 before running more than one worker per container.
3. Move `work/web/` to attached storage or object storage; it currently holds
   uploads, intermediates and outputs together under the repo.
4. Keep the CPU gate close to the user and the GPU workers behind the queue. The
   gate is the cheap half and rejects work before it costs GPU-hours, which is
   the main lever on cost per delivered video.

## Calibration debt

The gate's thresholds rest on three clips (see `core/gate.py`). They separate
the known-good from the known-bad correctly, but the boundary between them is
interpolated across a gap from 6% to 23% near-black. `gate.record_outcome()`
appends source-score/human-rating pairs to JSONL; once enough real uploads
accumulate, refit the thresholds instead of trusting the current guess.
