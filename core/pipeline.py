"""Chains script -> voice -> lip-sync -> optional captions for a single job.

CPU-only orchestration: each GPU stage is invoked via ``core.envs.run_stage``,
which runs it as a subprocess inside its own isolated environment. This module
never imports torch.

Resumable: see ``core.state``. Re-running the same job after a crash skips
stages whose inputs haven't changed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from core import envs
from core.config import JobConfig, load_config
from core.consent import ConsentError, assert_allowed
from core.scriptgen import resolve_script, write_script
from core.state import JobState, fingerprint


class PipelineError(RuntimeError):
    pass


def _check_source_quality(cfg: JobConfig) -> None:
    """Score the presenter clip before committing a GPU to it.

    Runs ahead of every stage deliberately. The measurement costs about ten
    seconds against a twenty-minute render, and the failure it predicts - a
    black gap between the lips - is invisible until the render finishes and is
    then unfixable by any setting.
    """
    mode = cfg.runtime.get("gate", "advisory")
    if mode == "off":
        return

    from core import gate

    result = gate.evaluate(cfg.presenter_clip)
    q = result.measured.get("quality") or {}
    detail = ", ".join(
        f"{k} {v}" for k, v in (
            ("mouth near-black", f"{q['near_black_pct']:.1f}%" if q.get("near_black_pct") is not None else None),
            ("face", f"{q['face_frac'] * 100:.0f}% of frame" if q.get("face_frac") is not None else None),
        ) if v
    )
    print(f"[pipeline] [{cfg.name}] source quality: {result.verdict.upper()}"
          + (f" ({detail})" if detail else ""), flush=True)
    for reason in result.reasons:
        print(f"[pipeline]   - {reason}", flush=True)

    if result.verdict == gate.REJECT:
        for advice in result.advice:
            print(f"[pipeline]   > {advice}", flush=True)
        if mode == "strict":
            raise PipelineError(
                f"source clip scored REJECT and runtime.gate is 'strict': "
                f"{'; '.join(result.reasons)}"
            )
        print(
            "[pipeline]   (runtime.gate is 'advisory' - rendering anyway; "
            "set it to 'strict' to refuse)", flush=True,
        )


def run_job(cfg: JobConfig, *, force: list[str] | None = None) -> Path:
    """Run every stage needed to produce ``cfg``'s final video. Returns its path."""
    force = set(force or [])

    problems = cfg.validate()
    if problems:
        raise PipelineError("invalid job config:\n" + "\n".join(f"  - {p}" for p in problems))

    assert_allowed(cfg)
    _check_source_quality(cfg)

    cfg.work_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = cfg.work_dir / "logs"
    state = JobState(cfg.artifact("state"))

    # -- 1. script ----------------------------------------------------------
    script_fp = fingerprint(cfg.script)
    script_path = cfg.artifact("script")
    if "script" in force or not state.is_fresh("script", script_fp, script_path):
        print(f"[pipeline] [{cfg.name}] resolving script", flush=True)
        text = resolve_script(cfg)
        write_script(cfg, text)
        state.mark_done("script", script_fp, words=len(text.split()))
    else:
        print(f"[pipeline] [{cfg.name}] script up to date, skipping", flush=True)
    script_text = script_path.read_text(encoding="utf-8")

    # -- 2. voice -------------------------------------------------------
    voice_fp = fingerprint({
        "script": script_fp,
        "voice": cfg.voice,
        "reference_audio": str(cfg.reference_audio),
    })
    voice_path = cfg.artifact("voice")
    if "voice" in force or not state.is_fresh("voice", voice_fp, voice_path):
        print(f"[pipeline] [{cfg.name}] synthesising voice", flush=True)
        envs.run_stage(
            "voice", "gpu.voice_chatterbox",
            [
                "--text-file", str(script_path),
                "--reference-audio", str(cfg.reference_audio),
                "--out", str(voice_path),
                "--device", cfg.runtime["device"],
                "--exaggeration", str(cfg.voice["exaggeration"]),
                "--cfg-weight", str(cfg.voice["cfg_weight"]),
                "--temperature", str(cfg.voice["temperature"]),
                "--seed", str(cfg.voice["seed"]),
                "--max-chars-per-chunk", str(cfg.voice["max_chars_per_chunk"]),
                "--speed", str(cfg.voice.get("speed", 1.0)),
            ],
            log_path=log_dir / "voice.log",
        )
        state.mark_done("voice", voice_fp)
    else:
        print(f"[pipeline] [{cfg.name}] voice up to date, skipping", flush=True)

    # -- 3. video (lip-sync) ---------------------------------------------
    video_fp = fingerprint({
        "voice": voice_fp,
        "video": cfg.video,
        "presenter_clip": str(cfg.presenter_clip),
    })
    video_path = cfg.artifact("video")
    if "video" in force or not state.is_fresh("video", video_fp, video_path):
        print(f"[pipeline] [{cfg.name}] running lip-sync", flush=True)
        args = [
            "--presenter-clip", str(cfg.presenter_clip),
            "--voice-audio", str(voice_path),
            "--out", str(video_path),
            "--device", cfg.runtime["device"],
            "--resolution", str(cfg.video["resolution"]),
            "--inference-steps", str(cfg.video["inference_steps"]),
            "--guidance-scale", str(cfg.video["guidance_scale"]),
            "--seed", str(cfg.video["seed"]),
            "--extend-mode", cfg.video["extend_mode"],
            "--work-dir", str(cfg.work_dir / "lipsync_scratch"),
        ]
        if not cfg.video.get("enable_deepcache", True):
            args.append("--no-deepcache")
        envs.run_stage("lipsync", "gpu.lipsync_latentsync", args, log_path=log_dir / "lipsync.log")
        state.mark_done("video", video_fp)
    else:
        print(f"[pipeline] [{cfg.name}] lip-sync up to date, skipping", flush=True)

    # -- 4. captions (optional) ------------------------------------------
    srt_path = None
    if cfg.captions.get("enabled"):
        srt_path = cfg.artifact("srt")
        captions_fp = fingerprint({"voice": voice_fp, "captions": cfg.captions})
        if "captions" in force or not state.is_fresh("captions", captions_fp, srt_path):
            print(f"[pipeline] [{cfg.name}] generating captions", flush=True)
            cap_args = [
                "--audio", str(voice_path),
                "--srt-out", str(srt_path),
                "--device", cfg.runtime["device"],
                "--model-size", cfg.captions["model"],
                "--language", cfg.captions["language"],
                "--report-out", str(cfg.work_dir / "captions_report.json"),
            ]
            if cfg.captions.get("verify_against_script"):
                cap_args += ["--script-file", str(script_path)]
            envs.run_stage("captions", "gpu.captions_whisperx", cap_args, log_path=log_dir / "captions.log")
            state.mark_done("captions", captions_fp)
        else:
            print(f"[pipeline] [{cfg.name}] captions up to date, skipping", flush=True)

    # -- 5. assemble ------------------------------------------------------
    from iolib import assembly, drive

    final_path = cfg.artifact("final")
    print(f"[pipeline] [{cfg.name}] assembling final output", flush=True)
    assembly.finalize(
        video_path, voice_path, final_path,
        srt=srt_path, burn=(cfg.captions.get("mode") == "burn"),
        work_dir=cfg.work_dir,
    )

    manifest_path = assembly.write_manifest(
        cfg.artifact("manifest"), cfg=cfg,
        artifacts={
            "script": script_path, "voice": voice_path, "video": video_path,
            "srt": srt_path, "final": final_path,
        },
    )

    drive_dir = drive.resolve_drive_dir(cfg.drive_dir)
    if drive_dir:
        to_sync = [final_path, manifest_path] + ([srt_path] if srt_path else [])
        written = drive.sync_out(to_sync, drive_dir, subdir=cfg.name)
        print(f"[pipeline] [{cfg.name}] synced {len(written)} artifact(s) to {drive_dir}", flush=True)

    print(f"[pipeline] [{cfg.name}] done -> {final_path}", flush=True)
    return final_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run one presenter-video job end to end.")
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--force", nargs="*", default=None, choices=["script", "voice", "video", "captions"],
        help="re-run these stages even if cached output looks fresh (bare --force reruns all)",
    )
    ap.add_argument("--audit-licenses", action="store_true", help="print the license audit and exit")
    args = ap.parse_args(argv)

    if args.audit_licenses:
        from core import doctor
        return doctor.audit_licenses()

    force = args.force
    if force == []:
        force = ["script", "voice", "video", "captions"]

    cfg = load_config(args.config)
    try:
        run_job(cfg, force=force)
    except (PipelineError, ConsentError, envs.StageEnvError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
