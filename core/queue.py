"""Resumable batch runner: chains core.pipeline.run_job over many jobs.

A failure in one job never aborts the batch by default — it's logged and the
queue moves on, so an unattended overnight run doesn't stall at item 3 of 40.
Re-running the same batch config skips jobs whose final video already exists
(coarse, job-level resumability; core.pipeline adds finer, per-stage
resumability underneath).
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from core.config import REPO_ROOT, JobConfig, config_from_dict, load_config
from core.consent import ConsentError
from core.envs import StageEnvError
from core.pipeline import PipelineError, run_job


class BatchError(RuntimeError):
    pass


def _load_jobs(batch_path: Path) -> list[JobConfig]:
    data = yaml.safe_load(batch_path.read_text(encoding="utf-8")) or {}
    entries = data.get("jobs")
    if not entries:
        raise BatchError(f"{batch_path} has no 'jobs' list")

    jobs: list[JobConfig] = []
    for entry in entries:
        if isinstance(entry, str):
            candidate = batch_path.parent / entry
            path = candidate if candidate.exists() else (REPO_ROOT / entry)
            jobs.append(load_config(path))
        elif isinstance(entry, dict):
            jobs.append(config_from_dict(entry))
        else:
            raise BatchError(f"bad job entry in {batch_path}: {entry!r}")
    return jobs


def run_batch(
    batch_path: Path,
    *,
    force: list[str] | None = None,
    stop_on_error: bool = False,
) -> dict[str, Any]:
    jobs = _load_jobs(Path(batch_path))
    results: list[dict[str, Any]] = []

    print(f"[queue] {len(jobs)} job(s) in {batch_path}", flush=True)
    for i, cfg in enumerate(jobs, 1):
        final_path = cfg.artifact("final")
        if not force and final_path.is_file():
            print(f"[queue] ({i}/{len(jobs)}) {cfg.name}: already done -> {final_path}", flush=True)
            results.append({"job": cfg.name, "status": "skipped", "output": str(final_path)})
            continue

        print(f"[queue] ({i}/{len(jobs)}) {cfg.name}: running", flush=True)
        try:
            out = run_job(cfg, force=force)
            results.append({"job": cfg.name, "status": "ok", "output": str(out)})
        except (PipelineError, ConsentError, StageEnvError) as exc:
            print(f"[queue] ({i}/{len(jobs)}) {cfg.name}: FAILED - {exc}", file=sys.stderr, flush=True)
            results.append({"job": cfg.name, "status": "failed", "error": str(exc)})
            if stop_on_error:
                break
        except Exception as exc:  # noqa: BLE001 - one bad job must not kill the batch
            print(f"[queue] ({i}/{len(jobs)}) {cfg.name}: FAILED (unexpected) - {exc}", file=sys.stderr, flush=True)
            traceback.print_exc()
            results.append({"job": cfg.name, "status": "failed", "error": str(exc)})
            if stop_on_error:
                break

    ok = sum(1 for r in results if r["status"] == "ok")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    failed = sum(1 for r in results if r["status"] == "failed")
    print(f"[queue] done: {ok} ok, {skipped} skipped, {failed} failed", flush=True)

    return {
        "batch": str(batch_path),
        "ran_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "results": results,
        "totals": {"ok": ok, "skipped": skipped, "failed": failed},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run a batch of presenter-video jobs.")
    ap.add_argument("--config", required=True, help="batch YAML with a 'jobs' list")
    ap.add_argument("--force", nargs="*", default=None, choices=["script", "voice", "video", "captions"])
    ap.add_argument("--stop-on-error", action="store_true")
    ap.add_argument("--summary-out", default=None)
    args = ap.parse_args(argv)

    force = args.force
    if force == []:
        force = ["script", "voice", "video", "captions"]

    try:
        summary = run_batch(Path(args.config), force=force, stop_on_error=args.stop_on_error)
    except BatchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.summary_out:
        Path(args.summary_out).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return 1 if summary["totals"]["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
