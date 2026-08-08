"""Commercially-clean face detector, drop-in for LatentSync's stock InsightFace one.

Why this exists: LatentSync's ``latentsync/utils/face_detector.py`` uses
``insightface.app.FaceAnalysis`` with the ``buffalo_l`` model pack. InsightFace's
*code* is MIT, but the project states its pretrained models are for
**non-commercial research use only** — that fails this project's "every
component's weights must permit commercial use" rule (see docs/licenses.md).
MediaPipe Face Mesh (Apache 2.0, code and weights) provides equivalent landmarks
and is used here instead. ``gpu/latentsync_runner.py`` monkey-patches this class
in before LatentSync's ``ImageProcessor`` is constructed, so no vendored file is
edited.

Runs MediaPipe in a **subprocess**, not in-process. MediaPipe's FaceMesh uses
TFLite's XNNPACK delegate, which builds its own native thread pool; constructing
it in the same process as PyTorch/CUDA (as happens here, inside LatentSync's
inference loop) triggers a real glibc assertion failure in the priority-protected
mutex implementation (``tpp.c:83 __pthread_tpp_change_priority``) — confirmed by
reproducing it directly, including a variant where the process hangs indefinitely
instead of crashing cleanly. Capping OpenMP thread counts only delays the same
failure. The reliable fix is process isolation: ``mediapipe_worker.py`` never
imports torch, so its thread pool never coexists with PyTorch's. See that
module's docstring for the wire protocol.

Matching the interface, not the semantics: downstream code
(``latentsync/utils/image_processor.py::ImageProcessor.affine_transform``) reads
exactly three anchor points out of whatever 106-slot array we return:

    pt_left_eye  = mean(landmark_2d_106[[43, 48, 49, 51, 50]])
    pt_right_eye = mean(landmark_2d_106[101:106])
    pt_nose      = mean(landmark_2d_106[[74, 77, 83, 86]])

and warps the source frame so pt_left_eye lands at the smaller-x template point,
pt_right_eye at the larger-x one, and pt_nose below both. InsightFace's own
left/right naming for those index groups is undocumented outside its source, and
guessing wrong would silently rotate every cropped face 180 degrees before it
reaches the UNet. To sidestep that risk entirely, this detector doesn't try to
replicate InsightFace's index semantics: it computes two eyebrow-cluster
centroids from MediaPipe's face mesh, sorts them by their actual pixel x
position each frame, and writes the smaller-x one into the "left" slots and the
larger-x one into the "right" slots. The result is correct by construction
regardless of camera mirroring or which side is anatomically which. (Verified
directly against the placeholder clip's first frame.)
"""

from __future__ import annotations

import os
import struct
import subprocess
import sys

import numpy as np

_HEADER = struct.Struct("<III")
_BBOX = struct.Struct("<4i")
_LANDMARKS_BYTES = 106 * 2 * 4  # int32


class MediaPipeFaceDetector:
    """Same call contract as LatentSync's InsightFace-backed ``FaceDetector``."""

    def __init__(self, device: str = "cuda"):
        repo_root = _repo_root()
        env = os.environ.copy()
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = os.pathsep.join(p for p in (str(repo_root), existing) if p)

        self._proc = subprocess.Popen(
            [sys.executable, "-m", "gpu.patches.mediapipe_worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,  # let the worker's stderr (mediapipe's own logging) pass through
            cwd=str(repo_root),
            env=env,
        )

    def __call__(self, frame: np.ndarray, threshold: float = 0.5):
        """``frame`` is an RGB uint8 HxWx3 array. Returns ``(bbox, landmarks)``
        or ``(None, None)`` if no face was found — same contract as upstream."""
        if self._proc.poll() is not None:
            raise RuntimeError(
                f"mediapipe worker subprocess exited unexpectedly (code {self._proc.returncode})"
            )

        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        h, w, c = frame.shape

        assert self._proc.stdin is not None and self._proc.stdout is not None
        self._proc.stdin.write(_HEADER.pack(h, w, c))
        self._proc.stdin.write(frame.tobytes())
        self._proc.stdin.flush()

        flag = self._proc.stdout.read(1)
        if not flag:
            raise RuntimeError("mediapipe worker subprocess closed its output unexpectedly")
        if flag == b"\x00":
            return None, None

        landmark_bytes = _read_exact(self._proc.stdout, _LANDMARKS_BYTES)
        bbox_bytes = _read_exact(self._proc.stdout, _BBOX.size)
        landmarks = np.frombuffer(landmark_bytes, dtype=np.int32).reshape(106, 2)
        bbox = _BBOX.unpack(bbox_bytes)
        return bbox, landmarks

    def close(self) -> None:
        proc = getattr(self, "_proc", None)
        if proc is None or proc.poll() is not None:
            return
        try:
            proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    def __del__(self):
        self.close()


def _read_exact(stream, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            raise RuntimeError("mediapipe worker subprocess closed its output mid-response")
        buf.extend(chunk)
    return bytes(buf)


def _repo_root():
    from pathlib import Path

    return Path(__file__).resolve().parent.parent.parent
