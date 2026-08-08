"""Standalone MediaPipe FaceMesh worker, run as a subprocess.

Why this is a separate process rather than an in-process call: MediaPipe's
FaceMesh uses TFLite's XNNPACK delegate, which spins up its own native thread
pool. When that pool is built in the same process as PyTorch's OpenMP/MKL
threads and CUDA's driver threads (as happens inside LatentSync's inference
loop), glibc's priority-protected mutex implementation hits a real assertion
failure:

    Fatal glibc error: tpp.c:83 (__pthread_tpp_change_priority): assertion
    failed: new_prio == -1 || (new_prio >= fifo_min_prio && new_prio <= fifo_max_prio)

Capping thread counts via OMP_NUM_THREADS etc. does not reliably avoid it — it
just changes the timing, and in testing the process still ends up stuck. The
robust fix is process isolation: this worker never imports torch, so its
thread pool never coexists with PyTorch's in one process. Communication with
the parent is a tiny synchronous binary protocol over stdin/stdout.

Protocol (little-endian):
  request:  <III  height, width, channels        (12 bytes)
            + height*width*channels raw uint8 RGB bytes
  response: <B    1 if a face was found, else 0   (1 byte)
            if found: 106*2 int32 landmarks + 4 int32 bbox (x1,y1,x2,y2)
"""

from __future__ import annotations

import struct
import sys

import numpy as np

_HEADER = struct.Struct("<III")
_BBOX = struct.Struct("<4i")

# Kept in sync with mediapipe_face_detector.py's contract deliberately —
# see that module's docstring for why these specific slots.
_EYEBROW_A = [46, 53, 52, 65, 55, 70, 63, 105, 66, 107]
_EYEBROW_B = [276, 283, 282, 295, 285, 300, 293, 334, 296, 336]
_NOSE = [1, 4, 5, 195, 197]
_LEFT_SLOTS = [43, 48, 49, 51, 50]
_RIGHT_SLOTS = [101, 102, 103, 104, 105]
_NOSE_SLOTS = [74, 77, 83, 86]
_ARRAY_LEN = 106


def _read_exact(stream, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            return bytes(buf)  # EOF — caller checks length
        buf.extend(chunk)
    return bytes(buf)


def main() -> int:
    import mediapipe as mp

    mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer

    while True:
        header = _read_exact(stdin, _HEADER.size)
        if len(header) < _HEADER.size:
            break  # parent closed the pipe
        h, w, c = _HEADER.unpack(header)
        frame_bytes = _read_exact(stdin, h * w * c)
        if len(frame_bytes) < h * w * c:
            break

        frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(h, w, c)
        result = mesh.process(frame)

        if not result.multi_face_landmarks:
            stdout.write(b"\x00")
            stdout.flush()
            continue

        lm = result.multi_face_landmarks[0].landmark
        pts = np.array([(p.x * w, p.y * h) for p in lm], dtype=np.float64)

        centroid_a = pts[_EYEBROW_A].mean(axis=0)
        centroid_b = pts[_EYEBROW_B].mean(axis=0)
        nose = pts[_NOSE].mean(axis=0)
        left, right = (centroid_a, centroid_b) if centroid_a[0] <= centroid_b[0] else (centroid_b, centroid_a)

        landmarks = np.zeros((_ARRAY_LEN, 2), dtype=np.int32)
        landmarks[_LEFT_SLOTS] = np.round(left).astype(np.int32)
        landmarks[_RIGHT_SLOTS] = np.round(right).astype(np.int32)
        landmarks[_NOSE_SLOTS] = np.round(nose).astype(np.int32)

        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)
        bbox = (max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2)))

        stdout.write(b"\x01")
        stdout.write(landmarks.tobytes())
        stdout.write(_BBOX.pack(*bbox))
        stdout.flush()

    return 0


if __name__ == "__main__":
    sys.exit(main())
