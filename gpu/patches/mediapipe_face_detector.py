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
reaches the UNet — a bug that would not raise an exception, just quietly wreck
lip-sync quality. To sidestep that risk entirely, this detector doesn't try to
replicate InsightFace's index semantics: it computes two eyebrow-cluster
centroids from MediaPipe's face mesh, sorts them by their actual pixel x
position each frame, and writes the smaller-x one into the "left" slots and the
larger-x one into the "right" slots. The result is correct by construction
regardless of camera mirroring or which side is anatomically which.
"""

from __future__ import annotations

import numpy as np

#: MediaPipe canonical face mesh indices, official FACEMESH_LEFT_EYEBROW /
#: FACEMESH_RIGHT_EYEBROW sets (subject-anatomical naming; we re-sort by pixel
#: position below, so which literal set is "left" here doesn't matter).
_EYEBROW_A = [46, 53, 52, 65, 55, 70, 63, 105, 66, 107]
_EYEBROW_B = [276, 283, 282, 295, 285, 300, 293, 334, 296, 336]
_NOSE = [1, 4, 5, 195, 197]

# image_processor.py reads these exact slots out of a 106-length array.
_LEFT_SLOTS = [43, 48, 49, 51, 50]
_RIGHT_SLOTS = [101, 102, 103, 104, 105]
_NOSE_SLOTS = [74, 77, 83, 86]
_ARRAY_LEN = 106


class MediaPipeFaceDetector:
    """Same call contract as LatentSync's InsightFace-backed ``FaceDetector``."""

    def __init__(self, device: str = "cuda"):
        # mediapipe's CPU FaceMesh is fast enough that it's not the pipeline's
        # bottleneck (the diffusion UNet is); no GPU delegate wiring needed.
        import mediapipe as mp

        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,   # sequential per-frame calls -> let it track
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def __call__(self, frame: np.ndarray, threshold: float = 0.5):
        """``frame`` is an RGB uint8 HxWx3 array. Returns ``(bbox, landmarks)``
        or ``(None, None)`` if no face was found — same contract as upstream."""
        h, w = frame.shape[:2]
        result = self._mesh.process(frame)
        if not result.multi_face_landmarks:
            return None, None

        lm = result.multi_face_landmarks[0].landmark
        pts = np.array([(p.x * w, p.y * h) for p in lm], dtype=np.float64)

        centroid_a = pts[_EYEBROW_A].mean(axis=0)
        centroid_b = pts[_EYEBROW_B].mean(axis=0)
        nose = pts[_NOSE].mean(axis=0)

        left, right = (centroid_a, centroid_b) if centroid_a[0] <= centroid_b[0] else (centroid_b, centroid_a)

        landmarks = np.zeros((_ARRAY_LEN, 2), dtype=np.float64)
        landmarks[_LEFT_SLOTS] = left
        landmarks[_RIGHT_SLOTS] = right
        landmarks[_NOSE_SLOTS] = nose

        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)
        bbox = (max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2)))

        return bbox, np.round(landmarks).astype(np.int_)
