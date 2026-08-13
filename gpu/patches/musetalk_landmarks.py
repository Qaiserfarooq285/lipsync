"""MediaPipe replacement for MuseTalk's mmpose/dwpose landmark step.

MuseTalk's ``musetalk/utils/preprocessing.py`` imports mmpose at module scope
purely to obtain 68 face landmarks (COCO-WholeBody keypoints 23:91, which follow
the classic dlib 68-point layout) plus a face bounding box from face_alignment.
Nothing else in the inference path needs the MMLab stack.

Pulling that stack in would mean mmengine + mmcv 2.0.1 + mmdet + mmpose, all
version-locked to torch 2.0.1 and requiring a compiled mmcv - a fragile ~3 GB
install, for landmarks this project already computes another way. MediaPipe
FaceMesh is already installed, is Apache 2.0, and gives a denser mesh than the
68 points being asked for.

So this module reimplements the one function the pipeline calls,
``get_landmark_and_bbox``, and ``gpu/musetalk_runner.py`` injects it before
MuseTalk's preprocessing module is imported. No vendored file is edited.

Only four of the 68 landmarks actually matter to the caller:

    face_land_mark[28], [29], [30]  nose-bridge points. [29] is the "half face"
                                    anchor; the [28]-[30] spread sets how far
                                    above it the crop starts.
    min/max over all 68             horizontal extent and the chin line.

The returned box is ``(x1, upper_bond, x2, y2)`` where ``upper_bond`` is derived
by mirroring the chin distance above the nose anchor - that is what centres the
mouth in MuseTalk's 256x256 crop, so the mapping to MediaPipe indices below is
chosen to reproduce those specific points rather than the whole contour.
"""

from __future__ import annotations

import numpy as np

#: MediaPipe FaceMesh indices approximating dlib-68 nose-bridge points 28/29/30.
#:
#: The anchor is landmark 29, and its height decides everything: upstream sets
#: the crop's top edge by mirroring the nose-to-chin distance above it, so an
#: anchor placed too high produces a tall box that then gets squashed into
#: MuseTalk's square 256x256 input - blurring and distorting the result. In
#: dlib-68 point 29 sits low on the bridge, just above the tip, so these map to
#: the lower bridge (195/5) rather than the upper (6/197).
_NOSE_BRIDGE = {28: 197, 29: 195, 30: 5}

#: Face oval, used for the horizontal extent and the chin line that the original
#: takes as min/max over all 68 landmarks.
_FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
    379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
    234, 127, 162, 21, 54, 103, 67, 109,
]

COORD_PLACEHOLDER = (0.0, 0.0, 0.0, 0.0)


class _Mesh:
    """Lazily-created FaceMesh, reused across frames."""

    _inst = None

    @classmethod
    def get(cls):
        if cls._inst is None:
            import mediapipe as mp

            cls._inst = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False, max_num_faces=1, refine_landmarks=False,
                min_detection_confidence=0.5, min_tracking_confidence=0.5,
            )
        return cls._inst


def _landmarks_for(frame):
    """Return (68x2 int array, ok). Indices other than those the caller reads
    are filled with the face-oval points so min/max still describe the face."""
    import cv2

    h, w = frame.shape[:2]
    res = _Mesh.get().process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:
        return None, False

    lm = res.multi_face_landmarks[0].landmark
    oval = np.array([[lm[i].x * w, lm[i].y * h] for i in _FACE_OVAL])

    pts = np.zeros((68, 2), dtype=np.float64)
    # Spread the oval across the array so min/max match the real face extent.
    for i in range(68):
        pts[i] = oval[i % len(oval)]
    for dlib_idx, mp_idx in _NOSE_BRIDGE.items():
        pts[dlib_idx] = [lm[mp_idx].x * w, lm[mp_idx].y * h]

    return pts.astype(np.int32), True


def get_landmark_and_bbox(img_list, upperbondrange=0):
    """Drop-in for MuseTalk's mmpose-backed version.

    Mirrors upstream's geometry exactly - including the fallback when the
    landmark-derived box comes out degenerate - so downstream cropping behaves
    identically. ``img_list`` may be paths or already-decoded frames.
    """
    import cv2

    frames = []
    for item in img_list:
        frames.append(cv2.imread(item) if isinstance(item, str) else item)

    coords_list = []
    range_minus, range_plus = [], []

    for frame in frames:
        if frame is None:
            coords_list.append(COORD_PLACEHOLDER)
            continue

        face_land_mark, ok = _landmarks_for(frame)
        if not ok:
            coords_list.append(COORD_PLACEHOLDER)
            continue

        half_face_coord = face_land_mark[29].astype(np.int32).copy()
        range_minus.append((face_land_mark[30] - face_land_mark[29])[1])
        range_plus.append((face_land_mark[29] - face_land_mark[28])[1])
        if upperbondrange != 0:
            half_face_coord[1] = upperbondrange + half_face_coord[1]

        half_face_dist = np.max(face_land_mark[:, 1]) - half_face_coord[1]
        upper_bond = max(0, half_face_coord[1] - half_face_dist)

        x1 = int(np.min(face_land_mark[:, 0]))
        x2 = int(np.max(face_land_mark[:, 0]))
        y1, y2 = int(upper_bond), int(np.max(face_land_mark[:, 1]))

        if y2 - y1 <= 0 or x2 - x1 <= 0 or x1 < 0:
            h, w = frame.shape[:2]
            coords_list.append((max(0, x1), max(0, y1), min(w, x2), min(h, y2)))
        else:
            coords_list.append((x1, y1, x2, y2))

    if range_minus and range_plus:
        print(f"[musetalk] bbox_shift suggested range: "
              f"[-{int(np.mean(range_minus))} ~ {int(np.mean(range_plus))}], "
              f"current: {upperbondrange}", flush=True)

    return coords_list, frames


def read_imgs(img_list):
    """Upstream exports this from the same module; keep the name available."""
    import cv2

    return [cv2.imread(p) for p in img_list]
