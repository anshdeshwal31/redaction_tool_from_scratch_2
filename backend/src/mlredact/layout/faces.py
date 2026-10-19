"""Faces (OpenCV Zoo YuNet, MIT) - photographs of people, ID-card photos (plan §7.1 VISUAL)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from mlredact.core.canonical import content_id
from mlredact.core.geometry import BBox
from mlredact.core.model import Region
from mlredact.core.types import RegionKind
from mlredact.imaging.ink import U8

# Detection runs on a copy whose long side is at most this; faces in documents (ID photos, clinical
# photographs) are large at 300 dpi, so this bounds cost without losing them.
_MAX_SIDE = 1600
_PAD = 0.35  # hair, ears, chin: the box covers the head, not just the facial landmarks


class FaceDetector:
    def __init__(self, model: Path, score_threshold: float) -> None:
        self._model = str(model)
        self._threshold = score_threshold

    def find(self, rgb: U8, page: int) -> list[Region]:
        h, w = rgb.shape[:2]
        scale = min(1.0, _MAX_SIDE / max(h, w))
        small = cv2.resize(rgb, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
        bgr = np.ascontiguousarray(small[:, :, ::-1])
        detector = cv2.FaceDetectorYN.create(self._model, "", (bgr.shape[1], bgr.shape[0]), self._threshold, 0.3, 5000)
        result: Any = detector.detect(bgr)
        faces = result[1]
        if faces is None:  # OpenCV returns None when nothing is found
            return []
        bounds = BBox(0, 0, w, h)
        out: dict[tuple[int, int, int, int], Region] = {}
        for row in np.asarray(faces, dtype=np.float64):
            x, y, fw, fh, score = row[0] / scale, row[1] / scale, row[2] / scale, row[3] / scale, float(row[-1])
            box = BBox(int(np.floor(x)), int(np.floor(y)), int(np.ceil(x + fw)), int(np.ceil(y + fh)))
            box = box.pad(round(_PAD * box.width), round(_PAD * box.height), bounds)
            if box.is_empty:
                continue
            out[box.as_tuple()] = Region(
                region_id=content_id("G", page, RegionKind.FACE.value, *box.as_tuple()),
                page=page,
                kind=RegionKind.FACE,
                bbox=box,
                score=round(score, 4),
                detector="yunet",
                rule="face",
            )
        return [out[k] for k in sorted(out)]
