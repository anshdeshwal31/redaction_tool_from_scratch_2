"""Geometry in the canonical raster coordinate system.

Convention: pixel coordinates of the page's canonical raster, origin top-left, x to the right,
y downwards.  Boxes are *half-open integer* rectangles ``[x0, x1) x [y0, y1)``.  Every conversion
from floating point rounds **outward** (floor for minima, ceil for maxima) so a redaction region
can only grow, never shrink, through rounding.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True, order=True)
class BBox:
    x0: int
    y0: int
    x1: int
    y1: int

    def __post_init__(self) -> None:
        if self.x1 < self.x0 or self.y1 < self.y0:
            raise ValueError("BBox maxima must not be smaller than minima")

    @staticmethod
    def outward(x0: float, y0: float, x1: float, y1: float) -> BBox:
        lo_x, hi_x = (x0, x1) if x0 <= x1 else (x1, x0)
        lo_y, hi_y = (y0, y1) if y0 <= y1 else (y1, y0)
        return BBox(math.floor(lo_x), math.floor(lo_y), math.ceil(hi_x), math.ceil(hi_y))

    @staticmethod
    def from_points(points: Iterable[tuple[float, float]]) -> BBox:
        pts = list(points)
        if not pts:
            raise ValueError("cannot build a BBox from no points")
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return BBox.outward(min(xs), min(ys), max(xs), max(ys))

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def is_empty(self) -> bool:
        return self.width == 0 or self.height == 0

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0)

    def union(self, other: BBox) -> BBox:
        return BBox(min(self.x0, other.x0), min(self.y0, other.y0), max(self.x1, other.x1), max(self.y1, other.y1))

    def intersection(self, other: BBox) -> BBox | None:
        x0, y0 = max(self.x0, other.x0), max(self.y0, other.y0)
        x1, y1 = min(self.x1, other.x1), min(self.y1, other.y1)
        if x1 <= x0 or y1 <= y0:
            return None
        return BBox(x0, y0, x1, y1)

    def iou(self, other: BBox) -> float:
        inter = self.intersection(other)
        if inter is None:
            return 0.0
        union_area = self.area + other.area - inter.area
        return inter.area / union_area if union_area else 0.0

    def overlap_fraction(self, other: BBox) -> float:
        """Fraction of *this* box covered by ``other``."""
        inter = self.intersection(other)
        if inter is None or self.area == 0:
            return 0.0
        return inter.area / self.area

    def vertical_overlap(self, other: BBox) -> float:
        """Overlap of the y-extents relative to the smaller height (0..1)."""
        lo, hi = max(self.y0, other.y0), min(self.y1, other.y1)
        denom = min(self.height, other.height)
        return max(0, hi - lo) / denom if denom > 0 else 0.0

    def pad(self, dx: int, dy: int, clip: BBox | None = None) -> BBox:
        box = BBox(self.x0 - dx, self.y0 - dy, self.x1 + dx, self.y1 + dy)
        if clip is not None:
            box = BBox(max(box.x0, clip.x0), max(box.y0, clip.y0), min(box.x1, clip.x1), min(box.y1, clip.y1))
            if box.x1 < box.x0 or box.y1 < box.y0:
                return BBox(clip.x0, clip.y0, clip.x0, clip.y0)
        return box

    def contains_point(self, x: float, y: float) -> bool:
        return self.x0 <= x < self.x1 and self.y0 <= y < self.y1

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x0, self.y0, self.x1, self.y1)


Quad = tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]


def quad_from_array(arr: npt.ArrayLike) -> Quad:
    a = np.asarray(arr, dtype=np.float64).reshape(4, 2)
    return (
        (float(a[0, 0]), float(a[0, 1])),
        (float(a[1, 0]), float(a[1, 1])),
        (float(a[2, 0]), float(a[2, 1])),
        (float(a[3, 0]), float(a[3, 1])),
    )


def union_all(boxes: Sequence[BBox]) -> BBox:
    if not boxes:
        raise ValueError("union of no boxes")
    out = boxes[0]
    for b in boxes[1:]:
        out = out.union(b)
    return out


@dataclass(frozen=True, slots=True)
class Affine:
    """2-D affine/projective transform as a 3x3 float64 matrix (row-major tuple)."""

    m: tuple[float, float, float, float, float, float, float, float, float]

    @staticmethod
    def identity() -> Affine:
        return Affine((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))

    @staticmethod
    def from_matrix(mat: npt.ArrayLike) -> Affine:
        a = np.asarray(mat, dtype=np.float64)
        if a.shape == (2, 3):
            a = np.vstack([a, [0.0, 0.0, 1.0]])
        if a.shape != (3, 3):
            raise ValueError("transform matrix must be 2x3 or 3x3")
        flat = tuple(float(v) for v in a.reshape(9))
        return Affine(flat)  # type: ignore[arg-type]

    def matrix(self) -> npt.NDArray[np.float64]:
        return np.array(self.m, dtype=np.float64).reshape(3, 3)

    def compose(self, then: Affine) -> Affine:
        """Return the transform that applies ``self`` first and ``then`` second."""
        return Affine.from_matrix(then.matrix() @ self.matrix())

    def inverse(self) -> Affine:
        return Affine.from_matrix(np.linalg.inv(self.matrix()))

    def apply(self, x: float, y: float) -> tuple[float, float]:
        m = self.m
        w = m[6] * x + m[7] * y + m[8]
        return ((m[0] * x + m[1] * y + m[2]) / w, (m[3] * x + m[4] * y + m[5]) / w)

    def apply_box(self, box: BBox) -> BBox:
        corners = [(box.x0, box.y0), (box.x1, box.y0), (box.x1, box.y1), (box.x0, box.y1)]
        return BBox.from_points(self.apply(x, y) for x, y in corners)
