"""Ink masks for ink accounting (plan §6).

Darkness is measured against a *local* background (paper colour varies across a scan), so
yellowed paper, shadows and uneven exposure do not read as ink.  Everything here is a pure,
deterministic function of the raster (OpenCV with fixed thread counts, integer arithmetic).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt

from mlredact.core.geometry import BBox

U8 = npt.NDArray[np.uint8]
Mask = npt.NDArray[np.bool_]


def scaled(px_at_300: float, dpi: int) -> int:
    """A length given for 300 dpi, at ``dpi``."""
    return max(1, round(px_at_300 * dpi / 300.0))


def to_gray(rgb: U8) -> U8:
    return np.asarray(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), dtype=np.uint8)


def background(gray: U8, dpi: int) -> U8:
    """Local paper level: a max filter at quarter resolution (wider than any stroke) + median."""
    h, w = gray.shape
    small = cv2.resize(gray, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA)
    k = scaled(9, dpi) | 1
    small = cv2.dilate(small, np.ones((k, k), np.uint8))
    small = cv2.medianBlur(small, 5)
    return np.asarray(cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR), dtype=np.uint8)


@dataclass(frozen=True, slots=True)
class InkMaps:
    gray: U8
    background: U8
    ink: Mask  # clearly dark marks
    faint: Mask  # light marks (after smoothing): show-through, pencil, faded ink


def ink_maps(rgb: U8, dpi: int, ink_darkness: int, faint_darkness: int) -> InkMaps:
    gray = to_gray(rgb)
    bg = background(gray, dpi)
    darkness = bg.astype(np.int16) - gray.astype(np.int16)
    ink = darkness >= ink_darkness
    # Faint ink is judged on a smoothed image: scanner noise is pixel-sized, show-through is not.
    k = max(3, scaled(5, dpi) | 1)  # 5 px at 300 dpi; thin strokes at lower dpi must survive smoothing
    smooth = cv2.blur(gray, (k, k))
    faint_dark = bg.astype(np.int16) - smooth.astype(np.int16)
    near_ink = cv2.dilate(ink.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    faint = (faint_dark >= faint_darkness) & ~near_ink
    return InkMaps(gray, bg, ink, faint)


def ruling_lines(ink: Mask, dpi: int) -> Mask:
    """Long straight horizontal / vertical strokes (form rules, table borders, underlines)."""
    m = ink.astype(np.uint8)
    length = scaled(120, dpi)  # ~1 cm: no glyph has a straight run this long
    horizontal = np.asarray(cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((1, length), np.uint8)), dtype=np.uint8)
    vertical = np.asarray(cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((length, 1), np.uint8)), dtype=np.uint8)
    rules = cv2.dilate(np.maximum(horizontal, vertical), np.ones((3, 3), np.uint8))
    return np.asarray(rules, dtype=bool)


def box_mask(shape: tuple[int, int], boxes: Iterable[BBox]) -> Mask:
    out = np.zeros(shape, dtype=bool)
    for b in boxes:
        out[max(0, b.y0) : max(0, b.y1), max(0, b.x0) : max(0, b.x1)] = True
    return out


def drop_specks(mask: Mask, min_area: int, min_extent: int) -> Mask:
    """Remove connected components that are both small in area and in extent (scanner dust)."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, dtype=bool)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        extent = max(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
        keep[i] = area >= min_area or extent >= min_extent
    return keep[labels]  # type: ignore[no-any-return]


@dataclass(frozen=True, slots=True)
class Blob:
    box: BBox  # tight box of the blob's own pixels
    pixels: int
    components: int


def blobs(mask: Mask, join_x: int, join_y: int) -> list[Blob]:
    """Group nearby marks (a signature's strokes, a word's letters) into blobs, in reading order."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return []
    joined = cv2.dilate(mask.astype(np.uint8), np.ones((max(1, join_y), max(1, join_x)), np.uint8))
    _n, labels = cv2.connectedComponents(joined, connectivity=8)
    _m, comp_labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    ids = labels[ys, xs]
    order = np.argsort(ids, kind="stable")
    ids, xs, ys, comps = ids[order], xs[order], ys[order], comp_labels[ys, xs][order]
    out: list[Blob] = []
    for seg in np.split(np.arange(len(ids)), np.flatnonzero(np.diff(ids)) + 1):
        bx, by = xs[seg], ys[seg]
        box = BBox(int(bx.min()), int(by.min()), int(bx.max()) + 1, int(by.max()) + 1)
        out.append(Blob(box, len(seg), len(np.unique(comps[seg]))))
    return sorted(out, key=lambda b: (b.box.y0, b.box.x0, b.box.y1, b.box.x1))
