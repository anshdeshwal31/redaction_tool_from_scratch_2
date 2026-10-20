"""Synthetic stand-ins for removed regions: pen strokes for a signature, a silhouette for a face or
photo (plan §10; the client's policy asks for a fake of the same kind rather than a grey box).

A stand-in is a pure function of the region id (derived from position and already in the manifest),
so it carries nothing from the original and is identical on every rerun.  It is drawn without
anti-aliasing over a uniform fill, so the pixel check can confirm that every pixel in the box is one
of two known colours: no original pixel survives.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw

from mlredact.core.geometry import BBox
from mlredact.core.types import RegionKind
from mlredact.render.typeset import estimate_background

PEN_INK = (28, 40, 96)  # blue-black ballpoint
PHOTO_BACKGROUND = (208, 214, 222)
PHOTO_FIGURE = (146, 156, 172)
_PORTRAITS = frozenset({RegionKind.FACE.value, RegionKind.PHOTO.value})


def _rng(ref_id: str) -> np.random.Generator:
    seed = int.from_bytes(hashlib.sha256(ref_id.encode("utf-8")).digest()[:8], "big")
    return np.random.Generator(np.random.PCG64(seed))


def _signature_mask(w: int, h: int, rng: np.random.Generator) -> npt.NDArray[np.bool_]:
    """A cursive-looking scribble: a run of loops of varying height along a slightly sloping
    baseline, a tall opening stroke, and sometimes an underline flourish."""
    img = Image.new("1", (w, h), 0)
    draw = ImageDraw.Draw(img)
    stroke = max(2, round(min(h, w / 4) / 16))
    margin = max(stroke, round(0.06 * w))
    span = max(1, w - 2 * margin)
    humps = max(3, min(14, round(span / max(1, 0.32 * h)) + int(rng.integers(-1, 2))))
    heights = rng.uniform(0.45, 1.0, humps)
    base = h * rng.uniform(0.58, 0.68)
    amp = h * 0.3
    slope = h * rng.uniform(-0.08, 0.08)
    loop = min(span / humps * 0.35, h * 0.12)
    points = []
    steps = 24 * humps
    for i in range(steps + 1):
        t = i / steps
        k = min(humps - 1, int(t * humps))
        x = margin + t * span - loop * math.sin(2 * math.pi * humps * t)
        y = (
            base
            + slope * t
            - amp * heights[k] * abs(math.sin(math.pi * humps * t))
            + 0.5 * loop * math.cos(2 * math.pi * humps * t)
        )
        points.append((round(x), round(y)))
    draw.line(points, fill=1, width=stroke, joint="curve")
    lead = (margin + round(0.04 * span), round(base))
    draw.line([lead, (lead[0] + round(0.05 * span), round(h * 0.12)), points[0]], fill=1, width=stroke)
    if rng.uniform() < 0.6:
        y0 = round(base + 0.18 * h)
        draw.line(
            [(margin, y0), (margin + span // 2, y0 + round(0.06 * h)), (margin + span, y0 - round(0.04 * h))],
            fill=1,
            width=max(1, stroke - 1),
            joint="curve",
        )
    return np.asarray(img, dtype=bool)


def _silhouette_mask(w: int, h: int) -> npt.NDArray[np.bool_]:
    """Head and shoulders, centred: the generic 'no photo' portrait."""
    img = Image.new("1", (w, h), 0)
    draw = ImageDraw.Draw(img)
    r = 0.2 * min(w, h)
    cx, cy = w / 2, h * 0.4
    draw.ellipse((round(cx - r), round(cy - r), round(cx + r), round(cy + r)), fill=1)
    draw.ellipse((round(w * 0.14), round(h * 0.68), round(w * 0.86), round(h * 1.4)), fill=1)
    return np.asarray(img, dtype=bool)


def _free_band(free_rows: npt.NDArray[np.bool_]) -> tuple[int, int]:
    """The longest run of free rows (first one on ties); (0, 0) when there is none."""
    best, start = (0, 0), None
    for y, ok in enumerate([*free_rows.tolist(), False]):
        if ok and start is None:
            start = y
        elif not ok and start is not None:
            if y - start > best[1] - best[0]:
                best = (start, y)
            start = None
    return best


def draw_stand_in(
    img: npt.NDArray[np.uint8], box: BBox, reason: str, ref_id: str, keep_out: Sequence[BBox] = ()
) -> bool:
    """Replace the box with the stand-in for region kind ``reason``.  Returns the pixel check.

    ``keep_out``: other operations' boxes (already drawn) that the stand-in must not paint over,
    such as a name line the signature region overlaps.  The stand-in is drawn in the tallest band of
    rows they leave free; every other free pixel of the box becomes the fill."""
    w, h = box.x1 - box.x0, box.y1 - box.y0
    if w <= 0 or h <= 0:
        return True
    free = np.ones((h, w), dtype=bool)
    for other in keep_out:
        inter = box.intersection(other)
        if inter is not None:
            free[inter.y0 - box.y0 : inter.y1 - box.y0, inter.x0 - box.x0 : inter.x1 - box.x0] = False
    y0, y1 = _free_band(free.all(axis=1))
    mask = np.zeros((h, w), dtype=bool)
    if reason in _PORTRAITS:
        fill, ink = PHOTO_BACKGROUND, PHOTO_FIGURE
        if y1 > y0:
            mask[y0:y1] = _silhouette_mask(w, y1 - y0)
    else:
        fill, ink = estimate_background(img, box), PEN_INK
        if y1 > y0:
            mask[y0:y1] = _signature_mask(w, y1 - y0, _rng(ref_id))
    region = img[box.y0 : box.y1, box.x0 : box.x1]
    region[free] = np.array(fill, dtype=np.uint8)
    region[free & mask] = np.array(ink, dtype=np.uint8)
    return bool(np.all(region[free & mask] == ink) and np.all(region[free & ~mask] == fill))
