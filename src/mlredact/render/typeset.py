"""Erase-and-typeset rendering of surrogate text (plan §10).

For each surrogate segment:

1. choose the erase box: start from the planned (padded) box and move its edges to the centre of
   the widest nearby whitespace gap, so whole PII glyphs are erased without clipping neighbouring
   words; if no clear gap exists the conservative padded edge is kept;
2. estimate the local paper colour from a ring of light pixels around the box and the ink colour
   from the original glyphs;
3. **erase** the box with the paper colour — fully opaque;
4. calibrate the font size by rendering the *original* text in our font and matching its ink
   height to the measured one, and place the surrogate on the original baseline;
5. draw the surrogate, condensing horizontally (and if needed shrinking) so it never leaves the
   erased box.

Rendering uses Pillow/FreeType with the BASIC layout engine, which is deterministic for pinned
library and font versions.  The erased box and glyph mask are returned for verification (V3).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw, ImageFont

from mlredact.core.geometry import BBox
from mlredact.core.model import RenderOp
from mlredact.resources.loader import font_path

_RING = 5
_MIN_CONDENSE = 0.62
_DEFAULT_INK = (24, 24, 24)
_DARK_DELTA = 60.0


@dataclass(frozen=True, slots=True)
class TypesetResult:
    box: BBox  # the box actually erased
    background: tuple[int, int, int]
    glyph_mask: npt.NDArray[np.bool_]  # shape == (box.height, box.width)
    # Pixels inside ``box`` left alone because an earlier operation drew glyphs there (shape as above).
    protected: npt.NDArray[np.bool_] | None = None


@cache
def _font(family: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(font_path(family, "Regular")), size, layout_engine=ImageFont.Layout.BASIC)


def _luminance(px: npt.NDArray[np.uint8]) -> npt.NDArray[np.float64]:
    p = px.astype(np.float64)
    return 0.299 * p[..., 0] + 0.587 * p[..., 1] + 0.114 * p[..., 2]


def _lum_of(rgb: tuple[int, int, int]) -> float:
    return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]


def estimate_background(img: npt.NDArray[np.uint8], box: BBox) -> tuple[int, int, int]:
    h, w = img.shape[:2]
    outer = box.pad(_RING, _RING, BBox(0, 0, w, h))
    region = img[outer.y0 : outer.y1, outer.x0 : outer.x1]
    mask = np.ones(region.shape[:2], dtype=bool)
    mask[box.y0 - outer.y0 : box.y1 - outer.y0, box.x0 - outer.x0 : box.x1 - outer.x0] = False
    ring = region[mask]
    if ring.size == 0:
        return (255, 255, 255)
    lum = _luminance(ring)
    light = ring[lum >= np.percentile(lum, 50)]
    sample = light if light.shape[0] >= 8 else ring
    med = np.median(sample, axis=0)
    return (int(med[0]), int(med[1]), int(med[2]))


def _dark(region: npt.NDArray[np.uint8], background: tuple[int, int, int]) -> npt.NDArray[np.bool_]:
    return _luminance(region) < min(_lum_of(background) - _DARK_DELTA, 190.0)


def ink_extent(
    img: npt.NDArray[np.uint8], box: BBox, background: tuple[int, int, int]
) -> tuple[BBox, tuple[int, int, int]] | None:
    """Extent and colour of the segment's own glyphs: the main text band only, so descenders or
    ascenders of adjacent lines that intrude into the token box do not inflate the measurement."""
    region = img[box.y0 : box.y1, box.x0 : box.x1]
    if region.size == 0:
        return None
    dark = _dark(region, background)
    if int(dark.sum()) < 6:
        return None
    row_ink = dark.sum(axis=1)
    runs: list[tuple[int, int]] = []
    start = None
    for i, count in enumerate(row_ink.tolist()):
        if count and start is None:
            start = i
        elif not count and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(row_ink)))
    # Merge runs separated by tiny gaps (dots of i/j, accents), then keep the heaviest band.
    merged: list[tuple[int, int]] = []
    for r in runs:
        if merged and r[0] - merged[-1][1] <= 2:
            merged[-1] = (merged[-1][0], r[1])
        else:
            merged.append(r)
    band = max(merged, key=lambda r: (int(row_ink[r[0] : r[1]].sum()), -r[0]))
    core = dark[band[0] : band[1]]
    ys, xs = np.nonzero(core)
    ink = BBox(
        box.x0 + int(xs.min()),
        box.y0 + band[0] + int(ys.min()),
        box.x0 + int(xs.max()) + 1,
        box.y0 + band[0] + int(ys.max()) + 1,
    )
    med = np.median(region[band[0] : band[1]][core], axis=0)
    return ink, (int(med[0]), int(med[1]), int(med[2]))


def free_space_right(img: npt.NDArray[np.uint8], box: BBox, band: BBox, background: tuple[int, int, int]) -> int:
    """Blank columns immediately right of ``box`` within the text band (up to ~6 line heights)."""
    h_img, w_img = img.shape[:2]
    limit = min(w_img, box.x1 + 6 * max(1, band.height))
    if limit <= box.x1:
        return 0
    stripe = _dark(img[max(0, band.y0) : min(h_img, band.y1), box.x1 : limit], background)
    cols = stripe.any(axis=0)
    filled = np.nonzero(cols)[0]
    free = int(filled[0]) if filled.size else limit - box.x1
    # Keep a word gap before whatever ink follows.
    return max(0, free - max(2, round(0.3 * band.height))) if filled.size else free


def _empty_runs(profile: npt.NDArray[np.bool_], offset: int) -> list[tuple[int, int]]:
    """Runs of False in a boolean profile, as half-open (start, end) in page coordinates."""
    runs: list[tuple[int, int]] = []
    start = None
    for i, filled in enumerate(profile.tolist()):
        if not filled and start is None:
            start = i
        elif filled and start is not None:
            runs.append((offset + start, offset + i))
            start = None
    if start is not None:
        runs.append((offset + start, offset + len(profile)))
    return runs


def _best_gap(runs: list[tuple[int, int]], boundary: int) -> tuple[int, int] | None:
    if not runs:
        return None
    return max(runs, key=lambda r: (r[1] - r[0], -abs((r[0] + r[1]) / 2.0 - boundary)))


def refine_erase_box(img: npt.NDArray[np.uint8], op: RenderOp, background: tuple[int, int, int]) -> BBox:
    """Move the planned box's edges into the widest nearby whitespace gaps (never shrinking it so
    that any of the segment's own ink would survive)."""
    box, ink = op.box, op.ink_box
    if ink is None or ink.is_empty:
        return box
    h_img, w_img = img.shape[:2]
    line_h = ink.height
    window = max(6, round(0.6 * line_h))
    min_word_gap = max(2, round(0.12 * line_h))
    x_lo, x_hi = max(0, box.x0 - window), min(w_img, box.x1 + window)
    band = _dark(img[ink.y0 : ink.y1, x_lo:x_hi], background)
    cols = band.any(axis=0)
    runs = _empty_runs(cols, x_lo)
    own = np.nonzero(cols[ink.x0 - x_lo : ink.x1 - x_lo])[0]
    own_first = ink.x0 + int(own[0]) if own.size else ink.x0
    own_last = ink.x0 + int(own[-1]) + 1 if own.size else ink.x1
    x0, x1 = box.x0, box.x1
    left = _best_gap([r for r in runs if r[0] <= own_first and r[1] >= ink.x0 - window], ink.x0)
    if left and left[1] - left[0] >= min_word_gap:
        x0 = min((left[0] + left[1]) // 2, own_first)
    right = _best_gap([r for r in runs if r[1] >= own_last and r[0] <= ink.x1 + window], ink.x1)
    if right and right[1] - right[0] >= min_word_gap:
        x1 = max((right[0] + right[1] + 1) // 2, own_last)
    # Vertical: stay inside the line band, but never cut into adjacent lines' ink.
    v_window = max(3, round(0.35 * line_h))
    y_lo, y_hi = max(0, box.y0 - v_window), min(h_img, box.y1 + v_window)
    stripe = _dark(img[y_lo:y_hi, max(0, x0) : min(w_img, x1)], background)
    rows = stripe.any(axis=1)
    row_runs = _empty_runs(rows, y_lo)
    y0, y1 = box.y0, box.y1
    top = _best_gap([r for r in row_runs if r[1] <= ink.y0 + 1 and r[1] >= box.y0 - v_window], ink.y0)
    if top:
        y0 = min((top[0] + top[1]) // 2, ink.y0)
    bottom = _best_gap([r for r in row_runs if r[0] >= ink.y1 - 1 and r[0] <= box.y1 + v_window], ink.y1)
    if bottom:
        y1 = max((bottom[0] + bottom[1] + 1) // 2, ink.y1)
    refined = BBox(max(0, x0), max(0, y0), min(w_img, x1), min(h_img, y1))
    # Fail safe: the erase must contain all of the segment's own ink; otherwise keep the plan.
    if refined.x0 > own_first or refined.x1 < own_last or refined.y0 > ink.y0 or refined.y1 < ink.y1:
        return box
    return refined


def _render(text: str, family: str, size: int) -> tuple[Image.Image, float, float]:
    """Render text on a tight canvas; returns (canvas, left, top) of its bbox relative to origin."""
    font = _font(family, size)
    left, top, right, bottom = font.getbbox(text)
    canvas = Image.new("L", (max(1, math.ceil(right - left) + 2), max(1, math.ceil(bottom - top) + 2)), 0)
    ImageDraw.Draw(canvas).text((1 - left, 1 - top), text, fill=255, font=font)
    return canvas, left, top


def _blocked_extension(box: BBox, extra: int, blockers: Sequence[BBox]) -> int:
    """Limit a rightward extension so it never enters another operation's box on the same band."""
    limit = extra
    for b in blockers:
        if b.y1 <= box.y0 or b.y0 >= box.y1 or b.x0 < box.x1:
            continue
        limit = min(limit, max(0, b.x0 - box.x1 - 1))
    return limit


def typeset(
    img: npt.NDArray[np.uint8],
    op: RenderOp,
    family: str = "Sans",
    blockers: Sequence[BBox] = (),
    *,
    fill: tuple[int, int, int] | None = None,
    ink_colour: tuple[int, int, int] | None = None,
    protect: npt.NDArray[np.bool_] | None = None,
) -> TypesetResult:
    """Erase and draw ``op.text`` in place (mutates ``img``).

    ``blockers`` are the boxes of the page's other operations: a surrogate that is wider than the
    original may extend into blank paper to its right, but never into another operation's box.
    ``fill`` / ``ink_colour`` override the paper and ink colours (used for labelled boxes).
    ``protect`` (page-sized) marks glyphs drawn by earlier operations: they are not erased again (the
    earlier operation already erased and verified that area), so tightly spaced re-typeset lines do
    not wipe each other's descenders."""

    def erase(b: BBox) -> None:
        if protect is None:
            img[b.y0 : b.y1, b.x0 : b.x1] = background
        else:
            region = img[b.y0 : b.y1, b.x0 : b.x1]
            region[~protect[b.y0 : b.y1, b.x0 : b.x1]] = background

    def protected_in(b: BBox) -> npt.NDArray[np.bool_] | None:
        return None if protect is None else protect[b.y0 : b.y1, b.x0 : b.x1].copy()

    planned_bg = estimate_background(img, op.box)
    box = refine_erase_box(img, op, planned_bg)
    paper = estimate_background(img, box)
    background = fill if fill is not None else paper
    ink_box = op.ink_box or op.box
    measured = ink_extent(img, ink_box, paper)
    band = measured[0] if measured else ink_box
    extra = _blocked_extension(box, free_space_right(img, box, band, paper), blockers)
    erase(box)
    mask = np.zeros((box.height, box.width), dtype=bool)
    text = (op.text or "").strip()
    if not text:
        return TypesetResult(box, background, mask, protected_in(box))
    colour = ink_colour if ink_colour is not None else (measured[1] if measured else _DEFAULT_INK)
    source = (op.source_text or "").strip()
    if measured is not None and source:
        ink = measured[0]
        _l, ref_top, _r, ref_bottom = _font(family, 100).getbbox(source)
        size = max(6, min(400, round(100 * ink.height / max(1.0, ref_bottom - ref_top))))
        s_left, s_top, _sr, _sb = _font(family, size).getbbox(source)
        origin_x, origin_y = ink.x0 - s_left, ink.y0 - s_top  # original glyph box -> same baseline
    else:
        tall = max(4, round(ink_box.height * 0.62))
        size = max(6, round(tall / 0.72))
        origin_x, origin_y = float(ink_box.x0), float(ink_box.y0 + (ink_box.height - tall) // 2)
    canvas, left, top = _render(text, family, size)
    page_x0 = math.floor(origin_x + left) - 1
    page_y0 = math.floor(origin_y + top) - 1
    avail = max(1, box.x1 - max(page_x0, box.x0))
    if canvas.width > avail and extra > 0:
        # Use blank paper to the right first (it is erased too, so V3 still covers the whole box).
        grow = min(extra, canvas.width - avail)
        erase(BBox(box.x1, box.y0, box.x1 + grow, box.y1))
        box = BBox(box.x0, box.y0, box.x1 + grow, box.y1)
        mask = np.zeros((box.height, box.width), dtype=bool)
        avail += grow
    if canvas.width > avail:
        scale = avail / canvas.width
        if scale < _MIN_CONDENSE:
            size = max(6, int(size * scale / _MIN_CONDENSE))
            canvas, left, top = _render(text, family, size)
            page_y0 = math.floor(origin_y + top) - 1
            scale = min(1.0, avail / canvas.width)
        if scale < 1.0:
            canvas = canvas.resize((max(1, int(canvas.width * scale)), canvas.height), Image.Resampling.LANCZOS)
    alpha_full = np.asarray(canvas, dtype=np.float64) / 255.0
    cx0, cy0 = max(page_x0, box.x0), max(page_y0, box.y0)
    cx1, cy1 = min(page_x0 + canvas.width, box.x1), min(page_y0 + canvas.height, box.y1)
    if cx1 <= cx0 or cy1 <= cy0:
        return TypesetResult(box, background, mask, protected_in(box))
    alpha = alpha_full[cy0 - page_y0 : cy1 - page_y0, cx0 - page_x0 : cx1 - page_x0][..., None]
    bg = np.array(background, dtype=np.float64)
    fg = np.array(colour, dtype=np.float64)
    img[cy0:cy1, cx0:cx1] = np.rint(bg * (1.0 - alpha) + fg * alpha).astype(np.uint8)
    mask[cy0 - box.y0 : cy1 - box.y0, cx0 - box.x0 : cx1 - box.x0] = alpha[..., 0] > 0.0
    return TypesetResult(box, background, mask, protected_in(box))


def erased_cleanly(img: npt.NDArray[np.uint8], result: TypesetResult) -> bool:
    """V3 for a surrogate op, checked right after drawing it: every pixel of the erased box that is
    not a glyph pixel is exactly the paper colour (no original pixel survived)."""
    b = result.box
    region = img[b.y0 : b.y1, b.x0 : b.x1]
    if region.size == 0:
        return True
    keep = result.glyph_mask if result.protected is None else (result.glyph_mask | result.protected)
    others = region[~keep]
    return bool(np.all(others == np.array(result.background, dtype=np.uint8)))
