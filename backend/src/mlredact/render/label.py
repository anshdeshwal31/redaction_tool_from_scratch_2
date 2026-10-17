"""Labelled boxes (render style ``label``) and the page notice banner (plan §10).

A label op erases its box with an opaque light fill and writes a short, consistent tag such as
``[PERSON 1]``.  The notice is drawn only into a blank page margin, so it can never cover content.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw

from mlredact.config.schema import RedactionRenderConfig
from mlredact.core.geometry import BBox
from mlredact.core.model import RenderOp
from mlredact.render.typeset import TypesetResult, _font, _luminance, typeset

NOTICE_SURROGATE = "De-identified copy - names, identifiers and contact details replaced with fictitious values"
NOTICE_REMOVED = "Redacted copy - personal details removed"


def _draw_text_in(
    img: npt.NDArray[np.uint8],
    box: BBox,
    text: str,
    family: str,
    size: int,
    fill: tuple[int, int, int],
    colour: tuple[int, int, int],
    centre: bool,
) -> npt.NDArray[np.bool_]:
    """Draw ``text`` inside ``box`` on a solid ``fill`` (already painted); returns the glyph mask."""
    mask = np.zeros((box.height, box.width), dtype=bool)
    font = _font(family, max(6, size))
    left, top, right, bottom = font.getbbox(text)
    width, height = max(1, math.ceil(right - left) + 2), max(1, math.ceil(bottom - top) + 2)
    canvas = Image.new("L", (width, height), 0)
    ImageDraw.Draw(canvas).text((1 - left, 1 - top), text, fill=255, font=font)
    avail_w = max(1, box.width - 4)
    if canvas.width > avail_w:
        canvas = canvas.resize((avail_w, canvas.height), Image.Resampling.LANCZOS)
    x0 = box.x0 + ((box.width - canvas.width) // 2 if centre else 2)
    y0 = box.y0 + max(0, (box.height - canvas.height) // 2)
    cx0, cy0 = max(x0, box.x0), max(y0, box.y0)
    cx1, cy1 = min(x0 + canvas.width, box.x1), min(y0 + canvas.height, box.y1)
    if cx1 <= cx0 or cy1 <= cy0:
        return mask
    alpha = np.asarray(canvas, dtype=np.float64)[cy0 - y0 : cy1 - y0, cx0 - x0 : cx1 - x0][..., None] / 255.0
    bg = np.array(fill, dtype=np.float64)
    fg = np.array(colour, dtype=np.float64)
    img[cy0:cy1, cx0:cx1] = np.rint(bg * (1.0 - alpha) + fg * alpha).astype(np.uint8)
    mask[cy0 - box.y0 : cy1 - box.y0, cx0 - box.x0 : cx1 - box.x0] = alpha[..., 0] > 0.0
    return mask


def draw_label(
    img: npt.NDArray[np.uint8],
    op: RenderOp,
    cfg: RedactionRenderConfig,
    family: str,
    blockers: Sequence[BBox] = (),
    protect: npt.NDArray[np.bool_] | None = None,
) -> TypesetResult:
    """Same erase geometry and text calibration as surrogates, on an opaque label fill."""
    return typeset(img, op, family, blockers, fill=cfg.label_fill_rgb, ink_colour=cfg.label_text_rgb, protect=protect)


def place_notice(
    img: npt.NDArray[np.uint8], text: str, dpi: int, family: str = "Sans"
) -> tuple[BBox, npt.NDArray[np.bool_], tuple[int, int, int]] | None:
    """Draw the notice into a blank band at the bottom (else top) margin.  None if no room."""
    h, w = img.shape[:2]
    size = max(8, round(7.0 * dpi / 72.0))  # 7 pt
    band_h = round(size * 1.8)
    side = max(1, round(0.06 * w))
    lum = _luminance(img)
    ink = lum < 170.0
    for y0 in (h - band_h - round(0.012 * h), round(0.008 * h)):
        if y0 < 0 or y0 + band_h > h:
            continue
        box = BBox(side, y0, w - side, y0 + band_h)
        if bool(ink[box.y0 : box.y1, box.x0 : box.x1].any()):
            continue  # not blank: never draw over content
        region = img[box.y0 : box.y1, box.x0 : box.x1].reshape(-1, 3)
        med = np.median(region, axis=0)
        paper = (int(med[0]), int(med[1]), int(med[2]))
        img[box.y0 : box.y1, box.x0 : box.x1] = paper
        mask = _draw_text_in(img, box, text, family, size, paper, (120, 120, 120), centre=True)
        return box, mask, paper
    return None
