"""Engine D: the PDF's own text layer (plan §6 "embedded text"), as words in canonical pixels.

Character boxes come from PDFium's text page and are mapped to raster pixels with PDFium's own
page-to-device transform (the same one rendering uses), so /Rotate and the CropBox are honoured.
Words are only *hypotheses*: a word with no ink under it (hidden or white text) is dropped, and
the fusion step decides how far a word is trusted against the OCR reading.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from mlredact.core.geometry import BBox


@dataclass(frozen=True, slots=True)
class EmbeddedWord:
    text: str
    bbox: BBox
    line: int  # index of the PDF text line the word belongs to (breaks at generated line ends)


def _to_device(page: Any, width: int, height: int, x: float, y: float) -> tuple[int, int]:
    import pypdfium2.raw as pdfium_c

    dx, dy = ctypes.c_int(), ctypes.c_int()
    pdfium_c.FPDF_PageToDevice(
        page.raw, 0, 0, width, height, 0, ctypes.c_double(x), ctypes.c_double(y), ctypes.byref(dx), ctypes.byref(dy)
    )
    return int(dx.value), int(dy.value)


def embedded_words(page: Any, textpage: Any, width: int, height: int) -> list[EmbeddedWord]:
    """Words of the text layer with their pixel boxes (whitespace and generated breaks split words)."""
    import pypdfium2.raw as pdfium_c

    n = textpage.count_chars()
    if n <= 0:
        return []
    text = textpage.get_text_range(0, n)
    bounds = BBox(0, 0, width, height)
    words: list[EmbeddedWord] = []
    chars: list[str] = []
    box: BBox | None = None
    line = 0

    def flush() -> None:
        nonlocal chars, box
        if chars and box is not None:
            clipped = box.intersection(bounds)
            if clipped is not None and not clipped.is_empty:
                words.append(EmbeddedWord("".join(chars), clipped, line))
        chars, box = [], None

    for i in range(min(n, len(text))):
        ch = text[i]
        if ch in "\r\n":
            flush()
            if ch == "\n":
                line += 1
            continue
        if ch.isspace() or pdfium_c.FPDFText_IsGenerated(textpage.raw, i) == 1:
            flush()
            continue
        left, bottom, right, top = textpage.get_charbox(i)
        if right <= left or top <= bottom:
            continue
        xa, ya = _to_device(page, width, height, left, bottom)
        xb, yb = _to_device(page, width, height, right, top)
        cbox = BBox(min(xa, xb), min(ya, yb), max(xa, xb) + 1, max(ya, yb) + 1)
        box = cbox if box is None else box.union(cbox)
        chars.append(ch)
    flush()
    return words


def visible(word: EmbeddedWord, rgb: npt.NDArray[np.uint8], min_ink: float = 0.02) -> bool:
    """Is there ink under the word on the rendered page?  (White / hidden text has none.)

    Paper level is measured around the word (a box one word-height larger on every side), so a
    word box that is solid ink still reads as visible."""
    b = word.bbox
    crop = rgb[b.y0 : b.y1, b.x0 : b.x1]
    if crop.size == 0:
        return False
    m = max(2, b.height)
    h, w = rgb.shape[:2]
    around = rgb[max(0, b.y0 - m) : min(h, b.y1 + m), max(0, b.x0 - m) : min(w, b.x1 + m)]
    paper = float(np.percentile(around.astype(np.int32).mean(axis=2), 95))
    gray = crop.astype(np.int32).mean(axis=2)
    return bool((gray < paper - 60).mean() >= min_ink)
