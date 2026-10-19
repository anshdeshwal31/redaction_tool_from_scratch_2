"""Canonical rasterisation with PDFium (plan §6).

The canonical raster is what a viewer shows: CropBox, /Rotate applied, visible annotations and form
field appearances drawn.  Rendering is deterministic for a pinned PDFium build and fixed flags, so a
page can be re-rendered later (redaction, verification) instead of keeping rasters in memory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from mlredact.config.schema import LimitsConfig, RenderConfig
from mlredact.core.errors import ProcessingError, ReasonCode
from mlredact.core.model import PageGeometry
from mlredact.ingest.embedded import EmbeddedWord, embedded_words

_MIN_DPI = 150


@dataclass(frozen=True, slots=True)
class RenderedPage:
    geometry: PageGeometry
    rgb: npt.NDArray[np.uint8]  # (H, W, 3), C-contiguous
    embedded_char_count: int
    image_object_count: int
    words: tuple[EmbeddedWord, ...] = ()  # engine D, only when requested


def effective_dpi(width_pt: float, height_pt: float, render: RenderConfig, limits: LimitsConfig) -> int:
    """Configured DPI, reduced deterministically for oversized pages to respect the pixel budget."""
    dpi = render.dpi
    area_in2 = (width_pt / 72.0) * (height_pt / 72.0)
    if area_in2 <= 0:
        raise ProcessingError(ReasonCode.RENDER_FAILED)
    if area_in2 * dpi * dpi > limits.max_page_pixels:
        dpi = math.floor(math.sqrt(limits.max_page_pixels / area_in2))
        if dpi < _MIN_DPI:
            raise ProcessingError(ReasonCode.PAGE_TOO_LARGE)
    return dpi


def open_document(data: bytes, render: RenderConfig) -> Any:
    import pypdfium2 as pdfium

    try:
        doc = pdfium.PdfDocument(data)
        if render.draw_forms:
            doc.init_forms()
    except Exception:
        raise ProcessingError(ReasonCode.RENDER_FAILED) from None
    return doc


def render_page(
    doc: Any, page_index: int, render: RenderConfig, limits: LimitsConfig, *, with_words: bool = False
) -> RenderedPage:
    import pypdfium2 as pdfium
    import pypdfium2.raw as pdfium_c

    try:
        page = doc[page_index]
    except Exception:
        raise ProcessingError(ReasonCode.RENDER_FAILED, page=page_index) from None
    try:
        width_pt, height_pt = page.get_size()
        dpi = effective_dpi(width_pt, height_pt, render, limits)
        bitmap = page.render(
            scale=dpi / 72.0,
            rotation=0,
            may_draw_forms=render.draw_forms,
            draw_annots=render.draw_annotations,
            fill_color=(255, 255, 255, 255),
            rev_byteorder=True,  # RGB
            prefer_bgrx=False,
            grayscale=False,
        )
        rgb = np.array(bitmap.to_numpy(), dtype=np.uint8, copy=True, order="C")
        bitmap.close()
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ProcessingError(ReasonCode.RENDER_FAILED, page=page_index)
        textpage = page.get_textpage()
        words: tuple[EmbeddedWord, ...] = ()
        try:
            chars = textpage.count_chars()
            text = textpage.get_text_range() if chars > 0 else ""
            embedded = sum(1 for c in text if not c.isspace())
            if with_words and embedded:
                words = tuple(embedded_words(page, textpage, int(rgb.shape[1]), int(rgb.shape[0])))
        finally:
            textpage.close()
        images = sum(1 for _ in page.get_objects(filter=[pdfium_c.FPDF_PAGEOBJ_IMAGE], max_depth=2))
    except ProcessingError:
        raise
    except Exception:
        raise ProcessingError(ReasonCode.RENDER_FAILED, page=page_index) from None
    finally:
        page.close()
    del pdfium
    geometry = PageGeometry(
        index=page_index,
        width_pt=round(float(width_pt), 4),
        height_pt=round(float(height_pt), 4),
        dpi=dpi,
        width_px=int(rgb.shape[1]),
        height_px=int(rgb.shape[0]),
    )
    return RenderedPage(geometry, rgb, embedded, images, words)
