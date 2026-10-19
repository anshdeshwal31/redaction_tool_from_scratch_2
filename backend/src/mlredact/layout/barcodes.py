"""Barcodes, QR codes, DataMatrix, PDF417 (zxing-cpp, Apache-2.0).

Symbols are removed whether or not they decode: ``return_errors`` also reports symbols that were
located but could not be read (damaged or partly obscured codes still carry data).
"""

from __future__ import annotations

from mlredact.core.canonical import content_id
from mlredact.core.geometry import BBox
from mlredact.core.model import Region
from mlredact.core.types import RegionKind
from mlredact.imaging.ink import U8, scaled, to_gray


def find_barcodes(rgb: U8, page: int, dpi: int) -> list[Region]:
    import zxingcpp

    h, w = rgb.shape[:2]
    bounds = BBox(0, 0, w, h)
    found: dict[tuple[int, int, int, int], Region] = {}
    for result in zxingcpp.read_barcodes(to_gray(rgb), return_errors=True):
        p = result.position
        xs = [p.top_left.x, p.top_right.x, p.bottom_left.x, p.bottom_right.x]
        ys = [p.top_left.y, p.top_right.y, p.bottom_left.y, p.bottom_right.y]
        box = BBox(min(xs), min(ys), max(xs) + 1, max(ys) + 1)
        # Quiet zone + finder patterns: pad by 15% of the symbol size (at least ~2 mm).
        pad = max(scaled(24, dpi), round(0.15 * max(box.width, box.height)))
        box = box.pad(pad, pad, bounds)
        if box.is_empty:
            continue
        decoded = bool(result.valid)
        found[box.as_tuple()] = Region(
            region_id=content_id("G", page, RegionKind.BARCODE.value, *box.as_tuple()),
            page=page,
            kind=RegionKind.BARCODE,
            bbox=box,
            score=1.0 if decoded else 0.5,
            detector="zxing",
            rule="barcode.decoded" if decoded else "barcode.located",
        )
    return [found[k] for k in sorted(found)]
