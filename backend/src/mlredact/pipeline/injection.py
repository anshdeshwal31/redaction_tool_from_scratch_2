"""Stage inputs injected by the benchmark (``mlredact bench`` oracle modes): for the evaluator, never production.

The benchmark can give the tool perfect upstream input, from its ground truth or from another tool's output: the
text (``oracle_text``), the mentions to redact (``oracle_mentions``), their clusters (``oracle_entities``) and the
regions to apply (``oracle_regions``).  This module turns the injected text into tool 2's lines.

Injected coordinates are display points (``display_pt``: the page as displayed, origin top-left, y down).  The
canonical raster is the displayed page, so a point converts to a pixel with the page's own dpi alone:
px = pt * dpi / 72.

Pure, no I/O.  It imports only ``mlredact.core``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from mlredact.core.geometry import BBox, Quad, union_all
from mlredact.core.model import Line, PageGeometry, Token, token_id

INJECTED_ENGINE = "inject"
# A pixel coordinate this close to a whole pixel is that pixel: display points written with 3 decimals (the
# benchmark's files, ``mlredact.bench.artifacts.display_poly``) then convert back to the box they came from.
_SNAP_PX = 0.01


@dataclass(frozen=True, slots=True)
class InjectedToken:
    """An injected token, numbered as tool 2 numbers its tokens.  ``points``: its polygon in display points (at least
    3 corners, in the benchmark's order: the top edge first in reading direction); ``confidence``: the injected one,
    or 1.0."""

    line_no: int
    word_no: int
    points: tuple[tuple[float, float], ...]
    confidence: float
    text: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class InjectedPage:
    """One page's injected tokens in reading order, line after line (each line's tokens one after another)."""

    index: int
    tokens: tuple[InjectedToken, ...] = field(default=(), repr=False)


def _px(value: float, dpi: int) -> float:
    px = value * dpi / 72.0
    nearest = round(px)
    return float(nearest) if abs(px - nearest) <= _SNAP_PX else px


def _within(lo: int, hi: int, size: int) -> tuple[int, int]:
    """A half-open pixel extent clipped to ``[0, size)``, at least 1 px."""
    lo = min(max(lo, 0), size - 1)
    return lo, min(max(hi, lo + 1), size)


def pt_box(points: Sequence[tuple[float, float]], geometry: PageGeometry) -> BBox:
    """Display points -> the canonical-raster pixel box around them, rounded outward (a coordinate within 0.01 px of
    a whole pixel is that pixel), clipped to the page and at least 1 px wide and high."""
    box = BBox.from_points((_px(x, geometry.dpi), _px(y, geometry.dpi)) for x, y in points)
    x0, x1 = _within(box.x0, box.x1, geometry.width_px)
    y0, y1 = _within(box.y0, box.y1, geometry.height_px)
    return BBox(x0, y0, x1, y1)


def _quad(first: InjectedToken, last: InjectedToken, box: BBox, dpi: int) -> Quad:
    """A line's corners in pixels: the first token's left edge and the last token's right edge when both polygons
    are quadrilaterals (so a sideways line stays sideways for planning), otherwise the line's box."""
    if len(first.points) == 4 and len(last.points) == 4:
        (x0, y0), (x1, y1) = first.points[0], last.points[1]
        (x2, y2), (x3, y3) = last.points[2], first.points[3]
        s = dpi / 72.0
        return ((x0 * s, y0 * s), (x1 * s, y1 * s), (x2 * s, y2 * s), (x3 * s, y3 * s))
    return (
        (float(box.x0), float(box.y0)),
        (float(box.x1), float(box.y0)),
        (float(box.x1), float(box.y1)),
        (float(box.x0), float(box.y1)),
    )


def injected_lines(page: InjectedPage, geometry: PageGeometry) -> tuple[Line, ...]:
    """The page's injected tokens as tool 2's lines, built as OCR fusion builds new lines: one line per run of
    tokens with the same line number, pixel boxes from :func:`pt_box`, the injected confidence as both the mean and
    the weakest character confidence, no alternate readings."""
    if page.index != geometry.index:
        raise ValueError("the injected page is not the rendered page")
    runs: list[list[tuple[InjectedToken, BBox]]] = []
    for tok in page.tokens:
        box = pt_box(tok.points, geometry)
        if runs and runs[-1][0][0].line_no == tok.line_no:
            runs[-1].append((tok, box))
        else:
            runs.append([(tok, box)])
    words = {(t.line_no, t.word_no) for t in page.tokens}
    if len({run[0][0].line_no for run in runs}) != len(runs) or len(words) != len(page.tokens):
        raise ValueError("injected line and word numbers must be unique on a page")
    lines = []
    for run in runs:
        line_no = run[0][0].line_no
        tokens = tuple(
            Token(
                token_id(page.index, line_no, t.word_no),
                page.index,
                line_no,
                t.word_no,
                box,
                t.confidence,
                t.confidence,
                INJECTED_ENGINE,
                t.text,
            )
            for t, box in run
        )
        bbox = union_all([t.bbox for t in tokens])
        conf = round(sum(t.confidence for t in tokens) / len(tokens), 5)
        quad = _quad(run[0][0], run[-1][0], bbox, geometry.dpi)
        lines.append(
            Line(f"p{page.index:04d}.l{line_no:04d}", page.index, line_no, bbox, quad, conf, INJECTED_ENGINE, tokens)
        )
    return tuple(lines)
