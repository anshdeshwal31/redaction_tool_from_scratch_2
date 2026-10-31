"""Pages that display sideways or upside down: the pipeline does not redact them, so their document is held for review.

Everything downstream of page analysis reads a page as it is displayed, left to right: the detectors and the address
grammar read lines across the page, and the planner draws each fake across its line's box, or blacks the mention out
when the line is tilted (``render.plan``).  A page whose text mostly runs sideways or upside down therefore comes out
wrecked (broken text, misplaced fakes and bars, fakes typeset upright over upside-down lines, a mention only partly
removed), yet it passed verification and was released.  It is held instead, with a reason that names the cause:

- ``E_PAGE_ROTATED_SIDEWAYS``: the page's /Rotate setting (90 or 270) turns its text on its side;
- ``E_PAGE_CONTENT_SIDEWAYS``: the setting leaves the page upright (0 or 180), but its text was drawn or scanned on
  its side;
- ``E_PAGE_ROTATED_UPSIDE_DOWN``: the page's /Rotate setting (180) turns its text upside down;
- ``E_PAGE_CONTENT_UPSIDE_DOWN``: the setting does not turn the page half a turn, but its text was drawn or scanned
  upside down.

The judgement is on the text as displayed, so a page whose /Rotate turns sideways or upside-down content upright (a
landscape scan stored on its side, a scan stored upside down) is processed as before, and so is a page with a little
turned text (a margin note, a turned table header, a stamp).

Sideways.  A line's text runs along the long side of its box, the way OCR reads it (``ocr.ppocr`` turns a detected box
at least 1.5 times taller than wide and reads it as vertical text, giving each of its words the whole line's box).
This holds for every kind of line: OCR's boxes (quadrilaterals, possibly tilted), and the plain boxes of lines read
from the PDF's own text layer or added by a second reader.  The line runs sideways when that long side is more than 45
degrees from horizontal.  A box less than 1.5 times longer than wide shows no direction and counts as upright.

Upside down.  A box shows no difference between text that runs left to right and text turned half a turn, so this
comes from OCR: its line classifier turns each line it finds upside down before reading it, and the line keeps that
mark (``Line.upside_down``).  OCR reads every page, the PDF's own text layer included, so the mark covers a page set
upside down by /Rotate as well as one scanned upside down.  A line added only by a second reader carries no mark.  A
line that runs sideways counts as sideways only.

A page displays sideways when at least half of its characters, and at least ``MIN_SIDEWAYS_CHARS`` of them, lie on
sideways lines; it displays upside down, by the same rule, on upside-down lines.  Sideways is judged first.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from mlredact.core.errors import ReasonCode
from mlredact.core.model import Line, PageAnalysis

SIDEWAYS_DEGREES = 45.0  # text running further than this from horizontal runs sideways
MIN_SIDEWAYS_CHARS = 30  # about a line of text: fewer turned characters never hold a page
ELONGATION = 1.5  # a box at least this many times longer than wide runs along its long side (OCR's own rule)

SIDEWAYS_CODES = frozenset({ReasonCode.PAGE_ROTATED_SIDEWAYS, ReasonCode.PAGE_CONTENT_SIDEWAYS})
UPSIDE_DOWN_CODES = frozenset({ReasonCode.PAGE_ROTATED_UPSIDE_DOWN, ReasonCode.PAGE_CONTENT_UPSIDE_DOWN})


@dataclass(frozen=True, slots=True)
class SidewaysPage:
    """A page held for review, and why (one of the four codes above)."""

    page: int
    reason: ReasonCode

    @property
    def upside_down(self) -> bool:
        return self.reason in UPSIDE_DOWN_CODES


def line_runs_sideways(line: Line) -> bool:
    """Does the line's text run sideways as displayed: is the long side of its box more than ``SIDEWAYS_DEGREES`` from
    horizontal?  A box too square to show a direction runs across (see the module docstring)."""
    (ax, ay), (bx, by), _, (dx, dy) = line.quad
    top, side = (bx - ax, by - ay), (dx - ax, dy - ay)  # the edges along and across the quad's first side
    top_len, side_len = math.hypot(*top), math.hypot(*side)
    if top_len >= ELONGATION * side_len:
        run = top
    elif side_len >= ELONGATION * top_len:
        run = side
    else:
        return False
    angle = abs(math.degrees(math.atan2(run[1], run[0])))
    return min(angle, 180.0 - angle) > SIDEWAYS_DEGREES


def _mostly(part: int, total: int) -> bool:
    return part >= MIN_SIDEWAYS_CHARS and 2 * part >= total


def sideways_pages(pages: Sequence[PageAnalysis], rotations: Sequence[int]) -> list[SidewaysPage]:
    """The pages whose text mostly runs sideways or upside down as displayed, in page order.  ``rotations``: each
    page's /Rotate as the renderer applies it (``InspectionReport.rotations``; a page it does not cover counts as 0)."""
    held: list[SidewaysPage] = []
    for page in pages:
        total = sideways = upside_down = 0
        for line in page.lines:
            chars = sum(len(t.text) for t in line.tokens)
            total += chars
            if not chars:
                continue
            if line_runs_sideways(line):
                sideways += chars
            elif line.upside_down:
                upside_down += chars
        index = page.geometry.index
        rotation = rotations[index] if index < len(rotations) else 0
        if _mostly(sideways, total):
            turned = rotation in (90, 270)
            held.append(
                SidewaysPage(index, ReasonCode.PAGE_ROTATED_SIDEWAYS if turned else ReasonCode.PAGE_CONTENT_SIDEWAYS)
            )
        elif _mostly(upside_down, total):
            turned = rotation == 180
            held.append(
                SidewaysPage(
                    index, ReasonCode.PAGE_ROTATED_UPSIDE_DOWN if turned else ReasonCode.PAGE_CONTENT_UPSIDE_DOWN
                )
            )
    return held
