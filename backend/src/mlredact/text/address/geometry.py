"""Line geometry of the grammar: the neighbours of a line (below, above, on its row, first on the next page).

Lines relate only within one page and one reading frame.  Every query reads a bisected window of the lines of its
page, never the whole document, so a 30-page document costs time linear in its lines.
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence

from mlredact.text.address.model import Band
from mlredact.text.address.tables import FURNITURE
from mlredact.text.address.words import Line

Span = tuple[float, float]


def x_overlap(a: Span, b: Span) -> float:
    """The horizontal overlap of two spans relative to the narrower one."""
    ov = min(a[1], b[1]) - max(a[0], b[0])
    narrow = max(1.0, min(a[1] - a[0], b[1] - b[0]))
    return ov / narrow


class _Group:
    """The non-empty lines of one page and reading frame."""

    __slots__ = ("by_y0", "by_y1", "max_h", "y0s", "y1s")

    def __init__(self, lines: list[Line]) -> None:
        self.by_y0 = sorted(lines, key=lambda o: (o.y0, o.x0, o.li))
        self.y0s = [o.y0 for o in self.by_y0]
        self.by_y1 = sorted(lines, key=lambda o: (o.y1, o.li))
        self.y1s = [o.y1 for o in self.by_y1]
        self.max_h = max((o.h for o in lines), default=1.0)


class Geo:
    """Neighbour queries over the lines of a document."""

    def __init__(self, lines: Sequence[Line]) -> None:
        self.lines = lines
        groups: dict[tuple[int, int], list[Line]] = {}
        for ln in lines:
            if ln.words:
                groups.setdefault((ln.page, ln.frame), []).append(ln)
        self._groups = {k: _Group(v) for k, v in groups.items()}
        # the body lines of each page that are not furniture, for the page-break join
        self._body: dict[tuple[int, int], tuple[list[float], list[Line]]] = {}
        for key, g in self._groups.items():
            body = [o for o in g.by_y0 if not o.furniture]
            self._body[key] = ([o.y0 for o in body], body)
        self._row_cache: dict[int, list[Line]] = {}
        self._rows: dict[int, int] | None = None

    # ------------------------------------------------------------------ vertical neighbours
    def below(
        self, ln: Line, span: Span | None = None, max_gap: float = 0.8, skip_furniture: bool = False
    ) -> Line | None:
        """The nearest line below ``ln`` in the same column: a gap of at most ``max_gap`` line heights and a horizontal
        overlap of 0.3 or more."""
        span = span or (ln.x0, ln.x1)
        g = self._groups.get((ln.page, ln.frame))
        if g is None:
            return None
        lo = bisect.bisect_left(g.y0s, ln.y1 - 0.3 * ln.h)
        hi = bisect.bisect_right(g.y0s, ln.y1 + max_gap * ln.h)
        best: Line | None = None
        for o in g.by_y0[lo:hi]:
            if o.li == ln.li:
                continue
            if skip_furniture and self._skippable(ln, o):
                continue
            if o.y0 < ln.y1 - 0.3 * ln.h:
                continue
            gap = o.y0 - ln.y1
            if gap > max_gap * min(ln.h, o.h):
                continue
            if x_overlap(span, (o.x0, o.x1)) < 0.3:
                continue
            if best is None or (o.y0, o.x0) < (best.y0, best.x0):
                best = o
        return best

    @staticmethod
    def _skippable(ln: Line, o: Line) -> bool:
        return bool(FURNITURE.fullmatch(o.text.strip())) or (
            ln.band is Band.BODY and o.band in (Band.HEADER, Band.FOOTER)
        )

    def above(self, ln: Line, span: Span | None = None) -> Line | None:
        """The nearest line above ``ln`` in the same column (a gap of at most 0.8 line heights)."""
        span = span or (ln.x0, ln.x1)
        g = self._groups.get((ln.page, ln.frame))
        if g is None:
            return None
        lo = bisect.bisect_left(g.y1s, ln.y0 - 0.8 * ln.h)
        hi = bisect.bisect_right(g.y1s, ln.y0 + 0.3 * ln.h)
        best: Line | None = None
        for o in g.by_y1[lo:hi]:
            if o.li == ln.li:
                continue
            if o.y1 > ln.y0 + 0.3 * ln.h:
                continue
            gap = ln.y0 - o.y1
            if gap > 0.8 * min(ln.h, o.h):
                continue
            if x_overlap(span, (o.x0, o.x1)) < 0.3:
                continue
            if best is None or o.y1 > best.y1:
                best = o
        return best

    def next_page_first(self, ln: Line) -> Line | None:
        """The first body line of the next page when ``ln`` ends its page (nothing but furniture below it), and the two
        share a column."""
        key = (ln.page, ln.frame)
        y0s, body = self._body.get(key, ([], []))
        i = bisect.bisect_right(y0s, ln.y1 - 0.3 * ln.h)
        if any(o.li != ln.li for o in body[i:]):
            return None
        _, nxt = self._body.get((ln.page + 1, ln.frame), ([], []))
        if not nxt:
            return None
        first = min(nxt, key=lambda o: (o.y0, o.x0))
        if x_overlap((ln.x0, ln.x1), (first.x0, first.x1)) < 0.3:
            return None
        return first

    # ------------------------------------------------------------------ the same row
    def same_row(self, a: Line, b: Line) -> bool:
        ov = min(a.y1, b.y1) - max(a.y0, b.y0)
        return a.page == b.page and a.frame == b.frame and ov >= 0.5 * min(a.h, b.h)

    def row_mates(self, ln: Line) -> list[Line]:
        """The other lines on ``ln``'s row (side-by-side form boxes and table cells), in document order."""
        got = self._row_cache.get(ln.li)
        if got is not None:
            return got
        out: list[Line] = []
        g = self._groups.get((ln.page, ln.frame))
        if g is not None:
            lo = bisect.bisect_left(g.y0s, ln.y0 - g.max_h)
            hi = bisect.bisect_right(g.y0s, ln.y1)
            out = [o for o in g.by_y0[lo:hi] if o.li != ln.li and self.same_row(o, ln)]
            out.sort(key=lambda o: o.li)
        self._row_cache[ln.li] = out
        return out

    def left_of(self, ln: Line) -> list[Line]:
        """The row mates wholly to the left of ``ln`` (right edge within a point of its left edge), nearest first."""
        return sorted((o for o in self.row_mates(ln) if o.x1 <= ln.x0 + 1), key=lambda o: (-o.x1, o.li))

    def right_of(self, ln: Line, x1: float) -> list[Line]:
        """The row mates whose left edge lies at or beyond ``x1`` minus a point."""
        return [o for o in self.row_mates(ln) if o.x0 >= x1 - 1 and o.words]

    def row_index(self, ln: Line) -> int:
        """The row number of a line: lines whose vertical centres lie within half a line height share one."""
        if self._rows is None:
            order = sorted((o for o in self.lines), key=lambda o: (o.page, o.frame, (o.y0 + o.y1) / 2, o.li))
            rows: dict[int, int] = {}
            row = -1
            cur: tuple[tuple[int, int], float, float] | None = None
            for o in order:
                yc = (o.y0 + o.y1) / 2
                if cur is None or (o.page, o.frame) != cur[0] or abs(yc - cur[1]) > 0.5 * min(o.h, cur[2]):
                    row += 1
                    cur = ((o.page, o.frame), yc, o.h)
                rows[o.li] = row
            self._rows = rows
        return self._rows[ln.li]
