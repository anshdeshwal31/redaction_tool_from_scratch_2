"""The state of one parse: the lines, their neighbours, the words already used, the units and kept places found.

The parse is a chain of classes, each adding the methods of one stage (labels, anchors, streets, joins, the form
layer, place-only units, kept places, repeats, keys).  They share this state; ``parser.Parser`` is the last link.
"""

from __future__ import annotations

from collections.abc import Sequence

from mlredact.text.address.geometry import Geo, Span
from mlredact.text.address.internal import Anchor, KeptB, Label, Street, UnitB
from mlredact.text.address.model import AddrLine, Pos, Role
from mlredact.text.address.words import Line, Word
from mlredact.text.gazetteer import Gazetteer


class ParseState:
    def __init__(self, lines: Sequence[AddrLine], gaz: Gazetteer) -> None:
        self.gaz = gaz
        self.L: list[Line] = [Line(src, li) for li, src in enumerate(lines)]
        self.geo = Geo(self.L)
        self.used: set[Pos] = set()
        self.units: list[UnitB] = []
        self.unit_at: dict[Pos, UnitB] = {}
        self.unit_lines: set[int] = set()
        self.kept: list[KeptB] = []
        self.notes: list[str] = []
        self.labels: dict[int, list[Label]] = {}
        self.anchors: dict[int, list[Anchor]] = {}
        self.line_streets: dict[int, list[Street]] = {}
        self._cue_scans: dict[int, list[int]] = {}
        self._cell_cues: dict[int, bool] = {}
        self._stops: dict[int, set[int]] = {}

    # ------------------------------------------------------------------ positions
    def word(self, p: Pos) -> Word:
        return self.L[p[0]].words[p[1]]

    def span(self, li: int, i: int, j: int) -> Span:
        ws = self.L[li].words[i : j + 1]
        return (min(w.x0 for w in ws), max(w.x1 for w in ws))

    @staticmethod
    def is_prose(ln: Line) -> bool:
        """Running text: three or more lower-case words."""
        return sum(1 for w in ln.words if w.lowr and len(w.b) >= 2 and w.b.isalpha()) >= 3

    # ------------------------------------------------------------------ units
    def add_unit(self, u: UnitB) -> None:
        u.seq = len(self.units)
        self.units.append(u)
        for p in u.roles:
            self.used.add(p)
            self.unit_at[p] = u
            self.unit_lines.add(p[0])

    def extend_unit(self, u: UnitB, p: Pos, role: Role) -> None:
        """Add one word to a unit already added (the locality boxes of a form)."""
        u.roles[p] = role
        self.used.add(p)
        self.unit_at[p] = u
        self.unit_lines.add(p[0])
