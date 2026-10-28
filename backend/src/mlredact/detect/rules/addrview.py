"""The address grammar over a line view: the rule detectors that read addresses (address, labels) share one parse.

The grammar (:mod:`mlredact.text.address`) reads words with positions and line geometry.  A :class:`TextView` holds the
tokens of a document in reading order, each with its page, line and box, so the lines the grammar needs follow from it
directly.  The parse is cached for the view it was made from (the last one), so the two detectors parse a document
once.  Nothing here holds document text beyond the view it was given.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from mlredact.core.model import Token
from mlredact.text.address import AddressParse, AddrLine, AddrWord, Role, parse_document
from mlredact.text.address.model import Pos
from mlredact.text.views import TextView

_SIDEWAYS_DEGREES = 25.0

#: The labels of the form layer's lexicon.  Any other colon phrase the grammar reads as a label is only a guess.
FORM_LABEL_ROLES = frozenset({"address", "street", "street_name", "street_no", "unit", "suburb", "state", "postcode"})


@dataclass(frozen=True, slots=True)
class AddressView:
    """The grammar's parse of a view, with the map between grammar positions ``(line, word)`` and token indices of
    the view."""

    parse: AddressParse
    lines: tuple[AddrLine, ...]
    token_at: Mapping[Pos, int]
    pos_of: Mapping[int, Pos]
    role_at: Mapping[int, Role]  # every token of every unit, with its role
    label_tokens: frozenset[int]  # tokens of every label of the form layer's lexicon


def _frame(tokens: list[Token]) -> int:
    """The reading frame of a line from the direction of its words: 0 unless they run vertically."""
    if len(tokens) < 2:
        return 0
    first, last = tokens[0].bbox, tokens[-1].bbox
    dx = (last.x0 + last.x1) - (first.x0 + first.x1)
    dy = (last.y0 + last.y1) - (first.y0 + first.y1)
    angle = math.degrees(math.atan2(dy, dx))
    if abs(angle) <= _SIDEWAYS_DEGREES:
        return 0
    if abs(angle) >= 180.0 - _SIDEWAYS_DEGREES:
        return 180
    return 90 if angle > 0 else 270


def _build(view: TextView) -> AddressView:
    lines: list[AddrLine] = []
    token_at: dict[Pos, int] = {}
    pos_of: dict[int, Pos] = {}
    i, n = 0, len(view.tokens)
    while i < n:
        key = (view.tokens[i].page, view.tokens[i].line_no)
        j = i
        while j < n and (view.tokens[j].page, view.tokens[j].line_no) == key:
            j += 1
        toks = list(view.tokens[i:j])
        li = len(lines)
        for wi in range(len(toks)):
            token_at[(li, wi)] = i + wi
            pos_of[i + wi] = (li, wi)
        lines.append(
            AddrLine(
                tuple(AddrWord(t.text, float(t.bbox.x0), float(t.bbox.x1)) for t in toks),
                page=key[0],
                x0=float(min(t.bbox.x0 for t in toks)),
                y0=float(min(t.bbox.y0 for t in toks)),
                x1=float(max(t.bbox.x1 for t in toks)),
                y1=float(max(t.bbox.y1 for t in toks)),
                frame=_frame(toks),
            )
        )
        i = j
    parse = parse_document(lines)
    return AddressView(
        parse,
        tuple(lines),
        token_at,
        pos_of,
        {token_at[p]: role for u in parse.units for p, role in u.roles.items()},
        frozenset(
            token_at[(lab.line, wi)]
            for lab in parse.labels
            if lab.role in FORM_LABEL_ROLES
            for wi in range(lab.first, lab.last + 1)
        ),
    )


_last: tuple[TextView, AddressView] | None = None


def address_view(view: TextView) -> AddressView:
    """The parse of ``view``.  The most recent one is kept, so the detectors of one document share it."""
    global _last
    cached = _last
    if cached is not None and cached[0] is view:
        return cached[1]
    built = _build(view)
    _last = (view, built)
    return built
