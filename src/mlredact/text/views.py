"""Text views over canonical tokens (plan §7.2).

A view is a string plus an exact map from character offsets back to tokens.  Derived views
(title-cased, OCR-normalised) are *length-preserving* transforms of the base view, so detector hits
in any view map to the same tokens without any re-alignment.
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence
from dataclasses import dataclass, field

from mlredact.core.model import PageAnalysis, Token
from mlredact.core.types import ViewKind
from mlredact.text.normalize import ocr_digit_normalise, titlecase_allcaps


@dataclass(frozen=True, slots=True)
class TextView:
    kind: ViewKind
    text: str = field(repr=False)
    starts: tuple[int, ...]
    ends: tuple[int, ...]
    tokens: tuple[Token, ...]

    def token_range(self, start: int, end: int) -> range:
        """Indices of tokens overlapping the half-open character span ``[start, end)``."""
        if end <= start:
            return range(0)
        first = bisect.bisect_right(self.ends, start)
        last = bisect.bisect_left(self.starts, end)
        return range(first, last)

    def token_at(self, offset: int) -> int | None:
        i = bisect.bisect_right(self.starts, offset) - 1
        if i >= 0 and self.starts[i] <= offset < self.ends[i]:
            return i
        return None

    def derive(self, kind: ViewKind, transform: str) -> TextView:
        if len(transform) != len(self.text):
            raise ValueError("derived views must be length-preserving")
        return TextView(kind, transform, self.starts, self.ends, self.tokens)


PAGE_SEPARATOR = "\n\n"


def build_line_view(pages: Sequence[PageAnalysis]) -> TextView:
    """Tokens joined by single spaces, lines by ``\\n``, pages by a blank line.

    Line-local patterns use ``[ \\t]`` instead of ``\\s`` so they never cross a line break.
    """
    parts: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    tokens: list[Token] = []
    pos = 0
    for p_i, page in enumerate(pages):
        if p_i > 0:
            parts.append(PAGE_SEPARATOR)
            pos += len(PAGE_SEPARATOR)
        for l_i, line in enumerate(page.lines):
            if l_i > 0:
                parts.append("\n")
                pos += 1
            for t_i, tok in enumerate(line.tokens):
                if t_i > 0:
                    parts.append(" ")
                    pos += 1
                starts.append(pos)
                parts.append(tok.text)
                pos += len(tok.text)
                ends.append(pos)
                tokens.append(tok)
    return TextView(ViewKind.LINE, "".join(parts), tuple(starts), tuple(ends), tuple(tokens))


def build_views(pages: Sequence[PageAnalysis]) -> dict[ViewKind, TextView]:
    line = build_line_view(pages)
    views = {
        ViewKind.LINE: line,
        ViewKind.TITLECASE: line.derive(ViewKind.TITLECASE, titlecase_allcaps(line.text)),
        ViewKind.OCR_NORMALISED: line.derive(ViewKind.OCR_NORMALISED, ocr_digit_normalise(line.text)),
    }
    return views
