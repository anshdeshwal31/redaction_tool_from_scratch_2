"""Test helpers: configuration for the current machine, and synthetic PageAnalysis objects (no OCR)."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlredact.config.loader import load_config
from mlredact.config.schema import AppConfig
from mlredact.core.geometry import BBox
from mlredact.core.model import Line, PageAnalysis, PageGeometry, Token, token_id
from mlredact.core.types import PageKind

# Machine-specific deployment files layered into every test configuration, e.g. on a laptop without
# the NER models:  MLREDACT_TEST_CONFIG=~/.mlredact/laptop-no-ner.yaml  (os.pathsep-separated list).
EXTRA_CONFIG = tuple(
    Path(p).expanduser() for p in os.environ.get("MLREDACT_TEST_CONFIG", "").split(os.pathsep) if p.strip()
)


def make_config(profile: str = "broad", overrides: Mapping[str, Any] | None = None) -> AppConfig:
    return load_config(profile, EXTRA_CONFIG, overrides)


def ner_enabled() -> bool:
    ner = make_config().detection.ner
    return ner.gliner.enabled or ner.privacy_filter.enabled


def pages_from_lines(
    lines: Sequence[str],
    *,
    confidences: dict[str, float] | None = None,
    alternates: dict[str, tuple[str, ...]] | None = None,
) -> list[PageAnalysis]:
    confidences = confidences or {}
    alternates = alternates or {}
    geometry = PageGeometry(0, 595.276, 841.89, 300, 2481, 3508)
    out_lines: list[Line] = []
    row = 0  # a blank input line leaves a vertical gap (paragraph break) but creates no Line
    line_no = -1
    for text in lines:
        row += 1
        if not text.strip():
            continue
        line_no += 1
        tokens: list[Token] = []
        x = 100
        y = 100 + (row - 1) * 60
        for word_no, word in enumerate(text.split()):
            w = 25 * len(word)
            conf = confidences.get(word, 0.99)
            tokens.append(
                Token(
                    token_id=token_id(0, line_no, word_no),
                    page=0,
                    line_no=line_no,
                    word_no=word_no,
                    bbox=BBox(x, y, x + w, y + 50),
                    confidence=conf,
                    min_confidence=conf,
                    engine="test",
                    text=word,
                    alternates=alternates.get(word, ()),
                )
            )
            x += w + 20
        out_lines.append(
            Line(
                line_id=f"p0000.l{line_no:04d}",
                page=0,
                line_no=line_no,
                bbox=BBox(100, y, max(x, 101), y + 50),
                quad=((100.0, y), (x, y), (x, y + 50.0), (100.0, y + 50.0)),
                confidence=0.99,
                engine="test",
                tokens=tuple(tokens),
            )
        )
    return [PageAnalysis(geometry, PageKind.IMAGE_ONLY, tuple(out_lines))]


@dataclass(frozen=True, slots=True)
class GridLine:
    """One line of a page laid out on a grid in points (A4, rows 14.4 points apart from 60, a line 12 points high):
    lines of the same row sit side by side, a skipped row number is a paragraph gap, ``x1`` defaults to 5.5 points a
    character."""

    text: str
    page: int = 0
    row: int = 0
    x0: float = 72.0
    x1: float | None = None


def column(*texts: str, page: int = 0, first_row: int = 0, x0: float = 72.0) -> list[GridLine]:
    """Lines one under the other, one row apart; an empty text leaves a paragraph gap."""
    return [GridLine(t, page, first_row + i, x0) for i, t in enumerate(texts) if t.strip()]


_PX_PER_PT = 300 / 72


def pages_from_grid(lines: Sequence[GridLine], *, page_count: int | None = None) -> list[PageAnalysis]:
    """Synthetic pages (300 dpi pixel boxes) for lines with a page, row and column: multi-page and side-by-side
    layouts that ``pages_from_lines`` cannot make.  Lines must be given in reading order, pages ascending; the
    ``(line, word)`` of the case table equal the flattened lines and words of the result."""
    count = page_count if page_count is not None else (max((ln.page for ln in lines), default=0) + 1)
    out: list[PageAnalysis] = []
    for page_no in range(count):
        geometry = PageGeometry(page_no, 595.276, 841.89, 300, 2481, 3508)
        page_lines: list[Line] = []
        for ln in (ln for ln in lines if ln.page == page_no):
            line_no = len(page_lines)
            right = ln.x1 if ln.x1 is not None else ln.x0 + 5.5 * len(ln.text)
            scale = (right - ln.x0) / max(1, len(ln.text))
            y0 = round((60 + ln.row * 14.4) * _PX_PER_PT)
            y1 = y0 + round(12 * _PX_PER_PT)
            tokens: list[Token] = []
            at = 0
            for word_no, word in enumerate(ln.text.split()):
                at = ln.text.index(word, at)
                bx0 = round((ln.x0 + at * scale) * _PX_PER_PT)
                bx1 = max(bx0 + 1, round((ln.x0 + (at + len(word)) * scale) * _PX_PER_PT))
                tokens.append(
                    Token(
                        token_id=token_id(page_no, line_no, word_no),
                        page=page_no,
                        line_no=line_no,
                        word_no=word_no,
                        bbox=BBox(bx0, y0, bx1, y1),
                        confidence=0.99,
                        min_confidence=0.99,
                        engine="test",
                        text=word,
                    )
                )
                at += len(word)
            bx0, bx1 = round(ln.x0 * _PX_PER_PT), max(round(right * _PX_PER_PT), round(ln.x0 * _PX_PER_PT) + 1)
            page_lines.append(
                Line(
                    line_id=f"p{page_no:04d}.l{line_no:04d}",
                    page=page_no,
                    line_no=line_no,
                    bbox=BBox(bx0, y0, bx1, y1),
                    quad=(
                        (float(bx0), float(y0)),
                        (float(bx1), float(y0)),
                        (float(bx1), float(y1)),
                        (float(bx0), float(y1)),
                    ),
                    confidence=0.99,
                    engine="test",
                    tokens=tuple(tokens),
                )
            )
        out.append(PageAnalysis(geometry, PageKind.IMAGE_ONLY, tuple(page_lines)))
    return out


# ------------------------------------------------------------- worker-pool tasks (importable by spawned children)
def task_echo(value: int) -> int:
    return value * 2


def task_reject(page: int) -> None:
    from mlredact.core.errors import InputRejected, ReasonCode

    raise InputRejected(ReasonCode.INPUT_ENCRYPTED, page=page)


def task_crash() -> None:
    import os

    os._exit(3)  # simulates a native crash (e.g. a hostile PDF segfaulting a parser)
