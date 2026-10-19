"""Fusing second readers into engine A's reading (plan §6 "fusion").

Second readers are engine D (the PDF's own text layer) and engine B (Tesseract).  The rules are the
same for both, with reader-specific thresholds:

* A word overlapping an engine-A token is a second reading of the same ink.  Where engine A was
  uncertain (weakest character below ``replace_below``) and the second reader is confident, its text
  becomes the token text and A's reading an alternate; otherwise A's text stays and the second
  reading is only an alternate.  A PDF with a broken font encoding, or a Tesseract misread, therefore
  cannot overwrite a good reading, while exact text fixes the confusions (0/O, 1/l, rn/m) that
  matter most for identifiers.
* A confident word with no engine-A token (text A's detector missed) becomes a new token, so
  detection sees it ("tokens found by only one engine are kept").
* Every reading stays available to propagation through ``Token.alternates``.
"""

from __future__ import annotations

import unicodedata
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import replace

import numpy as np
import numpy.typing as npt

from mlredact.core.geometry import BBox
from mlredact.core.model import Line, Token, token_id
from mlredact.ingest.embedded import EmbeddedWord, visible
from mlredact.ocr.tesseract import ReaderWord
from mlredact.text.normalize import match_key

ENGINE_ID = "pdf-text"
_EMBEDDED_CONFIDENCE = 0.99
_CONFIDENT_READER = 0.9


def plausible_text(text: str) -> bool:
    """Not the output of a broken font encoding: printable, no private-use or replacement chars."""
    if not text or "�" in text:
        return False
    for ch in text:
        if 0xE000 <= ord(ch) <= 0xF8FF or unicodedata.category(ch) in {"Cc", "Cf", "Co", "Cs", "Cn"}:
            return False
    good = sum(ch.isalnum() or ch in ".,;:'\"()[]/\\-–&@#%+*!?$_" for ch in text)
    return good / len(text) >= 0.8


def _overlap(a: BBox, b: BBox) -> float:
    inter = a.intersection(b)
    return 0.0 if inter is None else inter.area / max(1, min(a.area, b.area))


def fuse_reader(
    lines: Sequence[Line],
    words: Sequence[ReaderWord],
    page: int,
    *,
    engine_id: str,
    replace_below: float,
    min_new_confidence: float,
) -> tuple[Line, ...]:
    if not words:
        return tuple(lines)
    slots = [(li, ti, t) for li, line in enumerate(lines) for ti, t in enumerate(line.tokens)]
    matched: dict[tuple[int, int], list[ReaderWord]] = defaultdict(list)
    unmatched: list[ReaderWord] = []
    for w in words:
        best = max(slots, key=lambda s: (_overlap(w.bbox, s[2].bbox), -s[0], -s[1]), default=None)
        if best is not None and _overlap(w.bbox, best[2].bbox) >= 0.5:
            matched[(best[0], best[1])].append(w)
        elif w.confidence >= min_new_confidence:
            unmatched.append(w)

    out: list[Line] = []
    for li, line in enumerate(lines):
        tokens = []
        for ti, t in enumerate(line.tokens):
            ws = sorted(matched.get((li, ti), []), key=lambda w: (w.bbox.x0, w.bbox.y0))
            other = "".join(w.text for w in ws)
            if not ws or match_key(other) == match_key(t.text):
                tokens.append(t)
            elif t.min_confidence < replace_below and min(w.confidence for w in ws) >= _CONFIDENT_READER:
                tokens.append(replace(t, text=other, alternates=tuple(dict.fromkeys((t.text, *t.alternates)))))
            else:
                tokens.append(replace(t, alternates=tuple(dict.fromkeys((*t.alternates, other)))))
        out.append(replace(line, tokens=tuple(tokens)))

    next_line = max((line.line_no for line in lines), default=-1) + 1
    groups: dict[int, list[ReaderWord]] = defaultdict(list)
    for w in unmatched:
        groups[w.line].append(w)
    for k, key in enumerate(sorted(groups)):
        ws = sorted(groups[key], key=lambda w: (w.bbox.x0, w.bbox.y0))
        line_no = next_line + k
        new_tokens = tuple(
            Token(token_id(page, line_no, i), page, line_no, i, w.bbox, w.confidence, w.confidence, engine_id, w.text)
            for i, w in enumerate(ws)
        )
        box = new_tokens[0].bbox
        for t in new_tokens[1:]:
            box = box.union(t.bbox)
        quad = (
            (float(box.x0), float(box.y0)),
            (float(box.x1), float(box.y0)),
            (float(box.x1), float(box.y1)),
            (float(box.x0), float(box.y1)),
        )
        conf = round(sum(t.confidence for t in new_tokens) / len(new_tokens), 5)
        out.append(Line(f"p{page:04d}.l{line_no:04d}", page, line_no, box, quad, conf, engine_id, new_tokens))
    return tuple(out)


def fuse_embedded(
    lines: Sequence[Line],
    words: Sequence[EmbeddedWord],
    rgb: npt.NDArray[np.uint8],
    page: int,
    *,
    replace_below: float = 0.9,
) -> tuple[Line, ...]:
    """Engine D: only visible, plausibly encoded words; exact text, so always confident."""
    shown = [
        ReaderWord(w.text, w.bbox, _EMBEDDED_CONFIDENCE, w.line)
        for w in words
        if visible(w, rgb) and plausible_text(w.text)
    ]
    return fuse_reader(lines, shown, page, engine_id=ENGINE_ID, replace_below=replace_below, min_new_confidence=0.0)


def fuse_tesseract(
    lines: Sequence[Line],
    words: Sequence[ReaderWord],
    page: int,
    *,
    replace_below: float,
    min_new_confidence: float,
) -> tuple[Line, ...]:
    """Engine B: words below ``min_new_confidence`` may only add alternates, never new tokens."""
    readable = [w for w in words if plausible_text(w.text)]
    return fuse_reader(
        lines,
        readable,
        page,
        engine_id="tesseract",
        replace_below=replace_below,
        min_new_confidence=min_new_confidence,
    )
