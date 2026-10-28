"""Internal word and line types of the grammar, with the word-shape predicates it reads precomputed, and the
word-level readers of states and postcodes."""

from __future__ import annotations

import re
from typing import Final

from mlredact.text.address.model import AddrLine, AddrWord, Band, BoxRole
from mlredact.text.address.tables import (
    BOXNUM,
    DESIGNATORS,
    DOT_ABBREVIATIONS,
    DOTTED_STATES,
    FURNITURE,
    HN,
    HN_SLASH,
    NUMBERLESS_LEVELS,
    OCC_STOP,
    OCR_STATE_MAP,
    PC_OCR,
    SEPARATORS,
    SPELLED_STATES,
    SUFFIX_CODES,
    SUFFIX_WORDS,
    TITLE_ABBR_STATES,
    TITLES,
    TYPE_CANON,
)
from mlredact.text.gazetteer import STATE_CODES

_STATE_SET: Final = frozenset(STATE_CODES)
_INITIALS: Final = re.compile(r"(?:[A-Za-z]\.){1,4}")
_CLOCK: Final = re.compile(r"\d:\d")
_OCR_DIGITS: Final = str.maketrans("OoIlSB", "001158")


def bare(t: str) -> str:
    """A word without its surrounding brackets, quotes and trailing punctuation."""
    s = t.strip().lstrip("([{\"'‘“")
    s = s.rstrip(",;:)]}\"'’”!?")
    if s.endswith("."):
        s = s[:-1]
    return s.rstrip(",;:)]}\"'’”!?")


def pc_value(b: str) -> int | None:
    """The 4-digit postcode value of a bare word; OCR letters O/I/l/S/B are allowed when 3 or more digits are real."""
    if len(b) == 4 and b.isascii() and b.isdigit():
        return int(b)
    if PC_OCR.fullmatch(b) and sum(c.isdigit() for c in b) >= 3:
        return int(b.translate(_OCR_DIGITS))
    return None


class Word:
    """One word with its shape flags.  Immutable by convention: nothing assigns to it after construction."""

    __slots__ = (
        "alpha",
        "b",
        "blocked",
        "cap",
        "comma",
        "desig",
        "label",
        "level_type",
        "li",
        "lkey",
        "lo",
        "lowr",
        "name",
        "numw",
        "sbreak",
        "sep",
        "stop",
        "stopw",
        "suffix",
        "text",
        "title",
        "typ",
        "upper",
        "wi",
        "x0",
        "x1",
    )

    def __init__(self, src: AddrWord, li: int, wi: int) -> None:
        text = src.text
        self.text = text
        self.li = li
        self.wi = wi
        self.x0 = src.x0
        self.x1 = src.x1
        self.blocked = src.blocked
        b = bare(text)
        self.b = b
        self.lo = b.lower()
        self.lkey = text.strip().lower().rstrip(":").rstrip(".").rstrip(":")
        self.alpha = any(c.isalpha() for c in b)
        self.cap = bool(b) and b[0].isupper() and self.alpha
        letters = [c for c in b if c.isalpha()]
        self.upper = len(letters) >= 2 and all(c.isupper() for c in letters)
        self.lowr = bool(b) and b[0].islower()
        trimmed = text.rstrip()
        self.comma = trimmed.endswith(",")
        self.stop = trimmed.endswith((".", ";", "!", "?")) and not _INITIALS.fullmatch(text.strip())
        # a sentence break after the word: '; ! ?', or a full stop that does not end an abbreviation
        self.sbreak = self.stop and (
            not trimmed.endswith(".")
            or not (self.lo in TITLES or self.lo in DOT_ABBREVIATIONS or self._abbreviated_type())
        )
        self.label = trimmed.endswith(":") and not _CLOCK.search(trimmed)
        stripped = text.strip()
        self.sep = stripped in SEPARATORS or (not any(c.isalnum() for c in stripped) and stripped not in {"&", "&amp;"})
        self.title = self.lo in TITLES and (self.cap or self.upper)
        self.typ = self.lo in TYPE_CANON and (self.cap or self.upper or self.lowr)
        self.numw = bool(HN.fullmatch(b) or HN_SLASH.fullmatch(b))
        self.desig = self.lo in DESIGNATORS and (self.cap or self.upper)
        self.suffix = (self.lo in SUFFIX_WORDS and (self.cap or self.upper)) or b in SUFFIX_CODES
        self.level_type = self.lo in NUMBERLESS_LEVELS and (self.cap or self.upper)
        self.name = (self.cap or self.upper) and not self.sep and not self.label and not any(c.isdigit() for c in b)
        # words a suburb or building run never crosses
        self.stopw = self.blocked or self.lo in OCC_STOP or self.title or self.label or self.sep

    def _abbreviated_type(self) -> bool:
        canon = TYPE_CANON.get(self.lo)
        return canon is not None and self.lo != canon.lower()

    @property
    def digits(self) -> bool:
        return any(c.isdigit() for c in self.b)

    @property
    def alnum_text(self) -> bool:
        return any(c.isalnum() for c in self.text)

    def is_boxnum(self) -> bool:
        return bool(BOXNUM.fullmatch(self.b))


class Line:
    """One input line with internal words and geometry."""

    __slots__ = ("band", "box", "frame", "li", "page", "words", "x0", "x1", "y0", "y1")

    def __init__(self, src: AddrLine, li: int) -> None:
        self.li = li
        self.page = src.page
        self.x0 = src.x0
        self.y0 = src.y0
        self.x1 = src.x1
        self.y1 = src.y1
        self.band: Band = src.band
        self.box: BoxRole | None = src.box
        self.frame = src.frame
        self.words = [Word(w, li, wi) for wi, w in enumerate(src.words)]

    @property
    def h(self) -> float:
        return max(1.0, self.y1 - self.y0)

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    @property
    def furniture(self) -> bool:
        return self.band in (Band.HEADER, Band.FOOTER) or bool(FURNITURE.fullmatch(self.text.strip()))


def state_at(words: list[Word], j: int, *, ocr: bool = False) -> tuple[int, str] | None:
    """A state form ENDING at word ``j``: (start index, code).  Abbreviations count only in upper case or dotted form
    (plus Vic, Qld, Tas); spelled-out states in title or upper case; never 'Act'.  With ``ocr`` an upper-case form
    such as 'N5W' is read through the confusion table."""
    if j < 0 or j >= len(words):
        return None
    for n in (3, 2, 1):
        s = j - n + 1
        if s < 0:
            continue
        ws = words[s : j + 1]
        if n > 1 and any(w.comma or w.stop for w in ws[:-1]):
            continue
        key = tuple(w.lo for w in ws)
        if (
            key in SPELLED_STATES
            and all(w.b[:1].isupper() for w in ws)
            and all(w.cap and (w.b[1:].islower() or w.upper) for w in ws)
        ):
            return s, SPELLED_STATES[key]
        if n == 1:
            b = ws[0].b
            if b in _STATE_SET:
                return s, b
            if b in TITLE_ABBR_STATES:
                return s, TITLE_ABBR_STATES[b]
            if b in DOTTED_STATES:
                return s, DOTTED_STATES[b]
            if ocr and 2 <= len(b) <= 3 and any(c.isdigit() for c in b) and b.upper() == b:
                t = b.translate(OCR_STATE_MAP)
                if t in _STATE_SET:
                    return s, t
    return None
