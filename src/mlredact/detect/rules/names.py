"""Name-span scanning shared by the label and person-cue detectors."""

from __future__ import annotations

import re

from mlredact.detect.rules.lexicon import (
    MEDICAL_ABBREVIATIONS,
    NAME_TOKEN,
    NON_NAME_WORDS,
    PARTICLE,
    POST_NOMINALS,
    TITLES,
)

_TOKEN = re.compile(r"[^ \t\n]+")
_NAME_TOKEN_RE = re.compile(rf"(?:{NAME_TOKEN}|(?:[A-Z]\.){{2,3}})")
_PARTICLE_RE = re.compile(PARTICLE)
_TITLE_RE = re.compile(rf"(?i){TITLES}\.?")
_EDGE = ',;:()[]{}"“”'
_LABELISH = frozenset(
    w.lower()
    for w in """DOB D.O.B URN MRN UR DVA TFN ABN ACN CRN NDIS IHI HPI AHPRA MCN NOK ID NO REF ATTN PH MOB
    TEL FAX EMAIL CC BCC AKA NEE C/O RE DOD""".split()
)


def line_end(text: str, pos: int) -> int:
    end = text.find("\n", pos)
    return len(text) if end == -1 else end


def scan_name(
    text: str,
    start: int,
    limit: int,
    *,
    allow_title: bool = True,
    max_tokens: int = 6,
    allow_lower: bool = False,
) -> tuple[int, int] | None:
    """Return the span of a person name starting at ``start`` (titles excluded), or None.

    Accepts Capitalised / ALL-CAPS / initial tokens and particles, the inverted ``SMITH, John``
    form, and stops at post-nominals, label words, lowercase words and sentence punctuation.
    """
    limit = min(limit, line_end(text, start))
    spans: list[tuple[int, int]] = []
    seen_title = False
    for m in _TOKEN.finditer(text, start, limit):
        raw = m.group(0)
        lead = len(raw) - len(raw.lstrip(_EDGE))
        core = raw.strip(_EDGE)
        if not core:
            break
        c_start = m.start() + lead
        c_end = c_start + len(core)
        low = core.lower().rstrip(".")
        if allow_title and not spans and not seen_title and _TITLE_RE.fullmatch(core):
            seen_title = True
            continue
        if raw.endswith(":") or low in _LABELISH or low in POST_NOMINALS or low in NON_NAME_WORDS:
            break
        if len(core) <= 3 and core.isupper() and low in MEDICAL_ABBREVIATIONS:
            break
        # A sentence-final full stop ("Smith.") must not stop a word from matching the name shape;
        # initials ("J.", "J.S.") keep theirs.
        word = core[:-1] if core.endswith(".") and len(core) > 2 and not re.fullmatch(r"(?:[A-Z]\.)+", core) else core
        ok = bool(_NAME_TOKEN_RE.fullmatch(word))
        if not ok and spans and _PARTICLE_RE.fullmatch(word):
            ok = True
        if not ok and allow_lower and word.isalpha() and len(word) > 1:
            ok = True
        if not ok:
            break
        # Strip a sentence-final full stop from a normal word (keep it on initials like "J.").
        if core.endswith(".") and len(core) > 2 and not re.fullmatch(r"(?:[A-Z]\.)+", core):
            c_end -= 1
        spans.append((c_start, c_end))
        trailing = raw[len(raw.rstrip(_EDGE + ".")) :]
        if len(spans) >= max_tokens:
            break
        if trailing and "," in trailing and len(spans) == 1:
            continue  # inverted form: "SMITH, John"
        if trailing and any(ch in trailing for ch in ",;:)"):
            break
        if core.endswith(".") and len(core) > 2:
            break
    # A name must not end in a particle.
    while spans and _PARTICLE_RE.fullmatch(text[spans[-1][0] : spans[-1][1]]):
        spans.pop()
    if not spans:
        return None
    return spans[0][0], spans[-1][1]
