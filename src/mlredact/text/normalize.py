"""Length-preserving text normalisations and comparison keys."""

from __future__ import annotations

import re
import unicodedata

# Letters OCR commonly produces in place of digits.  Applied only inside digit-dominant words.
_DIGIT_CONFUSIONS = str.maketrans(
    {
        "O": "0",
        "o": "0",
        "Q": "0",
        "D": "0",
        "U": "0",
        "I": "1",
        "l": "1",
        "|": "1",
        "i": "1",
        "!": "1",
        "L": "1",
        "J": "1",
        "Z": "2",
        "z": "2",
        "E": "3",
        "A": "4",
        "S": "5",
        "s": "5",
        "G": "6",
        "b": "6",
        "T": "7",
        "B": "8",
        "g": "9",
        "q": "9",
    }
)
_WORD = re.compile(r"\S+")
_ALLCAPS_WORD = re.compile(r"\b[A-Z][A-Z'\-]+\b")


def ocr_digit_normalise(text: str) -> str:
    """Map digit-like letters to digits inside words that are at least half digits.

    ``"2953 7O15O 1"`` -> ``"2953 70150 1"``.  Words without enough digits are left untouched, so
    ordinary prose is unaffected.  Length-preserving.
    """

    def fix(m: re.Match[str]) -> str:
        word = m.group(0)
        digits = sum(c.isdigit() for c in word)
        alnum = sum(c.isalnum() for c in word)
        if alnum and digits * 2 >= alnum:
            return word.translate(_DIGIT_CONFUSIONS)
        return word

    return _WORD.sub(fix, text)


def titlecase_allcaps(text: str) -> str:
    """``"SMITH"`` -> ``"Smith"`` (length-preserving) so case-sensitive NER/regex see name shape."""
    return _ALLCAPS_WORD.sub(lambda m: m.group(0)[0] + m.group(0)[1:].lower(), text)


_STRIP = re.compile(r"[^\w]+", re.UNICODE)


def match_key(value: str) -> str:
    """Comparison key: NFKC, case-folded, punctuation/whitespace removed."""
    value = unicodedata.normalize("NFKC", value).casefold()
    return _STRIP.sub("", value)


def digits_only(value: str) -> str:
    return "".join(c for c in value if c.isdigit())


def collapse_ws(value: str) -> str:
    return " ".join(value.split())
