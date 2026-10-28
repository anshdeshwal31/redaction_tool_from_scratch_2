"""The identifying words and numbers of a document's address units, which no address fake may repeat (amendment D1).

No fake word or number of a unit may equal an identifying word or number of ANY address unit of the same document, in
any role.  Words compare after the name folding of the suburb list (case, curly apostrophes, dots, apostrophes and
hyphens folded; 'Mt' and 'Mount', 'St' and 'Saint', 'Pt' and 'Point' read as one word) and numbers compare as whole
digit runs.  A fake word also must not contain an identifying word of 3 or more letters (so it neither begins with one),
and must not be one edit from one of 5 or more letters.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

from mlredact.text.gazetteer import edit1, fold_name

_DIGITS = re.compile(r"\d+")


def number_key(run: str) -> str:
    """The comparison form of a digit run: leading zeros dropped ('02' and '2' are one number)."""
    return run.lstrip("0") or "0"


def digit_runs(text: str) -> list[str]:
    return _DIGITS.findall(text)


@dataclass(frozen=True, slots=True)
class Exclusions:
    """The folded identifying words (sorted) and the normalised identifying digit runs of a document's units."""

    words: tuple[str, ...] = field(default=(), repr=False)
    numbers: frozenset[str] = field(default=frozenset(), repr=False)

    @classmethod
    def of(cls, words: Iterable[str], numbers: Iterable[str]) -> Exclusions:
        folded = {w for text in words for w in fold_name(text).split()}
        return cls(tuple(sorted(folded)), frozenset(number_key(n) for n in numbers))

    def word_ok(self, word: str) -> bool:
        """True when no word of ``word`` equals, contains or is one edit from an identifying word."""
        for w in fold_name(word).split():
            for o in self.words:
                if w == o or (len(o) >= 3 and o in w) or (len(o) >= 5 and edit1(w, o)):
                    return False
        return True

    def number_ok(self, text: str) -> bool:
        """True when none of the digit runs of ``text`` is an identifying number."""
        return not any(number_key(r) in self.numbers for r in digit_runs(text))


NO_EXCLUSIONS = Exclusions()
