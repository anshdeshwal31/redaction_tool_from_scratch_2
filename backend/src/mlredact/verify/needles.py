"""What V2 searches an output's text for to find an address left behind (FINAL_PLAN TOOL2 section 6, amendment H).

An address unit is searched by its verification targets (:class:`mlredact.policy.addresses.UnitTargets`): compound
targets (consecutive words) and whole-token targets (the suburb, street-name and building words).  Matching is by whole
tokens: a token is a maximal run of letters or a maximal run of digits, case folded, so ``14/03/2021`` is the tokens 14,
03 and 2021, ``$12,750.00`` is 12, 750 and 00, and a digit never joins another number's digits.  There is no
concatenated-digit needle.

Every occurrence in the output's text must be accounted for by an exempt position: a word of a drawn fake, of a place
mentioned on its own or of a mention the profile keeps.  The check compares counts: the same extraction reads the text
the builder wrote (``found``) and that text with every word no exemption covers masked (``allowed``).  An occurrence
that survives only unmasked has a word nothing exempts: a hit.

Pure and independent of rendering and detection: it reads text and token tuples only.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

#: The token that stands in the exempt rendition for a word that no exemption covers.  It is its own token (never a
#: part of a target), so masking never makes two words adjacent.
MASK = "~"
#: Whole-token targets shorter than this are not searched alone (the compound targets hold them in context): a
#: two-letter word would match ordinary prose.
MIN_WHOLE = 3

_TOKEN = re.compile(r"\d+|[^\W\d_]+|" + re.escape(MASK))

Seq = tuple[str, ...]


def tokens_of(text: str) -> list[str]:
    """The whole tokens of ``text``: maximal runs of letters and maximal runs of digits, case folded."""
    return _TOKEN.findall(unicodedata.normalize("NFKC", text).casefold())


class _Targets(Protocol):
    @property
    def compound(self) -> Sequence[Sequence[str]]: ...

    @property
    def whole(self) -> Sequence[str]: ...


@dataclass(frozen=True, slots=True)
class AddressNeedles:
    """The token sequences V2 searches for: ``compound`` and ``whole`` targets of all the document's units, sorted and
    unique."""

    compound: tuple[Seq, ...] = field(default=(), repr=False)
    whole: tuple[Seq, ...] = field(default=(), repr=False)

    @property
    def sequences(self) -> tuple[Seq, ...]:
        return tuple(sorted({*self.compound, *self.whole}))

    def __bool__(self) -> bool:
        return bool(self.compound or self.whole)


def needles_of(targets: Iterable[_Targets]) -> AddressNeedles:
    """The needles of units' targets: words are cut into tokens (a hyphenated or apostrophe word is a sequence)."""
    compound: set[Seq] = set()
    whole: set[Seq] = set()
    for target in targets:
        for words in target.compound:
            seq = tuple(tok for word in words for tok in tokens_of(word) if tok != MASK)
            if len(seq) >= 2:
                compound.add(seq)
        for word in target.whole:
            seq = tuple(tok for tok in tokens_of(word) if tok != MASK)
            if seq and sum(len(t) for t in seq) >= MIN_WHOLE and any(c.isalpha() for t in seq for c in t):
                whole.add(seq)
    return AddressNeedles(tuple(sorted(compound)), tuple(sorted(whole)))


def count_sequences(needles: AddressNeedles, texts: Iterable[str]) -> Mapping[Seq, int]:
    """How many times each needle occurs in the texts (each text on its own: a sequence never crosses two texts)."""
    by_first: dict[str, list[Seq]] = {}
    for seq in needles.sequences:
        by_first.setdefault(seq[0], []).append(seq)
    counts: Counter[Seq] = Counter()
    for text in texts:
        toks = tokens_of(text)
        for i, tok in enumerate(toks):
            for seq in by_first.get(tok, ()):
                if tuple(toks[i : i + len(seq)]) == seq:
                    counts[seq] += 1
    return counts


@dataclass(frozen=True, slots=True)
class AddressText:
    """What V2 needs for address units: the needles and the exempt rendition of the output's text layer (the text layer
    with every word no exemption covers masked, as a PDF; None when the output has no text layer)."""

    needles: AddressNeedles
    exempt_pdf: bytes | None = field(default=None, repr=False)


def excess(needles: AddressNeedles, found: Mapping[Seq, int], allowed: Mapping[Seq, int]) -> int:
    """The number of needles that occur more often in the output than the exempt positions account for."""
    return sum(1 for seq in needles.sequences if found.get(seq, 0) > allowed.get(seq, 0))
