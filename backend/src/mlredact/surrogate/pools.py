"""Pinned surrogate pools (resources/surrogates, verified via SHA256SUMS) and the fake-locality pool.

Fake localities come from the suburb list (``resources/gazetteer``, ABS ASGS Edition 3, CC BY 4.0) and only from its
rows with ``fake_pool`` set: primary, in the state's postcode range, population 200 or more, an address area.
Detection-only and cross-border rows are never fakes.  The list is verified against its own ``SHA256SUMS`` when it is
first loaded."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

from mlredact.resources.loader import resource_root
from mlredact.text.gazetteer import Locality, load_gazetteer

__all__ = ["Locality", "Pools", "load_pools", "locality_pool"]


def _lines(name: str) -> tuple[str, ...]:
    text = (resource_root() / "surrogates" / name).read_text("utf-8")
    return tuple(line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#"))


@cache
def locality_pool(state: str | None = None, klass: str | None = None) -> tuple[Locality, ...]:
    """The localities a fake may use, of one state and one class ('metro' or 'regional') or of any, sorted by state,
    folded name and postcode (a stable order: keyed choices index into it)."""
    return load_gazetteer().fake_pool(state, klass)


@dataclass(frozen=True, slots=True)
class Pools:
    given_male: tuple[str, ...]
    given_female: tuple[str, ...]
    given_neutral: tuple[str, ...]
    surnames: tuple[str, ...]
    street_names: tuple[str, ...]
    org_stems: tuple[str, ...]
    eponyms: frozenset[str]

    @property
    def localities(self) -> tuple[Locality, ...]:
        """Every locality a fake may use (the list's ``fake_pool`` rows)."""
        return locality_pool()

    def localities_in(self, state: str | None = None, klass: str | None = None) -> tuple[Locality, ...]:
        """The fake localities of ``state`` and ``klass`` (either may be None for any); empty when none match."""
        return locality_pool(state, klass)


@cache
def load_pools() -> Pools:
    return Pools(
        given_male=_lines("given_male.txt"),
        given_female=_lines("given_female.txt"),
        given_neutral=_lines("given_neutral.txt"),
        surnames=_lines("surnames.txt"),
        street_names=_lines("street_names.txt"),
        org_stems=_lines("org_stems.txt"),
        eponyms=frozenset(e.lower() for e in _lines("eponyms.txt")),
    )
