"""Pinned surrogate pools (resources/surrogates, verified via SHA256SUMS)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

from mlredact.resources.loader import resource_root


def _lines(name: str) -> tuple[str, ...]:
    text = (resource_root() / "surrogates" / name).read_text("utf-8")
    return tuple(line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#"))


@dataclass(frozen=True, slots=True)
class Locality:
    state: str
    name: str
    postcode: str


@dataclass(frozen=True, slots=True)
class Pools:
    given_male: tuple[str, ...]
    given_female: tuple[str, ...]
    given_neutral: tuple[str, ...]
    surnames: tuple[str, ...]
    street_names: tuple[str, ...]
    org_stems: tuple[str, ...]
    localities: tuple[Locality, ...]
    eponyms: frozenset[str]

    def localities_in(self, state: str | None) -> tuple[Locality, ...]:
        if state:
            in_state = tuple(loc for loc in self.localities if loc.state == state)
            if in_state:
                return in_state
        return self.localities


@cache
def load_pools() -> Pools:
    locs = []
    for line in _lines("localities.tsv"):
        state, name, postcode = line.split("\t")
        locs.append(Locality(state, name, postcode))
    return Pools(
        given_male=_lines("given_male.txt"),
        given_female=_lines("given_female.txt"),
        given_neutral=_lines("given_neutral.txt"),
        surnames=_lines("surnames.txt"),
        street_names=_lines("street_names.txt"),
        org_stems=_lines("org_stems.txt"),
        localities=tuple(locs),
        eponyms=frozenset(e.lower() for e in _lines("eponyms.txt")),
    )
