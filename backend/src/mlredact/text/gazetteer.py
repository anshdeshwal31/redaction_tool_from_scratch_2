"""The Australian locality list (suburb list v2) and the state postcode ranges: the address grammar's gazetteer.

The list ships as ``resources/gazetteer`` with its attribution and provenance (ABS ASGS Edition 3, CC BY 4.0).
Every file is verified against ``SHA256SUMS`` when the list is first loaded, and the loaded list is cached for the
process (:func:`load_gazetteer`).

Detection never asks for an exact suburb and postcode pair.  A place counts when its suburb is a name of its state
(a row of either source, or an alias) and its postcode lies in the state's Australia Post ranges (PO-box blocks
included) or is one of the listed cross-border pairs.  Fakes may use only the rows with ``fake_pool`` set.

Name matching folds case, curly apostrophes, dots, apostrophes and hyphens, reads 'Mt' as 'Mount', 'St' as 'Saint',
'Pt' as 'Point', 'Nth' as 'North' and 'Sth' as 'South', and folds a reversed directional ('North Balwyn' for
'Balwyn North'), which Australia Post lists as a common error.
"""

from __future__ import annotations

import csv
import hashlib
import io
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache

from mlredact.core.errors import EnvironmentErrorMl, ReasonCode
from mlredact.resources.loader import resource_root

STATE_CODES = ("NSW", "VIC", "QLD", "SA", "WA", "TAS", "NT", "ACT")

# Australia Post's official postcode ranges (PO-box blocks included).
RANGES: Mapping[str, tuple[tuple[int, int], ...]] = {
    "ACT": ((200, 299), (2600, 2619), (2900, 2920)),
    "NSW": ((1000, 2599), (2620, 2899), (2921, 2999)),
    "NT": ((800, 999),),
    "QLD": ((4000, 4999), (9000, 9999)),
    "SA": ((5000, 5999),),
    "TAS": ((7000, 7999),),
    "VIC": ((3000, 3999), (8000, 8999)),
    "WA": ((6000, 6999),),
}

# The eight capitals, by the name Australians write (the list reaches Brisbane, Darwin and Canberra by alias).
CAPITALS: Mapping[str, str] = {
    "sydney": "NSW",
    "melbourne": "VIC",
    "brisbane": "QLD",
    "perth": "WA",
    "adelaide": "SA",
    "hobart": "TAS",
    "darwin": "NT",
    "canberra": "ACT",
}

DIRECTIONALS = ("north", "south", "east", "west")
_FOLD_WORDS = {"mt": "mount", "saint": "st", "pt": "point", "nth": "north", "sth": "south"}
_RESOURCE_DIR = "gazetteer"
_FILES = ("au_localities.csv", "aliases.csv", "cross_border.csv")


def in_range(code: str, postcode: int) -> bool:
    """True when ``postcode`` lies in the Australia Post ranges of the state ``code``."""
    return any(lo <= postcode <= hi for lo, hi in RANGES.get(code, ()))


def range_states(postcode: int) -> frozenset[str]:
    """The states whose ranges hold ``postcode`` (never inferred for a postcode followed by a foreign country)."""
    return frozenset(code for code in RANGES if in_range(code, postcode))


def fold_name(name: str) -> str:
    """The comparison key of a locality name: case, curly apostrophes, dots, apostrophes and hyphens folded, and the
    'Mt'/'Saint'/'Pt'/'Nth'/'Sth' spellings read as 'mount'/'st'/'point'/'north'/'south'."""
    s = name.casefold().replace("’", "'").replace("‘", "'").replace(".", " ").replace("'", "").replace("-", " ")
    return " ".join(_FOLD_WORDS.get(w, w) for w in s.split())


def name_variants(folded: str) -> list[str]:
    """A folded name and its reversed-directional forms ('north balwyn' -> 'balwyn north')."""
    out = [folded]
    ws = folded.split()
    if len(ws) >= 2 and ws[0] in DIRECTIONALS:
        out.append(" ".join(ws[1:] + ws[:1]))
    if len(ws) >= 2 and ws[-1] in DIRECTIONALS:
        out.append(" ".join(ws[-1:] + ws[:-1]))
    return out


def edit1(a: str, b: str) -> bool:
    """True when ``a`` and ``b`` are at most one edit apart (insert, delete or substitute)."""
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la > lb:
        a, b, la, lb = b, a, lb, la
    i = 0
    while i < la and a[i] == b[i]:
        i += 1
    if la == lb:
        return a[i + 1 :] == b[i + 1 :]
    return a[i:] == b[i + 1 :]


@dataclass(frozen=True, slots=True)
class Locality:
    """One row of the list: a locality and one of its postcodes."""

    state: str
    name: str  # display casing ('St Ives', 'Mount Victoria')
    folded: str = ""  # fold_name(name)
    postcode: str = ""  # four digits with leading zeros ('0800')
    metro: bool = False  # metro (at least half the population in a capital-city GCCSA), else regional
    primary: bool = False  # the locality's largest-share postcode
    in_state_range: bool = True
    population: int = 0
    capital_share: float = 0.0
    source: str = "sal"  # 'sal' (ABS Suburbs and Localities) or 'official_locality' (detection only)
    fake_pool: bool = False  # may be drawn as a fake address

    @property
    def klass(self) -> str:
        return "metro" if self.metro else "regional"


@dataclass(frozen=True, slots=True)
class Alias:
    """A postal name the ABS list lacks ('Brisbane' for 'Brisbane City'): it names the target's postcode."""

    state: str
    alias: str
    folded: str
    locality: str
    postcode: str


class Gazetteer:
    """Locality lookups by folded name per state, aliases, cross-border pairs, postcodes and classes."""

    def __init__(
        self,
        rows: Sequence[Locality],
        aliases: Sequence[Alias],
        cross_border: Iterable[tuple[str, str, str]],
    ) -> None:
        self.rows = tuple(rows)
        self.aliases = tuple(aliases)
        self._exceptions = frozenset((state, int(pc)) for state, _, pc in cross_border)
        self.cross_border = tuple(sorted((state, name, pc) for state, name, pc in cross_border))
        names: dict[str, set[str]] = {}
        postcodes: dict[tuple[str, str], set[int]] = {}
        by_name: dict[tuple[str, str], list[Locality]] = {}
        for r in self.rows:
            names.setdefault(r.state, set()).add(r.folded)
            postcodes.setdefault((r.state, r.folded), set()).add(int(r.postcode))
            by_name.setdefault((r.state, r.folded), []).append(r)
        for a in self.aliases:
            target = fold_name(a.locality)
            postcodes.setdefault((a.state, a.folded), set()).update(
                postcodes.get((a.state, target), set()) | {int(a.postcode)}
            )
            if (a.state, a.folded) not in by_name:
                by_name[(a.state, a.folded)] = list(by_name.get((a.state, target), ()))
            names.setdefault(a.state, set()).add(a.folded)
        self._postcodes = {k: frozenset(v) for k, v in postcodes.items()}
        self._rows_of = {k: tuple(v) for k, v in by_name.items()}
        index: dict[str, set[str]] = {}
        by_len: dict[tuple[str, int], list[str]] = {}
        for state in sorted(names):
            for n in sorted(names[state]):
                index.setdefault(n, set()).add(state)
                by_len.setdefault((state, len(n)), []).append(n)
        self._states_of = {n: frozenset(s) for n, s in index.items()}
        self._by_state_len = {k: tuple(v) for k, v in by_len.items()}
        self._fuzzy: dict[tuple[str, str | None], frozenset[str]] = {}
        self._fake_pool = tuple(
            sorted((r for r in self.rows if r.fake_pool), key=lambda r: (r.state, r.folded, r.postcode))
        )

    # ------------------------------------------------------------------ lookups
    def listed_form(self, folded: str) -> str:
        """The form of a folded name the list holds: the name itself when listed, else its reversed-directional form
        when that is listed ('north balwyn' -> 'balwyn north'), else the name."""
        return next((v for v in name_variants(folded) if v in self._states_of), folded)

    def states_of(self, words: Sequence[str], *, fuzzy: bool = False, state: str | None = None) -> frozenset[str]:
        """The states in which ``words`` name a real locality (exact, reversed directionals folded).  With ``fuzzy`` one
        edit is allowed on names of 5 or more letters (OCR damage).  ``state`` restricts the answer to that state."""
        n = fold_name(" ".join(words))
        if not n:
            return frozenset()
        got: set[str] = set()
        for v in name_variants(n):
            got |= self._states_of.get(v, frozenset())
        if state is not None:
            got &= {state}
        if got or not fuzzy or len(n.replace(" ", "")) < 5:
            return frozenset(got)
        key = (n, state)
        hit = self._fuzzy.get(key)
        if hit is None:
            out: set[str] = set()
            for st in [state] if state else list(RANGES):
                for ln in (len(n) - 1, len(n), len(n) + 1):
                    if any(edit1(n, cand) for cand in self._by_state_len.get((st, ln), ())):
                        out.add(st)
                        break
            hit = frozenset(out)
            self._fuzzy[key] = hit
        return hit

    def is_locality(self, words: Sequence[str], state: str | None = None) -> bool:
        return bool(self.states_of(words, state=state))

    def postcodes_of(self, words: Sequence[str], state: str) -> frozenset[int]:
        """Postcodes the list gives the named locality in ``state`` (an alias takes its target's)."""
        out: set[int] = set()
        for v in name_variants(fold_name(" ".join(words))):
            out |= self._postcodes.get((state, v), frozenset())
        return frozenset(out)

    def ok_pair(self, code: str, postcode: int) -> bool:
        """The postcode lies in the state's ranges, or the pair is a listed cross-border pair."""
        return in_range(code, postcode) or (code, postcode) in self._exceptions

    def klass_of(self, words: Sequence[str], state: str, postcode: str | None = None) -> str | None:
        """'metro' or 'regional' for the named locality of ``state`` (an alias takes its target's), else None.  A name
        with rows of both classes (Kingswood NSW: 2340 regional, 2747 metro) takes the class of the row of the written
        ``postcode``, else of its most populous primary row, else of its most populous row (the case table's rule)."""
        for v in name_variants(fold_name(" ".join(words))):
            rows = self._rows_of.get((state, v))
            if not rows:
                continue
            pick = [r for r in rows if postcode and r.postcode == postcode] or [r for r in rows if r.primary] or rows
            return max(pick, key=lambda r: (r.population, r.postcode)).klass
        return None

    def is_capital(self, words: Sequence[str]) -> bool:
        """True for the name of a state or territory capital ('Sydney', 'Canberra')."""
        return fold_name(" ".join(words)) in CAPITALS

    # ------------------------------------------------------------------ the fake pool
    def fake_pool(self, state: str | None = None, klass: str | None = None) -> tuple[Locality, ...]:
        """The rows a fake may use (``fake_pool`` set: primary, in range, population 200 or more, an address area),
        sorted by state, folded name and postcode; optionally of one state and one class."""
        return tuple(
            r for r in self._fake_pool if (state is None or r.state == state) and (klass is None or r.klass == klass)
        )


def _read_sums() -> dict[str, str]:
    root = resource_root() / _RESOURCE_DIR
    out: dict[str, str] = {}
    for raw in (root / "SHA256SUMS").read_text("utf-8").splitlines():
        line = raw.strip()
        if line:
            digest, name = line.split(maxsplit=1)
            out[name.lstrip("*")] = digest.lower()
    return out


def _read_verified(name: str, sums: Mapping[str, str]) -> str:
    data = (resource_root() / _RESOURCE_DIR / name).read_bytes()
    if hashlib.sha256(data).hexdigest() != sums.get(name):
        raise EnvironmentErrorMl(ReasonCode.RESOURCE_HASH_MISMATCH)
    return data.decode("utf-8")


def _flag(value: str) -> bool:
    return value.strip() == "1"


@cache
def load_gazetteer() -> Gazetteer:
    """The locality list, verified against SHA256SUMS and cached for the process."""
    sums = _read_sums()
    loc_text, alias_text, cross_text = (_read_verified(n, sums) for n in _FILES)
    rows: list[Locality] = []
    for r in csv.DictReader(io.StringIO(loc_text, newline="")):
        name = r["display_name"] or r["locality"]
        rows.append(
            Locality(
                state=r["state"],
                name=name,
                folded=fold_name(r["locality"]),
                postcode=r["postcode"],
                metro=r["class"] == "metro",
                primary=_flag(r["primary"]),
                in_state_range=_flag(r["in_state_range"]),
                population=int(r["population_2021"] or 0),
                capital_share=float(r["capital_share"] or 0.0),
                source=r["source"],
                fake_pool=_flag(r["fake_pool"]),
            )
        )
    aliases = [
        Alias(a["state"], a["alias"], fold_name(a["alias"]), a["locality"], a["postcode"])
        for a in csv.DictReader(io.StringIO(alias_text, newline=""))
    ]
    cross = [(c["state"], c["locality"], c["postcode"]) for c in csv.DictReader(io.StringIO(cross_text, newline=""))]
    return Gazetteer(rows, aliases, cross)
