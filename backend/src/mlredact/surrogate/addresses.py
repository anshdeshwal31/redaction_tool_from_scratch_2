"""One realistic, same-shape fake per address unit (FINAL_PLAN TOOL2 section 5, as amended by D, E and F; the owner's
ruling on overseas addresses).

The join (:mod:`mlredact.policy.addresses`) finds the units and gives every word its role; this module writes the fake.
Every part is chosen under the scope key from the unit's canonical key, never from an original part alone:

* the street line from (house number, street name, canonical type) or (box kind, number), so 'St' and 'Street', case,
  commas, line breaks, OCR forms and a street line seen without its locality give one fake street line;
* unit and level numbers from the street key plus the unit or level;
* suburb and postcode from the street key plus the suburb and state; a place-only unit from the place plus the
  canonical name of the person or organisation heading its block.

No choice depends on the rest of the document, with two exceptions that are rare by construction: the exclusion filter
of amendment D1 (no fake word or number may repeat an identifying word or number of ANY unit of the document), and the
rule that different originals never get the same whole fake (a clash is redrawn).  Every choice is a keyed draw whose
first candidate does not depend on the filters, so a filter that rejects other candidates leaves the fake as it was.

Per part: designators, box kinds, 'Cnr', '&', 'DX', corner words, 'via', the country and 'Australia' stay.  Numbers
keep their digit count with every digit different at its position (a non-zero lead; ranges ascending; a letter suffix
another letter).  Levels become another level.  The street name comes from the pool in the original's case, and the
street type becomes another canonical type in the original's style.  Building and property names get pool words and
another building word.  The suburb is a ``fake_pool`` locality of the same state and class in the original's case.  The
state is written as in the original (``address_state: keep``) or is the fake locality's, in the original's form, one
other state per original state (``replace``).  The postcode is the fake suburb's own, under amendment D3's prefix rule.
'GPO Box' becomes 'PO Box' unless the fake suburb is a capital.  An overseas unit keeps its shape: every number, street
name, town and postcode character changes, digit for digit and letter for letter, with no foreign gazetteer and no
state; the country word stays.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache
from typing import Literal

from mlredact.policy.addresses import JoinedUnit
from mlredact.surrogate.exclusions import Exclusions, digit_runs
from mlredact.surrogate.generators import STATE_NAMES, SurrogateFactory, apply_case, is_state_token
from mlredact.text.address import STRUCTURE_ROLES, AddressKey, Role
from mlredact.text.gazetteer import (
    STATE_CODES,
    Gazetteer,
    Locality,
    fold_name,
    load_gazetteer,
    name_variants,
    range_states,
)

AddressState = Literal["keep", "replace"]

# Amendment D3: the leading digits of a postcode that its state fixes ('08' for NT).
_POSTCODE_PREFIX = {"NSW": "2", "ACT": "2", "VIC": "3", "QLD": "4", "SA": "5", "WA": "6", "TAS": "7", "NT": "08"}
_OCR_DIGITS = str.maketrans("OoIlSB", "001158")
_NUMBER_ROLES = frozenset(
    {Role.NUMBER, Role.UNIT_NUMBER, Role.BOX_NUMBER, Role.DX_NUMBER, Role.LEVEL_NUMBER, Role.UNIT_NUMBER_AND_NUMBER}
)
_RUN_ROLES = frozenset({Role.STREET_NAME, Role.BUILDING, Role.BOX_EXTRA, Role.DX_EXCHANGE, Role.SUBURB})
_GLUED = re.compile(r"^([A-Za-z.]*?)(\d{4})$")
_GPO = re.compile(r"^g\.?p\.?o\.?$", re.IGNORECASE)
_FULL_STATE = {fold_name(name): code for code, name in STATE_NAMES.items()}
_MAX_REDRAWS = 8
_WORD_EDGES = ",;:()[]{}|!?\"'“”‘’"


@dataclass(frozen=True, slots=True)
class UnitWord:
    """One word of a unit: its token, text and role (None: a word a pairing forced into the unit that the grammar did
    not read; it is identifying), whether it is a word of the place, and, for a word that is partly structure, the
    letters that stay (``keep``) and the part that changes (``change``)."""

    token_id: str
    text: str = field(repr=False)
    role: Role | None
    place: bool = False
    keep: str = field(default="", repr=False)
    change: str | None = field(default=None, repr=False)


@dataclass(frozen=True, slots=True)
class UnitView:
    """What the faker needs of one unit.  ``heading`` is the canonical name of the person or organisation heading the
    block of a place-only unit ('' when none); ``repeat_of`` is the unit this one repeats (the join's link)."""

    unit_id: str
    key: AddressKey
    words: tuple[UnitWord, ...] = field(repr=False)
    overseas: bool = False
    heading: str = field(default="", repr=False)
    repeat_of: str | None = None


def view_of(unit: JoinedUnit, texts: Mapping[str, str], heading: str = "") -> UnitView:
    """The view of a joined unit; ``texts`` maps token ids to their text."""
    partial = {p.token_id: p for p in unit.partial}
    place = set(unit.locality)
    words = tuple(
        UnitWord(
            tid,
            texts[tid],
            role,
            tid in place,
            partial[tid].keep if tid in partial else "",
            partial[tid].change if tid in partial else None,
        )
        for tid, role in unit.roles.items()
    )
    return UnitView(unit.unit_id, unit.key, words, unit.overseas, heading, unit.repeat_of)


def postcode_ok(state: str | None, original: str, fake: str, fake_state: str | None = None) -> bool:
    """Amendment D3.  The original's leading digits that match its state's prefix may stay, matched from the left; every
    later digit of the fake differs at its position.  An original outside its state's usual prefix (a PO-box block, a
    cross-border code) shares no digit of its own.  The fake is never the original.  The digits the fake's state fixes
    ('08' for NT) are not a choice, so they may coincide when the fake lies in another state (``address_state:
    replace``) or the original lies outside its state's prefix (NT 4825: every NT postcode starts '08')."""
    if fake == original:
        return False
    prefix = _POSTCODE_PREFIX.get(state or "", "")
    k = 0
    while k < len(prefix) and k < len(original) and original[k] == prefix[k]:
        k += 1
    if (fake_state is not None and fake_state != state) or k < len(prefix):
        fixed = _POSTCODE_PREFIX.get(fake_state or state or "", "")
        if fake.startswith(fixed):
            k = max(k, len(fixed))
    return all(a != b for a, b in zip(original[k:], fake[k:], strict=False))


def _edges(text: str) -> tuple[str, str, str]:
    """``'(Smith,'`` -> ``('(', 'Smith', ',')``: leading punctuation, the word, trailing punctuation."""
    core = text.strip(_WORD_EDGES)
    if not core:
        return text, "", ""
    start = text.index(core)
    return text[:start], core, text[start + len(core) :]


def _bare(text: str) -> str:
    return _edges(text)[1].rstrip(".")


def _dot(original: str, fake: str) -> str:
    """The full stop an original word ends with (a sentence's, or an abbreviation's such as 'St.'), for a fake that
    does not end with one: punctuation is kept."""
    return "." if _edges(original)[1].endswith(".") and not fake.endswith(".") else ""


@cache
def _rows_by_name(gaz: Gazetteer) -> Mapping[str, tuple[Locality, ...]]:
    """Every row of the list by folded name, an alias reaching the rows of its target in its state."""
    by: dict[str, list[Locality]] = {}
    for r in gaz.rows:
        by.setdefault(r.folded, []).append(r)
    for a in gaz.aliases:
        by.setdefault(a.folded, []).extend(r for r in by.get(fold_name(a.locality), ()) if r.state == a.state)
    return {k: tuple(v) for k, v in by.items()}


@cache
def _class_by_postcode(gaz: Gazetteer) -> Mapping[tuple[str, str], bool]:
    """(state, postcode) -> metro, from the most populous row that carries the postcode."""
    best: dict[tuple[str, str], Locality] = {}
    for r in gaz.rows:
        k = (r.state, r.postcode)
        if k not in best or r.population > best[k].population:
            best[k] = r
    return {k: r.metro for k, r in best.items()}


def _likely_state(gaz: Gazetteer, names: Sequence[str], postcodes: Sequence[str]) -> str | None:
    """The state of an unqualified locality name: of its postcode's ranges when written, else of its most populous
    row (so the choice never depends on the rest of the document)."""
    rows = [r for v in name_variants(fold_name(" ".join(names))) for r in _rows_by_name(gaz).get(v, ())]
    if postcodes:
        allowed = {s for p in postcodes if p.isdigit() for s in range_states(int(p))}
        rows = [r for r in rows if r.state in allowed] or rows
    return max(rows, key=lambda r: (r.population, r.state)).state if rows else None


@dataclass(frozen=True, slots=True)
class _Origin:
    """What the unit's original place says: the state, the class (metro or regional) and the postcodes."""

    state: str | None
    klass: str | None
    written: tuple[str, ...]  # the postcodes the unit writes
    postcodes: tuple[str, ...]  # the written ones and the ones the list gives the original suburb


class AddressFaker:
    """The fakes of a document's address units.  ``fakes[unit_id]`` maps every token of the unit to its replacement text
    (an empty string erases a token whose words went into the first slot of its run).  Units are faked in a canonical
    order, so the result does not depend on the order they are given."""

    def __init__(
        self,
        factory: SurrogateFactory,
        units: Sequence[UnitView],
        *,
        address_state: AddressState = "keep",
        gazetteer: Gazetteer | None = None,
    ) -> None:
        self._f = factory
        self._gaz = gazetteer or load_gazetteer()
        self._replace_state = address_state == "replace"
        self._ex = self._exclusions(units)
        self._by_id = {u.unit_id: u for u in units}
        self.fakes: dict[str, Mapping[str, str]] = {}
        issued: dict[tuple[str, ...], list[AddressKey]] = {}
        order = sorted(
            units, key=lambda u: (u.key.street_text(), u.key.sub_text(), u.key.place_text(), u.heading, u.unit_id)
        )
        for u in order:
            for redraw in range(_MAX_REDRAWS + 1):
                out = self._fake(u, redraw)
                whole = tuple(fold_name(" ".join(out.values())).split())
                clash = any(not u.key.same_address(k) for k in issued.get(whole, ()))
                if not clash or redraw == _MAX_REDRAWS:
                    break
            issued.setdefault(whole, []).append(u.key)
            self.fakes[u.unit_id] = out

    # ------------------------------------------------------------------------------- the D1 filter
    @staticmethod
    def _exclusions(units: Sequence[UnitView]) -> Exclusions:
        """The identifying words and numbers of every unit: the words of every role outside STRUCTURE_ROLES (a word
        with a part that stays counts by its changing part), numbers as digit runs."""
        words: list[str] = []
        numbers: list[str] = []
        for u in units:
            for w in u.words:
                role = _effective(w)
                if role is None or role in STRUCTURE_ROLES:
                    continue
                text = w.change if w.change is not None else w.text
                if role is Role.STATE_AND_POSTCODE and w.change is None:
                    m = _GLUED.match(_edges(w.text)[1].rstrip("."))
                    text = m.group(2) if m else text
                runs = digit_runs(text)
                if runs:
                    numbers.extend(runs)
                elif _edges(text)[1]:
                    words.append(_edges(text)[1])
        return Exclusions.of(words, numbers)

    # ------------------------------------------------------------------------------------ one unit
    def _fake(self, u: UnitView, redraw: int) -> dict[str, str]:
        f, ex = self._f, self._ex
        words, roles = u.words, [_effective(w) for w in u.words]
        out = {w.token_id: w.text for w in words}
        has_place = any(r in (Role.SUBURB, Role.POSTCODE, Role.STATE_AND_POSTCODE) for r in roles)
        base = u.key.street_text() or f"{u.heading}\x1f{u.key.place_text()}"
        draw = "" if redraw == 0 else f"\x1f{redraw}"
        kb = base + (draw if not has_place else "")  # the street side redraws only when there is no place
        origin = None if u.overseas else self._origin(u, words, roles)
        ref = self._by_id.get(u.repeat_of or "")
        if origin is not None and u.key.state is None and ref is not None and not ref.overseas:
            # A repeat without a state is the address it repeats: its suburb takes that address's state and class
            # ('Richmond' after 'Richmond NSW 2753' stays in NSW, though the name's most populous row is in VIC).
            head = self._origin(ref, ref.words, [_effective(w) for w in ref.words])
            if head.state is not None:
                origin = _Origin(
                    head.state, head.klass, origin.written, tuple(sorted({*origin.written, *head.postcodes}))
                )
        if origin is None:
            place_key = f"{base}\x1f{u.key.place_text()}\x1f{u.heading}{draw}"
        else:  # the suburb and the state the unit writes or the list gives it (so a repeat without a state agrees)
            pcs = "" if u.key.suburb else ",".join(origin.written)
            place_key = f"{base}\x1f{u.key.suburb}|{origin.state or ''}|{pcs}\x1f{u.heading}{draw}"
        runs = _runs(roles)
        typed = Role.STREET_TYPE in roles
        loc: Locality | None = None
        fake_state: str | None = None
        if origin is not None:
            loc, fake_state = self._place(roles, origin, place_key)
        names_done = 0
        types_done = 0
        suffixes_done = 0
        used_names: set[str] = set()
        for role, idx in runs:
            texts = [_edges(words[i].text) for i in idx]
            template = " ".join(t[1] for t in texts)
            if role in _NUMBER_ROLES:
                i = idx[0]
                out[words[i].token_id] = self._number(words[i], role, kb)
            elif role is Role.STREET_NAME:
                name = f.street_name(
                    f"{kb}\x1fname\x1f{names_done}", lambda w: ex.word_ok(w) and w.lower() not in used_names
                )
                used_names.add(name.lower())
                names_done += 1
                value = apply_case(template, name)
                if not typed and not u.overseas and idx == _last_run(runs, role):  # an untyped street gets a type
                    value += " " + apply_case(template, f.street_type(None, f"{kb}\x1funtyped", ex))
                _put(out, words, idx, value)
            elif role is Role.STREET_TYPE:
                new_type = f.street_type(texts[0][1], f"{kb}\x1ftype\x1f{types_done}", ex)
                out[words[idx[0]].token_id] = _wrap(words[idx[0]], new_type)
                types_done += 1
            elif role is Role.STREET_SUFFIX:
                new_suffix = f.street_suffix(texts[0][1], f"{kb}\x1fsuffix\x1f{suffixes_done}", ex)
                out[words[idx[0]].token_id] = _wrap(words[idx[0]], new_suffix)
                suffixes_done += 1
            elif role is Role.BUILDING:
                key = f"{kb}\x1fbuilding\x1f{fold_name(template)}"
                for i, new in zip(idx, f.building_name([t[1] for t in texts], key, ex, used_names), strict=True):
                    out[words[i].token_id] = _wrap(words[i], new)
            elif role is Role.BOX_EXTRA:
                for n, i in enumerate(idx):
                    new = f.pool_word("box_extra", f"{kb}\x1f{fold_name(texts[n][1])}\x1f{n}", texts[n][1], ex)
                    out[words[i].token_id] = _wrap(words[i], new)
            elif role is Role.DX_EXCHANGE:
                exchange = self._exchange(u, words, f"{kb}\x1fdx\x1f{fold_name(template)}")
                _put(out, words, idx, apply_case(template, exchange.name))
            elif role is Role.BOX_KIND:
                i = idx[0]
                if loc is not None and not self._gaz.is_capital(loc.name.split()):
                    out[words[i].token_id] = _drop_g(words[i].text)
            elif role is Role.SUBURB:
                if u.overseas:
                    town = f.pool_word("town", f"{place_key}\x1f{fold_name(template)}", template, ex)
                    _put(out, words, idx, town)
                elif loc is not None:
                    _put(out, words, idx, apply_case(template, loc.name))
            elif role is Role.POSTCODE:
                i = idx[0]
                if u.overseas:
                    new = f.overseas_postcode(texts[0][1], f"{place_key}\x1fpc\x1f{texts[0][1].upper()}", ex)
                    out[words[i].token_id] = _wrap(words[i], new)
                elif loc is not None:
                    out[words[i].token_id] = _wrap(words[i], loc.postcode)
            elif role is Role.STATE_AND_POSTCODE:
                i = idx[0]
                if loc is not None:
                    out[words[i].token_id] = self._glued(words[i], loc, fake_state)
        if self._replace_state and fake_state and not u.overseas:
            self._write_states(words, roles, fake_state, out)
        return out

    # ---------------------------------------------------------------------------------- parts
    def _number(self, w: UnitWord, role: Role, kb: str) -> str:
        """A number word: the number rule, or the level rule; a slash form fakes its unit and house numbers apart."""
        f, ex = self._f, self._ex
        lead, core, trail = _edges(w.text)
        if core.endswith(".") and core[:-1]:  # a number that ends a sentence keeps its full stop
            core, trail = core[:-1], "." + trail
        keep, change = (w.keep, w.change) if w.change is not None else ("", None)
        if role is Role.LEVEL_NUMBER:
            new = f.level_number(core, f"{kb}\x1flevel\x1f{core.upper()}", ex)
        elif role is Role.UNIT_NUMBER_AND_NUMBER and "/" in core:
            left, _, right = core.partition("/")
            keep = keep if left.startswith(keep) else ""
            fake_left = keep + f.address_number(left[len(keep) :], f"{kb}\x1funit\x1f{left.upper()}", ex)
            fake_right = f.address_number(right, f"{kb}\x1fnumber\x1f{right.upper()}", ex)
            new = f"{fake_left}/{fake_right}"
        else:
            domain = {Role.UNIT_NUMBER: "unit", Role.BOX_NUMBER: "box", Role.DX_NUMBER: "dx"}.get(role, "number")
            if change is not None and core.startswith(keep):
                new = keep + f.address_number(core[len(keep) :], f"{kb}\x1f{domain}\x1f{core.upper()}", ex)
            else:
                new = f.address_number(core, f"{kb}\x1f{domain}\x1f{core.upper()}", ex)
        return lead + new + trail

    def _origin(self, u: UnitView, words: Sequence[UnitWord], roles: Sequence[Role | None]) -> _Origin:
        gaz = self._gaz
        names = [
            _bare(w.text)
            for w, r in zip(words, roles, strict=True)
            if r is Role.SUBURB or (r is Role.DX_EXCHANGE and not any(x is Role.SUBURB for x in roles))
        ]
        written = tuple(p for w, r in zip(words, roles, strict=True) if (p := _postcode_digits(w, r)) is not None)
        state = u.key.state
        if state is None and written:
            states = range_states(int(written[0]))
            state = next(iter(states)) if len(states) == 1 else None
        if state is None and names:
            state = _likely_state(gaz, names, written)
        klass = gaz.klass_of(names, state, written[0] if written else None) if names and state else None
        if klass is None and state and written:
            metro = _class_by_postcode(gaz).get((state, written[0]))
            klass = None if metro is None else ("metro" if metro else "regional")
        listed = tuple(f"{p:04d}" for p in sorted(gaz.postcodes_of(names, state))) if names and state else ()
        return _Origin(state, klass, written, tuple(sorted({*written, *listed})))

    def _target_state(self, origin: _Origin, key: str) -> str | None:
        """The state a fake is drawn from: the original's (keep), or the one other state all addresses of the original
        state move to (replace)."""
        if not self._replace_state:
            return origin.state
        if origin.state is None:
            return STATE_CODES[self._f.chooser.index("addr.state:any", key, len(STATE_CODES))]
        others = [s for s in STATE_CODES if s != origin.state]
        return others[self._f.chooser.index("addr.state", origin.state, len(others))]

    def _place(
        self, roles: Sequence[Role | None], origin: _Origin, place_key: str
    ) -> tuple[Locality | None, str | None]:
        """The fake locality of the unit's place (None when it writes neither a suburb nor a postcode) and the state a
        replaced state word is written as."""
        target = self._target_state(origin, place_key)
        if not any(r in (Role.SUBURB, Role.POSTCODE, Role.STATE_AND_POSTCODE) for r in roles):
            return None, target
        ex = self._ex

        def accept(loc: Locality) -> bool:
            return (
                ex.word_ok(loc.name)
                and ex.number_ok(loc.postcode)
                and self._f.not_original(loc.postcode)
                and all(postcode_ok(origin.state, p, loc.postcode, loc.state) for p in origin.postcodes)
            )

        loc = self._f.locality(place_key, target, origin.klass, key=place_key, accept=accept)
        return loc, loc.state if self._replace_state else (target or loc.state)

    def _exchange(self, u: UnitView, words: Sequence[UnitWord], key: str) -> Locality:
        """A DX exchange: a fake locality of the original exchange's state."""
        origin = self._origin(u, words, [_effective(w) for w in words])
        target = self._target_state(origin, key)
        return self._f.locality(key, target, origin.klass, key=key, accept=lambda loc: self._ex.word_ok(loc.name))

    def _glued(self, w: UnitWord, loc: Locality, fake_state: str | None) -> str:
        """A glued 'NSW2750': the letters stay (or become the replaced state), the digits are the fake postcode."""
        lead, core, trail = _edges(w.text)
        m = _GLUED.match(core.rstrip("."))
        letters = w.keep or (m.group(1) if m else "")
        if self._replace_state and fake_state:
            letters = self._f.state_form(letters, fake_state)
        return lead + letters + loc.postcode + _dot(w.text, loc.postcode) + trail

    def _write_states(
        self, words: Sequence[UnitWord], roles: Sequence[Role | None], fake_state: str, out: dict[str, str]
    ) -> None:
        """Replace mode: write the fake state in the original's form (spelled out, dotted, mixed case or upper case)."""
        i = 0
        while i < len(words):
            if roles[i] is not Role.STATE:
                i += 1
                continue
            j = i
            while j + 1 < len(words) and roles[j + 1] is Role.STATE:
                j += 1
            group = list(range(i, j + 1))
            joined = " ".join(_edges(words[k].text)[1] for k in group)
            if len(group) > 1 and fold_name(joined) in _FULL_STATE:  # 'New South Wales': one name over its words
                _put(out, words, group, self._f.state_form(joined, fake_state))
            else:
                for k in group:
                    out[words[k].token_id] = _wrap(words[k], self._f.state_form(_edges(words[k].text)[1], fake_state))
            i = j + 1


# ---------------------------------------------------------------------------------------------- helpers
def _effective(w: UnitWord) -> Role | None:
    return effective_role(w.role, w.text, w.place)


def effective_role(role: Role | None, text: str, place: bool) -> Role | None:
    """The role of a word.  A word the grammar did not read (a pairing forced it into the unit) is identifying, so it
    takes the role of its shape: a state code, a four-digit postcode or a suburb word in the place (``place``); a
    number or a street-name word in the street line; punctuation alone has none and stays.  V5 reads words by the same
    rule, so a kept state word of a forced unit is no collision."""
    if role is not None:
        return role
    core = _bare(text)
    if not core:
        return None
    if place:
        if is_state_token(core):
            return Role.STATE
        if core.isdigit() and len(core) == 4:
            return Role.POSTCODE
        return Role.SUBURB if not any(c.isdigit() for c in core) else Role.NUMBER
    return Role.NUMBER if any(c.isdigit() for c in core) else Role.STREET_NAME


def _runs(roles: Sequence[Role | None]) -> list[tuple[Role, list[int]]]:
    """The words grouped into runs: consecutive words of one name-like role form one run, any other word is its own.
    Words with no role (punctuation) are left out."""
    out: list[tuple[Role, list[int]]] = []
    for i, role in enumerate(roles):
        if role is None:
            continue
        if role in _RUN_ROLES and out and out[-1][0] is role and out[-1][1][-1] == i - 1:
            out[-1][1].append(i)
        else:
            out.append((role, [i]))
    return out


def _last_run(runs: Sequence[tuple[Role, list[int]]], role: Role) -> list[int]:
    return [idx for r, idx in runs if r is role][-1]


def _wrap(w: UnitWord, core: str) -> str:
    """``core`` with the leading and trailing punctuation of the original word, a final full stop included."""
    lead, _, trail = _edges(w.text)
    return lead + core + _dot(w.text, core) + trail


def _put(out: dict[str, str], words: Sequence[UnitWord], idx: Sequence[int], value: str) -> None:
    """Write ``value`` into the first slot of a run (with the run's punctuation, a final full stop included) and erase
    the others."""
    first, last = words[idx[0]], words[idx[-1]]
    out[first.token_id] = _edges(first.text)[0] + value + _dot(last.text, value) + _edges(last.text)[2]
    for i in idx[1:]:
        out[words[i].token_id] = ""


def _drop_g(text: str) -> str:
    """'GPO' becomes 'PO' (case and dots kept); any other box word stays."""
    lead, core, trail = _edges(text)
    if not _GPO.match(core):
        return text
    new = ("P.O" if "." in core else "PO") + ("." if core.endswith(".") else "")
    return lead + (new if core.isupper() else new.lower() if core.islower() else new.title()) + trail


def _postcode_digits(w: UnitWord, role: Role | None) -> str | None:
    """The four digits a postcode word writes (OCR letters read as digits), else None."""
    if role is Role.POSTCODE:
        core = _bare(w.text).translate(_OCR_DIGITS)
        return core if core.isdigit() and len(core) == 4 else None
    if role is Role.STATE_AND_POSTCODE:
        m = _GLUED.match(_edges(w.text)[1].rstrip("."))
        return m.group(2) if m else None
    return None
