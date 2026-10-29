"""Assign surrogate text to every SURROGATE mention, aligned token-by-token with the original.

Alignment matters because rendering happens per line segment: each original token's slot on the
page receives the surrogate token(s) that replace it.  When a surrogate has a different token count
than the original, the whole surrogate goes into the first slot and the others are only erased.

Street addresses.  Under ``addresses: whole`` the join hands in its layouts and units: every unit gets ONE fake
(:class:`mlredact.surrogate.addresses.AddressFaker`) and each mention takes the words of its own tokens, so the boxes
of a form get the matching parts of one fake.  A STREET_ADDRESS mention the join did not lay out (``addresses: parts``,
or a caller that does not pass the join) is replaced part by part as before.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from mlredact.core.model import Mention, Token
from mlredact.core.types import ActionKind, EntityType
from mlredact.policy.addresses import JoinedUnit, MentionLayout
from mlredact.policy.engine import DATE_TYPES, Decision
from mlredact.resolve.persons import Gender, NameToken, PersonResolution, Role
from mlredact.surrogate.addresses import AddressFaker, AddressState, view_of
from mlredact.surrogate.dates import DateShifter, generalise_age, leak_needles
from mlredact.surrogate.generators import (
    ORG_GENERIC,
    SurrogateFactory,
    apply_case,
    canonical_state,
    is_state_token,
    is_street_type,
)
from mlredact.surrogate.pools import Locality
from mlredact.text.address import UnitKind
from mlredact.text.normalize import match_key

_UNIT_WORDS = frozenset(
    "unit apt apartment flat suite shop level lvl lot u po p.o p.o. box gpo locked private bag pmb rmb rsd cmb".split()
)
_POSTCODE = re.compile(r"^\d{4}[,.]?$")
_SPLIT_LOCAL = re.compile(r"([._\-]+|\d+)")


@dataclass(frozen=True, slots=True)
class MentionSurrogate:
    mention_id: str
    tokens: tuple[str, ...] = field(repr=False)


def _aligned(surrogate: str, originals: Sequence[str]) -> tuple[str, ...]:
    parts = surrogate.split()
    if len(parts) == len(originals):
        return tuple(parts)
    return (surrogate, *[""] * (len(originals) - 1))


def forbidden_keys(mentions: Iterable[Mention], persons: PersonResolution) -> set[str]:
    keys: set[str] = set()
    for m in mentions:
        keys.add(match_key(m.value))
        keys.update(match_key(part) for part in m.value.split())
    for pm in persons.mentions.values():
        keys.update(t.key for t in pm.tokens)
    keys.discard("")
    return keys


class SurrogateAssigner:
    """``layouts`` and ``units`` are the address join's (``AddressJoin.layouts`` and ``.units``); without them every
    street address is replaced part by part.  ``address_state`` is the policy's ``address_state``."""

    def __init__(
        self,
        factory: SurrogateFactory,
        persons: PersonResolution,
        tokens: Mapping[str, Token],
        *,
        layouts: Mapping[str, MentionLayout] | None = None,
        units: Mapping[str, JoinedUnit] | None = None,
        address_state: AddressState = "keep",
    ) -> None:
        self._f = factory
        self._persons = persons
        factory.add_name_parts(
            t.key for pm in persons.mentions.values() for t in pm.tokens if t.role in (Role.GIVEN, Role.SURNAME)
        )
        self._tokens = tokens
        self._layouts = layouts or {}
        self._units = units or {}
        self._address_state = address_state
        self._faker: AddressFaker | None = None
        self._line_locality: dict[tuple[int, int], Locality] = {}
        self._lines: dict[tuple[int, int], list[Token]] = {}
        for tok in tokens.values():
            self._lines.setdefault((tok.page, tok.line_no), []).append(tok)
        for line in self._lines.values():
            line.sort(key=lambda t: t.word_no)
        self._given_by_cluster: dict[str, list[str]] = {}
        for pm in persons.mentions.values():
            if pm.cluster:
                bucket = self._given_by_cluster.setdefault(pm.cluster, [])
                bucket.extend(t.key for t in pm.tokens if t.role is Role.GIVEN and t.key not in bucket)

    # ---------------------------------------------------------------------------------- public
    @property
    def address_fakes(self) -> Mapping[str, Mapping[str, str]]:
        """After :meth:`assign`: the fake of every address unit, as unit id -> token id -> replacement text (an empty
        string erases a token whose words went into the first slot of its run).  Verification reads the drawn fakes
        from here; empty when no layouts were given."""
        return self._faker.fakes if self._faker is not None else {}

    def assign(self, decisions: Sequence[Decision]) -> dict[str, MentionSurrogate]:
        """Replacement text for every SURROGATE, DATE_SHIFT (dates) and GENERALISE (ages) decision."""
        out: dict[str, MentionSurrogate] = {}
        shifter: DateShifter | None = None
        if self._units and self._faker is None:
            self._faker = self._address_faker(decisions)
        for d in decisions:
            m = d.mention
            originals = [self._tokens[t].text for t in m.token_ids]
            if d.action is ActionKind.SURROGATE:
                out[m.mention_id] = MentionSurrogate(m.mention_id, self._one(m, originals))
            elif d.action is ActionKind.DATE_SHIFT and m.entity_type in DATE_TYPES:
                if shifter is None:
                    shifter = self._date_shifter(decisions)
                out[m.mention_id] = MentionSurrogate(m.mention_id, _aligned(shifter.shift(m.value), originals))
            elif d.action is ActionKind.GENERALISE and m.entity_type is EntityType.AGE:
                out[m.mention_id] = MentionSurrogate(m.mention_id, _aligned(generalise_age(m.value), originals))
        return out

    def _date_shifter(self, decisions: Sequence[Decision]) -> DateShifter:
        """One offset per scope/document; never produces text that V2 would read as a leak."""
        removed = [d.mention.value for d in decisions if d.action is not ActionKind.KEEP]
        dates = [d.mention.value for d in decisions if d.action is ActionKind.DATE_SHIFT]
        return DateShifter.choose(self._f.chooser, dates, leak_needles(removed))

    # --------------------------------------------------------------------------------- per type
    def _one(self, m: Mention, originals: list[str]) -> tuple[str, ...]:
        t = m.entity_type
        if t is EntityType.PERSON:
            return self._person(m)
        if t in (EntityType.DATE_OF_BIRTH, EntityType.DATE_OF_DEATH):
            return _aligned(self._f.date_of_birth(m.value), originals)
        if t is EntityType.PHONE:
            return _aligned(self._f.phone(m.value), originals)
        if t is EntityType.EMAIL:
            return _aligned(self._f.email(m.value, self._email_hint(m.value)), originals)
        if t is EntityType.URL:
            return _aligned(self._f.url(m.value), originals)
        if t is EntityType.FILE_PATH:
            return _aligned(self._f.file_path(m.value), originals)
        if t is EntityType.STREET_ADDRESS:
            return self._whole_address(m) or self._address(m, originals)
        if t is EntityType.LOCALITY:
            state = self._state_after(m)
            loc = self._f.locality(m.value, state)
            self._line_locality[self._line_of(m)] = loc
            return _aligned(apply_case(" ".join(o.strip(",.") for o in originals), loc.name), originals)
        if t is EntityType.POSTCODE:
            linked = self._line_locality.get(self._line_of(m))
            value = linked.postcode if linked else self._f.postcode(m.value, self._state_before(m))
            return _aligned(value, originals)
        if t is EntityType.ORGANISATION:
            return _aligned(self._f.organisation(m.value), originals)
        return _aligned(self._f.identifier(t, m.value), originals)

    def _person(self, m: Mention) -> tuple[str, ...]:
        pm = self._persons.mentions.get(m.mention_id)
        if pm is None:  # not resolved (should not happen): replace word-by-word as surnames
            return tuple(self._f.surname(match_key(w), w) for w in m.value.split())
        out: list[str] = []
        for nt in pm.tokens:
            out.append(nt.prefix + self._name_part(nt, pm.cluster) + nt.suffix if nt.core else nt.prefix + nt.suffix)
        return tuple(out)

    def _name_part(self, nt: NameToken, cluster: str | None) -> str:
        if nt.role is Role.SURNAME:
            return self._f.surname(nt.key, nt.core)
        if nt.role is Role.GIVEN:
            gender = self._persons.given_gender.get(nt.key, Gender.UNKNOWN)
            return self._f.given_name(nt.key, gender.value, nt.core)
        if nt.role is Role.INITIAL:
            letters = [c for c in nt.core if c.isalpha()]
            mapped = []
            givens = self._given_by_cluster.get(cluster or "", [])
            for ch in letters:
                match = next((g for g in givens if g.startswith(ch.lower())), None)
                if match is not None:
                    gender = self._persons.given_gender.get(match, Gender.UNKNOWN)
                    mapped.append(self._f.given_name(match, gender.value, match.title())[0].upper())
                else:
                    mapped.append(self._f.initial(f"{cluster}:{ch}"))
            out, k = [], 0
            for c in nt.core:  # keep the original punctuation pattern ("J.", "J.S.", "JS")
                if c.isalpha():
                    out.append(mapped[k])
                    k += 1
                else:
                    out.append(c)
            return "".join(out)
        return nt.core  # particles / other words that are not identifying on their own

    def _email_hint(self, value: str) -> str | None:
        local = value.split("@", 1)[0]
        surnames = {t.key for pm in self._persons.mentions.values() for t in pm.tokens if t.role is Role.SURNAME}
        givens = {t.key for pm in self._persons.mentions.values() for t in pm.tokens if t.role is Role.GIVEN}
        parts = [p for p in _SPLIT_LOCAL.split(local) if p]
        if not any(match_key(p) in surnames | givens for p in parts):
            return None
        out = []
        for p in parts:
            k = match_key(p)
            if k in surnames:
                out.append(self._f.surname(k, p.title()).lower())
            elif k in givens:
                gender = self._persons.given_gender.get(k, Gender.UNKNOWN)
                out.append(self._f.given_name(k, gender.value, p.title()).lower())
            elif p.isdigit():
                out.append(self._f.number_like("email.digits", p))
            elif p.isalpha():
                out.append(p[0])  # unknown alphabetic chunk (e.g. an initial): keep its shape only
            else:
                out.append(p)
        return "".join(out)

    def _address_faker(self, decisions: Sequence[Decision]) -> AddressFaker:
        """The fakes of all the document's units (D1 reads every unit, whatever its mention's action)."""
        owners = {
            t: d.mention
            for d in decisions
            if d.mention.entity_type in (EntityType.PERSON, EntityType.ORGANISATION)
            for t in d.mention.token_ids
        }
        views = [
            view_of(u, {t: self._tokens[t].text for t in u.roles}, self._heading(u, owners))
            for _, u in sorted(self._units.items())
            if all(t in self._tokens for t in u.roles)
        ]
        return AddressFaker(self._f, views, address_state=self._address_state)

    def _whole_address(self, m: Mention) -> tuple[str, ...] | None:
        """The words of the unit's fake for the tokens of this mention, or None when the mention has no unit."""
        layout = self._layouts.get(m.mention_id)
        fake = self._faker.fakes.get(layout.unit_id) if self._faker is not None and layout is not None else None
        if fake is None or not all(t in fake for t in m.token_ids):
            return None
        return tuple(fake[t] for t in m.token_ids)

    def _heading(self, unit: JoinedUnit, owners: Mapping[str, Mention]) -> str:
        """For a place-only unit, the canonical name of the person or organisation heading its block: the nearest one
        before it on its line, or on the lines right above it (a gap of more than a line height ends the block)."""
        if unit.kind is not UnitKind.PLACE_ONLY or not unit.roles:
            return ""
        first = self._tokens.get(next(iter(unit.roles)))
        if first is None:
            return ""
        line = self._lines.get((first.page, first.line_no), [])
        for tok in reversed([t for t in line if t.word_no < first.word_no]):
            if tok.token_id in owners:
                return self._canonical_name(owners[tok.token_id])
        lower, line_no = line, first.line_no - 1
        for _ in range(3):
            above = self._lines.get((first.page, line_no))
            if not above or not lower:
                break
            height = max(1, above[0].bbox.y1 - above[0].bbox.y0)
            if lower[0].bbox.y0 - above[0].bbox.y1 > height:
                break
            for tok in above:
                if tok.token_id in owners:
                    return self._canonical_name(owners[tok.token_id])
            lower, line_no = above, line_no - 1
        return ""

    def _canonical_name(self, m: Mention) -> str:
        """A person's surname and first initial, or an organisation's name without its generic words: the same whatever
        the spelling of the mention, for keying the fake of a place-only unit."""
        if m.entity_type is EntityType.ORGANISATION:
            words = [w.strip(",.()").lower() for w in m.value.split()]
            return "o:" + " ".join(w for w in words if w and w not in ORG_GENERIC)
        pm = self._persons.mentions.get(m.mention_id)
        if pm is None:
            return "p:" + match_key(m.value)
        surname = "".join(t.key for t in pm.tokens if t.role is Role.SURNAME)
        given = next((t.key for t in pm.tokens if t.role is Role.GIVEN), "")
        if pm.cluster:
            given = (self._given_by_cluster.get(pm.cluster) or [given])[0]
        return f"p:{surname},{given[:1]}"

    def _address(self, m: Mention, originals: list[str]) -> tuple[str, ...]:
        cores = [o.strip(",.") for o in originals]
        state_idx = next((i for i, c in enumerate(cores) if is_state_token(c)), None)
        state = canonical_state(cores[state_idx]) if state_idx is not None else None
        type_idx = max(
            (i for i, c in enumerate(cores) if is_street_type(c) and i < (state_idx or len(cores))), default=None
        )
        out = list(originals)
        locality_slots: list[int] = []
        street_slots: list[int] = []
        for i, (orig, core) in enumerate(zip(originals, cores, strict=True)):
            low = core.lower()
            if i == state_idx or (type_idx is not None and i == type_idx) or low in _UNIT_WORDS:
                continue
            if state_idx is not None and i == state_idx + 1 and _POSTCODE.match(core):
                continue  # postcode handled after the locality
            if any(c.isdigit() for c in core):
                out[i] = orig.replace(core, self._f.address_number(core, f"parts\x1f{core.upper()}"), 1)
            elif core[:1].isupper():
                if type_idx is not None and i < type_idx:
                    street_slots.append(i)
                elif state_idx is not None and i < state_idx:
                    locality_slots.append(i)
                else:
                    street_slots.append(i)
        if street_slots:
            name = self._f.street_name(match_key(" ".join(cores[i] for i in street_slots)))
            _fill(out, originals, street_slots, name)
        loc: Locality | None = None
        if locality_slots or state_idx is not None:
            loc_text = " ".join(cores[i] for i in locality_slots) or (state or "")
            loc = self._f.locality(loc_text, state)
            if locality_slots:
                _fill(out, originals, locality_slots, loc.name)
            self._line_locality[self._line_of(m)] = loc
        if state_idx is not None and state_idx + 1 < len(cores) and _POSTCODE.match(cores[state_idx + 1]):
            i = state_idx + 1
            out[i] = originals[i].replace(cores[i], loc.postcode if loc else self._f.postcode(cores[i], state), 1)
        return tuple(out)

    # ------------------------------------------------------------------------------- context
    def _line_of(self, m: Mention) -> tuple[int, int]:
        tok = self._tokens[m.token_ids[0]]
        return tok.page, tok.line_no

    def _line_tokens(self, m: Mention) -> list[Token]:
        return self._lines.get(self._line_of(m), [])

    def _state_after(self, m: Mention) -> str | None:
        last = self._tokens[m.token_ids[-1]]
        for t in self._line_tokens(m):
            if t.word_no > last.word_no and is_state_token(t.text):
                return canonical_state(t.text)
        return None

    def _state_before(self, m: Mention) -> str | None:
        first = self._tokens[m.token_ids[0]]
        for t in reversed(self._line_tokens(m)):
            if t.word_no < first.word_no and is_state_token(t.text):
                return canonical_state(t.text)
        return None


def _fill(out: list[str], originals: Sequence[str], slots: Sequence[int], value: str) -> None:
    """Write ``value`` into the first slot (keeping its trailing punctuation); erase the others."""
    first = slots[0]
    trailing = originals[slots[-1]][len(originals[slots[-1]].rstrip(",.")) :]
    template = " ".join(originals[i].strip(",.") for i in slots)
    out[first] = apply_case(template, value) + trailing
    for i in slots[1:]:
        out[i] = ""
