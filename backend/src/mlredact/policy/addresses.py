"""The address join: detector mentions and the address grammar become whole-address units (FINAL_PLAN TOOL2 section 3,
as amended by A, F, G and H, the owner's rulings and PENDING_CASES).

An address is one unit, from its first word to its last.  The grammar (:mod:`mlredact.text.address`) finds the
units from words and line geometry alone; this module is the adapter and the policy around it:

* it hands the grammar the page's lines, with the words of PERSON, ORGANISATION, PHONE, EMAIL and identifier mentions
  marked ``blocked`` (a suburb, building or place run never crosses them), and header and footer bands from the page
  position;
* it maps each unit back to mentions: one STREET_ADDRESS mention per unit (a block spans its lines), absorbing the
  STREET_ADDRESS, LOCALITY and POSTCODE mentions inside it and the unowned state and country words (retype, 3a); a
  unit with other mentions' words inside it, or with its parts in separate boxes or cells (forms), is carried by
  several mentions, each with its layout;
* it never absorbs PERSON, ORGANISATION, PHONE, EMAIL or identifier words, and it never drops or weakens a mention
  that the pipeline replaced before: the words of an address-type mention that lie outside every unit stay in a
  mention of the original type (a place after a birth, injury, accident, incident or declaration phrase stays a kept
  LOCALITY, as :func:`mlredact.policy.engine.split_kept_places` already made it), and date and age mentions lose only
  the words a unit now owns;
* it records the places mentioned on their own (kept under the client profile), the layouts the surrogate stage
  fills, and the verification targets and exempt positions (TOOL2 section 6, amendment H).

Everything is deterministic: ids come from :func:`mlredact.core.canonical.content_id` over token ids, and the output is
sorted by first token.  Nothing here holds document text in a repr or an exception.
"""

from __future__ import annotations

import dataclasses
import math
import re
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

from mlredact.config.schema import PolicyConfig
from mlredact.core.canonical import content_id
from mlredact.core.model import Evidence, Line, Mention, PageAnalysis, Token
from mlredact.core.types import ActionKind, EntityType, EvidenceStrength
from mlredact.policy.engine import split_kept_places
from mlredact.text.address import (
    LOCALITY_ROLES,
    STRUCTURE_ROLES,
    AddressKey,
    AddressUnit,
    AddrLine,
    AddrWord,
    Band,
    BoxRole,
    FormPart,
    KeptKind,
    PartialWord,
    Pos,
    Role,
    UnitKind,
    parse_document,
)
from mlredact.text.gazetteer import Gazetteer

E = EntityType

#: Mention types that carry an address or part of one.  Their words inside a unit are absorbed into it.
ADDRESS_TYPES = frozenset({E.STREET_ADDRESS, E.LOCALITY, E.POSTCODE})
#: Dates and ages are not address words; a unit that owns one of their words takes it (the unit is replaced whole).
TRIMMED_TYPES = frozenset({E.DATE, E.DATE_OF_BIRTH, E.DATE_OF_DEATH, E.AGE})
#: Every other type (persons, organisations, contact details, identifiers, paths) owns its words: they are marked
#: blocked for the grammar and never absorbed.

JOIN_DETECTOR = "policy.join"
JOIN_VERSION = "1"

_POSTCODE = re.compile(r"^\d{4}[,.]?$")
_CONNECTING = frozenset({"of", "the", "de", "la", "le", "van", "von", "du", "des"})
_NAME_ROLES = frozenset({Role.SUBURB, Role.STREET_NAME, Role.BUILDING})
_IDENTIFYING_LINE_ROLES = _NAME_ROLES | {Role.BOX_EXTRA, Role.DX_EXCHANGE}
#: Kept places a model may have read as an address: they become LOCALITY mentions again.  A street in prose and a word
#: repeated in prose stay replaced when a detector made them an address (fail closed).
_PLACE_KINDS = frozenset(
    {KeptKind.PLACE, KeptKind.STATE, KeptKind.POSTCODE, KeptKind.COUNTRY, KeptKind.PLACE_AFTER_CUE}
)
_NOT_THE_PLACE = frozenset({Role.COUNTRY, Role.CAPITAL_CITY_PHRASE})
_SIDEWAYS_DEGREES = 25.0


class UnitPart(StrEnum):
    """What one mention carries of its unit."""

    JOINED = "joined"  # street line and place together (a whole address in one mention)
    STREET = "street"  # only street-line words (a form's street boxes, a street run before a separator)
    PLACE = "place"  # only place words (a place-only unit, a form's suburb, state and postcode boxes)


@dataclass(frozen=True, slots=True)
class MentionLayout:
    """Where one mention sits in its unit.  ``roles`` has one entry per token of the mention, in the mention's token
    order: the grammar role of the word, or None for a word of a forced (paired) unit that the grammar did not read
    (identifying: fail closed).  ``place_start`` is the index in the mention's tokens of its first place word, None
    when the mention holds no place word."""

    unit_id: str
    part: UnitPart
    place_start: int | None
    roles: tuple[Role | None, ...]


@dataclass(frozen=True, slots=True)
class PartialToken:
    """A token that is partly structure (a glued ``NSW2750``: keep 'NSW', change '2750')."""

    token_id: str
    keep: str = field(repr=False)
    change: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class JoinedUnit:
    """One address as the join found it.

    ``roles`` maps every token of the unit to its role (None: not read by the grammar, forced by ``pairs``), in reading
    order.  ``street`` and ``locality`` split those tokens as the grammar does (everything from the first word to the
    start of the place, then the place).  ``mention_ids`` are the mentions that carry the unit, in reading order.
    ``excluded`` are the unit's tokens that another mention owns (a person, organisation, phone, email or identifier):
    that mention's own rule replaces them.  ``key`` is the canonical address identity (fakes are chosen from it);
    ``repeat_of`` is the unit id of the earlier address this one repeats.  ``kind`` is the grammar's unit kind."""

    unit_id: str
    kind: UnitKind
    key: AddressKey
    roles: Mapping[str, Role | None]
    street: tuple[str, ...]
    locality: tuple[str, ...]
    mention_ids: tuple[str, ...]
    excluded: tuple[str, ...]
    partial: tuple[PartialToken, ...]
    overseas: bool
    form: bool
    repeat_of: str | None
    forced: bool = False  # built from ``pairs``, not (only) from the grammar

    @property
    def token_ids(self) -> tuple[str, ...]:
        return tuple(self.roles)


@dataclass(frozen=True, slots=True)
class KeptRecord:
    """A place mentioned on its own, as the grammar recorded it, with its tokens."""

    kind: KeptKind
    token_ids: tuple[str, ...]
    after_cue: bool = False


@dataclass(frozen=True, slots=True)
class UnitTargets:
    """What verification searches for to find an address left behind (TOOL2 section 6 and amendment H).

    ``compound`` holds sequences of consecutive folded words: the unit text, each line of the unit that holds an
    identifying word, the street line, number and street name, box kind and number, suburb with state and postcode, and
    'label value' for form parts.  ``whole`` holds the suburb, street-name and building words as whole tokens (never
    connecting words such as 'of' and 'the', bare numbers, street types or postcodes).  Words are folded with
    :func:`fold_word`.  Both are sorted and unique."""

    unit_id: str
    compound: tuple[tuple[str, ...], ...] = field(repr=False)
    whole: tuple[str, ...] = field(repr=False)


@dataclass(frozen=True, slots=True)
class ExemptPositions:
    """Token ids that verification does not count as a leak of an address.

    ``kept_places``: places mentioned on their own that stay as written under the profile (the grammar's record, less
    the words a mention replaces).  ``kept_mentions``: tokens of mentions whose action is KEEP (dates, ages, kept
    localities and postcodes, ...).  The runner adds this run's drawn fakes."""

    kept_places: frozenset[str] = frozenset()
    kept_mentions: frozenset[str] = frozenset()

    @property
    def all(self) -> frozenset[str]:
        return self.kept_places | self.kept_mentions


@dataclass(frozen=True, slots=True)
class AddressJoin:
    """The result of the join.

    ``mentions``: the document's mentions after the join, sorted by first token.  ``layouts``: mention id -> layout, for
    the mentions that carry a unit.  ``units``: unit id -> unit, in reading order.  ``kept``: the places mentioned on
    their own.  ``targets``: unit id -> verification targets.  ``exempt``: the exempt positions."""

    mentions: tuple[Mention, ...]
    layouts: Mapping[str, MentionLayout] = field(default_factory=lambda: MappingProxyType({}))
    units: Mapping[str, JoinedUnit] = field(default_factory=lambda: MappingProxyType({}))
    kept: tuple[KeptRecord, ...] = ()
    targets: Mapping[str, UnitTargets] = field(default_factory=lambda: MappingProxyType({}))
    exempt: ExemptPositions = ExemptPositions()


# ---------------------------------------------------------------------------------------- folding and targets
def fold_word(text: str) -> str:
    """The comparison form of one word for verification targets: curly apostrophes straightened, edge punctuation
    stripped, case folded.  An empty result means the word is punctuation only."""
    return text.replace("’", "'").strip(",;:()|.!?'\"").casefold()


def fold_words(texts: Iterable[str]) -> tuple[str, ...]:
    """Fold words and drop the ones that are punctuation only."""
    return tuple(w for w in (fold_word(t) for t in texts) if w)


def _unit_targets(
    unit_id: str,
    roles: Mapping[str, Role | None],
    texts: Mapping[str, str],
    line_of: Mapping[str, tuple[int, int]],
    form_parts: Sequence[tuple[Sequence[str], Sequence[str]]],
    excluded: Collection[str],
    locality: Collection[str],
) -> UnitTargets:
    """``roles`` has one entry per unit token in reading order (None: a word a pairing forced into the unit that the
    grammar did not read, which is identifying); ``locality`` are the unit's place tokens; ``excluded`` the tokens
    another mention owns (its own rule replaces them, so they are no whole-token target)."""
    order = list(roles)
    folded = {t: fold_word(texts[t]) for t in texts}
    whole = {
        folded[t]
        for t in order
        if (roles[t] is None or roles[t] in _NAME_ROLES)
        and t not in excluded
        and folded[t]
        and folded[t] not in _CONNECTING
    }
    compound: set[tuple[str, ...]] = set()

    def add(token_ids: Sequence[str]) -> None:
        words = tuple(folded[t] for t in token_ids if folded[t])
        if len(words) >= 2:
            compound.add(words)

    add(order)
    by_line: dict[tuple[int, int], list[str]] = {}
    for t in order:
        by_line.setdefault((line_of[t][0], 0), []).append(t)
    for ids in by_line.values():
        if any((roles[t] is None or roles[t] in _IDENTIFYING_LINE_ROLES) and folded[t] for t in ids):
            add(ids)
    street = [t for t in order if t not in locality]
    if any((roles[t] is None or roles[t] in _IDENTIFYING_LINE_ROLES) and folded[t] for t in street):
        add(street)
    for i, t in enumerate(order):
        role = roles[t]
        nxt = roles[order[i + 1]] if i + 1 < len(order) else None
        if role in (Role.NUMBER, Role.UNIT_NUMBER_AND_NUMBER) and nxt is Role.STREET_NAME:
            j = i + 1
            while j < len(order) and roles[order[j]] is Role.STREET_NAME:
                j += 1
            add(order[i:j])
        if role is Role.BOX_KIND and nxt is Role.BOX_NUMBER:
            k = i
            while k > 0 and roles[order[k - 1]] is Role.BOX_KIND:
                k -= 1
            add(order[k : i + 2])
    place = [i for i, t in enumerate(order) if t in locality and roles[t] not in _NOT_THE_PLACE]
    if len(place) > 1 and any(roles[order[i]] in (Role.SUBURB, None) for i in place):
        add(order[place[0] : place[-1] + 1])
    for label, value in form_parts:
        if value and all(roles.get(t) in STRUCTURE_ROLES for t in value):
            continue  # a State box: its value stays, so 'State NSW' is no target
        add([*label, *value])
    return UnitTargets(unit_id, tuple(sorted(compound)), tuple(sorted(whole)))


# ---------------------------------------------------------------------------------------- the adapter
@dataclass(frozen=True, slots=True)
class _Doc:
    """The document's tokens and lines, indexed both ways."""

    lines: tuple[Line, ...]
    tokens: Mapping[str, Token]
    pos_of: Mapping[str, Pos]
    tid_at: Mapping[Pos, str]
    index: Mapping[Pos, int]  # reading order of every token


def _doc_of(pages: Sequence[PageAnalysis]) -> _Doc:
    lines = tuple(line for p in pages for line in p.lines)
    pos_of: dict[str, Pos] = {}
    tid_at: dict[Pos, str] = {}
    index: dict[Pos, int] = {}
    for li, line in enumerate(lines):
        for wi, tok in enumerate(line.tokens):
            pos_of[tok.token_id] = (li, wi)
            tid_at[(li, wi)] = tok.token_id
            index[(li, wi)] = len(index)
    return _Doc(lines, {t.token_id: t for p in pages for t in p.tokens()}, pos_of, tid_at, index)


@dataclass(slots=True)
class _Spec:
    """A unit under assembly; grammar units and forced (paired) units share it."""

    roles: dict[Pos, Role | None]
    kind: UnitKind
    key: AddressKey
    how: tuple[str, ...] = ()
    overseas: bool = False
    form: bool = False
    form_parts: tuple[FormPart, ...] = ()
    partial: tuple[PartialWord, ...] = ()
    uids: set[str] = field(default_factory=set)
    repeat_of: str | None = None  # a grammar uid
    forced: bool = False
    place: frozenset[Pos] = frozenset()  # words a pairing says are the place, whether or not the grammar read them


def _spec_of(u: AddressUnit) -> _Spec:
    return _Spec(
        dict(u.roles),
        u.kind,
        u.key,
        u.how,
        u.overseas,
        u.form or any(h in ("form", "form_locality") for h in u.how),
        u.form_parts,
        u.partial,
        {u.uid},
        u.repeat_of,
    )


def _frame(line: Line) -> int:
    """The reading frame of a line from the angle of its top edge: 0 when it is within 25 degrees of horizontal."""
    (x0, y0), (x1, y1) = line.quad[0], line.quad[1]
    angle = math.degrees(math.atan2(y1 - y0, x1 - x0))
    if abs(angle) <= _SIDEWAYS_DEGREES:
        return 0
    if abs(angle) >= 180.0 - _SIDEWAYS_DEGREES:
        return 180
    return 90 if angle > 0 else 270


def _band(line: Line, height: int, fraction: float) -> Band:
    if height <= 0 or fraction <= 0:
        return Band.BODY
    if line.bbox.y1 <= fraction * height:
        return Band.HEADER
    if line.bbox.y0 >= (1.0 - fraction) * height:
        return Band.FOOTER
    return Band.BODY


def _addr_lines(
    pages: Sequence[PageAnalysis],
    blocked: Collection[str],
    band_fraction: float,
    bands: Mapping[str, Band],
    boxes: Mapping[str, BoxRole],
) -> list[AddrLine]:
    out: list[AddrLine] = []
    for page in pages:
        for line in page.lines:
            words = tuple(
                AddrWord(t.text, float(t.bbox.x0), float(t.bbox.x1), t.token_id in blocked) for t in line.tokens
            )
            out.append(
                AddrLine(
                    words,
                    page=line.page,
                    x0=float(line.bbox.x0),
                    y0=float(line.bbox.y0),
                    x1=float(line.bbox.x1),
                    y1=float(line.bbox.y1),
                    band=bands.get(line.line_id) or _band(line, page.geometry.height_px, band_fraction),
                    box=boxes.get(line.line_id),
                    frame=_frame(line),
                )
            )
    return out


def _force(
    specs: list[_Spec],
    lines: Sequence[AddrLine],
    positions: set[Pos],
    place: set[Pos],
    gazetteer: Gazetteer | None,
    street: bool,
) -> None:
    """Make the words at ``positions`` one unit, merging the units that overlap them.  Roles come from the units
    merged, then from a parse of the words alone; a word that neither reads has no role (it is identifying)."""
    hit = [s for s in specs if s.roles.keys() & positions]
    merged: set[Pos] = set(positions)
    for s in hit:
        merged |= s.roles.keys()
    mini: list[AddrLine] = []
    back: list[list[Pos]] = []
    for li in sorted({p[0] for p in merged}):
        wis = sorted(p[1] for p in merged if p[0] == li)
        mini.append(dataclasses.replace(lines[li], words=tuple(lines[li].words[wi] for wi in wis)))
        back.append([(li, wi) for wi in wis])
    best: AddressUnit | None = None
    units = parse_document(mini, gazetteer).units
    if units:
        best = max(units, key=lambda u: (len(u.roles), tuple(-x for x in min(u.roles))))
    roles: dict[Pos, Role | None] = {p: None for p in merged}
    if best is not None:
        for (mli, mwi), role in best.roles.items():
            roles[back[mli][mwi]] = role
    for s in hit:
        for p, known_role in s.roles.items():
            if known_role is not None:
                roles[p] = known_role
    base = max(hit, key=lambda s: len(s.roles)) if hit else None
    if best is not None and len(best.roles) * 2 >= len(merged):
        kind, key = best.kind, best.key
    elif base is not None:
        kind, key = base.kind, base.key
    else:
        kind, key = (UnitKind.STREET if street else UnitKind.PLACE_ONLY), AddressKey()
    spec = _Spec(roles, kind, key, ("pairs",), forced=True, place=frozenset(place))
    for s in hit:
        spec.uids |= s.uids
        spec.overseas = spec.overseas or s.overseas
        spec.form = spec.form or s.form
        spec.repeat_of = spec.repeat_of or s.repeat_of
        spec.partial += s.partial
        spec.form_parts += s.form_parts
        spec.place |= s.place
        specs.remove(s)
    specs.append(spec)


def _runs(positions: Sequence[Pos], index: Mapping[Pos, int], split_lines: bool) -> list[list[Pos]]:
    """Positions in reading order grouped into runs of consecutive tokens.  A form's boxes split at every line."""
    runs: list[list[Pos]] = []
    for p in positions:
        last = runs[-1][-1] if runs else None
        if last is not None and index[last] + 1 == index[p] and not (split_lines and last[0] != p[0]):
            runs[-1].append(p)
        else:
            runs.append([p])
    return runs


def _make(m: Mention, ids: tuple[str, ...], etype: EntityType, tokens: Mapping[str, Token]) -> Mention:
    return Mention(
        mention_id=content_id("M", m.mention_id, ids[0], ids[-1], etype.value),
        entity_type=etype,
        token_ids=ids,
        evidence=m.evidence,
        value=" ".join(tokens[t].text for t in ids),
    )


def _place_mentions(
    m: Mention, ids: tuple[str, ...], tokens: Mapping[str, Token], policy: PolicyConfig
) -> list[Mention]:
    """Words of an address mention that lie in a place mentioned on its own: a LOCALITY (suburb, state, postcode), and
    the postcode as a mention of its own when it stands alone or the profile does not keep postcodes."""
    if len(ids) == 1 and _POSTCODE.match(tokens[ids[0]].text):
        return [_make(m, ids, E.POSTCODE, tokens)]
    end = len(ids)
    if policy.entity_actions[E.POSTCODE] is not ActionKind.KEEP and end > 1 and _POSTCODE.match(tokens[ids[-1]].text):
        return [_make(m, ids[:-1], E.LOCALITY, tokens), _make(m, ids[-1:], E.POSTCODE, tokens)]
    return [_make(m, ids, E.LOCALITY, tokens)]


def _residual(
    m: Mention, ids: tuple[str, ...], tokens: Mapping[str, Token], policy: PolicyConfig, kept: Collection[str]
) -> list[Mention]:
    """The words of an address-type mention that no unit owns, as mentions.  A STREET_ADDRESS mention whose words lie
    in a place mentioned on its own (the model reads a suburb as an address) becomes a LOCALITY, as it did before the
    join."""
    if m.entity_type is not E.STREET_ADDRESS:
        return [_make(m, ids, m.entity_type, tokens)]
    out: list[Mention] = []
    seg: list[str] = []
    for t in ids:
        if seg and (t in kept) != (seg[-1] in kept):
            out.extend(_segment(m, tuple(seg), tokens, policy, seg[0] in kept))
            seg = []
        seg.append(t)
    out.extend(_segment(m, tuple(seg), tokens, policy, seg[0] in kept))
    return out


def _segment(
    m: Mention, ids: tuple[str, ...], tokens: Mapping[str, Token], policy: PolicyConfig, place: bool
) -> list[Mention]:
    return _place_mentions(m, ids, tokens, policy) if place else [_make(m, ids, E.STREET_ADDRESS, tokens)]


def _residual_runs(ids: Sequence[str], claimed: Collection[str]) -> list[tuple[str, ...]]:
    runs: list[tuple[str, ...]] = []
    cur: list[str] = []
    for t in ids:
        if t in claimed:
            if cur:
                runs.append(tuple(cur))
            cur = []
        else:
            cur.append(t)
    if cur:
        runs.append(tuple(cur))
    return runs


def _is_place(role: Role | None, pos: Pos, spec: _Spec) -> bool:
    """A word is in the place when its role says so, or a pairing named it part of the place and nothing read it."""
    return (role is not None and role in LOCALITY_ROLES) or (role is None and pos in spec.place)


def _layout_part(run: Sequence[Pos], spec: _Spec) -> tuple[UnitPart, int | None]:
    at_place = [_is_place(spec.roles[p], p, spec) for p in run]
    start = at_place.index(True) if True in at_place else None
    if start is None:
        return UnitPart.STREET, None
    return (UnitPart.JOINED if not all(at_place) else UnitPart.PLACE), start


def _apply_pairs(
    specs: list[_Spec],
    pairs: Mapping[str, str | None],
    known: Sequence[Mention],
    doc: _Doc,
    lines: Sequence[AddrLine],
    gazetteer: Gazetteer | None,
) -> None:
    """Make each pair one unit (see :func:`join_addresses`), in the order of the place mention ids."""
    by_id = {m.mention_id: m for m in known}
    for place_id in sorted(pairs):
        place = by_id.get(place_id)
        street_id = pairs[place_id]
        street = by_id.get(street_id) if street_id is not None else None
        if place is None:
            continue
        place_at = {doc.pos_of[t] for t in place.token_ids}
        positions = place_at | ({doc.pos_of[t] for t in street.token_ids} if street is not None else set())
        hit = [s for s in specs if s.roles.keys() & positions]
        if len(hit) == 1 and positions <= hit[0].roles.keys():
            continue  # the grammar already reads these words as one unit
        _force(specs, lines, positions, place_at, gazetteer, street is not None)


@dataclass(slots=True)
class _Built:
    """What the units become: the units, the mentions that carry them, their layouts and targets, and the tokens the
    units took from other mentions."""

    units: dict[str, JoinedUnit] = field(default_factory=dict)
    layouts: dict[str, MentionLayout] = field(default_factory=dict)
    mentions: list[Mention] = field(default_factory=list)
    targets: dict[str, UnitTargets] = field(default_factory=dict)
    claimed: set[str] = field(default_factory=set)


def _build_units(
    specs: Sequence[_Spec], doc: _Doc, owners: Mapping[str, Sequence[Mention]], owned: Collection[EntityType]
) -> _Built:
    """One STREET_ADDRESS mention per run of a unit's tokens (the words another mention owns are left to it)."""
    built = _Built()
    held: dict[str, str] = {}  # grammar uid -> the unit that holds it
    repeats: dict[str, str] = {}  # unit id -> grammar uid of the unit it repeats

    def foreign(tid: str) -> bool:
        return any(m.entity_type not in owned for m in owners.get(tid, ()))

    for s in specs:
        pos = sorted(s.roles)
        carried = [p for p in pos if not foreign(doc.tid_at[p])]
        if not carried:
            continue
        unit_id = content_id("AU", doc.tid_at[pos[0]], doc.tid_at[pos[-1]])
        evidence = {Evidence(JOIN_DETECTOR, JOIN_VERSION, f"join.{s.kind.value}", EvidenceStrength.STRONG, 1.0)}
        for p in carried:
            for m in owners.get(doc.tid_at[p], ()):
                evidence.update(m.evidence)
        mention_ids: list[str] = []
        for run in _runs(carried, doc.index, s.form):
            ids = tuple(doc.tid_at[p] for p in run)
            part, place_start = _layout_part(run, s)
            mention = Mention(
                mention_id=content_id("M", unit_id, ids[0], ids[-1], E.STREET_ADDRESS.value),
                entity_type=E.STREET_ADDRESS,
                token_ids=ids,
                evidence=tuple(sorted(evidence)),
                value=" ".join(doc.tokens[t].text for t in ids),
            )
            built.mentions.append(mention)
            mention_ids.append(mention.mention_id)
            roles = tuple(s.roles[p] for p in run)
            built.layouts[mention.mention_id] = MentionLayout(unit_id, part, place_start, roles)
        built.claimed.update(doc.tid_at[p] for p in carried)
        tid_roles = {doc.tid_at[p]: s.roles[p] for p in pos}
        excluded = tuple(doc.tid_at[p] for p in pos if foreign(doc.tid_at[p]))
        unit = JoinedUnit(
            unit_id=unit_id,
            kind=s.kind,
            key=s.key,
            roles=MappingProxyType(tid_roles),
            street=tuple(doc.tid_at[p] for p in pos if not _is_place(s.roles[p], p, s)),
            locality=tuple(doc.tid_at[p] for p in pos if _is_place(s.roles[p], p, s)),
            mention_ids=tuple(mention_ids),
            excluded=excluded,
            partial=tuple(PartialToken(doc.tid_at[w.pos], w.keep, w.change) for w in s.partial if w.pos in s.roles),
            overseas=s.overseas,
            form=s.form,
            repeat_of=None,
            forced=s.forced,
        )
        built.units[unit_id] = unit
        labelled = [
            (fp, [doc.tid_at[p] for p in fp.label]) for fp in s.form_parts if all(p in doc.tid_at for p in fp.label)
        ]
        built.targets[unit_id] = _unit_targets(
            unit_id,
            tid_roles,
            {t: doc.tokens[t].text for t in [*tid_roles, *(t for _fp, ids in labelled for t in ids)]},
            {t: doc.pos_of[t] for t in tid_roles},
            [(ids, [doc.tid_at[p] for p in fp.value if p in s.roles]) for fp, ids in labelled],
            excluded,
            unit.locality,
        )
        for uid in s.uids:
            held[uid] = unit_id
        if s.repeat_of is not None:
            repeats[unit_id] = s.repeat_of
    for unit_id, uid in repeats.items():
        src = held.get(uid)
        if src is not None and src != unit_id:
            built.units[unit_id] = dataclasses.replace(built.units[unit_id], repeat_of=src)
    return built


def _remaining_mentions(
    mentions: Sequence[Mention],
    claimed: Collection[str],
    kept_places: Collection[str],
    tokens: Mapping[str, Token],
    policy: PolicyConfig,
) -> list[Mention]:
    """The mentions the units did not take whole.  Words of an address-type mention outside every unit stay in a mention
    of that type (a STREET_ADDRESS over a place on its own becomes a LOCALITY again); a date or age mention loses only
    the words a unit owns; everything else comes out as it went in."""
    out: list[Mention] = []
    for m in mentions:
        if not any(t in claimed or (t in kept_places and m.entity_type is E.STREET_ADDRESS) for t in m.token_ids):
            out.append(m)
            continue
        for piece in _residual_runs(m.token_ids, claimed):
            if m.entity_type in ADDRESS_TYPES:
                out.extend(_residual(m, piece, tokens, policy, kept_places))
            else:
                out.append(_make(m, piece, m.entity_type, tokens))
    return out


def join_addresses(
    mentions: Sequence[Mention],
    pages: Sequence[PageAnalysis],
    policy: PolicyConfig,
    pairs: Mapping[str, str | None] | None = None,
    *,
    repeat_sweep: bool = True,
    band_fraction: float = 0.05,
    bands: Mapping[str, Band] | None = None,
    boxes: Mapping[str, BoxRole] | None = None,
    gazetteer: Gazetteer | None = None,
) -> AddressJoin:
    """Join the document's address mentions into whole-address units.

    ``mentions`` are the document's mentions (any types), ``pages`` its analysed pages (their lines in reading order),
    ``policy`` the profile's policy.  With ``policy.addresses == "parts"`` the join only calls
    :func:`mlredact.policy.engine.split_kept_places` (no layouts, units or targets).

    ``pairs`` is the explicit pairing the benchmark's oracle modes pass in: place mention id -> street-line mention id
    (an injected LOCATION and the injected ADDRESS of its entity), or None for a place that is a place-only unit by
    itself.  Each pair becomes one unit whatever the grammar finds (the grammar's units that overlap it merge into it),
    and every word of the paired mentions is part of it.  An id that is not among ``mentions`` is ignored.
    ``repeat_sweep=False`` drops the units that only the repeat sweep found (the oracle modes list every mention, so
    there is nothing to sweep).  ``band_fraction`` is the share of the page height, at the top and the bottom, that
    counts as header and footer (0 turns the position rule off).  It is small on purpose: a continuation line at the top
    margin of a page must stay body text for the page-break join.  ``bands`` and ``boxes`` tag lines by ``line_id``
    where the caller knows better: a letterhead band, or a form's label, value and cell boxes (tool 2 has neither
    detector yet)."""
    doc = _doc_of(pages)
    if policy.addresses == "parts":
        return AddressJoin(tuple(split_kept_places(mentions, doc.tokens, policy)))
    known = [m for m in mentions if all(t in doc.pos_of for t in m.token_ids)]
    unknown = [m for m in mentions if not all(t in doc.pos_of for t in m.token_ids)]
    owned = ADDRESS_TYPES | TRIMMED_TYPES
    owners: dict[str, list[Mention]] = {}
    for m in known:
        for t in m.token_ids:
            owners.setdefault(t, []).append(m)
    blocked = {t for m in known if m.entity_type not in owned for t in m.token_ids}

    lines = _addr_lines(pages, blocked, band_fraction, bands or {}, boxes or {})
    parse = parse_document(lines, gazetteer)
    specs = [_spec_of(u) for u in parse.units if repeat_sweep or "repeat_sweep" not in u.how]
    if pairs:
        _apply_pairs(specs, pairs, known, doc, lines, gazetteer)
    specs.sort(key=lambda s: min(s.roles))
    built = _build_units(specs, doc, owners, owned)

    kept = tuple(
        KeptRecord(k.kind, ids, k.after_cue)
        for k in parse.kept
        if (ids := tuple(doc.tid_at[p] for p in k.positions if p in doc.tid_at and doc.tid_at[p] not in built.claimed))
    )
    kept_places = frozenset(t for k in kept if k.kind in _PLACE_KINDS for t in k.token_ids)
    out = [*unknown, *_remaining_mentions(known, built.claimed, kept_places, doc.tokens, policy), *built.mentions]
    big = len(doc.index)

    def sort_key(m: Mention) -> tuple[int, int, str, str]:
        first, last = doc.pos_of.get(m.token_ids[0]), doc.pos_of.get(m.token_ids[-1])
        return (doc.index[first] if first else big, doc.index[last] if last else big, m.entity_type.value, m.mention_id)

    out.sort(key=sort_key)
    actions = policy.entity_actions
    replaced = {t for m in out if actions[m.entity_type] is not ActionKind.KEEP for t in m.token_ids}
    exempt = ExemptPositions(
        frozenset(t for k in kept for t in k.token_ids if t not in replaced),
        frozenset(t for m in out if actions[m.entity_type] is ActionKind.KEEP for t in m.token_ids),
    )
    return AddressJoin(
        tuple(out),
        MappingProxyType(built.layouts),
        MappingProxyType(built.units),
        kept,
        MappingProxyType(built.targets),
        exempt,
    )
