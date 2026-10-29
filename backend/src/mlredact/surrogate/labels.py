"""Consistent tags for the ``label`` render style: ``[PERSON 1]``, ``[MEDICARE 1]``, ``[ADDRESS 2]``.

Numbering is per tag name in order of first appearance.  People are numbered per resolved
identity (cluster, else surname group, else value), so every mention of one person shares a tag.
An address is numbered per address the join found: a repeat (another spelling, a street line seen
without its place) takes the tag of the address it repeats.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from mlredact.core.types import ActionKind, EntityType
from mlredact.policy.addresses import JoinedUnit, MentionLayout
from mlredact.policy.engine import Decision
from mlredact.resolve.persons import PersonResolution, Role
from mlredact.surrogate.assign import MentionSurrogate
from mlredact.text.normalize import match_key

_NAMES = {
    EntityType.PERSON: "PERSON",
    EntityType.DATE_OF_BIRTH: "DOB",
    EntityType.DATE_OF_DEATH: "DATE OF DEATH",
    EntityType.STREET_ADDRESS: "ADDRESS",
    EntityType.ORGANISATION: "ORGANISATION",
    EntityType.OTHER_ID: "ID",
}


def tag_name(etype: EntityType) -> str:
    return _NAMES.get(etype, etype.value.upper().replace("_", " "))


def assign_labels(
    decisions: Sequence[Decision],
    persons: PersonResolution,
    layouts: Mapping[str, MentionLayout] | None = None,
    units: Mapping[str, JoinedUnit] | None = None,
) -> dict[str, MentionSurrogate]:
    """``layouts`` and ``units``: the address join's; without them an address is numbered by its written value."""
    numbers: dict[str, dict[str, int]] = {}
    out: dict[str, MentionSurrogate] = {}
    same = _same_addresses(units or {})
    for d in decisions:
        if d.action is ActionKind.KEEP:
            continue
        m = d.mention
        name = tag_name(m.entity_type)
        key = match_key(m.value)
        layout = (layouts or {}).get(m.mention_id)
        if m.entity_type is EntityType.STREET_ADDRESS and layout is not None and layout.unit_id in same:
            key = "unit:" + same[layout.unit_id]
        if m.entity_type is EntityType.PERSON:
            pm = persons.mentions.get(m.mention_id)
            if pm is not None and pm.cluster:
                key = "cluster:" + pm.cluster
            elif pm is not None:
                surnames = [t.key for t in pm.tokens if t.role is Role.SURNAME]
                key = "surname:" + "".join(surnames) if surnames else key
        bucket = numbers.setdefault(name, {})
        n = bucket.setdefault(key, len(bucket) + 1)
        texts = [f"[{name} {n}]"] + [""] * (len(m.token_ids) - 1)
        out[m.mention_id] = MentionSurrogate(m.mention_id, tuple(texts))
    return out


def _same_addresses(units: Mapping[str, JoinedUnit]) -> dict[str, str]:
    """Unit id -> the id of the first unit of its address: the unit it repeats (the join's link), or the first unit
    with the same whole canonical key (street and place)."""
    first_of: dict[object, str] = {}
    out: dict[str, str] = {}
    for unit_id, unit in units.items():  # in reading order
        head = unit.repeat_of or unit_id
        if unit.key.has_street and unit.key.has_place:
            head = first_of.setdefault(unit.key, out.get(head, head))
        out[unit_id] = out.get(head, head)
    return out
