"""Consistent tags for the ``label`` render style: ``[PERSON 1]``, ``[MEDICARE 1]``, ``[ADDRESS 2]``.

Numbering is per tag name in order of first appearance.  People are numbered per resolved
identity (cluster, else surname group, else value), so every mention of one person shares a tag.
"""

from __future__ import annotations

from collections.abc import Sequence

from mlredact.core.types import ActionKind, EntityType
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


def assign_labels(decisions: Sequence[Decision], persons: PersonResolution) -> dict[str, MentionSurrogate]:
    numbers: dict[str, dict[str, int]] = {}
    out: dict[str, MentionSurrogate] = {}
    for d in decisions:
        if d.action is ActionKind.KEEP:
            continue
        m = d.mention
        name = tag_name(m.entity_type)
        key = match_key(m.value)
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
