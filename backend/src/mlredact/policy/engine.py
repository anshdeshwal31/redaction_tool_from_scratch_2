"""Policy: map each mention to an action according to the active profile (plan §7.1, §10)."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass

from mlredact.config.schema import PolicyConfig
from mlredact.core.canonical import content_id
from mlredact.core.model import Mention, Token
from mlredact.core.types import ActionKind, EntityType
from mlredact.surrogate.dates import generalise_age, shift_date
from mlredact.surrogate.generators import is_state_token, is_street_type

_POSTCODE = re.compile(r"^\d{4}[,.]?$")


@dataclass(frozen=True, slots=True)
class Decision:
    mention: Mention
    action: ActionKind


DATE_TYPES = frozenset({EntityType.DATE, EntityType.DATE_OF_BIRTH, EntityType.DATE_OF_DEATH})


def decide(
    mentions: Sequence[Mention], policy: PolicyConfig, professionals: AbstractSet[str] = frozenset()
) -> list[Decision]:
    """``professionals``: person mention ids resolved as professionals (kept only if the profile says so)."""
    out = []
    for m in mentions:
        if policy.keep_professionals and m.entity_type is EntityType.PERSON and m.mention_id in professionals:
            out.append(Decision(m, ActionKind.KEEP))
        else:
            out.append(_settle(Decision(m, policy.entity_actions[m.entity_type])))
    return out


def _settle(d: Decision) -> Decision:
    """Transformations that would not change the text are KEEPs, in every render style: ages below
    the top-coding threshold, and bare years under date shifting (an offset of under a year cannot
    move them)."""
    m = d.mention
    if d.action is ActionKind.GENERALISE and m.entity_type is EntityType.AGE and generalise_age(m.value) == m.value:
        return Decision(m, ActionKind.KEEP)
    if d.action is ActionKind.DATE_SHIFT and m.entity_type in DATE_TYPES and shift_date(m.value, 1) == m.value:
        return Decision(m, ActionKind.KEEP)
    return d  # anything else (e.g. GENERALISE of a non-age) is rendered fail-safe, never kept


def split_kept_places(mentions: Sequence[Mention], tokens: Mapping[str, Token], policy: PolicyConfig) -> list[Mention]:
    """Street addresses whose place part the profile keeps ("12 Smith St, Parramatta NSW 2150" when
    localities are kept): the place part (suburb, state, postcode) becomes its own mention, so only
    the street line is replaced and only the street line counts as a removed value in verification.
    A postcode the profile does not keep becomes a mention of its own."""
    actions = policy.entity_actions
    if actions[EntityType.LOCALITY] is not ActionKind.KEEP or actions[EntityType.STREET_ADDRESS] is ActionKind.KEEP:
        return list(mentions)
    out: list[Mention] = []
    for m in mentions:
        start = None
        if m.entity_type is EntityType.STREET_ADDRESS:
            start = _place_start([tokens[t].text for t in m.token_ids])
        if start is None:
            out.append(m)
            continue
        ids = m.token_ids
        end = len(ids)
        postcode = None
        if actions[EntityType.POSTCODE] is not ActionKind.KEEP and _POSTCODE.match(tokens[ids[-1]].text):
            end -= 1
            postcode = ids[end:]
        for part, etype in ((ids[:start], EntityType.STREET_ADDRESS), (ids[start:end], EntityType.LOCALITY)):
            if part:
                out.append(_part(m, part, etype, tokens))
        if postcode:
            out.append(_part(m, postcode, EntityType.POSTCODE, tokens))
    return out


def _place_start(texts: Sequence[str]) -> int | None:
    """Index of the first token of the place part of an address, or None when it cannot be told
    apart from the street line (the whole address is then replaced).

    The state is the one that ends the address (before any postcode), so a street named after a
    state ("12 Victoria Road, ...") is not mistaken for it.  The street line ends at its last comma
    ("8 Broadway, Ultimo NSW"; "3 Smith St, Castle Hill NSW"), or else after its street type.  A comma
    with a number after it, or a street type only after it, is inside the street line ("Unit 5, ...")."""
    state = len(texts) - (1 if texts and _POSTCODE.match(texts[-1]) else 0) - 1
    if state < 0 or not is_state_token(texts[state]):
        return None
    street_types = [i for i in range(state) if is_street_type(texts[i])]
    comma = max((i for i in range(state - 1) if texts[i].endswith(",")), default=None)  # not "Parramatta, NSW"
    if (
        comma is not None
        and not any(c.isdigit() for t in texts[comma + 1 : state] for c in t)
        and (not street_types or street_types[0] <= comma)
    ):
        return comma + 1
    street_type = street_types[-1] if street_types else None
    if street_type is not None:
        return street_type + 1
    numbered = max((i for i in range(state) if any(c.isdigit() for c in texts[i])), default=None)
    if numbered is None:
        return 0  # "Parramatta NSW 2150": place only
    if numbered > 0 and texts[numbered - 1].lower() == "box":
        return numbered + 1  # "PO Box 12 Parramatta NSW"
    return None  # "8 Broadway Ultimo NSW": no way to tell where the street name ends


def _part(m: Mention, ids: tuple[str, ...], etype: EntityType, tokens: Mapping[str, Token]) -> Mention:
    return Mention(
        mention_id=content_id("M", m.mention_id, ids[0], ids[-1], etype.value),
        entity_type=etype,
        token_ids=ids,
        evidence=m.evidence,
        value=" ".join(tokens[t].text for t in ids),
    )


def acted(decisions: Sequence[Decision]) -> list[Decision]:
    return [d for d in decisions if d.action is not ActionKind.KEEP]
