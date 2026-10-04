"""Policy: map each mention to an action according to the active profile (plan §7.1, §10)."""

from __future__ import annotations

from collections.abc import Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass

from mlredact.config.schema import PolicyConfig
from mlredact.core.model import Mention
from mlredact.core.types import ActionKind, EntityType
from mlredact.surrogate.dates import generalise_age, shift_date


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


def acted(decisions: Sequence[Decision]) -> list[Decision]:
    return [d for d in decisions if d.action is not ActionKind.KEEP]
