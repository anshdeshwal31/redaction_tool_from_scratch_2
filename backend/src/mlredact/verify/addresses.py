"""Verification of whole addresses beyond the text layer: V4's probe and re-detection, and V5 (FINAL_PLAN TOOL2
section 6, as amended by D and H).

* V4.  The removed-value probe searches the re-read output for the per-part targets of every unit (compound and
  whole-token targets, form parts as 'label value'), with the same exempt positions as V2.  The join runs again on the
  re-read output: a unit, or an anchored place, whose identifying words are not this run's fakes is a residual even
  though LOCALITY and POSTCODE are KEEP types.  A residual that repeats a found unit is re-planned with that unit's
  fake (:attr:`AddressReread.swaps`) instead of a black box.
* V5.  Every non-structure word of a unit's fake is compared with ALL identifying words and numbers of the document's
  units (amendment D1), the fake postcode lies in the state's ranges and follows D3, and no two different originals
  share a whole fake.

Pure: it reads units, texts and fakes.  Document text never enters a message or a repr.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field

from mlredact.core.geometry import BBox
from mlredact.policy.addresses import JoinedUnit, UnitTargets, fold_word
from mlredact.surrogate.addresses import postcode_ok
from mlredact.surrogate.exclusions import Exclusions, digit_runs
from mlredact.surrogate.generators import canonical_state
from mlredact.text.address import STRUCTURE_ROLES, Role
from mlredact.text.gazetteer import fold_name, in_range, range_states

_GLUED = re.compile(r"^([A-Za-z.]*?)(\d{4})$")
_OCR_DIGITS = str.maketrans("OoIlSB", "001158")
_EDGES = ",;:()[]{}|!?\"'“”‘’"
_POSTCODE_ROLES = frozenset({Role.POSTCODE, Role.STATE_AND_POSTCODE})


# ------------------------------------------------------------------------------------------------------- V4
@dataclass(frozen=True, slots=True)
class AddressReread:
    """What V4 knows of the document's addresses.

    ``targets``: the units' verification targets.  ``exempt``: per page, the padded boxes of the exempt positions
    (kept places, kept mentions of other types).  ``fake_words``: the folded words of every fake drawn in this run.
    ``structure``: the folded words that stay as written (designators, the state, box kinds).  ``swaps``: the folded
    original word -> its fake, for the words one fake covers unambiguously (a word two units fake differently is left
    out): a residual whose words all swap is re-planned with their fakes.  ``plain``: the folded words the document
    writes outside every unit and that no unit's identifying words include: a re-read unit that took such a word in
    (a ward, a heading) did not leave it behind."""

    targets: tuple[UnitTargets, ...] = field(default=(), repr=False)
    exempt: Mapping[int, Sequence[BBox]] = field(default_factory=dict, repr=False)
    fake_words: frozenset[str] = field(default=frozenset(), repr=False)
    structure: frozenset[str] = field(default=frozenset(), repr=False)
    swaps: Mapping[str, str] = field(default_factory=dict, repr=False)
    plain: frozenset[str] = field(default=frozenset(), repr=False)


def _edges(text: str) -> tuple[str, str, str]:
    core = text.strip(_EDGES)
    if not core:
        return text, "", ""
    start = text.index(core)
    return text[:start], core, text[start + len(core) :]


def carry_punctuation(original: str, fake: str) -> str:
    """``fake`` with the leading and trailing punctuation and the case (upper or lower) of the ``original`` word."""
    lead, core, trail = _edges(original)
    if len(core) > 1 and core.isupper():
        fake = fake.upper()
    elif core.islower():
        fake = fake.lower()
    return lead + fake + trail


def reread_of(
    units: Mapping[str, JoinedUnit],
    targets: Mapping[str, UnitTargets],
    texts: Mapping[str, str],
    fakes: Mapping[str, Mapping[str, str]],
    applied: AbstractSet[str],
    fake_words: frozenset[str],
    exempt: Mapping[int, Sequence[BBox]],
    outside: Iterable[str] = (),
) -> AddressReread:
    """The V4 data of a run.  ``applied``: the tokens of the unit mentions that carry a fake (a token another mention
    owns is replaced by that mention's rule); ``fakes``: unit id -> token id -> replacement ('' erases a token);
    ``outside``: the text of the document's tokens that no unit holds."""
    structure: set[str] = set()
    swap: dict[str, dict[str, str]] = {}
    for unit_id, unit in sorted(units.items()):
        fake = fakes.get(unit_id)
        for tid, role in unit.roles.items():
            word = fold_word(texts[tid])
            if not word:
                continue
            if role is not None and role in STRUCTURE_ROLES:
                structure.add(word)
            elif fake is not None and tid in applied and tid in fake:
                swap.setdefault(word, {}).setdefault(fold_word(fake[tid]), fake[tid])
    swaps = {w: next(iter(v.values())) for w, v in sorted(swap.items()) if len(v) == 1 and w not in structure}
    identifying = {
        fold_word(texts[t])
        for unit in units.values()
        for t, role in unit.roles.items()
        if role is None or role not in STRUCTURE_ROLES
    }
    plain = frozenset(w for text in outside if (w := fold_word(text)) and w not in identifying)
    return AddressReread(
        tuple(targets[u] for u in sorted(targets)), exempt, fake_words, frozenset(structure), swaps, plain
    )


def swapped(words: Sequence[str], reread: AddressReread) -> str | None:
    """The text that replaces a run of words with the fakes of the units they belong to: a structure word stays, a
    word with a fake takes it (with the original's punctuation).  None when some word has neither, or no word takes a
    fake (or the text would be empty): the run is then blacked out."""
    out: list[str] = []
    changed = False
    for text in words:
        word = fold_word(text)
        if not word or word in reread.structure:
            out.append(text)
        elif word in reread.swaps:
            changed = True
            if reread.swaps[word]:
                out.append(carry_punctuation(text, reread.swaps[word]))
        else:
            return None
    drawn = " ".join(out).strip()
    return drawn if changed and any(c.isalnum() for c in drawn) else None


# ------------------------------------------------------------------------------------------------------- V5
@dataclass(frozen=True, slots=True)
class AddressFakes:
    """The fakes of a run as V5 reads them: the units, the original text of every token, the fake text of every token
    (unit id -> token id -> text) and the tokens that carry a fake (a mention that is replaced, not kept or owned by
    another rule)."""

    units: Mapping[str, JoinedUnit] = field(repr=False)
    texts: Mapping[str, str] = field(repr=False)
    fakes: Mapping[str, Mapping[str, str]] = field(repr=False)
    applied: AbstractSet[str] = field(repr=False)


def _identifying(unit: JoinedUnit, tid: str, text: str) -> str | None:
    """The part of a token's text that is identifying (None for a structure word): a partial word counts by its
    changing part."""
    role = unit.roles[tid]
    if role is not None and role in STRUCTURE_ROLES:
        return None
    for p in unit.partial:
        if p.token_id == tid:
            return p.change
    if role is Role.STATE_AND_POSTCODE:
        m = _GLUED.match(_edges(text)[1])
        return m.group(2) if m else text
    return text


def exclusions_of(addresses: AddressFakes) -> Exclusions:
    """Amendment D1: the identifying words and numbers of ALL the document's units, in any role."""
    words: list[str] = []
    numbers: list[str] = []
    for _uid, unit in sorted(addresses.units.items()):
        for tid in unit.roles:
            part = _identifying(unit, tid, addresses.texts[tid])
            if part is None:
                continue
            runs = digit_runs(part)
            if runs:
                numbers.extend(runs)
            elif _edges(part)[1]:
                words.append(_edges(part)[1])
    return Exclusions.of(words, numbers)


def _digits(text: str) -> str | None:
    core = _edges(text)[1]
    m = _GLUED.match(core)
    core = m.group(2) if m else core.translate(_OCR_DIGITS)
    return core if core.isdigit() and len(core) == 4 else None


def _state_word(role: Role | None, text: str) -> str | None:
    """The state code a word writes: a state word, or the letters of a glued 'NSW2750'."""
    core = _edges(text)[1]
    if role is Role.STATE:
        return canonical_state(core)
    if role is Role.STATE_AND_POSTCODE:
        m = _GLUED.match(core)
        return canonical_state(m.group(1)) if m and m.group(1) else None
    return None


def _state_of(unit: JoinedUnit, texts: Mapping[str, str], original: str | None) -> str | None:
    """The state a unit's text writes (canonical code), else the key's, else the one its postcode's ranges name."""
    for tid, role in unit.roles.items():
        state = _state_word(role, texts[tid])
        if state is not None:
            return state
    if unit.key.state is not None:
        return unit.key.state
    if original is not None:
        states = range_states(int(original))
        if len(states) == 1:
            return next(iter(states))
    return None


def _fake_state(unit: JoinedUnit, texts: Mapping[str, str], fake: Mapping[str, str]) -> str | None:
    """The state the fake writes: the fake text of the unit's state word (replace mode), else the original's."""
    for tid, role in unit.roles.items():
        state = _state_word(role, fake.get(tid) or texts[tid])
        if state is not None:
            return state
    return unit.key.state


def check_address_fakes(addresses: AddressFakes) -> dict[str, int]:
    """V5 for address units; the counts of the problems found.

    * ``address_word_collision``: a word of a fake equals, begins with or contains an identifying word of 3 or more
      letters of any unit, or is one edit from one of 5 or more letters (D1);
    * ``address_number_collision``: a number of a fake is a whole digit run that any unit's identifying numbers hold;
    * ``address_postcode``: the fake postcode is outside the fake state's ranges, equals the original or breaks D3;
    * ``address_shared_fake``: two different originals got one whole fake."""
    problems = {
        "address_word_collision": 0,
        "address_number_collision": 0,
        "address_postcode": 0,
        "address_shared_fake": 0,
    }
    exclusions = exclusions_of(addresses)
    issued: dict[tuple[str, ...], list[str]] = {}
    for unit_id, unit in sorted(addresses.units.items()):
        fake = addresses.fakes.get(unit_id)
        carried = [t for t in unit.roles if t in addresses.applied]
        if fake is None or not carried:
            continue
        for tid in carried:
            part = _identifying(unit, tid, addresses.texts[tid])
            new = fake.get(tid, "")
            if part is None or not new:
                continue
            keep = next((p.keep for p in unit.partial if p.token_id == tid), "")
            new = new[len(keep) :] if keep and new.startswith(keep) else new
            if digit_runs(new):
                if not exclusions.number_ok(new):
                    problems["address_number_collision"] += 1
            elif _edges(new)[1] and not exclusions.word_ok(_edges(new)[1]):
                problems["address_word_collision"] += 1
        if not unit.overseas:
            problems["address_postcode"] += _postcode_problems(unit, addresses.texts, fake, carried)
        whole = tuple(fold_name(" ".join(fake[t] for t in unit.roles if t in fake)).split())
        problems["address_shared_fake"] += sum(
            1 for other in issued.get(whole, ()) if not unit.key.same_address(addresses.units[other].key)
        )
        issued.setdefault(whole, []).append(unit_id)
    return problems


def _postcode_problems(
    unit: JoinedUnit, texts: Mapping[str, str], fake: Mapping[str, str], carried: Collection[str]
) -> int:
    """The fake postcode of a unit against the original: in the fake state's ranges, never the original, D3."""
    problems = 0
    fake_state = _fake_state(unit, texts, fake)
    for tid, role in unit.roles.items():
        if role not in _POSTCODE_ROLES or tid not in carried:
            continue
        original, new_text = _digits(texts[tid]), fake.get(tid, "")
        if original is None or not new_text:
            continue
        new = _digits(new_text)
        state = _state_of(unit, texts, original)
        if (
            new is None
            or (fake_state is not None and not in_range(fake_state, int(new)))
            or not postcode_ok(state, original, new, fake_state)
        ):
            problems += 1
    return problems
