"""Working types of the parse: anchors (places read from the end), labels and fields (the form layer), street
candidates and the unit under construction.  Nothing here leaves the package."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from mlredact.text.address.model import AddressKey, FormPart, KeptKind, PartialWord, Pos, Role, UnitKind


class Flag(StrEnum):
    """Properties of a street candidate."""

    TYPED = "typed"
    UNTYPED = "untyped"
    MONTH = "month"
    DR_NAME = "dr_name"
    SPLIT = "split"
    BOX = "box"
    NEEDS_CTX = "needs_ctx"
    NUMBERLESS = "numberless"
    VIA = "via"
    CORNER = "corner"
    DX = "dx"
    PROSE = "prose"


class AnchorKind(StrEnum):
    STRONG = "strong"  # a state and an in-range postcode (or a glued 'NSW2750')
    STATELESS = "stateless"  # a four-digit postcode with no state before it
    WEAK = "weak"  # a state form with no postcode, ending a phrase
    OVERSEAS = "overseas"  # a postcode shape and a foreign country (owner ruling on overseas addresses)


@dataclass(slots=True)
class Anchor:
    """A place read from the end of its words: country, postcode, state."""

    li: int
    kind: AnchorKind
    state: tuple[int, int] | None  # first and last word of the state form
    pc: int | None  # the (last) postcode word
    glued: bool
    code: str | None
    pcv: int | None
    country: tuple[int, int] | None = None  # first and last word of the country
    pc_first: int | None = None  # the first postcode word (a two-word postcode: 'OX1 4AA')

    @property
    def start(self) -> int:
        if self.state is not None:
            return self.state[0]
        if self.pc_first is not None:
            return self.pc_first
        assert self.pc is not None
        return self.pc

    @property
    def end(self) -> int:
        if self.country is not None:
            return self.country[1]
        if self.pc is not None:
            return self.pc
        assert self.state is not None
        return self.state[1]


class LabelRole(StrEnum):
    ADDRESS = "address"
    STREET = "street"
    STREET_NAME = "street_name"
    STREET_NO = "street_no"
    UNIT = "unit"
    SUBURB = "suburb"
    STATE = "state"
    POSTCODE = "postcode"
    OTHER = "other"
    UNREADABLE = "unreadable"


STREET_PART_LABELS = frozenset(
    {LabelRole.UNIT, LabelRole.STREET_NO, LabelRole.STREET_NAME, LabelRole.STREET, LabelRole.ADDRESS}
)
LOCALITY_PART_LABELS = frozenset({LabelRole.SUBURB, LabelRole.STATE, LabelRole.POSTCODE})


@dataclass(slots=True)
class Label:
    """A label span on a line."""

    start: int
    end: int
    role: LabelRole
    colon: bool
    whole: bool


@dataclass(slots=True)
class Field:
    """A label with the positions of its value, in reading order, and the boxes it occupies."""

    li: int
    lab: Label
    role: LabelRole
    vals: list[Pos]
    boxes: list[tuple[int, int, int]]  # (line, first word, last word)


@dataclass(slots=True)
class Street:
    """A street-line candidate: the words of one line from ``start`` to ``end``."""

    li: int
    start: int
    end: int
    kind: UnitKind
    roles: dict[int, Role]
    flags: set[Flag] = field(default_factory=set)
    above: list[tuple[int, dict[int, Role]]] = field(default_factory=list)
    name_start: int | None = None


@dataclass(slots=True)
class UnitB:
    """A unit under construction."""

    kind: UnitKind
    roles: dict[Pos, Role]
    street: list[Pos] = field(default_factory=list)
    locality: list[Pos] = field(default_factory=list)
    how: list[str] = field(default_factory=list)
    partial: list[PartialWord] = field(default_factory=list)
    key: AddressKey = field(default_factory=AddressKey)
    repeat_src: UnitB | None = None
    overseas: bool = False
    form: bool = False
    form_parts: list[FormPart] = field(default_factory=list)
    seq: int = 0  # creation order, set when the unit is added

    def positions(self) -> set[Pos]:
        return set(self.roles)


@dataclass(slots=True)
class KeptB:
    kind: KeptKind
    positions: list[Pos]
    after_cue: bool = False
