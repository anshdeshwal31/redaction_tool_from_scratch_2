"""Types of the address grammar: the document it reads (words and lines with geometry), and what it finds (units, roles,
keys and places kept on their own).  See the package docstring for how they fit together."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

Pos = tuple[int, int]  # (line index, word index) in the lines the caller passed, both counted from 0


class Role(StrEnum):
    """The role of one word of an address unit.  The values are the case table's role names."""

    BUILDING = "building"  # building or property name words, the building word included
    DESIGNATOR = "designator"  # Unit, Suite, Level, Floor, Ground, Building (before a letter or number) ...
    UNIT_NUMBER = "unit_number"
    LEVEL_NUMBER = "level_number"
    NUMBER = "number"  # house number, ranges and letter suffixes included
    UNIT_NUMBER_AND_NUMBER = "unit_number+number"  # '5/3', 'B02/53'
    STREET_NAME = "street_name"
    STREET_TYPE = "street_type"
    STREET_SUFFIX = "street_suffix"  # the directional after the type ('Smith Street North')
    CORNER = "corner"  # 'Cnr', '&', 'Corner', 'of', 'and' inside a corner form
    BOX_KIND = "box_kind"
    BOX_NUMBER = "box_number"
    BOX_EXTRA = "box_extra"  # words between a box number and the place that are not a road ('City East')
    VIA = "via"
    DX = "dx"
    DX_NUMBER = "dx_number"
    DX_EXCHANGE = "dx_exchange"
    SUBURB = "suburb"  # the town, in an overseas unit
    STATE = "state"
    POSTCODE = "postcode"
    STATE_AND_POSTCODE = "state+postcode"  # glued 'NSW2750'
    COUNTRY = "country"
    CAPITAL_CITY_PHRASE = "capital_city_phrase"  # 'IN YOUR CAPITAL CITY' written in place of a locality


class UnitKind(StrEnum):
    STREET = "street"
    STREET_NUMBERLESS = "street_numberless"
    CORNER = "corner"
    BOX = "box"
    DX = "dx"
    PLACE_ONLY = "place_only"


class Band(StrEnum):
    BODY = "body"
    LETTERHEAD = "letterhead"
    HEADER = "header"
    FOOTER = "footer"


class BoxRole(StrEnum):
    """The kind of form or table box a line is."""

    LABEL = "label"
    VALUE = "value"
    CELL = "cell"


class KeptKind(StrEnum):
    """A place mentioned on its own, which the client profile keeps."""

    PLACE = "place"  # a locality, with its state when written
    STATE = "state"
    POSTCODE = "postcode"  # 'postcode 2750'
    COUNTRY = "country"
    STREET_IN_PROSE = "street_in_prose"  # a street named without a house number in running text
    PLACE_AFTER_CUE = "place_after_cue"  # a locality, state and postcode right after a birth, injury, accident,
    # incident or declaration label or phrase (owner ruling on question 2, option c)
    REPEAT_IN_PROSE = "repeat_in_prose"  # a word of a found unit repeated in running text


@dataclass(frozen=True, slots=True)
class AddrWord:
    """One word of the input.  ``x0``/``x1`` are its horizontal extent in the line's units (page points or pixels).
    ``blocked`` marks a word that another detector owns (a person, organisation or contact mention): a suburb,
    building name or place run never crosses it.  A street name may still be blocked (a surname is a street name)."""

    text: str = field(repr=False)
    x0: float = 0.0
    x1: float = 0.0
    blocked: bool = False


@dataclass(frozen=True, slots=True)
class AddrLine:
    """One line (or form box, or table cell) of the input, with its geometry.  ``frame`` is the reading frame (0
    upright, 270 for text read bottom to top): lines relate to each other only within one frame and one page.  Lines
    must be passed in reading order."""

    words: tuple[AddrWord, ...]
    page: int = 0
    x0: float = 0.0
    y0: float = 0.0
    x1: float = 0.0
    y1: float = 0.0
    band: Band = Band.BODY
    box: BoxRole | None = None
    frame: int = 0

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    @classmethod
    def from_text(
        cls,
        text: str,
        *,
        page: int = 0,
        row: int = 0,
        x0: float = 72.0,
        x1: float | None = None,
        band: Band = Band.BODY,
        box: BoxRole | None = None,
        frame: int = 0,
    ) -> AddrLine:
        """A line of whitespace-separated words laid out on a regular grid: row ``row`` of its page (rows sit 14.4
        points apart, a line is 12 points high) from ``x0`` to ``x1`` (default 5.5 points per character).  Lines of the
        same row sit side by side.  A convenience for callers that have text but no geometry."""
        right = x1 if x1 is not None else x0 + 5.5 * len(text)
        scale = (right - x0) / max(1, len(text))
        words: list[AddrWord] = []
        pos = 0
        for t in text.split():
            at = text.index(t, pos)
            words.append(AddrWord(t, x0 + at * scale, x0 + (at + len(t)) * scale))
            pos = at + len(t)
        y0 = 60.0 + row * 14.4
        return cls(tuple(words), page, x0, y0, right, y0 + 12.0, band, box, frame)


@dataclass(frozen=True, slots=True)
class PartialWord:
    """A word that is partly structure: the ``keep`` letters stay and the ``change`` part follows its rule (glued
    'NSW2750': keep 'NSW', change '2750'; a level code 'B02/53': keep 'B', change '02/53')."""

    pos: Pos
    keep: str = field(repr=False)
    change: str = field(repr=False)


@dataclass(frozen=True, slots=True)
class AddressKey:
    """The canonical identity of an address (FINAL_PLAN section 5, amendment I): the house number, street name,
    canonical street type and suffix, or the box kind and number, or the property's name; the unit and level numbers
    sit beneath it; the suburb and state give the place.  'St' and 'Street', case, commas, line breaks, glued forms
    and OCR variants all give one key.  The key holds name parts, never positions.

    ``street_kind`` is 'ST' (street), 'BOX', 'DX', 'PROP' (a property line with no street) or '' (no street line);
    ``street_parts`` holds the parts for that kind (ST: numbers, names, types, suffix; BOX: the kind, the number;
    DX: the number; PROP: the names).  ``suburb`` is the folded suburb in upper case; ``state`` is None when it is
    missing and cannot be told (never inferred for an overseas unit)."""

    street_kind: str = ""
    street_parts: tuple[tuple[str, ...], ...] = field(default=(), repr=False)
    sub: tuple[tuple[str, str], ...] = field(default=(), repr=False)
    suburb: str = field(default="", repr=False)
    state: str | None = None

    @property
    def has_street(self) -> bool:
        return bool(self.street_kind)

    @property
    def has_place(self) -> bool:
        return bool(self.suburb)

    def street_text(self) -> str:
        """A stable string of the street part, for keyed choices ('' without a street line)."""
        return "|".join([self.street_kind, *(",".join(p) for p in self.street_parts)]) if self.street_kind else ""

    def place_text(self) -> str:
        return f"{self.suburb}|{self.state or ''}" if self.suburb else ""

    def sub_text(self) -> str:
        return ",".join(f"{r}={v}" for r, v in self.sub)

    def same_address(self, other: AddressKey) -> bool:
        """Two keys name one address when every part both hold agrees and they share a street or a place."""
        shared = False
        if self.street_kind and other.street_kind:
            if (self.street_kind, self.street_parts) != (other.street_kind, other.street_parts):
                return False
            shared = True
        if self.suburb and other.suburb:
            if self.suburb != other.suburb or (self.state and other.state and self.state != other.state):
                return False
            shared = True
        if self.sub and other.sub and self.sub != other.sub:
            return False
        return shared


@dataclass(frozen=True, slots=True, eq=False)
class AddressUnit:
    """One address, from its first word to its last.

    ``roles`` maps every word of the unit to its role, in reading order.  ``street`` holds the street-line words
    (everything from the unit's first word to the start of its place) and ``locality`` the place words (suburb,
    state, postcode, country); a place-only unit is all locality and a DX line all street.  Words with no role are
    not in the unit.  ``overseas`` marks a unit whose country is not Australia: its fake keeps the shape, with no
    state and no gazetteer.  ``form`` marks a unit whose lines are all form value boxes.  ``repeat_of`` names the
    earlier unit this one repeats (the repeat sweep found it, or its street part and place agree with that unit's).
    ``how`` lists the routes that built the unit (diagnostics, never text)."""

    uid: str
    kind: UnitKind
    roles: Mapping[Pos, Role]
    street: tuple[Pos, ...]
    locality: tuple[Pos, ...]
    how: tuple[str, ...]
    partial: tuple[PartialWord, ...]
    key: AddressKey
    repeat_of: str | None = None
    overseas: bool = False
    form: bool = False

    @property
    def positions(self) -> tuple[Pos, ...]:
        """Every word of the unit, in reading order."""
        return tuple(self.roles)

    def role_positions(self, *roles: Role) -> tuple[Pos, ...]:
        return tuple(p for p, r in self.roles.items() if r in roles)

    @property
    def place_start(self) -> Pos | None:
        """The first word of the place (the street line ends right before it), or None without a place."""
        return self.locality[0] if self.locality else None


@dataclass(frozen=True, slots=True)
class KeptPlace:
    """A place mentioned on its own, with the positions of its words.  Verification exempts these positions."""

    kind: KeptKind
    positions: tuple[Pos, ...]
    after_cue: bool = False  # right after a place-kind label or phrase


@dataclass(frozen=True, slots=True, eq=False)
class AddressParse:
    """What the grammar found in a document: its units in reading order, the places kept on their own, and notes
    (diagnostic strings holding positions only, never text)."""

    units: tuple[AddressUnit, ...]
    kept: tuple[KeptPlace, ...]
    notes: tuple[str, ...] = ()

    def unit_index(self) -> Mapping[Pos, AddressUnit]:
        """Every word of every unit mapped to its unit."""
        return MappingProxyType({p: u for u in self.units for p in u.roles})

    def kept_positions(self) -> frozenset[Pos]:
        return frozenset(p for k in self.kept for p in k.positions)
