"""The shared address grammar: one parser for detection, policy, fakes, verification and bench.

Tool 2 replaces every postal or street address as one unit, from its first word (a building or property name,
floor or unit words, a house number, a corner form, a street, or a box kind and number) to its last (suburb, state,
postcode, country), and keeps places mentioned on their own.  This package finds those units.  It is pure: it reads
words with positions and line geometry, plus the packaged locality list (:mod:`mlredact.text.gazetteer`), and it
knows nothing of mentions, tokens, policy or models.  The same lines give the same result.

What it reads (FINAL_PLAN TOOL2 section 2, as amended by A1-A5, B, D6, E and F; the owner's rulings; PENDING_CASES)
    * the place, from the end: 'Australia' or a foreign country, the postcode (OCR letters, a glued ``NSW2750``), the
      LAST state form (never 'Act'; abbreviations only in upper case or dotted; spelled-out states; ``N5W`` through a
      confusion table; a doubled state), the postcode gate (Australia Post ranges, PO-box blocks, the listed
      cross-border pairs) and the suburb (the gazetteer, the postcode's owner, 'St'/'Mt' suburbs, 'Mount Victoria');
    * the street line and its roles: building and property names, designators (all of Australia Post's flat/unit and
      floor/level types, ordinal floors, 'Building C'), house numbers (ranges, letters, ``5/3``, ``B02/53``), corner
      forms, street name, canonical type (AS 4590 / AS/NZS 4819) and suffix, the box kinds (PO Box, GPO Box, Locked Bag,
      Reply Paid, RMB, RSD, CMB, PMB, MS, RMS, CMA, CPA, Care PO, with their case and context rules), DX lines;
    * the street guards (months and weekdays, measure words, 'Dr', the address-cue rule, ``and``/``&``, numberless
      'Name Type' lines, Court/Plaza/Mall/Gardens), the lower-case whole-line pattern, and sentence breaks (a unit never
      spans a ``.``, ``;``, ``!`` or ``?`` that ends a sentence);
    * overseas addresses (the owner's ruling): a street line, the town, a foreign postcode shape and a country are one
      unit; the country word is a structure word; no Australian state is ever inferred (``AddressUnit.overseas``);
    * the join over lines: the block line below and above (by geometry), page breaks, a place wrapped onto the next
      line, the suburb-only tail, place-only lines, form boxes under compound or colon-less labels, the repeat sweep,
      and the places kept on their own.

Public API (everything below is re-exported here)
    ``parse_document(lines, gazetteer=None) -> AddressParse``
        ``lines`` is a sequence of :class:`AddrLine` in reading order, each a tuple of :class:`AddrWord` with a page,
        a box, a band (body, letterhead, header, footer), an optional form role (label, value, cell) and a reading
        frame.  Geometry is relative: any unit works (points or pixels) as long as one document uses one.  A word
        may be ``blocked`` (another detector owns it: a person, organisation or contact mention): suburb, building and
        place runs never cross it.  A street name may still be blocked, because a surname is a street name.
    ``parse_texts(texts, gazetteer=None) -> AddressParse``
        The same over plain lines of text laid out as one column.  ``AddrLine.from_text`` builds one line on a grid.
    ``parse_mention(words, gazetteer=None) -> AddressUnit | None`` and ``place_start(words) -> int | None``
        The unit that covers most of one detected mention's words, and the index of the first word of its place.

    A result, :class:`AddressParse`, holds
        ``units``: :class:`AddressUnit` in reading order, each with
            ``uid`` ('U1', ...), ``kind`` (:class:`UnitKind`: street, street_numberless, corner, box, dx, place_only),
            ``roles``: every word of the unit mapped, in reading order, to its :class:`Role` (the case table's role
            names: building, designator, unit_number, level_number, number, unit_number+number, street_name,
            street_type, street_suffix, corner, box_kind, box_number, box_extra, via, dx, dx_number, dx_exchange,
            suburb, state, postcode, state+postcode, country, capital_city_phrase);
            ``street`` and ``locality``: the street-line words (everything from the unit's first word to the start of
            its place) and the place words, as positions; ``place_start`` is the first of the latter;
            ``partial``: words that are partly structure (``NSW2750``: keep 'NSW', change '2750'; ``B02/53``);
            ``key``: the canonical :class:`AddressKey`; ``repeat_of``: the unit this one repeats; ``overseas``;
            ``form`` (built from form value boxes); ``how``: the routes that built it (diagnostics, never text).
        ``kept``: :class:`KeptPlace` positions of places mentioned on their own (verification exempts them), with the
            kind (place, state, postcode, country, street_in_prose, place_after_cue, repeat_in_prose).
        ``notes``: diagnostic strings (positions only).
    Positions are ``(line index, word index)`` into the lines the caller passed.  No result, repr or exception holds
    document text.

    The roles that stay when an address is replaced are :data:`STRUCTURE_ROLES` (designator, box kind, corner words,
    'via', 'DX', the state, the country, 'IN YOUR CAPITAL CITY'); every other word of a unit is identifying (fail
    closed).  :data:`STREET_ROLES` and :data:`LOCALITY_ROLES` split the roles into the street line and the place.

    Address identity (FINAL_PLAN section 5, amendment I).  ``AddressUnit.key`` is the canonical key: the house number,
    street name, canonical street type and suffix (or the box kind and number, the DX number, or the property's name),
    with the unit and level numbers beneath it, and the suburb and state for the place.  'St' and 'Street', case,
    commas, line breaks, glued forms and OCR variants give one key; ``key.same_address(other)`` compares two (parts
    both hold must agree, and they must share a street or a place).  ``street_text()``, ``sub_text()`` and
    ``place_text()`` are stable strings for keyed choices.  An overseas unit's key has no state.

The gazetteer, name folding and postcode ranges live in :mod:`mlredact.text.gazetteer`.

The package is a chain of stages sharing one state (``state`` -> ``labels`` -> ``places`` -> ``streets`` -> ``join``
-> ``forms`` -> ``sweep`` -> ``parser``); ``tables`` holds the lexicons, ``geometry`` the neighbour queries (windowed by
bisection, so a 30-page document costs time linear in its lines).

Where the line to the policy join falls: everything that needs only words, their positions and line geometry is
here (blocks, place-only lines, form boxes, repeats, kept places).  What the policy adds is the adapter from
tokens and lines to :class:`AddrLine` (with ``blocked`` and ``band`` tags), the retyping and merging of detector
mentions, the profile's action, and the rule for places on their own.
"""

from mlredact.text.address.model import (
    AddressKey,
    AddressParse,
    AddressUnit,
    AddrLine,
    AddrWord,
    Band,
    BoxRole,
    KeptKind,
    KeptPlace,
    PartialWord,
    Pos,
    Role,
    UnitKind,
)
from mlredact.text.address.parser import parse_document, parse_mention, parse_texts, place_start
from mlredact.text.address.tables import (
    LOCALITY_ROLES,
    STREET_ROLES,
    STREET_TYPES,
    STRUCTURE_ROLES,
    TYPE_FORMS,
    canonical_street_type,
    is_abbreviated_type,
)

__all__ = [
    "LOCALITY_ROLES",
    "STREET_ROLES",
    "STREET_TYPES",
    "STRUCTURE_ROLES",
    "TYPE_FORMS",
    "AddrLine",
    "AddrWord",
    "AddressKey",
    "AddressParse",
    "AddressUnit",
    "Band",
    "BoxRole",
    "KeptKind",
    "KeptPlace",
    "PartialWord",
    "Pos",
    "Role",
    "UnitKind",
    "canonical_street_type",
    "is_abbreviated_type",
    "parse_document",
    "parse_mention",
    "parse_texts",
    "place_start",
]
