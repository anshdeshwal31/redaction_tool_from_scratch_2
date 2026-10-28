"""The shared address grammar (``mlredact.text.address``): the case table, the plan's named cases and the shapes the
reference parser still got wrong.  Pure: no models, no OCR."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import pytest

from mlredact.text.address import (
    STRUCTURE_ROLES,
    AddressKey,
    AddressParse,
    AddressUnit,
    AddrLine,
    AddrWord,
    Band,
    BoxRole,
    KeptKind,
    Role,
    UnitKind,
    canonical_street_type,
    is_abbreviated_type,
    parse_document,
    parse_mention,
    parse_texts,
    place_start,
)
from mlredact.text.address.geometry import Geo, x_overlap
from mlredact.text.address.tables import FURNITURE
from mlredact.text.address.words import Line
from support import pages_from_lines

_TABLE = json.loads((Path(__file__).resolve().parents[1] / "data" / "address_cases.json").read_text("utf-8"))
_CASES: list[dict[str, Any]] = _TABLE["cases"]
_BY_ID = {c["id"]: c for c in _CASES}


# ------------------------------------------------------------------------------------------------ helpers
def case_lines(case: dict[str, Any]) -> list[AddrLine]:
    """The case's lines on the grid its ``page``, ``row``, ``x0``, ``x1``, ``band`` and ``box`` tags describe."""
    return [
        AddrLine.from_text(
            ln["text"],
            page=ln.get("page", 0),
            row=ln["row"],
            x0=float(ln.get("x0", 72)),
            x1=float(ln["x1"]) if "x1" in ln else None,
            band=Band(ln.get("band", "body")),
            box=BoxRole(ln["box"]) if ln.get("box") else None,
        )
        for ln in case["lines"]
    ]


def word_at(texts: list[str], pos: tuple[int, int]) -> str:
    return texts[pos[0]].split()[pos[1]]


def unit_text(texts: list[str], unit: AddressUnit) -> str:
    return " ".join(word_at(texts, p) for p in unit.roles)


def role_text(texts: list[str], unit: AddressUnit) -> list[tuple[str, str]]:
    return [(word_at(texts, p), r.value) for p, r in unit.roles.items()]


def parse_lines(*texts: str) -> AddressParse:
    return parse_texts(list(texts))


def match_units(case: dict[str, Any], parse: AddressParse) -> tuple[dict[str, AddressUnit], list[str]]:
    """Each expected unit paired with the unit that has exactly its words, and the errors of the pairing."""
    errs: list[str] = []
    matched: dict[str, AddressUnit] = {}
    for e in case["expect"]["addresses"]:
        want = {tuple(p) for p in e["street_line"]} | {tuple(p) for p in e["locality"]}
        hits = [u for u in parse.units if set(u.roles) == want]
        if len(hits) != 1:
            over = [u for u in parse.units if set(u.roles) & want]
            detail = (
                f"missing {sorted(want - set(over[0].roles))}, extra {sorted(set(over[0].roles) - want)}"
                if over
                else "missed"
            )
            errs.append(f"{e['id']}: {detail}")
            continue
        matched[e["id"]] = hits[0]
    return matched, errs


def check_case(case: dict[str, Any], parse: AddressParse) -> list[str]:
    """README 'What to assert per case', item 1 (units), plus the roles, the structure words, the keys and the kept
    places the table records."""
    expect = case["expect"]
    matched, errs = match_units(case, parse)
    for e in expect["addresses"]:
        u = matched.get(e["id"])
        if u is None:
            continue
        if set(u.street) != {tuple(p) for p in e["street_line"]} or set(u.locality) != {
            tuple(p) for p in e["locality"]
        }:
            errs.append(f"{e['id']}: street/locality split differs")
        for line, word, _text, role in e["words"]:
            if u.roles.get((line, word)) != role:
                errs.append(f"{e['id']}: role at {[line, word]} is {u.roles.get((line, word))}, not {role}")
        if u.kind.value != e["kind"]:
            errs.append(f"{e['id']}: kind {u.kind.value}, not {e['kind']}")
        structure = {tuple(p) for p in e["structure"]}
        if structure != {p for p, r in u.roles.items() if r in STRUCTURE_ROLES}:
            errs.append(f"{e['id']}: structure words differ")
        partial = {(tuple(x["at"]), x["keep"], x["change"]) for x in e.get("partial_structure", [])}
        if partial != {(x.pos, x.keep, x.change) for x in u.partial}:
            errs.append(f"{e['id']}: partial structure differs")
        if u.form != e["form"]:
            errs.append(f"{e['id']}: form {u.form}, not {e['form']}")
        if u.overseas != e.get("overseas", False):
            errs.append(f"{e['id']}: overseas {u.overseas}")
        if u.overseas and u.key.state is not None:
            errs.append(f"{e['id']}: an overseas unit has a state in its key")
        if len({p[0] for p in u.roles}) != e["lines"]:
            errs.append(f"{e['id']}: spans {len({p[0] for p in u.roles})} lines, not {e['lines']}")
        if e.get("repeat_of"):
            src = matched.get(e["repeat_of"])
            if src is None or u.repeat_of != src.uid:
                errs.append(f"{e['id']}: should repeat {e['repeat_of']}")
    expected_positions = {p for u in matched.values() for p in u.roles}
    for u in parse.units:
        if not set(u.roles) <= expected_positions:
            errs.append(f"extra unit {u.kind.value} at {sorted(u.roles)[:3]}")
    # kept places, look-alikes and neighbours stay out of every unit
    for key in ("kept_places", "lookalikes", "not_address"):
        for item in expect[key]:
            if {tuple(p) for p in item["words"]} & {p for u in parse.units for p in u.roles}:
                errs.append(f"{key[:-1]} '{item['kind']}' inside a unit")
    # kept places are recorded, so verification can exempt them
    kept = parse.kept_positions()
    for item in expect["kept_places"]:
        if not {tuple(p) for p in item["words"]} <= kept:
            errs.append(f"kept place '{item['kind']}' not recorded")
    # same_fake groups share a key, other units of the case do not
    groups = [set(g) for g in expect["same_fake"]]
    ids = sorted(matched)
    for a in ids:
        for b in ids:
            if a >= b:
                continue
            same = matched[a].key.same_address(matched[b].key)
            grouped = any(a in g and b in g for g in groups)
            if grouped and not same:
                errs.append(f"same_fake {a}/{b}: keys differ")
            if not grouped and same and case["id"] != "matter-d7-st-and-street-in-one-document":
                errs.append(f"{a}/{b}: different addresses share a key")
    return errs


# ------------------------------------------------------------------------------------------------ the case table
def test_case_table_is_the_expected_version() -> None:
    assert _TABLE["version"] == "0.3.0"
    assert len(_CASES) == _TABLE["case_count"] == 293
    assert sum(c["tier"] == "core" for c in _CASES) == 288
    assert sum(c["tier"] == "stretch" for c in _CASES) == 5


@pytest.mark.parametrize("case", _CASES, ids=[c["id"] for c in _CASES])
def test_case(case: dict[str, Any]) -> None:
    """Every expected unit is exactly one unit with exactly its street-line and locality words and roles, no other
    unit exists, and no unit touches a kept place, a look-alike or a neighbour.  All 288 core cases, the 4 overseas
    stretch cases, and the letterhead drawn as outlines (the grammar reads its text; tool 1 must OCR the outlines)."""
    parse = parse_document(case_lines(case))
    assert check_case(case, parse) == []


def _is_plain(case: dict[str, Any]) -> bool:
    """A single-column, single-page case with no band, box or x tags: ``pages_from_lines`` can lay it out."""
    rows = [ln["row"] for ln in case["lines"]]
    return len(rows) == len(set(rows)) and all(
        ln.get("page", 0) == 0
        and ln.get("band", "body") == "body"
        and "box" not in ln
        and ln.get("x0", 72) == 72
        and "x1" not in ln
        for ln in case["lines"]
    )


_PLAIN = [c for c in _CASES if _is_plain(c)]


def tool2_lines(case: dict[str, Any]) -> list[AddrLine]:
    """The case laid out by the tool 2 test helper (pixel boxes, 25 px a character, 60 px a row), as a caller with
    tool 2's pages would pass it."""
    by_row = {ln["row"]: ln["text"] for ln in case["lines"]}
    texts = [by_row.get(r, "") for r in range(max(by_row) + 1)]
    (page,) = pages_from_lines(texts)
    return [
        AddrLine(
            tuple(AddrWord(t.text, t.bbox.x0, t.bbox.x1) for t in line.tokens),
            page=0,
            x0=line.bbox.x0,
            y0=line.bbox.y0,
            x1=line.bbox.x1,
            y1=line.bbox.y1,
        )
        for line in page.lines
    ]


def test_plain_cases_are_many_enough_to_matter() -> None:
    assert len(_PLAIN) > 150


@pytest.mark.parametrize("case", _PLAIN, ids=[c["id"] for c in _PLAIN])
def test_case_on_tool2_pixel_geometry(case: dict[str, Any]) -> None:
    """Geometry is relative: the same single-column case laid out in tool 2's pixel boxes gives the same units."""
    assert check_case(case, parse_document(tool2_lines(case))) == []


def test_matters_give_one_fake_per_address() -> None:
    """Matter MX: nine spellings of one address share a canonical key, a different address in the same suburb does
    not.  MY: another matter key changes the fake, not the parse."""
    parses = {cid: parse_document(case_lines(_BY_ID[cid])) for m in _TABLE["matters"] for cid in m["cases"]}

    def unit(ref: str) -> AddressUnit:
        cid, uid = ref.split("/")
        matched, errs = match_units(_BY_ID[cid], parses[cid])
        assert errs == []
        return matched[uid]

    for m in _TABLE["matters"]:
        for group in m.get("same_fake", []):
            first = unit(group[0])
            for ref in group[1:]:
                assert first.key.same_address(unit(ref).key), ref
        for a, b in m.get("different_fake", []):
            ua, ub = unit(a), unit(b)
            if a.split("/")[0] in m["cases"] and b.split("/")[0] in m["cases"]:
                assert not ua.key.same_address(ub.key), (a, b)


# ------------------------------------------------------------------------------------------------ the plan's list
NON_ADDRESSES = [
    "Invoice 2024",
    "June 2024",
    "6 Minute Walk Test",
    "30 Second Chair Rise",
    "Level 2 evidence",
    "She saw 2 Court Orders",
    "3 Local Court convictions",
    "12 March Dr Lee reports",
    "RSD 3 months",
    "CRPS (RSD 2)",
    "Dx 1 lumbar strain",
    "Workers Compensation Act 1987",
    "WORKERS COMPENSATION ACT 1987",
    "Civil Liability Act 2002 (NSW)",
    "WorkCover NSW 2006 Guides",
    "SIRA NSW 2021 Guidelines",
]


@pytest.mark.parametrize("text", NON_ADDRESSES)
def test_non_addresses_form_no_unit(text: str) -> None:
    assert parse_lines(text).units == ()


def test_injured_on_a_date_has_no_street_line_and_the_place_stays() -> None:
    """'Injured 12 June, Penrith NSW 2750': '12 June,' is a date, so there is no street line; the place follows an
    injury phrase, so it stays (owner ruling on question 2, option c) and its position is recorded."""
    parse = parse_lines("Injured 12 June, Penrith NSW 2750")
    assert parse.units == ()
    (kept,) = parse.kept
    assert kept.kind is KeptKind.PLACE_AFTER_CUE and kept.after_cue
    assert kept.positions == ((0, 3), (0, 4), (0, 5))


BOUNDED_PLACES = [
    ("Orthopaedic Surgeon Penrith NSW 2750", "Penrith NSW 2750"),
    ("Royal North Shore Hospital St Leonards NSW 2065", "St Leonards NSW 2065"),
    ("Local Court, Penrith NSW 2750", "Penrith NSW 2750"),
    ("Penrith Plaza, Penrith NSW 2750", "Penrith NSW 2750"),
]


@pytest.mark.parametrize(("text", "place"), BOUNDED_PLACES)
def test_bounded_places_stop_at_the_name(text: str, place: str) -> None:
    """The occupation, hospital and court names stay outside the unit; only the place is replaced."""
    (unit,) = parse_lines(text).units
    assert unit.kind is UnitKind.PLACE_ONLY
    assert unit_text([text], unit) == place


SHAPES: list[tuple[str, list[str], list[list[tuple[str, str]]]]] = [
    (
        "mount-victoria-with-postcode",
        ["Mount Victoria NSW 2786"],
        [[("Mount", "suburb"), ("Victoria", "suburb"), ("NSW", "state"), ("2786", "postcode")]],
    ),
    (
        "mount-victoria-without-postcode",
        ["12 Smith Street, Mount Victoria"],
        [
            [
                ("12", "number"),
                ("Smith", "street_name"),
                ("Street,", "street_type"),
                ("Mount", "suburb"),
                ("Victoria", "suburb"),
            ]
        ],
    ),
    (
        "po-box-with-words-before-the-comma",
        ["PO Box 123 City East, Brisbane QLD 4002"],
        [
            [
                ("PO", "box_kind"),
                ("Box", "box_kind"),
                ("123", "box_number"),
                ("City", "box_extra"),
                ("East,", "box_extra"),
                ("Brisbane", "suburb"),
                ("QLD", "state"),
                ("4002", "postcode"),
            ]
        ],
    ),
    (
        "suite-and-building-before-the-number",
        ["Suite 3 Level 2 Riverside Medical Chambers 45 Smith Street, Penrith NSW 2750"],
        [
            [
                ("Suite", "designator"),
                ("3", "unit_number"),
                ("Level", "designator"),
                ("2", "level_number"),
                ("Riverside", "building"),
                ("Medical", "building"),
                ("Chambers", "building"),
                ("45", "number"),
                ("Smith", "street_name"),
                ("Street,", "street_type"),
                ("Penrith", "suburb"),
                ("NSW", "state"),
                ("2750", "postcode"),
            ]
        ],
    ),
    (
        "ordinal-floor",
        ["1st Floor, 45 Smith Street, Penrith NSW 2750"],
        [
            [
                ("1st", "level_number"),
                ("Floor,", "designator"),
                ("45", "number"),
                ("Smith", "street_name"),
                ("Street,", "street_type"),
                ("Penrith", "suburb"),
                ("NSW", "state"),
                ("2750", "postcode"),
            ]
        ],
    ),
    (
        "corner-with-plural-type",
        ["Cnr Smith & Jones Streets, Penrith NSW 2750"],
        [
            [
                ("Cnr", "corner"),
                ("Smith", "street_name"),
                ("&", "corner"),
                ("Jones", "street_name"),
                ("Streets,", "street_type"),
                ("Penrith", "suburb"),
                ("NSW", "state"),
                ("2750", "postcode"),
            ]
        ],
    ),
    (
        "corner-spelled-over-a-place-line",
        ["Corner of Derby and Somerset Streets", "KINGSWOOD NSW 2747"],
        [
            [
                ("Corner", "corner"),
                ("of", "corner"),
                ("Derby", "street_name"),
                ("and", "corner"),
                ("Somerset", "street_name"),
                ("Streets", "street_type"),
                ("KINGSWOOD", "suburb"),
                ("NSW", "state"),
                ("2747", "postcode"),
            ]
        ],
    ),
    (
        "lot-with-a-long-street-name",
        ["Lot 3 Old Bells Line of Road, Kurrajong NSW 2758"],
        [
            [
                ("Lot", "designator"),
                ("3", "unit_number"),
                ("Old", "street_name"),
                ("Bells", "street_name"),
                ("Line", "street_name"),
                ("of", "street_name"),
                ("Road,", "street_type"),
                ("Kurrajong", "suburb"),
                ("NSW", "state"),
                ("2758", "postcode"),
            ]
        ],
    ),
    (
        "dx-line",
        ["DX 1234 Sydney"],
        [[("DX", "dx"), ("1234", "dx_number"), ("Sydney", "dx_exchange")]],
    ),
    (
        "month-as-a-street-name",
        ["3 March Road, Richmond NSW 2753"],
        [
            [
                ("3", "number"),
                ("March", "street_name"),
                ("Road,", "street_type"),
                ("Richmond", "suburb"),
                ("NSW", "state"),
                ("2753", "postcode"),
            ]
        ],
    ),
]


@pytest.mark.parametrize(("texts", "units"), [(s[1], s[2]) for s in SHAPES], ids=[s[0] for s in SHAPES])
def test_shapes(texts: list[str], units: list[list[tuple[str, str]]]) -> None:
    parse = parse_texts(texts)
    assert [role_text(texts, u) for u in parse.units] == units


# ------------------------------------------------------------------------------------------------ pending shapes
def test_pending_numberless_street_after_a_label_or_on_its_own_line_is_an_address() -> None:
    """The direct-follow guard applies to running text only: after an address label or on a line of its own, a
    numberless street followed by its place is replaced whole (PENDING_CASES, first section)."""
    for text, last in (
        ("Address: Smith Street, Nth Ryde 2113", "2113"),
        ("Smith Street, Nth Ryde NSW", "NSW"),
        ("Address: Smith Street, Sth Hurstville NSW 2221", "2221"),
    ):
        (unit,) = parse_lines(text).units
        words = text.split()
        assert unit.kind is UnitKind.STREET_NUMBERLESS
        assert [word_at([text], p) for p in unit.street] == words[words.index("Smith") : words.index("Smith") + 2], text
        assert word_at([text], unit.locality[-1]) == last
        assert unit.roles[(0, words.index("Smith"))] is Role.STREET_NAME
        # 'Nth' and 'Sth' are the suburb's own first word
        assert unit.roles[(0, words.index("Smith") + 2)] is Role.SUBURB


@pytest.mark.parametrize(
    ("text", "street", "place"),
    [
        ("She walked along Smith Street. Penrith NSW 2750 is where she lives.", "Smith Street.", "Penrith NSW 2750"),
        (
            "She was injured on Victoria Road. Ryde NSW 2112 is where she lives.",
            "Victoria Road.",
            "Ryde NSW 2112",
        ),
        ("She lives on Smith Street North; Ryde NSW 2112 is nearby.", "Smith Street North;", "Ryde NSW 2112"),
    ],
)
def test_pending_sentence_break_one_line(text: str, street: str, place: str) -> None:
    """Never join across '.', ';' or a sentence break: the street is kept (owner ruling 3) and the place is a
    place-only unit (ruling 2, option c); no unit holds both."""
    parse = parse_lines(text)
    (unit,) = parse.units
    assert unit.kind is UnitKind.PLACE_ONLY
    assert unit_text([text], unit) == place
    (kept,) = parse.kept
    assert kept.kind is KeptKind.STREET_IN_PROSE
    assert " ".join(word_at([text], p) for p in kept.positions) == street


@pytest.mark.parametrize(
    ("texts", "place"),
    [
        (["She walked along Smith Street.", "Penrith NSW 2750 is where she lives."], "Penrith NSW 2750"),
        (["She was injured on Victoria Road.", "Ryde NSW 2112 is where she lives."], "Ryde NSW 2112"),
        (["She lives on Smith Street North;", "Ryde NSW 2112 is nearby."], "Ryde NSW 2112"),
    ],
)
def test_pending_sentence_break_over_two_lines(texts: list[str], place: str) -> None:
    """The same sentences split over two lines join no better."""
    parse = parse_texts(texts)
    (unit,) = parse.units
    assert unit.kind is UnitKind.PLACE_ONLY and unit_text(texts, unit) == place
    assert [k.kind for k in parse.kept] == [KeptKind.STREET_IN_PROSE]


@pytest.mark.parametrize(
    ("texts", "whole", "kind"),
    [
        (
            ["She was injured at 12 Victoria Road,", "Ryde NSW 2112 in May."],
            "12 Victoria Road, Ryde NSW 2112",
            UnitKind.STREET,
        ),
        (
            ["She was injured on Victoria Road,", "Ryde NSW 2112 in May."],
            "Victoria Road, Ryde NSW 2112",
            UnitKind.STREET_NUMBERLESS,
        ),
        (
            ["She was injured on Victoria Road, Ryde", "NSW 2112 in May."],
            "Victoria Road, Ryde NSW 2112",
            UnitKind.STREET_NUMBERLESS,
        ),
    ],
)
def test_pending_wrapped_running_text_still_joins(texts: list[str], whole: str, kind: UnitKind) -> None:
    """A place that directly follows across a line wrap joins, also when the line below starts at the margin."""
    parse = parse_texts(texts)
    (unit,) = parse.units
    assert unit.kind is kind
    assert unit_text(texts, unit) == whole
    assert parse.kept == ()


def test_pending_numbered_street_before_a_later_place() -> None:
    """'She walked from 12 George Street to Parramatta NSW 2150': the street is one unit, the place a separate
    place-only unit, and 'to' is in neither."""
    text = "She walked from 12 George Street to Parramatta NSW 2150"
    street, place = parse_lines(text).units
    assert unit_text([text], street) == "12 George Street" and street.kind is UnitKind.STREET
    assert unit_text([text], place) == "Parramatta NSW 2150" and place.kind is UnitKind.PLACE_ONLY
    assert (0, 6) not in street.roles and (0, 6) not in place.roles  # 'to'


def test_pending_a_street_type_that_starts_the_suburb() -> None:
    """'She visited Smith Lane Cove NSW 2066': 'Smith' is not a street; 'Lane Cove NSW 2066' is a place-only unit."""
    text = "She visited Smith Lane Cove NSW 2066"
    (unit,) = parse_lines(text).units
    assert unit.kind is UnitKind.PLACE_ONLY
    assert unit_text([text], unit) == "Lane Cove NSW 2066"


def test_pending_numberless_street_keeps_its_type_when_the_suburb_starts_with_one() -> None:
    text = "Smith Road Lane Cove NSW 2066"
    (unit,) = parse_lines(text).units
    assert unit.kind is UnitKind.STREET_NUMBERLESS
    assert [(w, r) for w, r in role_text([text], unit)][:2] == [("Smith", "street_name"), ("Road", "street_type")]


# ------------------------------------------------------------------------------------------------ new grammar
@pytest.mark.parametrize(
    "text",
    [
        "12 Queen Street, Auckland 1010, New Zealand",
        "She now lives at 12 Queen Street, Auckland 1010, New Zealand.",
        "10 High Street, Oxford OX1 4AA, United Kingdom",
        "5 Elm Road, Toronto M5V 2T6, Canada",
    ],
)
def test_overseas_addresses_are_one_unit_with_no_state(text: str) -> None:
    """The owner's ruling: replaced whole, the country word stays, and no Australian state is ever inferred (1010
    lies in NSW's range, and Toronto is no Australian suburb)."""
    (unit,) = parse_lines(text).units
    assert unit.overseas
    assert unit.key.state is None
    assert Role.STATE not in unit.roles.values()
    assert Role.COUNTRY in unit.roles.values()
    assert Role.POSTCODE in unit.roles.values()
    assert unit.kind is UnitKind.STREET
    assert unit.roles[(0, text.split().index("Street," if "Street," in text else "Road,"))] is Role.STREET_TYPE


def test_overseas_block_with_the_country_on_its_own_line() -> None:
    texts = ["Ms Jane Citizen", "12 Queen Street", "AUCKLAND 1010", "NEW ZEALAND"]
    (unit,) = parse_texts(texts).units
    assert unit.overseas
    assert [r for _, r in role_text(texts, unit)] == [
        "number",
        "street_name",
        "street_type",
        "suburb",
        "postcode",
        "country",
        "country",
    ]


def test_an_australian_place_is_not_overseas() -> None:
    (unit,) = parse_lines("12 Queen Street, Auckland 1010").units  # no country: 1010 is in NSW's range, not a place
    assert not unit.overseas


def test_a_foreign_country_alone_is_kept_not_a_unit() -> None:
    parse = parse_lines("She was born in New Zealand and moved here in 2001.")
    assert parse.units == ()
    assert [k.kind for k in parse.kept] == [KeptKind.COUNTRY]


# ------------------------------------------------------------------------------------------------ name folding
@pytest.mark.parametrize(
    "text",
    [
        "12 Smith Street, Mt Druitt NSW 2770",
        "12 Smith Street, MT DRUITT NSW 2770",
        "12 Smith Street, Mount Druitt NSW 2770",
        "12 Smith Street, Nth Ryde NSW 2113",
        "12 Smith Street, North Ryde NSW 2113",
        "12 Smith Street, Ryde North NSW 2113",
    ],
)
def test_name_folding_finds_the_suburb(text: str) -> None:
    (unit,) = parse_lines(text).units
    assert unit.key.state == "NSW"
    assert unit.kind is UnitKind.STREET
    assert len(unit.locality) == len(text.split()) - 3


def test_keys_ignore_abbreviation_case_commas_and_line_breaks() -> None:
    variants = [
        ["12 Smith Street, Penrith NSW 2750"],
        ["12 SMITH ST PENRITH NSW 2750"],
        ["12 Smith St.,", "Penrith NSW 2750"],
        ["12 Smith Street", "PENRITH NSW2750"],
        ["12 smith st, penrith nsw 2750"],
    ]
    keys = []
    for texts in variants:
        (unit,) = parse_texts(texts).units
        keys.append(unit.key)
    assert all(k.same_address(keys[0]) for k in keys)
    assert len({(k.street_text(), k.place_text()) for k in keys}) == 1
    other = parse_lines("14 Smith Street, Penrith NSW 2750").units[0].key
    assert not other.same_address(keys[0])


def test_key_text_holds_no_state_for_an_unreadable_suburb_and_prints_no_names() -> None:
    (unit,) = parse_lines("12 Smith Street, Penrith NSW 2750").units
    assert unit.key.street_text() == "ST|12|SMITH|STREET|"
    assert unit.key.place_text() == "PENRITH|NSW"
    assert "SMITH" not in repr(unit.key) and "PENRITH" not in repr(unit)


def test_a_unit_never_repeats_a_different_address() -> None:
    parse = parse_lines("12 Smith Street, Penrith NSW 2750", "14 Smith Street, Penrith NSW 2750")
    assert [u.repeat_of for u in parse.units] == [None, None]


# ------------------------------------------------------------------------------------------------ geometry
def _line(text: str, **kw: Any) -> AddrLine:
    return AddrLine.from_text(text, **kw)


def test_a_place_line_under_a_street_line_joins_only_in_the_same_column() -> None:
    near = parse_document([_line("12 Smith Street", row=0), _line("PENRITH NSW 2750", row=1)])
    far = parse_document([_line("12 Smith Street", row=0), _line("PENRITH NSW 2750", row=30)])
    beside = parse_document([_line("12 Smith Street", row=0, x0=72), _line("PENRITH NSW 2750", row=1, x0=400)])
    assert [u.kind for u in near.units] == [UnitKind.STREET]
    assert len(near.units[0].roles) == 6
    assert [len(u.roles) for u in far.units] == [3, 3]  # the street alone, and a place-only unit
    assert [len(u.roles) for u in beside.units] == [3, 3]


def test_a_paragraph_gap_up_to_two_line_heights_still_joins_a_strong_place_line() -> None:
    gap = parse_document([_line("12 Smith Street", row=0), _line("PENRITH NSW 2750", row=2)])
    assert [len(u.roles) for u in gap.units] == [6]


def test_lines_on_different_pages_relate_only_across_a_page_break() -> None:
    a = _line("12 Smith Street", page=0, row=40)
    b = _line("Page 1 of 2", page=0, row=48, band=Band.FOOTER)
    c = _line("PENRITH NSW 2750", page=1, row=0)
    parse = parse_document([a, b, c])
    assert [len(u.roles) for u in parse.units] == [6]
    assert parse.units[0].how[-1] == "page_break"


def test_blocked_words_stop_a_suburb_run() -> None:
    """A word another detector owns (a person or organisation mention) is never part of a suburb."""

    def lines(blocked: bool) -> list[AddrLine]:
        words = tuple(
            AddrWord(t, blocked=blocked and t == "Medical") for t in "Penrith Medical Penrith NSW 2750".split()
        )
        return [AddrLine(words, 0, 72, 60, 300, 72)]

    free = parse_document(lines(False))
    held = parse_document(lines(True))
    assert [len(u.roles) for u in free.units] == [5]
    assert [len(u.roles) for u in held.units] == [3]


# ------------------------------------------------------------------------------------------------ the mention API
def test_parse_mention_and_place_start() -> None:
    words = "12 Smith St, Penrith NSW 2750".split()
    unit = parse_mention(words)
    assert unit is not None and unit.kind is UnitKind.STREET
    assert place_start(words) == 3
    assert place_start("Parramatta NSW 2150".split()) == 0
    assert place_start("PO Box 123 Parramatta NSW".split()) == 3
    assert place_start("8 Broadway Ultimo NSW 2007".split()) == 2
    assert place_start("Workers Compensation Act 1987".split()) is None
    assert parse_mention([]) is None


def test_street_type_helpers() -> None:
    assert canonical_street_type("Cres.") == canonical_street_type("Cr") == "CRESCENT"
    assert canonical_street_type("Streets") == "STREET"
    assert canonical_street_type("Penrith") is None
    assert is_abbreviated_type("Rd") and not is_abbreviated_type("Road") and not is_abbreviated_type("Streets")


# ------------------------------------------------------------------------------------------------ robustness
_VOCAB = (
    "12 14-16 5/3 B02/53 3A Unit Suite Level Floor 1st Ground Shop Lot Smith Wattle Street St Road Rd Ave Lane Cove "
    "Penrith Sydney NSW 2750 NSW2750 VIC 3000 QLD ACT Australia New Zealand Auckland 1010 OX1 4AA PO Box 123 GPO RMB "
    "4600 RSD DX 1234 Cnr & and Corner of Mount Victoria St Ives Dr Mr Ms Jane Citizen , | : Address: Suburb: State: "
    "Postcode: Street name Unit/flat no. the in at lives resides born injured declared 2024 March Monday Minute Walk "
    "Test 3 N5W Mt Nth via Station Kilbride IN YOUR CAPITAL CITY Hospital Pty Ltd Surgeon Local Court Plaza . ; ! ? "
    "Street. North North; 2112 Ryde"
).split()


def test_random_documents_give_disjoint_deterministic_units() -> None:
    rng = random.Random(20241024)
    for _ in range(120):
        lines = [
            AddrLine.from_text(
                " ".join(rng.choice(_VOCAB) for _ in range(rng.randint(0, 14))),
                page=rng.choice([0, 0, 1]),
                row=i,
                x0=72.0 + rng.choice([0, 0, 120]),
                band=rng.choice([Band.BODY] * 6 + [Band.FOOTER, Band.LETTERHEAD]),
                box=rng.choice([None, None, None, BoxRole.LABEL, BoxRole.VALUE, BoxRole.CELL]),
            )
            for i in range(rng.randint(1, 10))
        ]
        first = parse_document(lines)
        assert [dict(u.roles) for u in first.units] == [dict(u.roles) for u in parse_document(lines).units]
        seen: set[tuple[int, int]] = set()
        for u in first.units:
            for p in u.roles:
                assert 0 <= p[0] < len(lines) and 0 <= p[1] < len(lines[p[0]].words)
                assert p not in seen
                seen.add(p)
            assert set(u.street) | set(u.locality) == set(u.roles)
            assert list(u.roles) == sorted(u.roles)
        assert all(not set(k.positions) & seen for k in first.kept)


def test_an_empty_document_and_empty_lines() -> None:
    assert parse_document([]).units == ()
    assert parse_document([AddrLine(()), AddrLine(()), _line("12 Smith Street, Penrith NSW 2750", row=3)]).units[0].uid


def test_a_thirty_page_document_parses_in_linear_time() -> None:
    texts = [ln["text"] for c in _CASES for ln in c["lines"]]
    lines = [AddrLine.from_text(texts[(p * 60 + r) % len(texts)], page=p, row=r) for p in range(30) for r in range(60)]
    parse_document(lines[:60])  # warm the gazetteer
    start = time.perf_counter()
    parse = parse_document(lines)
    elapsed = time.perf_counter() - start
    assert parse.units
    assert elapsed < 20.0, f"{elapsed:.1f} s for 30 pages"


def test_results_and_keys_carry_no_document_text() -> None:
    text = "12 Wattle Street, Katoomba NSW 2780"
    parse = parse_lines(text)
    for obj in (parse, parse.units[0], parse.units[0].key, _line(text), AddressKey("ST", (("12",),))):
        shown = repr(obj)
        assert "Wattle" not in shown and "Katoomba" not in shown


# ------------------------------------------------------------------------------------------------ neighbour queries
def _naive_below(lines: list[Line], ln: Line, span: tuple[float, float], max_gap: float, skip: bool) -> Line | None:
    """The reference scan over every line of the document."""
    best = None
    for o in lines:
        if o.page != ln.page or o.frame != ln.frame or o.li == ln.li or not o.words:
            continue
        if skip and (
            FURNITURE.fullmatch(o.text.strip()) or (ln.band is Band.BODY and o.band in (Band.HEADER, Band.FOOTER))
        ):
            continue
        if o.y0 < ln.y1 - 0.3 * ln.h or o.y0 - ln.y1 > max_gap * min(ln.h, o.h):
            continue
        if x_overlap(span, (o.x0, o.x1)) < 0.3:
            continue
        if best is None or (o.y0, o.x0) < (best.y0, best.x0):
            best = o
    return best


def _naive_above(lines: list[Line], ln: Line) -> Line | None:
    best = None
    for o in lines:
        if o.page != ln.page or o.frame != ln.frame or o.li == ln.li or not o.words:
            continue
        if o.y1 > ln.y0 + 0.3 * ln.h or ln.y0 - o.y1 > 0.8 * min(ln.h, o.h):
            continue
        if x_overlap((ln.x0, ln.x1), (o.x0, o.x1)) < 0.3:
            continue
        if best is None or o.y1 > best.y1:
            best = o
    return best


def test_windowed_neighbour_queries_equal_a_full_scan() -> None:
    rng = random.Random(7)
    for _ in range(40):
        src = []
        for i in range(rng.randint(5, 40)):
            y0 = rng.choice([0, 3, 14, 15, 28, 29, 43, 60, 120]) + rng.randint(0, 2) * 14.4 + 60.0 * (i // 12)
            height = rng.choice([8.0, 12.0, 12.0, 30.0])
            x0 = rng.choice([72.0, 72.0, 90.0, 300.0])
            words = tuple(AddrWord("w") for _ in range(rng.randint(0, 3)))
            src.append(
                AddrLine(
                    words,
                    page=i // 20,
                    x0=x0,
                    y0=y0,
                    x1=x0 + rng.choice([40.0, 120.0]),
                    y1=y0 + height,
                    band=rng.choice([Band.BODY, Band.BODY, Band.FOOTER]),
                    frame=rng.choice([0, 0, 270]),
                )
            )
        lines = [Line(s, i) for i, s in enumerate(src)]
        geo = Geo(lines)
        for ln in lines:
            if not ln.words:
                continue
            span = (ln.x0, ln.x1)
            for gap, skip in ((0.8, False), (0.8, True), (1.2, False), (2.0, True)):
                assert geo.below(ln, span, gap, skip) is _naive_below(lines, ln, span, gap, skip)
            assert geo.above(ln) is _naive_above(lines, ln)
            mates = [o for o in lines if o.words and o.li != ln.li and geo.same_row(o, ln)]
            assert geo.row_mates(ln) == mates


def test_first_line_of_the_next_page_equals_a_full_scan() -> None:
    rng = random.Random(11)
    for _ in range(40):
        src = []
        for i in range(rng.randint(4, 30)):
            y0 = rng.choice([40.0, 60.0, 400.0, 760.0, 790.0]) + rng.randint(0, 2) * 14.4
            x0 = rng.choice([72.0, 72.0, 300.0])
            text = rng.choice(["12 Smith Street", "Page 2 of 3", "7", "body text here", ""])
            src.append(
                AddrLine(
                    tuple(AddrWord(t) for t in text.split()),
                    page=i // 8,
                    x0=x0,
                    y0=y0,
                    x1=x0 + 90.0,
                    y1=y0 + 12.0,
                    band=rng.choice([Band.BODY, Band.BODY, Band.FOOTER, Band.HEADER]),
                )
            )
        lines = [Line(s, i) for i, s in enumerate(src)]
        geo = Geo(lines)
        for ln in lines:
            if not ln.words:
                continue
            same = [o for o in lines if o.page == ln.page and o.frame == ln.frame and not o.furniture and o.words]
            expected = None
            if not any(o.y0 > ln.y1 - 0.3 * ln.h and o.li != ln.li for o in same):
                nxt = [
                    o for o in lines if o.page == ln.page + 1 and o.frame == ln.frame and not o.furniture and o.words
                ]
                if nxt:
                    first = min(nxt, key=lambda o: (o.y0, o.x0))
                    if x_overlap((ln.x0, ln.x1), (first.x0, first.x1)) >= 0.3:
                        expected = first
            assert geo.next_page_first(ln) is expected


def test_empty_and_odd_words_do_not_break_the_parse() -> None:
    texts = [
        "",
        "12",
        "",
        "Smith",
        "Street,",
        "",
        "Penrith",
        "NSW",
        "2750",
        "",
        "\u00a0",
        "…",
        "\U0001f600",
        "A" * 5000,
    ]
    line = AddrLine(tuple(AddrWord(t, i * 10.0, i * 10.0 + 9) for i, t in enumerate(texts)), 0, 0, 0, 200, 12)
    parse = parse_document([line, line])
    assert parse.units
    assert all(p[1] < len(texts) for u in parse.units for p in u.roles)
