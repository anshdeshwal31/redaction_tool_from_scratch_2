"""Detection on token streams (no OCR): coverage of planted values, precision guards, determinism."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from synthdocs.generator import _content, medicare_number, provider_number
from test_address_join import Doc, case_doc

from mlredact.config.loader import load_config
from mlredact.core.types import EntityType
from mlredact.detect.engine import Seed, detect_document
from mlredact.detect.rules.checksums import (
    abn_valid,
    acn_valid,
    ihi_valid,
    luhn_valid,
    medicare_valid,
    provider_number_valid,
    tfn_valid,
)
from mlredact.security.sensitive import Sensitive
from mlredact.text.normalize import match_key, ocr_digit_normalise, titlecase_allcaps
from support import pages_from_lines

CFG = load_config("broad")


def _covered_keys(lines: list[str], **kw: object) -> tuple[set[str], list[tuple[EntityType, str]]]:
    pages = pages_from_lines(lines, **kw)  # type: ignore[arg-type]
    result = detect_document(pages, CFG, "doc")
    acted = [
        (m.entity_type, m.value) for m in result.mentions if CFG.policy.entity_actions[m.entity_type].value != "keep"
    ]
    return {match_key(v) for _t, v in acted}, acted


def _is_covered(value: str, covered: set[str]) -> bool:
    key = match_key(value)
    return any(key in c for c in covered)


class TestChecksums:
    def test_generators_produce_valid_numbers(self) -> None:
        assert medicare_valid(medicare_number("29537015").replace(" ", ""))
        assert provider_number_valid(provider_number("212345", "A"))

    def test_known_values(self) -> None:
        assert abn_valid("51824753556")  # ATO's published example ABN
        assert not abn_valid("51824753557")
        assert acn_valid("000000019")
        assert tfn_valid("123456782")  # canonical valid test TFN
        assert luhn_valid("4111111111111111")
        assert ihi_valid("8003608166690503")  # Luhn-valid test IHI shape
        assert not medicare_valid("2953701511")

    def test_medicare_requires_nonzero_issue(self) -> None:
        assert not medicare_valid("2953701500")


class TestNormalisation:
    def test_ocr_digit_normalise_only_digit_words(self) -> None:
        assert ocr_digit_normalise("Medicare 2953 7O15O 1 Olive") == "Medicare 2953 70150 1 Olive"

    def test_titlecase_is_length_preserving(self) -> None:
        s = "Re: Mr John SMITH-JONES"
        assert titlecase_allcaps(s) == "Re: Mr John Smith-jones" and len(titlecase_allcaps(s)) == len(s)


class TestLetterCoverage:
    def test_all_phase1_planted_values_are_covered(self) -> None:
        lines, planted = _content()
        covered, _ = _covered_keys(lines)
        missing = [p.kind for p in planted if p.phase <= 1 and not _is_covered(p.text, covered)]
        assert missing == []

    def test_clinical_content_is_not_redacted(self) -> None:
        lines, _ = _content()
        covered, _ = _covered_keys(lines)
        for keep in (
            "Tinel's sign",
            "Phalen's test",
            "Panadeine Forte",
            "low back pain",
            "2 May 2024",
            "3 February 2023",
            "Sydney",
        ):
            assert not _is_covered(keep, covered), keep

    def test_detection_is_deterministic(self) -> None:
        lines, _ = _content()
        a = detect_document(pages_from_lines(lines), CFG, "doc")
        b = detect_document(pages_from_lines(lines), CFG, "doc")
        assert [(m.mention_id, m.entity_type, m.token_ids) for m in a.mentions] == [
            (m.mention_id, m.entity_type, m.token_ids) for m in b.mentions
        ]


@pytest.mark.parametrize(
    ("line", "value", "etype"),
    [
        ("Patient Name: SMITH, John Paul", "SMITH, John Paul", EntityType.PERSON),
        ("Surname: O'Brien  Given names: Mary Anne", "O'Brien", EntityType.PERSON),
        ("Next of Kin: Mrs Helen McDonald (wife)", "Helen McDonald", EntityType.PERSON),
        ("UR No: 00123456", "00123456", EntityType.CLINICAL_ID),
        ("MRN: A1234567", "A1234567", EntityType.CLINICAL_ID),
        ("D.O.B. 3/7/62", "3/7/62", EntityType.DATE_OF_BIRTH),
        ("Born on 14 March 1978 in Dubbo", "14 March 1978", EntityType.DATE_OF_BIRTH),
        ("Ph: 02 9999 1234", "02 9999 1234", EntityType.PHONE),
        ("Mob 0412 555 666", "0412 555 666", EntityType.PHONE),
        ("Contact +61 412 555 666 after hours", "+61 412 555 666", EntityType.PHONE),
        ("IHI: 8003 6081 6669 0503", "8003 6081 6669 0503", EntityType.IHI),
        ("ABN 51 824 753 556", "51 824 753 556", EntityType.ABN),
        ("TFN: 123 456 782", "123 456 782", EntityType.TFN),
        ("Unit 3/45 Smith Street, Blacktown NSW 2148", "3/45 Smith Street", EntityType.STREET_ADDRESS),
        ("PO Box 77, Wagga Wagga NSW 2650", "PO Box 77", EntityType.STREET_ADDRESS),
        ("lives at 9 Ocean Pde, Coffs Harbour NSW 2450", "Coffs Harbour", EntityType.LOCALITY),
        ("He was employed by Bunnings Warehouse in Penrith", "Bunnings Warehouse", EntityType.ORGANISATION),
        ("Our Ref: JS:kl:20231187", "JS:kl:20231187", EntityType.REFERENCE),
        ("Registration No: MED0001234567", "MED0001234567", EntityType.AHPRA),
        ("Specimen No 24-117733", "24-117733", EntityType.CLINICAL_ID),
        ("Email: j.citizen@bigpond.com.au", "j.citizen@bigpond.com.au", EntityType.EMAIL),
        ("Dear Mrs Nguyen,", "Nguyen", EntityType.PERSON),
        ("cc: Dr A. Wong, Orthopaedic Surgeon", "A. Wong", EntityType.PERSON),
    ],
)
def test_single_line_rules(line: str, value: str, etype: EntityType) -> None:
    covered, acted = _covered_keys([line])
    assert _is_covered(value, covered), (line, [t for t, _ in acted])


@pytest.mark.parametrize(
    "line",
    [
        "BP 120/80, HR 72, temperature 36.8 degrees.",
        "Range of motion was reduced to 50% of normal at L4-5 and C5/6.",
        "Ibuprofen 400mg three times daily; 12500IU vitamin D weekly.",
        "He was reviewed in 2019-2021 and again in March 2022.",
        "Tinel's sign and Phalen's test were negative; McMurray test positive.",
        "The claimant reports 7/10 pain.",
    ],
)
def test_clinical_lines_produce_no_redactions(line: str) -> None:
    _covered, acted = _covered_keys([line])
    assert acted == [], acted


def test_propagation_finds_unlabelled_and_ocr_variant_mentions() -> None:
    lines = [
        "Claim No: WC1234567",
        "Further to claim WC1234567 we enclose the report.",
        "Ref WC 1234567 was reopened.",
        "The file wc-1234567 is closed.",
    ]
    _covered, acted = _covered_keys(lines)
    refs = [v for t, v in acted if t is EntityType.REFERENCE]
    assert len(refs) == 4, refs


def test_propagation_uses_ocr_alternates() -> None:
    lines = ["Medicare No: 2953 70150 1", "card 2953 7O15O 1 presented"]
    _covered, acted = _covered_keys(lines, alternates={"7O15O": ("70150",)})
    assert sum(1 for t, _ in acted if t is EntityType.MEDICARE) == 2


def test_seeds_find_unlabelled_names() -> None:
    pages = pages_from_lines(["Zelphinia attended with her sister.", "zelphinia was anxious."])
    result = detect_document(pages, CFG, "doc", seeds=[Seed(EntityType.PERSON, Sensitive("Zelphinia"))])
    assert sum(1 for m in result.mentions if m.entity_type is EntityType.PERSON) == 2


def test_uncertain_neighbours_join_spans() -> None:
    covered, _acted = _covered_keys(["Re: Mr John Sm?th DOB 1/2/1980"], confidences={"Sm?th": 0.3})
    assert any("sm?th".replace("?", "") in c or "smth" in c for c in covered), covered


@pytest.mark.parametrize(
    ("line", "date", "etype"),
    [
        ("He was born on 14/03/1978.", "14/03/1978", EntityType.DATE_OF_BIRTH),
        ("DOB 3.7.62. Seen today.", "3.7.62", EntityType.DATE_OF_BIRTH),
        ("He was injured on 14/03/2023. Pain persists.", "14/03/2023", EntityType.DATE),
    ],
)
def test_dates_at_the_end_of_a_sentence_are_found(line: str, date: str, etype: EntityType) -> None:
    pages = pages_from_lines([line])
    tokens = {t.token_id: t for p in pages for t in p.tokens()}
    result = detect_document(pages, load_config("broad"), "doc")
    found = {(m.entity_type, " ".join(tokens[t].text for t in m.token_ids).rstrip(".")) for m in result.mentions}
    assert (etype, date) in found


def test_version_like_numbers_are_not_dates() -> None:
    pages = pages_from_lines(["Software build 2.14.03.2023 installed"])
    mentions = detect_document(pages, load_config("broad"), "doc").mentions
    assert not [m for m in mentions if m.entity_type is EntityType.DATE]


class TestFormGeometry:
    @staticmethod
    def _found(pages):  # type: ignore[no-untyped-def]
        tokens = {t.token_id: t for p in pages for t in p.tokens()}
        result = detect_document(pages, load_config("broad"), "doc")
        return {(m.entity_type, " ".join(tokens[t].text for t in m.token_ids)) for m in result.mentions}

    def test_values_on_the_line_below_a_label(self) -> None:
        found = self._found(
            pages_from_lines(
                ["Patient name:", "John Andrew SMITH", "Date of birth:", "14/03/1978", "Claim No:", "WC1234567"]
            )
        )
        assert (EntityType.PERSON, "John Andrew SMITH") in found
        assert (EntityType.DATE_OF_BIRTH, "14/03/1978") in found
        assert (EntityType.REFERENCE, "WC1234567") in found

    def test_values_in_a_separate_cell_on_the_same_row(self) -> None:
        from dataclasses import replace

        from mlredact.core.geometry import BBox
        from mlredact.core.model import Line, Token, token_id

        page = pages_from_lines(["Surname:", "Medicare No:"])[0]
        cells = []
        for k, (label_line, value) in enumerate(zip(page.lines, ("CITIZEN", "2953 70150 1"), strict=True)):
            y0, y1 = label_line.bbox.y0, label_line.bbox.y1
            words, x, toks = value.split(), 900, []
            for w_no, word in enumerate(words):
                box = BBox(x, y0, x + 25 * len(word), y1)
                toks.append(Token(token_id(0, 10 + k, w_no), 0, 10 + k, w_no, box, 0.99, 0.99, "test", word))
                x = box.x1 + 20
            bb = BBox(900, y0, x, y1)
            quad = ((bb.x0, bb.y0), (bb.x1, bb.y0), (bb.x1, bb.y1), (bb.x0, bb.y1))
            cells.append(Line(f"p0000.l{10 + k:04d}", 0, 10 + k, bb, quad, 0.99, "test", tuple(toks)))
        found = self._found([replace(page, lines=(*page.lines, *cells))])
        assert (EntityType.PERSON, "CITIZEN") in found
        assert (EntityType.MEDICARE, "2953 70150 1") in found

    def test_prose_does_not_pull_a_value_from_the_next_line(self) -> None:
        found = self._found(
            pages_from_lines(["He attended the appointment today with his partner", "and they left at noon"])
        )
        assert not any(etype is EntityType.PERSON and "they" in text for etype, text in found)


# ------------------------------------------------------------------------------------------------ addresses
# The rule detectors read addresses with the shared grammar (text/address): what is an address, where it starts and
# where it ends.  These tests run the real detectors (no models) on token streams, in every profile.
_PROFILES = ("client", "broad", "claimant_focused", "maximal")
_ADDRESS_TYPES = frozenset({EntityType.STREET_ADDRESS, EntityType.LOCALITY, EntityType.POSTCODE})
_CONFIGS = {name: load_config(name) for name in _PROFILES}
_TABLE = json.loads((Path(__file__).resolve().parents[1] / "data" / "address_cases.json").read_text("utf-8"))
_CASES: list[dict[str, Any]] = _TABLE["cases"]
# The policy rules that keep a word as it is: an address mention over a word they own would damage it.
_KEEPING_RULES = frozenset(
    name for name, rule in _TABLE["own_rules"].items() if rule["requirement"] in ("MUST_NOT", "MAY")
)


def _mentions(
    lines: list[str], profile: str = "broad", types: frozenset[EntityType] | None = None
) -> list[tuple[EntityType, str]]:
    pages = pages_from_lines(lines)
    tokens = {t.token_id: t for p in pages for t in p.tokens()}
    result = detect_document(pages, _CONFIGS[profile], "doc")
    return [
        (m.entity_type, " ".join(tokens[t].text for t in m.token_ids))
        for m in result.mentions
        if types is None or m.entity_type in types
    ]


def _address_mentions(lines: list[str], profile: str = "broad") -> list[tuple[EntityType, str]]:
    return _mentions(lines, profile, _ADDRESS_TYPES)


def _covered_words(lines: list[str], profile: str = "broad") -> set[str]:
    """The words an address, locality or postcode mention covers (punctuation stripped)."""
    return {w.strip(",.;:()") for _t, text in _address_mentions(lines, profile) for w in text.split()}


S, L, P = EntityType.STREET_ADDRESS, EntityType.LOCALITY, EntityType.POSTCODE


class TestAddressRules:
    @pytest.mark.parametrize("profile", _PROFILES)
    @pytest.mark.parametrize(
        "line",
        [
            "Invoice 2024",
            "June 2024",
            "The 6 Minute Walk Test was 410 m.",
            "She completed the 30 Second Chair Rise test.",
            "The 10 Metre Walk Test was timed.",
            "Level 2 evidence was reviewed.",
            "She saw 2 Court Orders.",
            "3 Local Court convictions were recorded.",
            "12 March Dr Lee reports a recovery.",
            "He developed RSD 3 months after the injury.",
            "Diagnosis: CRPS (RSD 2) of the right hand.",
            "Dx 1 lumbar strain",
            "diagnosed with MS 3 years ago",
            "MS JANE CITIZEN",
            "The RMS 2019 crash report",
            "Workers Compensation Act 1987",
            "WORKERS COMPENSATION ACT 1987",
            "Civil Liability Act 2002 (NSW)",
            "WorkCover NSW 2006 Guides",
            "SIRA NSW 2021 Guidelines",
            "WORKCOVER NSW 2006",
            "Personal Injury Commission NSW 2021",
            "She walked along Smith Street.",
            "injured on Victoria Road, Ryde",
        ],
    )
    def test_look_alikes_are_no_address_part(self, line: str, profile: str) -> None:
        assert _address_mentions([line], profile) == []

    @pytest.mark.parametrize("profile", _PROFILES)
    def test_a_street_in_prose_without_a_house_number_is_no_candidate_for_any_rule(self, profile: str) -> None:
        for line in ("She walked along Smith Street.", "injured on Victoria Road, Ryde", "He lives on Elm Court."):
            assert _address_mentions([line], profile) == [], line

    @pytest.mark.parametrize(
        ("line", "expected"),
        [
            # an untyped number-led street directly before an anchored place
            ("Visit 8 Broadway, Ultimo NSW 2007", [(S, "8 Broadway,"), (L, "Ultimo"), (P, "2007")]),
            # the locality cut: 'St', 'Castle', 'Victoria' open or sit inside a suburb, not the street
            ("She lives in St Ives NSW 2075", [(L, "St Ives"), (P, "2075")]),
            ("Seen at 5 Smith Street Castle Hill NSW 2154", [(S, "5 Smith Street"), (L, "Castle Hill"), (P, "2154")]),
            ("12 Queensland Avenue, Victoria Point QLD 4165", [(S, "12 Queensland Avenue,"), (L, "Victoria Point")]),
            # a glued state and postcode
            ("Penrith NSW2750", [(L, "Penrith"), (P, "NSW2750")]),
            # the street types that guards keep out of clinical phrases still make addresses
            ("12 Banksia Court", [(S, "12 Banksia Court")]),
            ("4 Heron Walk, Penrith NSW 2750", [(S, "4 Heron Walk,"), (L, "Penrith"), (P, "2750")]),
            ("9 Wattle Rise, Penrith NSW 2750", [(S, "9 Wattle Rise,"), (L, "Penrith"), (P, "2750")]),
            ("3 March Road, Richmond NSW 2753", [(S, "3 March Road,"), (L, "Richmond"), (P, "2753")]),
            # a street with its locality in prose is an address, with a house number or not
            ("injured on Victoria Road, Ryde NSW 2112", [(S, "Victoria Road,"), (L, "Ryde"), (P, "2112")]),
            # property lines and box kinds
            ("Lot 3 Old Bells Line of Road, Kurrajong NSW 2758", [(S, "Lot 3 Old Bells Line of Road,")]),
            ("PO Box 123 City East, Brisbane QLD 4002", [(S, "PO Box 123 City East,"), (P, "4002")]),
            ("RMB 4600 Lakes Road, EUROA VIC 3664", [(S, "RMB 4600 Lakes Road,"), (L, "EUROA"), (P, "3664")]),
            ("DX 1234 Sydney", [(S, "DX 1234 Sydney")]),
            ("MS 55, Bathurst NSW 2795", [(S, "MS 55,"), (L, "Bathurst"), (P, "2795")]),
        ],
    )
    @pytest.mark.parametrize("profile", _PROFILES)
    def test_shapes(self, line: str, expected: list[tuple[EntityType, str]], profile: str) -> None:
        found = _address_mentions([line], profile)
        for item in expected:
            assert item in found, (line, found)

    @pytest.mark.parametrize("profile", _PROFILES)
    def test_a_capital_city_is_no_locality_candidate_but_its_postcode_is(self, profile: str) -> None:
        found = _address_mentions(["Level 4, 123 George Street, Sydney NSW 2000"], profile)
        assert (S, "Level 4, 123 George Street,") in found and (P, "2000") in found
        assert all(t is not L for t, _v in found)

    @pytest.mark.parametrize("profile", _PROFILES)
    @pytest.mark.parametrize(
        ("line", "kept"),
        [
            ("Orthopaedic Surgeon Penrith NSW 2750", ("Orthopaedic", "Surgeon")),
            ("Royal North Shore Hospital St Leonards NSW 2065", ("Royal", "North", "Shore", "Hospital")),
            ("Local Court, Penrith NSW 2750", ("Local", "Court")),
            ("Consultant Psychiatrist Penrith NSW 2750", ("Consultant", "Psychiatrist")),
            ("Penrith Plaza, Penrith NSW 2750", ("Plaza",)),
        ],
    )
    def test_the_suburb_bound_stops_at_occupation_and_organisation_words(
        self, line: str, kept: tuple[str, ...], profile: str
    ) -> None:
        assert not set(kept) & _covered_words([line], profile)
        assert (P, line.split()[-1]) in _address_mentions([line], profile)

    @pytest.mark.parametrize("profile", _PROFILES)
    def test_a_place_after_a_birth_or_injury_phrase_is_no_address(self, profile: str) -> None:
        for line in ("Place of birth: Penrith NSW 2750", "Injured 12 June, Penrith NSW 2750"):
            assert S not in {t for t, _v in _address_mentions([line], profile)}, line

    @pytest.mark.parametrize("profile", _PROFILES)
    def test_a_surname_that_is_a_street_name_does_not_turn_the_address_into_a_person(self, profile: str) -> None:
        found = _mentions(
            ["Ms Jane Smith", "Address: 12 Smith Street, Penrith NSW 2750", "Jane Smith attended."], profile
        )
        assert (S, "12 Smith Street, Penrith NSW 2750") in found
        assert [v for t, v in found if t is EntityType.PERSON] == ["Jane Smith", "Jane Smith"]

    @pytest.mark.parametrize("profile", _PROFILES)
    def test_dr_after_a_street_name_is_the_street_type_not_a_title(self, profile: str) -> None:
        found = _mentions(["45 Smith Dr Penrith NSW 2750"], profile)
        assert (S, "45 Smith Dr") in found and (L, "Penrith") in found
        assert all(t is not EntityType.PERSON for t, _v in found)

    def test_detection_of_addresses_is_deterministic(self) -> None:
        lines = [
            "Address: 5 Smith Street, Penrith NSW 2750 T (02) 5550 6610",
            "Suburb: Penrith State: NSW",
            "8 Broadway,",
        ]
        assert _mentions(lines, "broad") == _mentions(lines, "broad")


class TestAddressLabels:
    @pytest.mark.parametrize("profile", _PROFILES)
    def test_an_address_value_ends_where_its_place_ends(self, profile: str) -> None:
        found = _mentions(["Address: 5 Smith Street, Penrith NSW 2750 T (02) 5550 6610"], profile)
        assert found == [(S, "5 Smith Street, Penrith NSW 2750"), (EntityType.PHONE, "(02) 5550 6610")]

    @pytest.mark.parametrize("profile", _PROFILES)
    def test_an_address_value_with_no_place_ends_before_a_phone_number(self, profile: str) -> None:
        found = _mentions(["Address: Unit 4 Rose Gardens Ph (02) 5550 6610"], profile)
        assert found[-1] == (EntityType.PHONE, "(02) 5550 6610")
        assert all("5550" not in v for t, v in found if t is S)

    @pytest.mark.parametrize("profile", _PROFILES)
    def test_the_boxes_of_a_form_are_labels_never_values(self, profile: str) -> None:
        lines = [
            "Address:",
            "Unit",
            "5",
            "Street No.",
            "12",
            "Street name",
            "Smith Street",
            "Suburb/Town",
            "Penrith",
            "State",
            "NSW",
            "Postcode",
            "2750",
        ]
        covered = _covered_words(lines, profile)
        assert not {"Address", "Unit", "No", "name", "Suburb/Town", "State", "Postcode", "NSW"} & covered
        assert {"5", "12", "Smith", "Penrith", "2750"} <= covered

    @pytest.mark.parametrize("profile", _PROFILES)
    def test_a_suburb_label_ends_at_the_next_label(self, profile: str) -> None:
        found = _address_mentions(["Suburb: Penrith State: NSW"], profile)
        assert found == [(L, "Penrith")]

    @pytest.mark.parametrize("profile", _PROFILES)
    def test_a_state_value_is_no_candidate_and_never_a_locality(self, profile: str) -> None:
        assert _mentions(["State: NSW"], profile) == []
        assert _mentions(["Suburb/Town: Penrith", "State/Territory: NSW", "Postcode: 2750"], profile) == [
            (L, "Penrith"),
            (P, "2750"),
        ]

    @pytest.mark.parametrize("profile", _PROFILES)
    def test_a_postcode_label_takes_only_a_postcode(self, profile: str) -> None:
        assert _mentions(["Residents of postcode 2750 were surveyed."], profile) == [(P, "2750")]
        # a text layer without spaces glues the next word on: the glued token is not a postcode
        assert _mentions(["Residents of postcode 2070weresurveyed."], profile) == []
        assert _mentions(["Postcode: 275O"], profile) == [(P, "275O")]


class TestPropagation:
    def test_a_repeated_address_is_found_once_not_shifted_over_its_neighbours(self) -> None:
        lines = [
            "Address: 5 Smith Street, Penrith NSW 2750 T (02) 5550 6610",
            "Please post to 5 Smith Street, Penrith NSW 2750 today.",
        ]
        found = [(t, v) for t, v in _mentions(lines, "broad") if t is S]
        assert found == [(S, "5 Smith Street, Penrith NSW 2750"), (S, "5 Smith Street, Penrith NSW 2750")]


def _case_pages(case: dict[str, Any]) -> tuple[Doc, list[Any]]:
    doc, _bands, _boxes = case_doc(case)
    return doc, doc.pages


# The expected-unit words the rule detectors cannot cover on their own, with the reason.
_NEEDS_THE_JOIN = {
    "form-unreadable-labels": "value boxes read by shape: the join gets the box tags",
    "footer-repeat-without-state-or-postcode": "a repeat found by the join's sweep",
}
_CAPITALS = {"sydney", "melbourne", "brisbane", "perth", "adelaide", "hobart", "darwin", "canberra"}


@pytest.mark.parametrize("profile", _PROFILES)
@pytest.mark.parametrize("case", _CASES, ids=[c["id"] for c in _CASES])
def test_case_table_through_the_rule_detectors(case: dict[str, Any], profile: str) -> None:
    doc, pages = _case_pages(case)
    result = detect_document(pages, _CONFIGS[profile], "doc")
    pos_of = {t.token_id: (li, wi) for li, line in enumerate(doc.lines) for wi, t in enumerate(line.tokens)}
    types_at: dict[tuple[int, int], set[EntityType]] = {}
    for m in result.mentions:
        if m.entity_type in _ADDRESS_TYPES:
            for t in m.token_ids:
                types_at.setdefault(pos_of[t], set()).add(m.entity_type)
    expect = case["expect"]
    # look-alikes and neighbours that keep their words (clinical phrases, citations, dates, occupations, labels):
    # no address, locality or postcode mention over them
    for key in ("lookalikes", "not_address"):
        for item in expect[key]:
            if item["follows"] in _KEEPING_RULES:
                hit = [p for p in map(tuple, item["words"]) if p in types_at]
                assert not hit, (key, item["kind"], item["text"])
    # a place kept on its own: never a street address
    for item in expect["kept_places"]:
        assert not any(S in types_at.get(tuple(p), ()) for p in item["words"]), item["kind"]
    # the client keeps places mentioned on their own: the rules make no address, locality or postcode mention of a
    # street in prose, a bare suburb, a state or a country
    if profile == "client":
        for item in expect["kept_places"]:
            if item["kind"] in ("street_in_prose", "suburb", "city", "state", "country"):
                assert not any(tuple(p) in types_at for p in item["words"]), item["kind"]
    # every identifying word of an expected unit is covered by an address-type mention, except the capital cities (a
    # place on its own is the profile's rule), overseas towns and what only the join can see
    structure = {"state", "country", "capital_city_phrase", "designator", "corner", "box_kind", "via", "dx"}
    for unit in expect["addresses"]:
        if case["id"] in _NEEDS_THE_JOIN or unit.get("overseas"):
            continue
        for li, wi, text, role in unit["words"]:
            if role in structure or (role == "suburb" and text.strip(",").lower() in _CAPITALS):
                continue
            assert (li, wi) in types_at, (unit["id"], text, role)


def test_phone_starts_finds_the_numbers_that_stand_on_their_own() -> None:
    from mlredact.detect.rules.contact import phone_starts

    text = "5 Smith Street NSW 2750 T (02) 5550 6610 or 0412 555 666, ref 2750 1234"
    assert phone_starts(text, 0, len(text)) == [text.index("(02)"), text.index("0412")]
    assert phone_starts(text, 0, text.index("(02)")) == []
