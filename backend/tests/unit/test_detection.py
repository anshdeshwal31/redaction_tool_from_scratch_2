"""Detection on token streams (no OCR): coverage of planted values, precision guards, determinism."""

from __future__ import annotations

import pytest
from synthdocs.generator import _content, medicare_number, provider_number

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
