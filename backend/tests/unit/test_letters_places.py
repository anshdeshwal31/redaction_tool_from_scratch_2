"""The synthetic letters' places, through the real runner with the in-process harness of test_bench_runner (page
analysis reads the PDF's own text layer, no OCR and no NER models).

The address letter holds three addresses; the places letter holds one address and places on their own.  The answer
key of each (``SynthDoc.planted``) marks the suburbs and postcodes that are part of an address (``in_address``): the
address is replaced whole in every profile, so those must not be readable in the output.  The client profile keeps
the places on their own, and they must be.
"""

from __future__ import annotations

import pytest
from synthdocs import SynthDoc, make_ime_letter, make_places_letter
from test_bench_runner import NO_MODELS, InProcessPool
from test_synthdocs_corpus import layer_text

from mlredact.config.loader import load_config
from mlredact.core.types import JobStatus
from mlredact.pipeline.runner import JobResult, Redactor
from mlredact.text.normalize import match_key

PROFILES = ("client", "broad", "claimant_focused", "maximal")


def redacted(
    doc: SynthDoc, profile: str = "client", overrides: dict[str, object] | None = None
) -> tuple[JobResult, str]:
    pool = InProcessPool()
    config = load_config(profile, (), {**NO_MODELS, "surrogate": {"allow_development_key": True}, **(overrides or {})})
    with pytest.MonkeyPatch.context() as mp:
        pool.install(mp)
        with Redactor(config) as redactor:
            result = redactor.run(doc.pdf, keep_held_back=True)
    pdf = result.pdf or result.held_back_pdf
    assert pdf is not None
    return result, match_key(layer_text(pdf))


def readable(doc: SynthDoc, text: str, *, in_address: bool) -> list[str]:
    return [
        p.text
        for p in doc.planted
        if p.kind in ("locality", "postcode") and p.in_address is in_address and match_key(p.text) in text
    ]


class TestTheAnswerKeys:
    def test_the_letters_address_places_are_marked_as_address_parts(self) -> None:
        planted = make_ime_letter(scanned=False).planted
        places = {p.text for p in planted if p.in_address}
        assert places == {"Sydney", "2000", "Parramatta", "2150", "Penrith", "2750"}
        assert {p.kind for p in planted if p.in_address} == {"locality", "postcode"}
        wholes = {p.text for p in planted if p.kind == "street_address" and "NSW" in p.text}
        assert wholes == {
            "Level 4, 123 George Street, Sydney NSW 2000",
            "PO Box 123, Parramatta NSW 2150",
            "42 Wattle Grove Road, Penrith NSW 2750",
        }

    def test_the_places_letter_plants_places_of_every_kind_on_their_own(self) -> None:
        doc = make_places_letter(scanned=False)
        alone = {p.text for p in doc.planted if p.kind in ("locality", "postcode") and not p.in_address}
        # after a place-of-birth phrase, after an injury phrase, in prose, a capital in prose and a postcode alone
        assert alone == {"Orange", "2800", "Bathurst", "2795", "Katoomba", "Lithgow", "Sydney", "2250"}
        assert {p.text for p in doc.planted if p.in_address} == {"Wollongong", "2500"}
        text = "\n".join(doc.lines)
        assert all(p.text in text for p in doc.planted if p.kind in ("locality", "postcode"))

    def test_the_letters_are_born_digital_or_scanned_alike(self) -> None:
        digital, scanned = make_places_letter(scanned=False), make_places_letter(scanned=True, degrade=True, seed=7)
        assert digital.planted == scanned.planted and digital.lines == scanned.lines and digital.pdf != scanned.pdf


class TestTheAddressLetter:
    @pytest.mark.parametrize("profile", PROFILES)
    def test_no_address_part_is_readable(self, profile: str) -> None:
        doc = make_ime_letter(scanned=False)
        result, text = redacted(doc, profile)
        assert result.status is JobStatus.RELEASED, result.manifest["reasons"]
        assert readable(doc, text, in_address=True) == []
        assert [p.text for p in doc.planted if p.kind == "street_address" and match_key(p.text) in text] == []

    def test_the_client_profile_keeps_the_date_of_birth(self) -> None:
        doc = make_ime_letter(scanned=False)
        _result, text = redacted(doc)
        assert match_key("14/03/1978") in text


class TestThePlacesLetter:
    def test_the_client_profile_replaces_the_address_and_keeps_every_place_on_its_own(self) -> None:
        doc = make_places_letter(scanned=False)
        result, text = redacted(doc)
        assert result.status is JobStatus.RELEASED, result.manifest["reasons"]
        assert readable(doc, text, in_address=True) == []
        assert [
            p.text
            for p in doc.planted
            if p.kind in ("locality", "postcode") and not p.in_address and match_key(p.text) not in text
        ] == []
        for kept in ("Bathurst NSW 2795", "Orange NSW 2800", "3 September 2024", "14 May 2023"):
            assert match_key(kept) in text, kept
        assert [p.text for p in doc.planted if p.kind == "street_address" and match_key(p.text) in text] == []

    @pytest.mark.parametrize("profile", PROFILES)
    def test_every_profile_replaces_the_address_whole(self, profile: str) -> None:
        doc = make_places_letter(scanned=False)
        result, text = redacted(doc, profile)
        assert result.status is JobStatus.RELEASED, result.manifest["reasons"]
        assert readable(doc, text, in_address=True) == []
        assert match_key("Marsh") not in text

    def test_under_parts_the_suburb_and_postcode_of_the_address_stay(self) -> None:
        doc = make_places_letter(scanned=False)
        _result, text = redacted(doc, overrides={"policy": {"addresses": "parts"}})
        assert match_key("Marsh") not in text
        assert sorted(readable(doc, text, in_address=True)) == ["2500", "Wollongong"]
