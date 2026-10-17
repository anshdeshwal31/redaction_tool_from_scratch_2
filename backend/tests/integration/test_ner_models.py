"""NER detectors on the pinned models (slow: loads ~1.6 GB of ONNX weights)."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from mlredact.config.loader import load_config
from mlredact.core.model import Candidate
from mlredact.core.types import EntityType, ViewKind
from mlredact.detect.ner.gliner import GlinerDetector
from mlredact.detect.ner.privacy_filter import PrivacyFilterDetector
from mlredact.registry.models import models_dir
from mlredact.text.views import TextView, build_views
from support import ner_enabled, pages_from_lines

pytestmark = [
    pytest.mark.slow,
    pytest.mark.models,
    pytest.mark.skipif(not ner_enabled(), reason="NER disabled on this machine (MLREDACT_TEST_CONFIG)"),
]

CFG = load_config("broad")
NER = CFG.detection.ner
LINES = (
    "Re: Mr John Andrew SMITH DOB: 14/03/1978",
    "Address: 42 Wattle Grove Road, Penrith NSW 2750 Phone 0412 345 678",
    "I examined Mr Smith at Westmead Hospital. His wife Mary attended with him.",
    "Tinel's sign and Phalen's test were positive. He takes Panadeine Forte.",
    "He lives in Blacktown and works for Bunnings Warehouse.",
)


def _found(cands: Sequence[Candidate], views: dict[ViewKind, TextView]) -> set[tuple[EntityType, str]]:
    return {(c.entity_type, views[c.view].text[c.start : c.end]) for c in cands}


@pytest.fixture(scope="module")
def views() -> dict[ViewKind, TextView]:
    return build_views(pages_from_lines(LINES))


@pytest.fixture(scope="module")
def gliner() -> GlinerDetector:
    return GlinerDetector(NER.gliner, NER.threads, models_dir(CFG.runtime.models_dir))


@pytest.fixture(scope="module")
def privacy_filter() -> PrivacyFilterDetector:
    return PrivacyFilterDetector(NER.privacy_filter, NER.threads, models_dir(CFG.runtime.models_dir))


def test_gliner_finds_people_places_and_organisations(gliner: GlinerDetector, views: dict[ViewKind, TextView]) -> None:
    cands = gliner.detect(views)
    found = _found(cands, views)
    assert any(t is EntityType.ORGANISATION and s.startswith("Westmead Hospital") for t, s in found)
    assert any(t is EntityType.PERSON and "Mary" in s for t, s in found)
    assert any(t is EntityType.LOCALITY and "Blacktown" in s for t, s in found)
    assert not any(w in s for _t, s in found for w in ("Tinel", "Phalen", "Panadeine"))
    assert gliner.detect(views) == cands  # deterministic


def test_privacy_filter_finds_identifiers_and_respects_allowlist(
    privacy_filter: PrivacyFilterDetector, views: dict[ViewKind, TextView]
) -> None:
    cands = privacy_filter.detect(views)
    found = _found(cands, views)
    assert (EntityType.PHONE, "0412 345 678") in found
    assert any(t is EntityType.PERSON and s.startswith("John") for t, s in found)  # title trimmed
    assert not any(w in s for _t, s in found for w in ("Tinel", "Phalen", "Panadeine", "Mr "))
    assert privacy_filter.detect(views) == cands


def test_privacy_filter_windowing_matches_whole_document_inference(privacy_filter: PrivacyFilterDetector) -> None:
    sentence = (
        "Mr Peter Walsh was reviewed at the clinic regarding his lumbar spine; he can be reached on 0491 570 006 "
        "and lives at 9 Example Street, Dubbo NSW 2830. "
    )
    text = " ".join(sentence + f"Note {i}." for i in range(70))
    ids, _offsets = privacy_filter.tokenize(text)
    assert 2304 < len(ids) < 4096
    stitched = privacy_filter.decode(privacy_filter.logits(ids, window_tokens=2304))  # two exact cores
    whole = privacy_filter.decode(privacy_filter.logits(ids, window_tokens=len(ids)))  # one pass
    assert len(stitched) > 100
    assert [(a, b, c, v) for a, b, c, _s, v in stitched] == [(a, b, c, v) for a, b, c, _s, v in whole]
