"""NER building blocks without models: windowing, span decoding, constrained Viterbi, allowlist."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from mlredact.config.schema import ViterbiBiases
from mlredact.core.types import EntityType, ViewKind
from mlredact.detect.allowlist import allowlisted, refine_span
from mlredact.detect.ner.gliner import Span, decode_span_logits, decode_token_logits, greedy_flat, word_windows
from mlredact.detect.ner.privacy_filter import core_windows
from mlredact.detect.ner.viterbi import TagSet, log_softmax, path_spans, posterior_runs, transitions, viterbi
from mlredact.text.views import build_views
from support import pages_from_lines

CATEGORIES = ("person", "phone")
ID2LABEL = {0: "O"} | {1 + 4 * c + k: f"{'BIES'[k]}-{cat}" for c, cat in enumerate(CATEGORIES) for k in range(4)}
TAGS = TagSet.from_id2label(ID2LABEL)


def label(name: str) -> int:
    return next(i for i, v in ID2LABEL.items() if v == name)


# ------------------------------------------------------------------------------------ windows
@given(st.integers(0, 3000), st.integers(2, 400), st.integers(0, 399))
def test_word_windows_cover_everything_with_the_given_overlap(n: int, window: int, overlap: int) -> None:
    if overlap >= window:
        return
    wins = word_windows(n, window, overlap)
    covered = set()
    for a, b in wins:
        assert 0 <= a < b <= n and b - a <= window
        covered.update(range(a, b))
    assert covered == set(range(n))
    for (a0, b0), (a1, _b1) in itertools.pairwise(wins):
        assert a1 - a0 == window - overlap and b0 - a1 == overlap


@given(st.integers(0, 20000), st.integers(0, 1500), st.integers(1, 3000))
def test_core_windows_tile_the_document_with_full_context(n: int, context: int, extra: int) -> None:
    window = 2 * context + extra
    wins = core_windows(n, window, context)
    position = 0
    for ws, we, cs, ce in wins:
        assert cs == position and cs < ce and 0 <= ws <= cs and ce <= we <= n and we - ws <= window
        assert ws == 0 or cs - ws >= context  # left context, unless at the document start
        assert we == n or we - ce >= context  # right context, unless at the document end
        position = ce
    assert position == n


def test_core_windows_rejects_windows_without_a_core() -> None:
    with pytest.raises(ValueError):
        core_windows(5000, 2048, 1024)


# ------------------------------------------------------------------------------ GLiNER decoding
def test_greedy_flat_prefers_higher_scores_and_keeps_ties_in_order() -> None:
    spans = [Span(0, 1, 0, 0.6), Span(1, 2, 1, 0.9), Span(3, 3, 0, 0.5), Span(3, 3, 1, 0.5)]
    assert greedy_flat(spans) == [Span(1, 2, 1, 0.9), Span(3, 3, 0, 0.5)]


def test_span_decoding_ignores_spans_past_the_text() -> None:
    logits = np.full((4, 3, 2), -9.0, dtype=np.float32)  # (L words, max_width, classes)
    logits[1, 1, 0] = 3.0  # words 1-2, class 0
    logits[3, 2, 1] = 5.0  # words 3-5: runs past a 4-word text
    assert decode_span_logits(logits, 4, 0.3) == [Span(1, 2, 0, pytest.approx(0.9526, abs=1e-4))]


def test_token_decoding_requires_inside_scores_throughout() -> None:
    logits = np.full((5, 1, 3), -9.0, dtype=np.float32)  # (L, classes, start/end/inside)
    logits[0, 0, 0] = logits[2, 0, 1] = 4.0  # start at 0, end at 2
    logits[0:3, 0, 2] = 2.0  # inside 0..2
    logits[3, 0, 0] = logits[4, 0, 1] = 4.0  # start 3, end 4 ...
    logits[3, 0, 2] = 2.0  # ... but word 4 is not "inside"
    (span,) = decode_token_logits(logits, 5, 0.3)
    assert (span.start, span.end, span.label) == (0, 2, 0)


# ---------------------------------------------------------------------------------- Viterbi
def test_tagset_parses_bioes_labels() -> None:
    assert TAGS.categories == CATEGORIES and TAGS.background == 0
    with pytest.raises(ValueError):
        TagSet.from_id2label({0: "O", 1: "X-person"})


def test_viterbi_repairs_ill_formed_argmax_paths() -> None:
    # Per-token argmax would be O, I-person, E-person (no B): the constrained path must be well formed.
    logits = np.full((3, len(ID2LABEL)), -5.0)
    logits[0, label("O")] = 2.0
    logits[0, label("B-person")] = 1.5
    logits[1, label("I-person")] = 2.0
    logits[2, label("E-person")] = 2.0
    start, trans, end = transitions(TAGS, ViterbiBiases())
    path = viterbi(log_softmax(logits), start, trans, end)
    assert [ID2LABEL[i] for i in path] == ["B-person", "I-person", "E-person"]
    assert path_spans(path, TAGS) == [(0, 2, 0)]


def test_biases_move_the_operating_point() -> None:
    logits = np.zeros((1, len(ID2LABEL)))
    logits[0, label("O")] = 1.0
    logits[0, label("S-phone")] = 0.6
    start, trans, end = transitions(TAGS, ViterbiBiases())
    assert viterbi(log_softmax(logits), start, trans, end) == [label("O")]
    # Over two tokens, a start bias tips "O O" into "O S-phone".
    two = np.vstack([np.eye(len(ID2LABEL))[label("O")] * 5.0, logits[0]])
    recall = transitions(TAGS, ViterbiBiases(background_to_start=1.0))
    assert viterbi(log_softmax(two), *recall)[1] == label("S-phone")


@given(st.lists(st.lists(st.floats(-8, 8), min_size=9, max_size=9), min_size=1, max_size=30))
def test_viterbi_paths_are_always_well_formed(rows: list[list[float]]) -> None:
    logits = np.array(rows)
    path = viterbi(log_softmax(logits), *transitions(TAGS, ViterbiBiases()))
    kinds = [TAGS.kinds[i] for i in path]
    assert kinds[0] in {"O", "B", "S"} and kinds[-1] in {"O", "E", "S"}
    for (a, ca), (b, cb) in itertools.pairwise([(TAGS.kinds[i], TAGS.cats[i]) for i in path]):
        if a in {"B", "I"}:
            assert b in {"I", "E"} and ca == cb
        else:
            assert b in {"O", "B", "S"}


def test_posterior_sweep_finds_uncovered_mass() -> None:
    probs = np.zeros((4, len(ID2LABEL)))
    probs[:, 0] = [0.9, 0.6, 0.7, 0.95]
    probs[1, label("I-phone")] = 0.4
    probs[2, label("E-phone")] = 0.3
    probs[[0, 3], label("S-person")] = [0.1, 0.05]
    runs = posterior_runs(probs, TAGS, 0.25, np.zeros(4, dtype=bool))
    assert [(a, b, TAGS.categories[c]) for a, b, c, _s in runs] == [(1, 2, "phone")]
    covered = np.array([False, True, False, False])
    assert [(a, b) for a, b, _c, _s in posterior_runs(probs, TAGS, 0.25, covered)] == [(2, 2)]


# -------------------------------------------------------------------------------- allowlist
def _line_view(text: str):  # type: ignore[no-untyped-def]
    return build_views(pages_from_lines([text]))[ViewKind.LINE]


@pytest.mark.parametrize(
    ("text", "words", "etype"),
    [
        ("Tinel's sign was positive", (0, 0), EntityType.PERSON),
        ("Positive Tinel and Phalen tests bilaterally", (1, 3), EntityType.PERSON),
        ("a Colles fracture of the wrist", (1, 1), EntityType.PERSON),
        ("He takes Panadeine Forte twice daily", (2, 3), EntityType.PERSON),
        ("Targin 10 mg at night", (0, 2), EntityType.ORGANISATION),
        ("referred by Medicare", (2, 2), EntityType.PERSON),
        ("assessed under SIRA guidelines", (2, 2), EntityType.ORGANISATION),
        ("He lives in Sydney", (3, 3), EntityType.LOCALITY),
    ],
)
def test_clinical_and_public_terms_are_allowlisted(text: str, words: tuple[int, int], etype: EntityType) -> None:
    assert allowlisted(_line_view(text), words[0], words[1], etype)


@pytest.mark.parametrize(
    ("text", "words", "etype"),
    [
        ("Mr Baker attended today", (1, 1), EntityType.PERSON),  # eponym word, person context
        ("Baker reported pain", (0, 0), EntityType.PERSON),
        ("He lives in Blacktown", (3, 3), EntityType.LOCALITY),
        ("employed by Bunnings Warehouse", (2, 3), EntityType.ORGANISATION),
        ("Tinel's sign was positive", (0, 0), EntityType.PHONE),  # only context types are allowlisted
    ],
)
def test_real_identifiers_are_not_allowlisted(text: str, words: tuple[int, int], etype: EntityType) -> None:
    assert not allowlisted(_line_view(text), words[0], words[1], etype)


def test_refine_span_trims_titles_and_post_nominals() -> None:
    view = _line_view("Dr Peter Brown FRACS, examined him")
    assert refine_span(view, 0, 3, EntityType.PERSON) == (1, 2, EntityType.PERSON)
    assert refine_span(view, 0, 0, EntityType.PERSON) is None  # a bare title is not a name
    assert refine_span(view, 0, 3, EntityType.ORGANISATION) == (0, 3, EntityType.ORGANISATION)  # persons only


def test_implausible_types_are_corrected_or_dropped() -> None:
    view = _line_view("treated at Westmead Hospital since 2019")
    assert refine_span(view, 2, 3, EntityType.CLINICAL_ID) == (2, 3, EntityType.ORGANISATION)
    assert refine_span(view, 4, 4, EntityType.PHONE) is None  # "since": no digits, not a proper name
    assert refine_span(view, 5, 5, EntityType.OTHER_ID) == (5, 5, EntityType.OTHER_ID)
