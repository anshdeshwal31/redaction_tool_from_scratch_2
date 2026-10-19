"""Maximal profile building blocks: date shifting, age top-coding, age detection."""

from __future__ import annotations

import datetime as dt
import hashlib

import pytest
from hypothesis import given
from hypothesis import strategies as st

from mlredact.core.types import EntityType, ViewKind
from mlredact.detect.rules.ages import AgeDetector
from mlredact.surrogate.dates import UNPARSEABLE_DATE, DateShifter, date_delta, generalise_age, leak_needles, shift_date
from mlredact.surrogate.drbg import Chooser
from mlredact.text.normalize import match_key
from mlredact.text.views import build_views
from support import pages_from_lines

KEY = hashlib.sha256(b"maximal").digest()


@pytest.mark.parametrize(
    ("value", "days", "expected"),
    [
        ("14/03/2023", 47, "30/04/2023"),
        ("3.7.62", 47, "19.8.62"),
        ("14-03-23", -14, "28-02-23"),
        ("2023-03-14", -40, "2023-02-02"),
        ("14th March 2023", 47, "30th April 2023"),
        ("14 SEPT 2021", 47, "31 OCT 2021"),
        ("March 14, 2023", 47, "April 30, 2023"),
        ("Mar 2023", 47, "May 2023"),
        ("28/02/2024", 1, "29/02/2024"),
        ("31/12/2023,", 1, "01/01/2024,"),
    ],
)
def test_shift_keeps_layout(value: str, days: int, expected: str) -> None:
    assert shift_date(value, days) == expected


def test_bare_years_are_unchanged_and_non_dates_become_placeholders() -> None:
    assert shift_date("2019", 200) == "2019"
    assert shift_date("31/02/2023", 10) == UNPARSEABLE_DATE
    assert shift_date("next Tuesday", 10) == UNPARSEABLE_DATE


@given(st.dates(min_value=dt.date(1900, 3, 1)), st.integers(-365, 365))
def test_intervals_are_preserved(date: dt.date, days: int) -> None:
    for layout in (f"{date.day:02d}/{date.month:02d}/{date.year}", f"{date.day} {date:%B} {date.year}"):
        assert date_delta(layout, shift_date(layout, days)) == days


def test_offset_is_keyed_deterministic_and_avoids_collisions() -> None:
    dates = ["14/03/2023", "21/03/2023", "02/05/2023"]
    forbidden = leak_needles(dates)
    a = DateShifter.choose(Chooser(KEY), dates, forbidden)
    assert a == DateShifter.choose(Chooser(KEY), dates, forbidden)
    assert 30 <= abs(a.days) <= 365
    assert all(match_key(a.shift(d)) not in forbidden for d in dates)
    others = {DateShifter.choose(Chooser(hashlib.sha256(bytes([i])).digest()), dates, forbidden).days for i in range(8)}
    assert len(others) > 1  # the key decides the offset


def test_collisions_force_a_redraw() -> None:
    chooser = Chooser(KEY)
    first = DateShifter.choose(chooser, ["14/03/2023"], frozenset()).days
    # Forbid exactly what the first offset would produce: the next candidate must be used.
    blocked = frozenset({match_key(shift_date("14/03/2023", first))})
    second = DateShifter.choose(Chooser(KEY), ["14/03/2023"], blocked)
    assert second.days != first and match_key(second.shift("14/03/2023")) not in blocked


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("aged 92", "aged 90+"),
        ("101 years old", "90+ years old"),
        ("a 45-year-old", "a 45-year-old"),
        ("90yo", "90+yo"),
    ],
)
def test_ages_are_top_coded(value: str, expected: str) -> None:
    assert generalise_age(value) == expected


def test_age_detector_matches_explicit_ages_only() -> None:
    lines = [
        "He is a 92-year-old retired fitter, aged 92.",
        "Age: 45; 45 years of age; 30yo; Mrs Smith (92 y.o.)",
        "Pain for 5 years; injured 3 years ago; 10 years of service.",
    ]
    views = build_views(pages_from_lines(lines))
    text = views[ViewKind.LINE].text
    found = sorted(text[c.start : c.end] for c in AgeDetector().detect(views))
    assert found == sorted(["92-year-old", "92", "45", "45 years of age", "30yo", "92 y.o."])
    assert all(c.entity_type is EntityType.AGE for c in AgeDetector().detect(views))


# ------------------------------------------------------------------------- end to end (no OCR)
def test_maximal_policy_shifts_dates_consistently_and_top_codes_ages() -> None:
    from mlredact.config.loader import load_config
    from mlredact.core.types import ActionKind
    from mlredact.detect.engine import detect_document
    from mlredact.policy.engine import decide
    from mlredact.surrogate.assign import SurrogateAssigner, forbidden_keys
    from mlredact.surrogate.generators import SurrogateFactory
    from mlredact.verify.surrogates import check_surrogates

    cfg = load_config("maximal")
    lines = [
        "Mr John SMITH, a 92-year-old retired fitter, was injured on 14/03/2023.",
        "He was reviewed on 21 March 2023 and again on 2023-05-02; his wife is aged 88.",
        "He retired in 2019.",
    ]
    pages = pages_from_lines(lines)
    result = detect_document(pages, cfg, "doc")
    decisions = decide(result.mentions, cfg.policy)
    by_value = {d.mention.value.rstrip(".,;"): d.action for d in decisions}
    assert by_value["92-year-old"] is ActionKind.GENERALISE
    assert by_value.get("88", ActionKind.KEEP) is ActionKind.KEEP  # under 90: unchanged, kept
    removed = [d.mention for d in decisions if d.action is not ActionKind.KEEP]
    tokens = {t.token_id: t for p in pages for t in p.tokens()}
    factory = SurrogateFactory(Chooser(KEY), forbidden_keys(removed, result.persons))
    sur = SurrogateAssigner(factory, result.persons, tokens).assign(decisions)
    out = {
        d.mention.value.rstrip(".,;"): " ".join(t for t in sur[d.mention.mention_id].tokens if t).rstrip(".,;")
        for d in decisions
        if d.mention.mention_id in sur
    }
    assert out["92-year-old"] == "90+-year-old"
    shifted = [(v, out[v]) for v in ("14/03/2023", "21 March 2023", "2023-05-02")]
    deltas = {date_delta(v, s) for v, s in shifted}
    assert len(deltas) == 1 and 30 <= abs(deltas.pop() or 0) <= 365  # one offset: intervals preserved
    assert check_surrogates([], [], shifted_dates=shifted, generalised=[out["92-year-old"]]).passed
    # V5 independently rejects an inconsistent shift and an age left un-coded.
    bad = [*shifted[:2], ("2023-05-02", "2023-05-09")]
    assert not check_surrogates([], [], shifted_dates=bad).passed
    assert not check_surrogates([], [], generalised=["aged 93"]).passed
