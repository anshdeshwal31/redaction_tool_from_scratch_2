"""Injected stage inputs of the benchmark's oracle modes: points to pixels, and tool 2's lines from injected tokens.

All text is fictitious.
"""

from __future__ import annotations

import pickle
import random

import pytest

from mlredact.bench.artifacts import display_poly
from mlredact.core.geometry import BBox
from mlredact.core.model import PageGeometry
from mlredact.pipeline.injection import InjectedPage, InjectedToken, injected_lines, pt_box
from mlredact.render.plan import rotated_token_ids

CANARY = "Zelphinia Quarrington"
# A4 at 150 dpi.
GEOM = PageGeometry(0, 595.276, 841.89, 150, 1240, 1754)


# ------------------------------------------------------------------------------ points -> pixels
def test_points_round_outward_and_written_points_convert_back() -> None:
    box = pt_box([(72, 30.325), (121.005, 30.325), (121.005, 42.691), (72, 42.691)], GEOM)
    assert box == BBox(150, 63, 253, 89)
    written = display_poly(box, GEOM)
    assert written == [[72.0, 30.24], [121.44, 30.24], [121.44, 42.72], [72.0, 42.72]]
    assert pt_box([(x, y) for x, y in written], GEOM) == box


@pytest.mark.parametrize("dpi", [150, 200, 300, 207])
def test_every_written_box_converts_back_to_itself(dpi: int) -> None:
    """207 dpi: a large page rendered below the configured resolution (``effective_dpi``)."""
    geometry = PageGeometry(0, 595.276, 841.89, dpi, round(595.276 * dpi / 72), round(841.89 * dpi / 72))
    rng = random.Random(dpi)
    for _ in range(400):
        x0, y0 = rng.randrange(geometry.width_px - 1), rng.randrange(geometry.height_px - 1)
        x1, y1 = rng.randrange(x0 + 1, geometry.width_px + 1), rng.randrange(y0 + 1, geometry.height_px + 1)
        box = BBox(x0, y0, x1, y1)
        assert pt_box([(x, y) for x, y in display_poly(box, geometry)], geometry) == box


def test_boxes_are_clipped_to_the_page_and_never_empty() -> None:
    assert pt_box([(-10, -10), (20, -10), (20, 20)], GEOM) == BBox(0, 0, 42, 42)
    assert pt_box([(590, 830), (700, 830), (700, 900)], GEOM) == BBox(1229, 1729, 1240, 1754)
    assert pt_box([(72, 72)] * 3, GEOM) == BBox(150, 150, 151, 151)
    assert pt_box([(1000, 1000), (1100, 1000), (1100, 1100)], GEOM) == BBox(1239, 1753, 1240, 1754)


# --------------------------------------------------------------------------------- tool 2 lines
def quad(x0: float, y0: float, x1: float, y1: float) -> tuple[tuple[float, float], ...]:
    return ((x0, y0), (x1, y0), (x1, y1), (x0, y1))


def test_a_polygon_that_is_not_a_quadrilateral_gives_the_box_corners() -> None:
    page = InjectedPage(
        0,
        (
            InjectedToken(0, 0, quad(72, 100, 96, 110), 1.0, "Ms"),
            InjectedToken(0, 1, ((100, 101), (130, 101), (115, 111)), 1.0, "Jane"),
        ),
    )
    (line,) = injected_lines(page, GEOM)
    b = line.bbox
    assert line.quad == ((b.x0, b.y0), (b.x1, b.y0), (b.x1, b.y1), (b.x0, b.y1))


def test_sideways_lines_stay_sideways_for_planning() -> None:
    page = InjectedPage(
        0,
        (
            InjectedToken(0, 0, quad(60, 100, 100, 110), 1.0, "Upright"),
            # Rotated +90 degrees: the text runs down the page.  -90 degrees: it runs up.
            InjectedToken(1, 0, ((200, 100), (200, 140), (190, 140), (190, 100)), 1.0, "Down"),
            InjectedToken(2, 0, ((300, 140), (300, 100), (310, 100), (310, 140)), 1.0, "Up"),
        ),
    )
    upright, down, up = injected_lines(page, GEOM)
    assert rotated_token_ids([upright, down, up]) == {down.tokens[0].token_id, up.tokens[0].token_id}


def test_numbering_must_be_unique_and_the_page_must_match() -> None:
    box = quad(60, 100, 100, 110)
    with pytest.raises(ValueError, match="rendered page"):
        injected_lines(InjectedPage(1, (InjectedToken(0, 0, box, 1.0, "A"),)), GEOM)
    for tokens in (
        (InjectedToken(0, 0, box, 1.0, "A"), InjectedToken(1, 0, box, 1.0, "B"), InjectedToken(0, 1, box, 1.0, "C")),
        (InjectedToken(0, 0, box, 1.0, "A"), InjectedToken(0, 0, box, 1.0, "B")),
    ):
        with pytest.raises(ValueError, match="unique"):
            injected_lines(InjectedPage(0, tokens), GEOM)


def test_injected_pages_travel_to_workers_and_show_no_text() -> None:
    page = InjectedPage(0, (InjectedToken(0, 0, ((1.0, 2.0), (3.0, 2.0), (3.0, 4.0)), 1.0, CANARY),))
    assert pickle.loads(pickle.dumps(page)) == page
    assert CANARY not in repr(page) + repr(page.tokens[0])
