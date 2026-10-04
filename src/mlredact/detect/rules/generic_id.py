"""Generic identifier shapes (weak evidence): long alphanumeric codes with >= 5 digits.

Catches MRNs, specimen numbers, invoice numbers and claim numbers that appear without a label.
Explicit exclusions keep dates, times, money, doses/measurements and blood-pressure-style ratios out.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

from mlredact.core.model import Candidate
from mlredact.core.types import EntityType, EvidenceStrength, ViewKind
from mlredact.detect.base import candidate
from mlredact.detect.rules.dates import find_dates
from mlredact.text.views import TextView

_TOKEN = re.compile(r"(?<![A-Za-z0-9$])([A-Za-z0-9][A-Za-z0-9/\-]{4,}[A-Za-z0-9])(?![A-Za-z0-9.%])")
_UNIT_SUFFIX = re.compile(
    r"(?i)(?:mg|mcg|ug|ml|l|iu|units?|kg|g|cm|mm|m|km|mmol|umol|mmhg|bpm|hrs?|mins?|yrs?|kj|cal)$"
)
_YEAR_RANGE = re.compile(r"^(?:19|20)\d{2}[-/](?:19|20)?\d{2}$")
_RATIO = re.compile(r"^\d{2,3}/\d{2,3}$")


class GenericIdDetector:
    detector_id = "rules.generic_id"
    version = "1"

    def detect(self, views: Mapping[ViewKind, TextView]) -> list[Candidate]:
        text = views[ViewKind.LINE].text
        date_spans = [(d.start, d.end) for d in find_dates(text)]
        out: list[Candidate] = []
        for m in _TOKEN.finditer(text):
            tok = m.group(1)
            if sum(c.isdigit() for c in tok) < 5:
                continue
            if _UNIT_SUFFIX.search(tok) or _YEAR_RANGE.match(tok) or _RATIO.match(tok):
                continue
            if any(s <= m.start(1) and m.end(1) <= e for s, e in date_spans):
                continue
            out.append(
                candidate(
                    self,
                    EntityType.OTHER_ID,
                    ViewKind.LINE,
                    m.start(1),
                    m.end(1),
                    "generic.alnum",
                    EvidenceStrength.WEAK,
                    0.5,
                )
            )
        return out
