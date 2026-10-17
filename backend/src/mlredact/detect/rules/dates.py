"""R2: dates (day-first, Australian), classified as DOB / date of death / other by context cues.

Only DOB and DoD are removed under the Broad profile; other dates are emitted as DATE mentions so
that profiles with date shifting (Maximal) can act on them.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from mlredact.core.model import Candidate
from mlredact.core.types import EntityType, EvidenceStrength, ViewKind
from mlredact.detect.base import candidate, line_prefix
from mlredact.detect.rules.lexicon import MONTHS
from mlredact.text.views import TextView

# A sentence-final period ("... on 14/03/2023.") must not block a match; only digit-period-digit
# continuations (version numbers, longer numeric runs) do.
_NUMERIC = re.compile(
    r"(?<![0-9/\-])(?<![0-9]\.)(\d{1,2})[ \t]?([/.\-])[ \t]?(\d{1,2})[ \t]?\2[ \t]?(\d{4}|\d{2})"
    r"(?![0-9/\-])(?!\.[0-9])"
)
_ISO = re.compile(r"(?<![0-9])((?:18|19|20)\d{2})-(\d{2})-(\d{2})(?![0-9])")
_TEXT_DMY = re.compile(
    rf"(?i)(?<![A-Za-z0-9])(\d{{1,2}})(?:st|nd|rd|th)?(?:[ \t]+of)?[ \t]*[\-.,]?[ \t]*({MONTHS})\.?,?[ \t\-.]*"
    r"((?:18|19|20)\d{2}|'?\d{2})(?![0-9])"
)
_TEXT_MDY = re.compile(
    rf"(?i)(?<![A-Za-z])({MONTHS})\.?[ \t]+(\d{{1,2}})(?:st|nd|rd|th)?,?[ \t]+((?:18|19|20)\d{{2}})(?![0-9])"
)
_DOB_CUE = re.compile(r"(?i)(\bd\.?[ \t]?o\.?[ \t]?b\b|date[ \t]+of[ \t]+birth|\bbirth[ \t]?date|\bborn\b|\bdob\b)")
_DOD_CUE = re.compile(r"(?i)(\bd\.?[ \t]?o\.?[ \t]?d\b|date[ \t]+of[ \t]+death|\bdied\b|\bdeceased\b)")


@dataclass(frozen=True, slots=True)
class DateMatch:
    start: int
    end: int


def _valid_dmy(d: int, m: int) -> bool:
    return (1 <= d <= 31 and 1 <= m <= 12) or (1 <= m <= 31 and 1 <= d <= 12)  # allow US order


def find_dates(text: str) -> list[DateMatch]:
    found: list[DateMatch] = []
    for m in _NUMERIC.finditer(text):
        if _valid_dmy(int(m.group(1)), int(m.group(3))):
            found.append(DateMatch(m.start(), m.end()))
    for m in _ISO.finditer(text):
        if 1 <= int(m.group(2)) <= 12 and 1 <= int(m.group(3)) <= 31:
            found.append(DateMatch(m.start(), m.end()))
    for m in _TEXT_DMY.finditer(text):
        if 1 <= int(m.group(1)) <= 31:
            found.append(DateMatch(m.start(), m.end()))
    for m in _TEXT_MDY.finditer(text):
        if 1 <= int(m.group(2)) <= 31:
            found.append(DateMatch(m.start(), m.end()))
    return found


class DateDetector:
    detector_id = "rules.dates"
    version = "1"

    def detect(self, views: Mapping[ViewKind, TextView]) -> list[Candidate]:
        text = views[ViewKind.LINE].text
        out: list[Candidate] = []
        for dm in find_dates(text):
            prefix = line_prefix(text, dm.start, 28)
            if _DOB_CUE.search(prefix):
                t, rule, s = EntityType.DATE_OF_BIRTH, "date.dob_cue", EvidenceStrength.STRONG
            elif _DOD_CUE.search(prefix):
                t, rule, s = EntityType.DATE_OF_DEATH, "date.dod_cue", EvidenceStrength.STRONG
            else:
                # A well-formed date is a confident *type* detection; whether it is identifying is
                # the policy's decision (Broad keeps it, Maximal shifts it).
                t, rule, s = EntityType.DATE, "date.format", EvidenceStrength.STRONG
            out.append(candidate(self, t, ViewKind.LINE, dm.start, dm.end, rule, s))
        return out
