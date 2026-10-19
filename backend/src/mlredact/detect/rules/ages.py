"""Ages ("aged 92", "92 years old", "a 92-year-old man", "Age: 92", "92yo").

Kept by the Broad profile; the Maximal profile top-codes ages of 90 and over (plan §7.1).  Only
explicit age phrasing is matched: a bare duration ("for 5 years") is not an age.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

from mlredact.core.model import Candidate
from mlredact.core.types import EntityType, EvidenceStrength, ViewKind
from mlredact.detect.base import candidate
from mlredact.text.views import TextView

_N = r"(1[0-2]\d|\d{1,2})"
_PATTERNS = (
    ("age.aged", re.compile(rf"(?i)(?<![A-Za-z])(?:aged|age[ \t]*[:\-]|age[ \t]+of)[ \t]*{_N}(?![0-9])")),
    (
        "age.years_old",
        re.compile(
            rf"(?i)(?<![0-9]){_N}[ \t]*[\-‐]?[ \t]*(?:years?|yrs?|y)[ \t]*[\-‐]?[ \t]*(?:old|of[ \t]+age)(?![A-Za-z])"
        ),
    ),
    ("age.yo", re.compile(rf"(?i)(?<![0-9A-Za-z]){_N}[ \t]?(?:yo|y\.o\.?|y/o)(?![A-Za-z])")),
)


class AgeDetector:
    detector_id = "rules.ages"
    version = "1"

    def detect(self, views: Mapping[ViewKind, TextView]) -> list[Candidate]:
        text = views[ViewKind.LINE].text
        out: list[Candidate] = []
        for rule_id, pattern in _PATTERNS:
            for m in pattern.finditer(text):
                out.append(
                    candidate(
                        self,
                        EntityType.AGE,
                        ViewKind.LINE,
                        m.start(1),
                        m.end(),
                        rule_id,
                        EvidenceStrength.STRONG,
                    )
                )
        return out
