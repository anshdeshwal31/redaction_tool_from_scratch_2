"""Person-name cues: honorific titles, salutations and sign-off blocks (strong evidence)."""

from __future__ import annotations

import re
from collections.abc import Mapping

from mlredact.core.model import Candidate
from mlredact.core.types import EntityType, EvidenceStrength, ViewKind
from mlredact.detect.base import candidate
from mlredact.detect.rules.lexicon import TITLES
from mlredact.detect.rules.names import line_end, scan_name
from mlredact.text.views import TextView

_TITLED = re.compile(rf"(?<![A-Za-z]){TITLES}\.?(?=[ \t]+[A-Z])")
_SALUTATION = re.compile(r"(?<![A-Za-z])(?:Dear|Hi|Hello|Attention|Attn\.?)[ \t]+(?=[A-Z])")
_SIGNOFF = re.compile(
    r"(?im)^[ \t]*(?:yours[ \t]+(?:sincerely|faithfully|truly)|kind[ \t]+regards|regards|with[ \t]+thanks"
    r"|signed|sincerely)[ \t,.]*$"
)


class PersonCueDetector:
    detector_id = "rules.persons"
    version = "1"

    def detect(self, views: Mapping[ViewKind, TextView]) -> list[Candidate]:
        text = views[ViewKind.LINE].text
        out: list[Candidate] = []
        for m in _TITLED.finditer(text):
            span = scan_name(text, m.end(), line_end(text, m.end()), allow_title=False, max_tokens=5)
            if span:
                out.append(
                    candidate(self, EntityType.PERSON, ViewKind.LINE, *span, "person.title", EvidenceStrength.STRONG)
                )
        for m in _SALUTATION.finditer(text):
            span = scan_name(text, m.end(), line_end(text, m.end()), allow_title=True, max_tokens=4)
            if span:
                out.append(
                    candidate(
                        self, EntityType.PERSON, ViewKind.LINE, *span, "person.salutation", EvidenceStrength.STRONG
                    )
                )
        for m in _SIGNOFF.finditer(text):
            # The first non-empty line after a sign-off (skipping up to 3 blank/short lines).
            pos = m.end()
            for _ in range(4):
                nl = text.find("\n", pos)
                if nl == -1:
                    break
                pos = nl + 1
                end = line_end(text, pos)
                if text[pos:end].strip():
                    span = scan_name(text, pos, end, allow_title=True, max_tokens=5)
                    if span:
                        out.append(
                            candidate(
                                self, EntityType.PERSON, ViewKind.LINE, *span, "person.signoff", EvidenceStrength.STRONG
                            )
                        )
                    break
        return out
