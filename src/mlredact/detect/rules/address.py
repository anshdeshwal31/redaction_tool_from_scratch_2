"""R4 (grammar part): Australian street addresses, PO boxes and locality/state/postcode triples.

Gazetteer validation (ABS SAL / G-NAF localities) is added with the resources stage; the grammar
alone is already strong evidence when a state + 4-digit postcode anchors the match.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

from mlredact.core.model import Candidate
from mlredact.core.types import EntityType, EvidenceStrength, ViewKind
from mlredact.detect.base import candidate
from mlredact.detect.rules.lexicon import CAPITAL_CITIES, STATES, STREET_TYPE_WORDS, STREET_TYPES
from mlredact.text.views import TextView

STRONG = EvidenceStrength.STRONG
_UNIT = r"(?:(?:Unit|Apt|Apartment|Flat|Suite|Shop|Level|Lvl|Lot|U)\.?[ \t]*\d+[A-Za-z]?[ \t]*[,/]?[ \t]*)"
_WORD = r"(?:[A-Z][A-Za-z'\-]+|[A-Z]{2,})"
_STREET = re.compile(
    rf"(?<![A-Za-z0-9])({_UNIT}?\d+[A-Za-z]?(?:[ \t]*[-/][ \t]*\d+[A-Za-z]?)?,?[ \t]+(?:{_WORD}[ \t]+){{1,3}}"
    rf"(?i:{STREET_TYPES})\.?)(?![A-Za-z])"
)
_PO_BOX = re.compile(
    r"(?i)(?<![A-Za-z])((?:P\.?[ \t]?O\.?[ \t]*Box|G\.?P\.?O\.?[ \t]+Box|Locked[ \t]+Bag|Private[ \t]+Bag|PMB|RMB"
    r"|RSD|CMB)[ \t]*\d+[A-Za-z]?)(?![A-Za-z0-9])"
)
_LOCALITY = re.compile(rf"(?<![A-Za-z])({_WORD}(?:[ \t]+{_WORD}){{0,3}}),?[ \t]+({STATES}),?[ \t]+(\d{{4}})(?![0-9])")


class AddressDetector:
    detector_id = "rules.address"
    version = "1"

    def detect(self, views: Mapping[ViewKind, TextView]) -> list[Candidate]:
        text = views[ViewKind.LINE].text
        out: list[Candidate] = []
        for m in _STREET.finditer(text):
            out.append(
                candidate(
                    self, EntityType.STREET_ADDRESS, ViewKind.LINE, m.start(1), m.end(1), "address.street", STRONG
                )
            )
        for m in _PO_BOX.finditer(text):
            out.append(
                candidate(
                    self, EntityType.STREET_ADDRESS, ViewKind.LINE, m.start(1), m.end(1), "address.po_box", STRONG
                )
            )
        for m in _LOCALITY.finditer(text):
            loc_start, loc_end = m.start(1), m.end(1)
            # A street-type word inside the captured words means the street ran on without a comma:
            # the locality starts after the last street-type word.
            words = list(re.finditer(r"\S+", text[loc_start:loc_end]))
            cut = 0
            for i, w in enumerate(words):
                if w.group(0).lower().rstrip(".") in STREET_TYPE_WORDS:
                    cut = i + 1
            if cut >= len(words):
                loc_start = loc_end
            elif cut:
                loc_start = loc_start + words[cut].start()
            locality = text[loc_start:loc_end]
            if loc_end > loc_start and locality.lower() not in CAPITAL_CITIES:
                out.append(
                    candidate(self, EntityType.LOCALITY, ViewKind.LINE, loc_start, loc_end, "address.locality", STRONG)
                )
            out.append(
                candidate(self, EntityType.POSTCODE, ViewKind.LINE, m.start(3), m.end(3), "address.postcode", STRONG)
            )
        return out
