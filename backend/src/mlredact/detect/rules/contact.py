"""R2: contact details — AU phone formats, email, URL."""

from __future__ import annotations

import re
from collections.abc import Mapping

from mlredact.core.model import Candidate
from mlredact.core.types import EntityType, EvidenceStrength, ViewKind
from mlredact.detect.base import candidate, line_prefix
from mlredact.text.views import TextView

STRONG = EvidenceStrength.STRONG
_S = r"[ \t\-]?"

_PHONES: tuple[tuple[str, re.Pattern[str], bool], ...] = (
    ("phone.mobile", re.compile(rf"(?<![0-9+])((?:\+?61{_S}|0)4\d{{2}}{_S}\d{{3}}{_S}\d{{3}})(?![0-9])"), False),
    (
        "phone.landline",
        re.compile(rf"(?<![0-9+])((?:\(0[2378]\)|\+?61{_S}\(?0?[2378]\)?|0[2378]){_S}\d{{4}}{_S}\d{{4}})(?![0-9])"),
        False,
    ),
    ("phone.business", re.compile(rf"(?<![0-9])(1[38]00{_S}\d{{3}}{_S}\d{{3}})(?![0-9])"), False),
    ("phone.thirteen", re.compile(rf"(?<![0-9])(13{_S}\d{{2}}{_S}\d{{2}})(?![0-9])"), True),
    ("phone.local8", re.compile(rf"(?<![0-9])(\d{{4}}{_S}\d{{4}})(?![0-9])"), True),
    (
        "phone.international",
        re.compile(rf"(?<![0-9])(\+\d{{1,3}}{_S}(?:\(\d{{1,4}}\){_S})?\d{{1,4}}(?:{_S}\d{{2,4}}){{2,4}})(?![0-9])"),
        False,
    ),
)
_PHONE_CTX = re.compile(
    r"(?i)(\b(ph|phone|tel|telephone|mob|mobile|cell|fax|facsimile|contact|call|landline)\b|\b[ptfm]\s*:)"
)
_EMAIL = re.compile(
    r"(?<![\w.+\-])([A-Za-z0-9][A-Za-z0-9._%+'\-]*[ \t]?@[ \t]?[A-Za-z0-9\-]+(?:[ \t]?\.[ \t]?[A-Za-z0-9\-]+)*"
    r"\.[A-Za-z]{2,})(?![A-Za-z0-9\-])"
)
_URL = re.compile(r"(?<![\w@])((?:https?://|www\.)[^\s<>\"']+[^\s<>\"'.,;:)\]])")


class ContactDetector:
    detector_id = "rules.contact"
    version = "1"

    def detect(self, views: Mapping[ViewKind, TextView]) -> list[Candidate]:
        out: list[Candidate] = []
        for kind in (ViewKind.LINE, ViewKind.OCR_NORMALISED):
            text = views[kind].text
            for rule, pattern, needs_ctx in _PHONES:
                for m in pattern.finditer(text):
                    if needs_ctx and not _PHONE_CTX.search(line_prefix(text, m.start(1), 30)):
                        continue
                    out.append(candidate(self, EntityType.PHONE, kind, m.start(1), m.end(1), rule, STRONG))
        text = views[ViewKind.LINE].text
        for m in _EMAIL.finditer(text):
            out.append(candidate(self, EntityType.EMAIL, ViewKind.LINE, m.start(1), m.end(1), "email", STRONG))
        for m in _URL.finditer(text):
            out.append(candidate(self, EntityType.URL, ViewKind.LINE, m.start(1), m.end(1), "url", STRONG))
        return out
