"""R3: label -> value extraction.

Forms, letterheads, "Re:" lines and patient labels carry most direct identifiers next to a label.
The value next to a PII label is marked **regardless of what any model thinks** (strong evidence).

The value is looked for on the label's own line first.  When a form label has nothing after it on
its line ("Patient name:" on its own, a label cell of a table) the value is found by *geometry*:
the separate cell to the right on the same row, else the line directly below in the same column.
The value must still parse as the label's kind (name, identifier, date, phone, e-mail...).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from mlredact.core.geometry import BBox
from mlredact.core.model import Candidate
from mlredact.core.types import EntityType, EvidenceStrength, ViewKind
from mlredact.detect.base import candidate, trim_span
from mlredact.detect.rules.dates import find_dates
from mlredact.detect.rules.names import line_end, scan_name
from mlredact.text.views import TextView

T = EntityType


class V(Enum):
    NAME = "name"
    ID = "id"
    DATE = "date"
    ADDRESS = "address"
    PHONE = "phone"
    EMAIL = "email"
    ORG = "org"
    FREE = "free"


@dataclass(frozen=True, slots=True)
class LabelSpec:
    rule_id: str
    pattern: str
    entity_type: EntityType
    value: V
    separator_required: bool


_W = "[ \\t]"
_NO = rf"(?:{_W}*(?:no\.?|number|num\.?|#))"

LABELS: tuple[LabelSpec, ...] = (
    LabelSpec(
        "label.name.party",
        rf"(?:patient|client|claimant|worker|injured{_W}+worker|employee|insured|applicant|plaintiff|defendant"
        rf"|respondent|appellant|deceased|examinee|participant|resident|veteran)(?:'?s)?(?:{_W}+(?:full{_W}+)?name)?",
        T.PERSON,
        V.NAME,
        True,
    ),
    LabelSpec(
        "label.name",
        rf"(?:(?:full|given|first|last|family|middle|maiden|preferred|previous|other|christian){_W}+)?names?"
        rf"|surname|forenames?|also{_W}+known{_W}+as|a\.?k\.?a\.?|n[ée]e|formerly",
        T.PERSON,
        V.NAME,
        True,
    ),
    LabelSpec(
        "label.re",
        rf"re|regarding|subject|our{_W}+client|your{_W}+client|in{_W}+the{_W}+matter{_W}+of",
        T.PERSON,
        V.NAME,
        True,
    ),
    LabelSpec(
        "label.kin",
        rf"next{_W}+of{_W}+kin|nok|emergency{_W}+contact|spouse|partner|mother|father|guardian|parent|husband|wife|carer",
        T.PERSON,
        V.NAME,
        True,
    ),
    LabelSpec(
        "label.professional",
        rf"(?:(?:referring|treating|attending|usual|nominated|reporting|requesting|examining){_W}+)?"
        rf"(?:doctor|dr|gp|practitioner|specialist|surgeon|physician|psychiatrist|psychologist|physiotherapist"
        rf"|case{_W}+manager|claims?{_W}+(?:officer|manager|consultant)|solicitor|lawyer|barrister|counsel"
        rf"|contact(?:{_W}+person)?|attention|attn|signed|witness|interpreter|author|reported{_W}+by"
        rf"|dictated{_W}+by|prepared{_W}+by|referred{_W}+by|copy{_W}+to|cc)",
        T.PERSON,
        V.NAME,
        True,
    ),
    LabelSpec("label.claim", rf"claim(?:{_NO}|{_W}*ref(?:erence)?)", T.REFERENCE, V.ID, False),
    LabelSpec("label.claim_colon", r"claim", T.REFERENCE, V.ID, True),
    LabelSpec("label.policy", rf"policy{_NO}?", T.REFERENCE, V.ID, True),
    LabelSpec(
        "label.ref",
        rf"(?:our|your|my|insurer'?s?|file|matter|case|job|client|scheme|agent'?s?){_W}*ref(?:erence)?\.?{_NO}?",
        T.REFERENCE,
        V.ID,
        False,
    ),
    LabelSpec("label.ref_generic", rf"ref(?:erence)?\.?{_NO}?", T.REFERENCE, V.ID, True),
    LabelSpec("label.file", rf"(?:file|matter|case|job|invoice|account){_NO}", T.REFERENCE, V.ID, False),
    LabelSpec(
        "label.court",
        rf"(?:court|tribunal|commission|proceedings?|registry){_W}*(?:file{_W}*)?{_NO}",
        T.COURT_FILE,
        V.ID,
        False,
    ),
    LabelSpec(
        "label.mrn",
        rf"(?:u\.?r\.?n?\.?|mrn|m\.r\.n\.?|medical{_W}+record|patient{_W}+(?:id|no\.?|number)|hospital{_W}+(?:no\.?|number|id)){_NO}?",
        T.CLINICAL_ID,
        V.ID,
        True,
    ),
    LabelSpec(
        "label.episode",
        rf"(?:episode|admission|encounter|visit|accession|lab|specimen|request|study|order|sample)(?:{_NO}|{_W}*id)",
        T.CLINICAL_ID,
        V.ID,
        False,
    ),
    LabelSpec("label.medicare", rf"medicare(?:{_W}*card)?{_NO}?", T.MEDICARE, V.ID, False),
    LabelSpec("label.dva", rf"(?:dva|veterans?'?{_W}+affairs)(?:{_W}*file)?{_NO}?", T.DVA_FILE, V.ID, False),
    LabelSpec("label.crn", rf"crn|centrelink(?:{_W}*crn|{_NO})?|customer{_W}+reference{_NO}?", T.CRN, V.ID, False),
    LabelSpec("label.tfn", rf"tfn|tax{_W}+file{_NO}", T.TFN, V.ID, False),
    LabelSpec("label.ndis", rf"ndis{_NO}?", T.NDIS, V.ID, False),
    LabelSpec(
        "label.licence",
        rf"(?:driver'?s?{_W}+)?licen[cs]e{_NO}|dl{_W}*(?:no\.?|#)|licen[cs]e{_W}+card{_NO}",
        T.DRIVER_LICENCE,
        V.ID,
        False,
    ),
    LabelSpec("label.passport", rf"passport{_NO}?", T.PASSPORT, V.ID, False),
    LabelSpec("label.provider", rf"provider{_NO}|prov\.?{_W}*no\.?|prescriber{_NO}", T.PROVIDER_NUMBER, V.ID, False),
    LabelSpec("label.ahpra", rf"ahpra(?:{_W}*reg(?:istration)?\.?)?{_NO}?|registration{_NO}", T.AHPRA, V.ID, False),
    LabelSpec("label.ihi", rf"ihi|hpi-?[io]|healthcare{_W}+identifier", T.IHI, V.ID, False),
    LabelSpec(
        "label.employee",
        rf"(?:employee|staff|payroll|member(?:ship)?|student|customer)(?:{_NO}|{_W}*id)",
        T.EMPLOYEE_ID,
        V.ID,
        False,
    ),
    LabelSpec("label.bsb", r"bsb", T.BANK_ACCOUNT, V.ID, False),
    LabelSpec("label.account", rf"(?:bank{_W}+)?(?:account|acc|a/c){_NO}", T.BANK_ACCOUNT, V.ID, False),
    LabelSpec(
        "label.vehicle",
        rf"(?:vehicle{_W}+)?(?:rego|registration{_W}+plate|reg\.?{_W}*no\.?|number{_W}+plate|plate{_NO}?)",
        T.VEHICLE_REG,
        V.ID,
        False,
    ),
    LabelSpec("label.abn", r"abn", T.ABN, V.ID, False),
    LabelSpec("label.acn", r"acn", T.ACN, V.ID, False),
    LabelSpec(
        "label.dob", rf"d\.?{_W}?o\.?{_W}?b\.?|date{_W}+of{_W}+birth|birth{_W}*date", T.DATE_OF_BIRTH, V.DATE, False
    ),
    LabelSpec("label.dod", rf"d\.?{_W}?o\.?{_W}?d\.?|date{_W}+of{_W}+death", T.DATE_OF_DEATH, V.DATE, False),
    LabelSpec(
        "label.address",
        rf"(?:(?:residential|postal|home|street|mailing|current|previous|work|business|correspondence){_W}+)?address|addr\.?",
        T.STREET_ADDRESS,
        V.ADDRESS,
        True,
    ),
    LabelSpec(
        "label.phone",
        rf"(?:(?:home|work|business|contact|after{_W}+hours){_W}+)?(?:phone|ph|tel|telephone|mobile|mob|cell|fax|facsimile)\.?{_NO}?"
        rf"|contact{_NO}",
        T.PHONE,
        V.PHONE,
        False,
    ),
    LabelSpec("label.email", rf"e-?mail(?:{_W}+address)?", T.EMAIL, V.EMAIL, False),
    LabelSpec(
        "label.employer",
        rf"employer|place{_W}+of{_W}+(?:employment|work)|employed{_W}+(?:by|at|with)|works?{_W}+(?:at|for)"
        rf"|working{_W}+(?:at|for)",
        T.ORGANISATION,
        V.ORG,
        False,
    ),
    LabelSpec("label.org", r"company|workplace|school|university|college|tafe", T.ORGANISATION, V.ORG, True),
    LabelSpec("label.locality", r"suburb|town|city|locality", T.LOCALITY, V.FREE, True),
    LabelSpec("label.postcode", r"post[ \t]?code|p/code|pcode", T.POSTCODE, V.ID, False),
)

_SEP_REQUIRED = r"[ \t]*[:#\-–][ \t]*"
_SEP_OPTIONAL = r"[ \t]*[:#\-–]?[ \t]*"
_COMPILED = tuple(
    (
        spec,
        re.compile(
            rf"(?i)(?<![A-Za-z0-9])(?:{spec.pattern})(?![A-Za-z])"
            + (_SEP_REQUIRED if spec.separator_required else _SEP_OPTIONAL)
        ),
    )
    for spec in LABELS
)

_ID_TOKEN = re.compile(r"[^ \t\n]+")
_PHONE_VALUE = re.compile(r"[+(]?\d[\d ()+\-]{4,}\d")
_EMAIL_VALUE = re.compile(r"[A-Za-z0-9._%+'\-]+[ \t]?@[ \t]?[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_ORG_CONNECTORS = frozenset(
    {"&", "and", "of", "the", "pty", "ltd", "p/l", "inc", "limited", "co", "(aust)", "t/as", "t/a"}
)


def _parse_id(text: str, start: int, limit: int) -> tuple[int, int] | None:
    toks = list(_ID_TOKEN.finditer(text, start, limit))
    spans: list[tuple[int, int]] = []
    for i, m in enumerate(toks[:4]):
        tok = m.group(0).strip(",;()[]")
        has_digit = any(c.isdigit() for c in tok)
        next_has_digit = i + 1 < len(toks) and any(c.isdigit() for c in toks[i + 1].group(0))
        short_caps = tok.isalnum() and tok.isupper() and len(tok) <= 4
        if has_digit or (short_caps and next_has_digit) or (tok.isupper() and len(tok) >= 4 and "/" in tok):
            spans.append((m.start(), m.end()))
        else:
            break
    if not spans:
        return None
    return trim_span(text, spans[0][0], spans[-1][1])


def _parse_org(text: str, start: int, limit: int) -> tuple[int, int] | None:
    spans: list[tuple[int, int]] = []
    for m in _ID_TOKEN.finditer(text, start, limit):
        tok = m.group(0)
        core = tok.strip(",;:()")
        if not core:
            break
        if core[0].isupper() or core[0].isdigit() or core.lower() in _ORG_CONNECTORS:
            spans.append((m.start(), m.end()))
            if tok.endswith((",", ";", ".")) and core.lower() not in {"pty", "ltd", "co", "inc"}:
                break
        else:
            break
        if len(spans) >= 8:
            break
    while spans and text[spans[-1][0] : spans[-1][1]].strip(",;:.").lower() in {"&", "and", "of", "the"}:
        spans.pop()
    if not spans:
        return None
    return trim_span(text, spans[0][0], spans[-1][1])


@dataclass(frozen=True, slots=True)
class _Row:
    first: int  # token index range in the line view (inclusive)
    last: int
    box: BBox


def _rows(view: TextView) -> list[_Row]:
    rows: list[_Row] = []
    for i, t in enumerate(view.tokens):
        if rows and (view.tokens[rows[-1].last].page, view.tokens[rows[-1].last].line_no) == (t.page, t.line_no):
            r = rows[-1]
            rows[-1] = _Row(r.first, i, r.box.union(t.bbox))
        else:
            rows.append(_Row(i, i, t.bbox))
    return rows


def _segment(view: TextView, row: _Row, x_from: float, gap: float) -> tuple[int, int] | None:
    """Tokens of ``row`` starting at ``x_from``, up to the first gap wider than ``gap`` (next cell)."""
    first = next((i for i in range(row.first, row.last + 1) if view.tokens[i].bbox.x1 > x_from), None)
    if first is None:
        return None
    last = first
    while last < row.last and view.tokens[last + 1].bbox.x0 - view.tokens[last].bbox.x1 <= gap:
        last += 1
    return first, last


class LabelValueDetector:
    detector_id = "rules.labels"
    version = "2"

    def _geometric(
        self, view: TextView, rows: list[_Row], row_of: dict[int, int], label_first: int, label_last: int
    ) -> list[tuple[int, int, str]]:
        """Value char spans for a label with nothing after it: right cell, then the line below."""
        tokens = view.tokens
        lab = tokens[label_first].bbox.union(tokens[label_last].bbox)
        lh = max(1, lab.height)
        here = rows[row_of[label_last]]
        page = tokens[label_last].page
        out: list[tuple[int, int, str]] = []
        # A separate cell on the same row, to the right (OCR often splits table rows into cells).
        right = [
            r
            for r in rows
            if r is not here
            and tokens[r.first].page == page
            and r.box.x0 >= lab.x1 - 0.5 * lh
            and min(r.box.y1, lab.y1) - max(r.box.y0, lab.y0) >= 0.5 * min(r.box.height, lh)
        ]
        if right:
            nearest = min(right, key=lambda r: (r.box.x0, r.first))
            seg = _segment(view, nearest, nearest.box.x0 - 1, 2 * lh)
            if seg is not None:
                out.append((view.starts[seg[0]], view.ends[seg[1]], "right"))
        # The line directly below, in the label's column.
        below = [
            r
            for r in rows
            if r is not here
            and tokens[r.first].page == page
            and r.box.y0 >= lab.y1 - 0.3 * lh
            and r.box.y0 - lab.y1 <= 1.8 * lh
            and r.box.x1 > lab.x0 - lh
            and r.box.x0 < lab.x1 + 4 * lh
        ]
        if below:
            nearest = min(below, key=lambda r: (r.box.y0, r.box.x0, r.first))
            seg = _segment(view, nearest, lab.x0 - lh, 2 * lh)
            if seg is not None:
                out.append((view.starts[seg[0]], view.ends[seg[1]], "below"))
        return out

    def detect(self, views: Mapping[ViewKind, TextView]) -> list[Candidate]:
        view = views[ViewKind.LINE]
        text = view.text
        rows = _rows(view)
        row_of = {i: k for k, r in enumerate(rows) for i in range(r.first, r.last + 1)}
        matches: list[tuple[int, int, LabelSpec]] = []
        for spec, pattern in _COMPILED:
            for m in pattern.finditer(text):
                matches.append((m.start(), m.end(), spec))
        matches.sort(key=lambda x: (x[0], -x[1], x[2].rule_id))
        starts = sorted({s for s, _e, _sp in matches})
        out: list[Candidate] = []
        for m_start, v_start, spec in matches:
            limit = line_end(text, v_start)
            for s in starts:
                if s >= v_start + 1 and s < limit:
                    limit = s
                    break
            span = self._value(spec.value, text, v_start, limit)
            if span is not None and span[1] > span[0]:
                out.append(
                    candidate(
                        self, spec.entity_type, ViewKind.LINE, span[0], span[1], spec.rule_id, EvidenceStrength.STRONG
                    )
                )
                continue
            # Nothing usable after the label on its line: a form label?  Look by geometry.
            rng = view.token_range(m_start, v_start)
            if len(rng) == 0 or text[v_start:limit].strip(" \t:#-–_.") or limit != line_end(text, v_start):
                continue
            label_first, label_last = rng.start, rng.stop - 1
            row = rows[row_of[label_last]]
            label_like = row.last - row.first <= 5 or text[m_start:v_start].rstrip().endswith((":", "#"))
            if not label_like:
                continue
            for s, e, where in self._geometric(view, rows, row_of, label_first, label_last):
                value = self._value(spec.value, text, s, e)
                if value is not None and value[1] > value[0]:
                    out.append(
                        candidate(
                            self,
                            spec.entity_type,
                            ViewKind.LINE,
                            value[0],
                            value[1],
                            f"{spec.rule_id}.{where}",
                            EvidenceStrength.STRONG,
                        )
                    )
        return out

    @staticmethod
    def _value(kind: V, text: str, start: int, limit: int) -> tuple[int, int] | None:
        if start >= limit:
            return None
        if kind is V.NAME:
            return scan_name(text, start, limit, allow_title=True, max_tokens=6)
        if kind is V.ID:
            return _parse_id(text, start, limit)
        if kind is V.DATE:
            for dm in find_dates(text[start:limit]):
                if dm.start <= 3:
                    return start + dm.start, start + dm.end
            return None
        if kind is V.ADDRESS or kind is V.FREE:
            s, e = trim_span(text, start, limit)
            if kind is V.FREE:
                words = text[s:e].split()
                if len(words) > 6:
                    return None
            return (s, e) if e - s >= 2 else None
        if kind is V.PHONE:
            m = _PHONE_VALUE.match(text, start, limit)
            if m and sum(c.isdigit() for c in m.group(0)) >= 6:
                return trim_span(text, m.start(), m.end())
            return None
        if kind is V.EMAIL:
            m = _EMAIL_VALUE.match(text, start, limit)
            return (m.start(), m.end()) if m else None
        return _parse_org(text, start, limit)  # V.ORG
