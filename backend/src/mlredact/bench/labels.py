"""Benchmark types and tool 2's types, for injected mentions (``mlredact bench`` oracle modes).

``LABEL_MAP`` is tool 2's published label map (``label_map`` in the benchmark's candidate manifest,
``candidates/mlredact-client.yaml``): every label ``mentions.json`` reports -> the benchmark type it is scored as.
Injected mentions carry benchmark types, or labels another tool's map did not cover, so they are mapped the other
way here: each benchmark type is a family of tool 2 types, and every member of a family maps back to that type, so
the label reported for an injected mention always names the type it was injected with.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from mlredact.core.model import Candidate
from mlredact.core.types import EntityType, RegionKind
from mlredact.detect.fusion import TYPE_PRIORITY

E = EntityType
R = RegionKind

# Tool 2's label -> the benchmark type it is scored as (the candidate manifest's label_map, in its order).
LABEL_MAP: dict[str, str] = {
    "person": "PERSON_NAME",
    "date_of_birth": "DATE",
    "date_of_death": "DATE",
    "date": "DATE",
    "age": "AGE",
    "street_address": "ADDRESS",
    "locality": "LOCATION",
    "postcode": "LOCATION",
    "phone": "PHONE",
    "email": "EMAIL",
    "url": "URL",
    "ip_address": "ID_OTHER",
    "medicare": "ID_GOVERNMENT",
    "ihi": "ID_GOVERNMENT",
    "dva_file": "ID_GOVERNMENT",
    "crn": "ID_GOVERNMENT",
    "tfn": "ID_GOVERNMENT",
    "ndis": "ID_GOVERNMENT",
    "driver_licence": "ID_GOVERNMENT",
    "passport": "ID_GOVERNMENT",
    "hpi": "ID_PROVIDER",
    "provider_number": "ID_PROVIDER",
    "ahpra": "ID_PROVIDER",
    "clinical_id": "ID_OTHER",
    "reference": "ID_CLAIM",
    "court_file": "ID_LEGAL_REF",
    "bank_account": "ID_FINANCIAL",
    "card_number": "ID_FINANCIAL",
    "abn": "ID_BUSINESS",
    "acn": "ID_BUSINESS",
    "employee_id": "ID_OTHER",
    "vehicle_reg": "ID_OTHER",
    "vin": "ID_OTHER",
    "other_id": "ID_OTHER",
    "organisation": "ORGANISATION",
    "file_path": "FILE_PATH",
    "signature": "SIGNATURE",
    "face": "IMAGE_PII",
    "photo": "IMAGE_PII",
    "handwriting": "HANDWRITTEN_MARK",
    "barcode": "ID_OTHER",
    "logo": "ORGANISATION",
    "stamp": "ORGANISATION",
    "illegible": "ID_OTHER",
}


@dataclass(frozen=True, slots=True)
class Family:
    """What an injected label becomes.

    ``types``: the tool 2 types a text mention with the label may get, the default first (none for a region label);
    ``region_kind``: the kind of the regions a region label becomes, and for a text label the kind of an anchor no
    token explains (or of the whole mention when it has no token); ``ambiguous``: tool 2's rule detectors choose the
    type among ``types`` (the identifier families); ``location``: the locality of an
    address (under ``addresses: whole`` it joins the address before it; under ``parts`` a trailing postcode becomes a
    mention of its own)."""

    types: tuple[EntityType, ...]
    region_kind: RegionKind
    ambiguous: bool = False
    location: bool = False

    @property
    def default(self) -> EntityType | None:
        return self.types[0] if self.types else None

    @property
    def reported(self) -> str:
        """The label reported for a mention of the family without a token of its own: its default type, or its region
        kind for a region label."""
        return self.types[0].value if self.types else self.region_kind.value


# Benchmark type -> family.  Defaults: a date takes the date-of-birth fake (same year and layout; a plain date would
# get an identifier's scramble), a government identifier the driver-licence one until the rules recognise it.
FAMILIES: dict[str, Family] = {
    "PERSON_NAME": Family((E.PERSON,), R.HANDWRITING),
    "SIGNATURE": Family((), R.SIGNATURE),
    "HANDWRITTEN_MARK": Family((), R.HANDWRITING),
    "IMAGE_PII": Family((), R.PHOTO),
    "ADDRESS": Family((E.STREET_ADDRESS,), R.HANDWRITING),
    "LOCATION": Family((E.LOCALITY, E.POSTCODE), R.HANDWRITING, location=True),
    "PHONE": Family((E.PHONE,), R.HANDWRITING),
    "EMAIL": Family((E.EMAIL,), R.HANDWRITING),
    "URL": Family((E.URL,), R.HANDWRITING),
    "DATE": Family((E.DATE_OF_BIRTH, E.DATE_OF_DEATH, E.DATE), R.HANDWRITING),
    "AGE": Family((E.AGE,), R.HANDWRITING),
    "ID_CLAIM": Family((E.REFERENCE,), R.HANDWRITING),
    "ID_LEGAL_REF": Family((E.COURT_FILE,), R.HANDWRITING),
    "ID_PROVIDER": Family((E.PROVIDER_NUMBER, E.HPI, E.AHPRA), R.HANDWRITING, ambiguous=True),
    "ID_GOVERNMENT": Family(
        (E.DRIVER_LICENCE, E.MEDICARE, E.IHI, E.DVA_FILE, E.CRN, E.TFN, E.NDIS, E.PASSPORT),
        R.HANDWRITING,
        ambiguous=True,
    ),
    "ID_BUSINESS": Family((E.ABN, E.ACN), R.HANDWRITING, ambiguous=True),
    "ID_FINANCIAL": Family((E.BANK_ACCOUNT, E.CARD_NUMBER), R.HANDWRITING, ambiguous=True),
    "ID_OTHER": Family(
        (E.OTHER_ID, E.IP_ADDRESS, E.CLINICAL_ID, E.EMPLOYEE_ID, E.VEHICLE_REG, E.VIN), R.BARCODE, ambiguous=True
    ),
    "FILE_PATH": Family((E.FILE_PATH,), R.HANDWRITING),
    "ORGANISATION": Family((E.ORGANISATION,), R.LOGO),
}
# QUASI_IDENTIFIER (never required under the benchmark's policy) and labels nobody maps.
FALLBACK = Family((E.OTHER_ID,), R.HANDWRITING)


def _own(label: str) -> Family:
    """One of tool 2's own labels, taken as itself."""
    if label in {k.value for k in RegionKind}:
        return Family((), RegionKind(label))
    return Family((EntityType(label),), FAMILIES[LABEL_MAP[label]].region_kind)


_OWN = {label: _own(label) for label in LABEL_MAP}
# Every region kind an injected mention can become: a profile that keeps one of them would keep injected PII.
REGION_KINDS_USED = frozenset(f.region_kind for f in (*FAMILIES.values(), *_OWN.values(), FALLBACK))
_RANK = {t: i for i, t in enumerate(TYPE_PRIORITY)}
# A postcode, as policy and surrogates recognise one.
_POSTCODE = re.compile(r"^\d{4}[,.]?$")


def family_of(label: str) -> Family:
    """The family of an injected label: a benchmark type, or one of tool 2's own labels (as itself); anything else,
    QUASI_IDENTIFIER included, falls back to :data:`FALLBACK`."""
    family = FAMILIES.get(label, _OWN.get(label))
    return family if family is not None else FALLBACK


def refine(family: Family, tokens: range, candidates: Sequence[tuple[range, Candidate]]) -> EntityType:
    """The type of a text mention of ``family`` over the line-view token indices ``tokens``.  In an ambiguous family
    the rule detectors' hits of the family's types that overlap the mention choose it, as detection fusion chooses a
    type (strongest evidence, then type priority, then rule id); otherwise, or without such a hit, the default.
    ``candidates``: every hit with its token indices.  Only the type is chosen: the mention keeps its tokens."""
    if family.default is None:
        raise ValueError("a region label has no text type")
    if not family.ambiguous:
        return family.default
    hits = [
        c
        for span, c in candidates
        if c.entity_type in family.types and max(span.start, tokens.start) < min(span.stop, tokens.stop)
    ]
    if not hits:
        return family.default
    best = min(hits, key=lambda c: (-int(c.evidence.strength), _RANK[c.entity_type], c.evidence.rule_id))
    return best.entity_type


def split_location(texts: Sequence[str]) -> list[tuple[EntityType, int, int]]:
    """A LOCATION mention's token texts as (type, start, end) parts, ``end`` exclusive: a trailing postcode is a
    POSTCODE part (its fake then follows the fake suburb on the line), the rest (suburb and state) a LOCALITY part.
    Used under ``addresses: parts`` only: ``addresses: whole`` makes a LOCATION a unit with its address."""
    if not texts:
        return []
    if not _POSTCODE.match(texts[-1]):
        return [(E.LOCALITY, 0, len(texts))]
    last = len(texts) - 1
    return [*([(E.LOCALITY, 0, last)] if last else []), (E.POSTCODE, last, len(texts))]
