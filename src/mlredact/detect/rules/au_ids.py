"""R1: Australian structured identifiers (regex + check digit + context)."""

from __future__ import annotations

import re
from collections.abc import Mapping

from mlredact.core.model import Candidate
from mlredact.core.types import EntityType, EvidenceStrength, ViewKind
from mlredact.detect.base import candidate, line_prefix
from mlredact.detect.rules.checksums import (
    abn_valid,
    acn_valid,
    ihi_valid,
    luhn_valid,
    medicare_valid,
    provider_number_valid,
    tfn_valid,
)
from mlredact.text.normalize import digits_only
from mlredact.text.views import TextView

STRONG, WEAK = EvidenceStrength.STRONG, EvidenceStrength.WEAK
_B = r"(?<![A-Za-z0-9])"  # left boundary
_E = r"(?![A-Za-z0-9])"  # right boundary
_SEP = r"[ \t\-]?"

_MEDICARE = re.compile(_B + rf"([2-6]\d{{3}}{_SEP}\d{{5}}{_SEP}\d(?:[ \t\-/]{{0,2}}\d)?)" + _E)
_IHI = re.compile(rf"(?<![0-9])(8003{_SEP}6[0-2]\d{{2}}{_SEP}\d{{4}}{_SEP}\d{{4}})(?![0-9])")
_NINE = re.compile(rf"(?<![0-9])(\d{{3}}{_SEP}\d{{3}}{_SEP}\d{{2,3}})(?![0-9])")
_ABN = re.compile(rf"(?<![0-9])(\d{{2}}{_SEP}\d{{3}}{_SEP}\d{{3}}{_SEP}\d{{3}})(?![0-9])")
_PROVIDER = re.compile(_B + r"(\d{5,6}[0-9A-HJ-NP-RT-Y][ABFHJKLTWXY])" + _E)
_AHPRA = re.compile(_B + r"([A-Z]{3}\d{10})" + _E)
_DVA_CTX = re.compile(r"(?i)\b(dva|veteran|gold card|white card|orange card)\b")
_DVA = re.compile(_B + r"([NVQWST][A-Z]{0,3}[ \t]?\d{1,6}[A-Z]?)" + _E)
_CRN = re.compile(_B + rf"(\d{{3}}{_SEP}\d{{3}}{_SEP}\d{{3}}[ \t]?[A-Za-z])" + _E)
_CARD = re.compile(r"(?<![0-9])([3-6](?:[ \t\-]?\d){12,18})(?![0-9])")
_VIN = re.compile(_B + r"([A-HJ-NPR-Z0-9]{17})" + _E)
_IP = re.compile(r"(?<![0-9.])((?:25[0-5]|2[0-4]\d|1?\d?\d)(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3})(?![0-9.])")

_CTX = {
    "medicare": re.compile(r"(?i)\b(medicare|mcn|medicare card)\b"),
    "ihi": re.compile(r"(?i)\b(ihi|hpi-?[io]|healthcare identifier)\b"),
    "tfn": re.compile(r"(?i)\b(tfn|tax file)\b"),
    "abn": re.compile(r"(?i)\babn\b"),
    "acn": re.compile(r"(?i)\bacn\b"),
    "provider": re.compile(r"(?i)\b(provider|prov\.?|prescriber)\b"),
    "crn": re.compile(r"(?i)\b(crn|centrelink|customer reference|concession|pension|health care card)\b"),
    "ndis": re.compile(r"(?i)\bndis\b"),
    "passport": re.compile(r"(?i)\bpassport\b"),
}
_PASSPORT = re.compile(_B + r"([A-Z]{1,2}[ \t]?\d{7})" + _E)
_NDIS = re.compile(rf"(?<![0-9])(\d{{3}}{_SEP}\d{{3}}{_SEP}\d{{3}})(?![0-9])")


class AuIdentifierDetector:
    detector_id = "rules.au_ids"
    version = "1"

    def detect(self, views: Mapping[ViewKind, TextView]) -> list[Candidate]:
        out: list[Candidate] = []
        for kind in (ViewKind.LINE, ViewKind.OCR_NORMALISED):
            out.extend(self._scan(views[kind].text, kind))
        return out

    def _has(self, key: str, text: str, pos: int, width: int = 40) -> bool:
        return bool(_CTX[key].search(line_prefix(text, pos, width)))

    def _scan(self, text: str, kind: ViewKind) -> list[Candidate]:
        out: list[Candidate] = []

        def add(t: EntityType, m: re.Match[str], rule: str, strength: EvidenceStrength) -> None:
            out.append(candidate(self, t, kind, m.start(1), m.end(1), rule, strength))

        for m in _MEDICARE.finditer(text):
            d = digits_only(m.group(1))
            if medicare_valid(d[:10]):
                add(EntityType.MEDICARE, m, "medicare.checksum", STRONG)
            elif self._has("medicare", text, m.start(1)):
                add(EntityType.MEDICARE, m, "medicare.context", STRONG)
        for m in _IHI.finditer(text):
            d = digits_only(m.group(1))
            t = EntityType.IHI if d.startswith("800360") else EntityType.HPI
            if ihi_valid(d):
                add(t, m, "ihi.luhn", STRONG)
            elif self._has("ihi", text, m.start(1)):
                add(t, m, "ihi.context", STRONG)
        for m in _NINE.finditer(text):
            d = digits_only(m.group(1))
            if self._has("tfn", text, m.start(1)):
                add(EntityType.TFN, m, "tfn.context", STRONG)
            elif self._has("acn", text, m.start(1)) and len(d) == 9:
                add(EntityType.ACN, m, "acn.context", STRONG)
            elif self._has("ndis", text, m.start(1)) and len(d) == 9:
                add(EntityType.NDIS, m, "ndis.context", STRONG)
            elif len(d) == 9 and acn_valid(d) and tfn_valid(d):
                add(EntityType.OTHER_ID, m, "nine_digit.checksums", WEAK)
            elif tfn_valid(d):
                add(EntityType.TFN, m, "tfn.checksum", WEAK)
        for m in _ABN.finditer(text):
            d = digits_only(m.group(1))
            if abn_valid(d):
                strength = STRONG if self._has("abn", text, m.start(1)) else WEAK
                add(EntityType.ABN, m, "abn.checksum", strength)
            elif self._has("abn", text, m.start(1)):
                add(EntityType.ABN, m, "abn.context", STRONG)
        for m in _PROVIDER.finditer(text):
            if provider_number_valid(m.group(1)):
                add(EntityType.PROVIDER_NUMBER, m, "provider.checksum", STRONG)
            elif self._has("provider", text, m.start(1)):
                add(EntityType.PROVIDER_NUMBER, m, "provider.context", STRONG)
        for m in _AHPRA.finditer(text):
            add(EntityType.AHPRA, m, "ahpra.format", STRONG)
        for m in _DVA.finditer(text):
            if _DVA_CTX.search(line_prefix(text, m.start(1), 40)) and any(c.isdigit() for c in m.group(1)):
                add(EntityType.DVA_FILE, m, "dva.context", STRONG)
        for m in _CRN.finditer(text):
            strength = STRONG if self._has("crn", text, m.start(1)) else WEAK
            add(EntityType.CRN, m, "crn.format", strength)
        for m in _PASSPORT.finditer(text):
            if self._has("passport", text, m.start(1)):
                add(EntityType.PASSPORT, m, "passport.context", STRONG)
        for m in _NDIS.finditer(text):
            if self._has("ndis", text, m.start(1)):
                add(EntityType.NDIS, m, "ndis.context", STRONG)
        for m in _CARD.finditer(text):
            d = digits_only(m.group(1))
            if 13 <= len(d) <= 19 and luhn_valid(d):
                add(EntityType.CARD_NUMBER, m, "card.luhn", STRONG)
        for m in _VIN.finditer(text):
            v = m.group(1)
            if sum(c.isdigit() for c in v) >= 3 and sum(c.isalpha() for c in v) >= 3:
                add(EntityType.VIN, m, "vin.format", STRONG)
        for m in _IP.finditer(text):
            add(EntityType.IP_ADDRESS, m, "ipv4", STRONG)
        return out
