"""R4 (grammar part): Australian street addresses, PO boxes, DX lines and anchored places, read by the shared address
grammar (:mod:`mlredact.text.address`).

The grammar owns every rule that decides what is an address: the postcode gate (Australia Post's ranges), the state case
rule (never 'Act'), the suburb bound (a gazetteer locality of the state, never across a title, occupation or
organisation word), the street guards (months, measure words, 'Dr', clinical and legal-record nouns), the box-kind
case and context rules ('RSD 3 months' is no box), DX lines, untyped number-led streets before an anchored place
('8 Broadway,'), glued 'NSW2750' and OCR damage.  This detector turns what it reads into candidates:

* a unit's street line (from its first word to the start of its place) is a STREET_ADDRESS, one candidate per line;
* its suburb is a LOCALITY (a capital city stays out, as before: places on their own are the profile's rule) and its
  postcode a POSTCODE;
* a place right after a birth, injury, accident, incident or declaration phrase is not an address (owner ruling,
  question 2), but its suburb and postcode are candidates as before: the profile decides what happens to them;
* a street named without a house number in running text is no candidate at all (owner ruling, question 3).

An overseas address contributes its street line only.  The units the grammar finds by repetition (a footer that repeats
the letterhead's street) are units like any other.  Every candidate is STRONG: the grammar alone is strong evidence.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from mlredact.core.model import Candidate
from mlredact.core.types import EntityType, EvidenceStrength, ViewKind
from mlredact.detect.base import candidate
from mlredact.detect.rules.addrview import AddressView, address_view
from mlredact.detect.rules.lexicon import CAPITAL_CITIES
from mlredact.text.address import AddressUnit, KeptKind, Pos, Role, UnitKind, parse_texts
from mlredact.text.views import TextView

STRONG = EvidenceStrength.STRONG

_STREET_RULES: Mapping[UnitKind, str] = {
    UnitKind.STREET: "address.street",
    UnitKind.STREET_NUMBERLESS: "address.street_numberless",
    UnitKind.CORNER: "address.corner",
    UnitKind.BOX: "address.po_box",
    UnitKind.DX: "address.dx",
}
_POSTCODE_ROLES = frozenset({Role.POSTCODE, Role.STATE_AND_POSTCODE})


def _runs(positions: Iterable[Pos]) -> list[list[Pos]]:
    """Positions grouped into runs of consecutive words of one line."""
    runs: list[list[Pos]] = []
    for p in sorted(positions):
        if runs and runs[-1][-1][0] == p[0] and runs[-1][-1][1] + 1 == p[1]:
            runs[-1].append(p)
        else:
            runs.append([p])
    return runs


class AddressDetector:
    detector_id = "rules.address"
    version = "2"

    def detect(self, views: Mapping[ViewKind, TextView]) -> list[Candidate]:
        view = views[ViewKind.LINE]
        doc = address_view(view)
        out: list[Candidate] = []
        for unit in doc.parse.units:
            if unit.kind is not UnitKind.PLACE_ONLY:
                rule = _STREET_RULES[unit.kind]
                for run in _runs(p for p in unit.street if p in unit.roles):
                    out.append(self._span(view, doc, run, EntityType.STREET_ADDRESS, rule))
            if unit.overseas:
                continue
            out.extend(self._place(view, doc, unit.roles))
        for kept in doc.parse.kept:
            if kept.kind is KeptKind.PLACE_AFTER_CUE:
                out.extend(self._place_after_cue(view, doc, kept.positions))
        return out

    def _span(self, view: TextView, doc: AddressView, run: Sequence[Pos], etype: EntityType, rule: str) -> Candidate:
        first, last = doc.token_at[run[0]], doc.token_at[run[-1]]
        return candidate(self, etype, ViewKind.LINE, view.starts[first], view.ends[last], rule, STRONG)

    def _place(self, view: TextView, doc: AddressView, roles: Mapping[Pos, Role]) -> list[Candidate]:
        """The suburb (a capital city excepted) and the postcode of one place."""
        out: list[Candidate] = []
        for run in _runs(p for p, r in roles.items() if r is Role.SUBURB):
            name = " ".join(view.tokens[doc.token_at[p]].text for p in run).strip(" \t,;:.").lower()
            if name not in CAPITAL_CITIES:
                out.append(self._span(view, doc, run, EntityType.LOCALITY, "address.locality"))
        for run in _runs(p for p, r in roles.items() if r in _POSTCODE_ROLES):
            out.append(self._span(view, doc, run, EntityType.POSTCODE, "address.postcode"))
        return out

    def _place_after_cue(self, view: TextView, doc: AddressView, positions: Sequence[Pos]) -> list[Candidate]:
        """The words of a place kept after a place-kind phrase, read again on their own for their roles."""
        words = [view.tokens[doc.token_at[p]].text for p in positions]
        unit: AddressUnit | None = None
        parsed = parse_texts([" ".join(words)]).units
        if parsed:
            unit = max(parsed, key=lambda u: len(u.roles))
        if unit is None:
            return []
        roles = {positions[wi]: role for (_li, wi), role in unit.roles.items()}
        return self._place(view, doc, roles)
