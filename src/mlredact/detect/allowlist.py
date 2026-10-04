"""Contextual allowlist (plan §7.3).

Applied to *statistical* evidence only (NER models readily tag eponyms, public bodies and capital
cities).  It never touches seed, label->value, identifier-rule or propagation evidence.  Every rule
is an exact pattern over canonical tokens, so a word that is both a surname and an eponym is only
suppressed in its clinical pattern: "Baker's cyst" is clinical, "Mr Baker" is a person.
"""

from __future__ import annotations

import re
from functools import cache

from mlredact.core.types import EntityType
from mlredact.detect.rules.lexicon import (
    CAPITAL_CITIES,
    CLINICAL_NOUNS,
    DOSE_FORM_WORDS,
    MEDICAL_ABBREVIATIONS,
    NON_NAME_WORDS,
    POST_NOMINALS,
    PUBLIC_BODIES,
    TITLES,
)
from mlredact.resolve.persons import split_token
from mlredact.resources.loader import resource_root
from mlredact.surrogate.pools import load_pools
from mlredact.text.normalize import match_key
from mlredact.text.views import TextView

_CONNECTORS = frozenset({"and", "&", "or", "/", "+", "-"})
_LOOKAHEAD = 6
_PLACES = CAPITAL_CITIES | frozenset(
    match_key(s)
    for s in (
        "Australia",
        "New South Wales",
        "Victoria",
        "Queensland",
        "South Australia",
        "Western Australia",
        "Tasmania",
        "Northern Territory",
        "Australian Capital Territory",
        "NSW",
        "VIC",
        "QLD",
        "SA",
        "WA",
        "TAS",
        "NT",
        "ACT",
        "New Zealand",
    )
)
_CONTEXT_TYPES = frozenset({EntityType.PERSON, EntityType.ORGANISATION, EntityType.LOCALITY})
_TITLE = re.compile(rf"(?i)^{TITLES}\.?$")
_DOSE = re.compile(r"(?i)^[\d.,/]+(?:mg|mcg|g|ml|%)?$")


@cache
def drug_names() -> frozenset[str]:
    """Curated medicine names (``resources/allowlists/drugs.txt``, hash-verified with the tree)."""
    text = (resource_root() / "allowlists" / "drugs.txt").read_text("utf-8")
    return frozenset(w for line in text.splitlines() if not line.startswith("#") for w in line.lower().split())


def _core(text: str) -> str:
    return split_token(text)[1]


def _clinical_follows(view: TextView, last: int, eponyms: frozenset[str]) -> bool:
    """After token ``last`` (same line): optional further eponyms/connectors, then a clinical noun."""
    tokens = view.tokens
    line = (tokens[last].page, tokens[last].line_no)
    for i in range(last + 1, min(len(tokens), last + 1 + _LOOKAHEAD)):
        if (tokens[i].page, tokens[i].line_no) != line:
            return False
        word = _core(tokens[i].text).lower()
        if word in CLINICAL_NOUNS:
            return True
        if word not in eponyms and word not in _CONNECTORS:
            return False
    return False


def allowlisted(view: TextView, first: int, last: int, entity_type: EntityType) -> bool:
    """True if the tokens ``first..last`` (inclusive) of ``view`` match an allowlist pattern."""
    if entity_type not in _CONTEXT_TYPES:
        return False
    cores = [_core(t.text) for t in view.tokens[first : last + 1]]
    words = [w for w in cores if any(ch.isalnum() for ch in w)]
    if not words:
        return True  # punctuation only
    lower = [w.lower() for w in words]
    # Nothing name-like: "Patient", "Re", "MRI", "FRACS", "Medicare".
    if all(w in NON_NAME_WORDS or w in MEDICAL_ABBREVIATIONS or w in POST_NOMINALS for w in lower):
        return True
    key = match_key(" ".join(words))
    if key in PUBLIC_BODIES:
        return True
    if entity_type is EntityType.LOCALITY and key in _PLACES:
        return True
    # Medicine names: "Panadeine Forte", "Targin 10 mg", "Lyrica".
    drugs = drug_names()
    if any(w in drugs for w in lower) and all(w in drugs or w in DOSE_FORM_WORDS or _DOSE.match(w) for w in lower):
        return True
    # Eponym pattern: "Tinel's sign", "Tinel and Phalen tests", "Colles fracture".
    eponyms = load_pools().eponyms
    names = [w for w in lower if w not in _CONNECTORS]
    clinical_inside = bool(names) and names[-1] in CLINICAL_NOUNS
    if clinical_inside:
        names = names[:-1]
    if names and all(w in eponyms for w in names):
        return clinical_inside or _clinical_follows(view, last, eponyms)
    return False


# Statistical models sometimes put the right span under the wrong label ("Westmead Hospital" as a
# "healthcare number").  Identifier and contact types must look like one; otherwise the hit is
# re-typed as an organisation when it reads like a proper name, and dropped when it does not.
_NEEDS_DIGIT = frozenset(
    {
        EntityType.MEDICARE,
        EntityType.IHI,
        EntityType.HPI,
        EntityType.TFN,
        EntityType.CRN,
        EntityType.NDIS,
        EntityType.DVA_FILE,
        EntityType.PASSPORT,
        EntityType.DRIVER_LICENCE,
        EntityType.PROVIDER_NUMBER,
        EntityType.AHPRA,
        EntityType.CARD_NUMBER,
        EntityType.BANK_ACCOUNT,
        EntityType.ABN,
        EntityType.ACN,
        EntityType.VIN,
        EntityType.VEHICLE_REG,
        EntityType.CLINICAL_ID,
        EntityType.COURT_FILE,
        EntityType.EMPLOYEE_ID,
        EntityType.REFERENCE,
        EntityType.OTHER_ID,
        EntityType.PHONE,
        EntityType.IP_ADDRESS,
        EntityType.POSTCODE,
    }
)


def _plausible_type(text: str, entity_type: EntityType) -> EntityType | None:
    if entity_type in _NEEDS_DIGIT and not any(ch.isdigit() for ch in text):
        return EntityType.ORGANISATION if any(w[:1].isupper() for w in text.split()) else None
    if entity_type is EntityType.EMAIL and "@" not in text:
        return None
    return entity_type


def refine_span(view: TextView, first: int, last: int, entity_type: EntityType) -> tuple[int, int, EntityType] | None:
    """Normalise a statistical hit (token indices ``first..last``).

    Person spans lose leading titles and trailing post-nominals (kept visible, plan §10); an
    implausible type is corrected or the hit dropped; ``None`` when nothing (or only allowlisted
    text) remains.
    """
    if entity_type is EntityType.PERSON:
        while first <= last and _TITLE.match(_core(view.tokens[first].text)):
            first += 1
        while first <= last and (
            (core := _core(view.tokens[last].text).lower()) in POST_NOMINALS or not any(ch.isalnum() for ch in core)
        ):
            last -= 1
        if first > last:
            return None
    etype = _plausible_type(" ".join(t.text for t in view.tokens[first : last + 1]), entity_type)
    if etype is None or allowlisted(view, first, last, etype):
        return None
    return first, last, etype
