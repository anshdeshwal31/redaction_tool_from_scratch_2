"""V5: surrogate safety (plan §12), re-checked independently of the generators.

* no surrogate equals, or is within one edit of, any original value or name component;
* identifier surrogates fail their check digit (a surrogate can never be a real identifier);
* phone surrogates lie in ACMA's ranges reserved for fiction;
* e-mail / URL surrogates use RFC 2606 reserved domains;
* shifted dates (Maximal) never equal an original value and all move by one common offset;
* generalised ages leave no age of 90 or over.
"""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Callable, Iterable, Sequence

from rapidfuzz.distance import Levenshtein

from mlredact.core.types import EntityType, VerificationCheck
from mlredact.detect.rules.checksums import (
    abn_valid,
    acn_valid,
    ihi_valid,
    luhn_valid,
    medicare_valid,
    provider_number_valid,
    tfn_valid,
)
from mlredact.surrogate.dates import UNPARSEABLE_DATE
from mlredact.text.normalize import digits_only, match_key
from mlredact.verify.checks import CheckResult

_ACMA_MOBILES = frozenset(
    {
        "0491570006", "0491570156", "0491570157", "0491570158", "0491570159", "0491570110", "0491570313",
        "0491570737", "0491571266", "0491571491", "0491571804", "0491572549", "0491572665", "0491572983",
        "0491573770", "0491573087", "0491574118", "0491574632", "0491575254", "0491575789", "0491576398",
        "0491576801", "0491577426", "0491577644", "0491578957", "0491578148", "0491578888", "0491579212",
        "0491579760", "0491579455",
    }
)  # fmt: skip
_ACMA_SPECIAL = frozenset(
    {"1800160401", "1800975707", "1800975708", "1800975709", "1800975710", "1800975711",
     "1300975707", "1300975708", "1300975709", "1300975710", "1300975711"}
)  # fmt: skip
_RESERVED_DOMAINS = ("example.com", "example.org", "example.net")
_CHECKSUMS: dict[EntityType, Callable[[str], bool]] = {
    EntityType.MEDICARE: lambda d: medicare_valid(d[:10]),
    EntityType.IHI: ihi_valid,
    EntityType.HPI: ihi_valid,
    EntityType.TFN: tfn_valid,
    EntityType.ABN: abn_valid,
    EntityType.ACN: acn_valid,
    EntityType.CARD_NUMBER: luhn_valid,
}


def acma_phone(value: str) -> bool:
    d = digits_only(value)
    if d.startswith("61"):
        d = "0" + d[2:]
    if d in _ACMA_MOBILES or d in _ACMA_SPECIAL:
        return True
    return len(d) == 10 and d[0] == "0" and d[1] in "2378" and d[2:6] in ("5550", "7010")


def near_any(key: str, originals: Iterable[str]) -> bool:
    for o in originals:
        if key == o:
            return True
        if (
            len(o) >= 5
            and len(key) >= 5
            and abs(len(o) - len(key)) <= 1
            and Levenshtein.distance(key, o, score_cutoff=1) <= 1
        ):
            return True
    return False


TOKENWISE = frozenset({EntityType.PERSON, EntityType.STREET_ADDRESS, EntityType.LOCALITY, EntityType.ORGANISATION})


_AGE_LEFT = re.compile(r"(?<!\d)(\d{2,3})(?![\d+])")


def _iso(value: str) -> dt.date | None:
    """Parse with the canonical (day-first) date parser, or as ISO 8601 - not with the shifter's code."""
    from mlredact.surrogate.generators import canonical_date

    text = value.strip().strip("()[]{}\"'.,;:!?")
    try:
        return dt.date.fromisoformat(canonical_date(text) or text)
    except ValueError:
        return None


def check_surrogates(
    items: Sequence[tuple[EntityType, str, tuple[str, ...]]],
    original_keys: Iterable[str],
    *,
    shifted_dates: Sequence[tuple[str, str]] = (),
    generalised: Sequence[str] = (),
) -> CheckResult:
    """``items``: (entity type, whole surrogate text, replaced tokens) per surrogated mention.

    Name-like types are checked token by token (only tokens that were actually replaced; kept
    structure words such as "Street" or "NSW" are identical to the original by design); identifier,
    contact and date types are checked as whole values.
    """
    originals = sorted({k for k in original_keys if k})
    problems = {"collision": 0, "checksum_valid": 0, "phone_not_reserved": 0, "domain_not_reserved": 0}
    for etype, text, replaced in items:
        parts = replaced if etype in TOKENWISE else (text,)
        for part in parts:
            key = match_key(part)
            if key and near_any(key, originals):
                problems["collision"] += 1
                break
        check = _CHECKSUMS.get(etype)
        if check is not None and check(digits_only(text)):
            problems["checksum_valid"] += 1
        if etype is EntityType.PROVIDER_NUMBER and provider_number_valid("".join(c for c in text if c.isalnum())):
            problems["checksum_valid"] += 1
        if etype is EntityType.PHONE and not acma_phone(text):
            problems["phone_not_reserved"] += 1
        if etype in (EntityType.EMAIL, EntityType.URL) and not text.lower().rstrip("/").endswith(_RESERVED_DOMAINS):
            problems["domain_not_reserved"] += 1
    offsets: set[int] = set()
    problems["date_collision"] = problems["date_shift_inconsistent"] = problems["age_not_top_coded"] = 0
    for original, shifted in shifted_dates:
        if match_key(shifted) in originals:
            problems["date_collision"] += 1
        before, after = _iso(original), _iso(shifted)
        if before is not None and after is not None:
            offsets.add((after - before).days)
        elif before is not None and shifted.strip() != UNPARSEABLE_DATE:
            problems["date_shift_inconsistent"] += 1  # a parseable date became something unexplained
    if len(offsets) > 1 or 0 in offsets:
        problems["date_shift_inconsistent"] += 1
    for text in generalised:
        if any(int(n) >= 90 for n in _AGE_LEFT.findall(text)):
            problems["age_not_top_coded"] += 1
    total = sum(problems.values())
    metrics: dict[str, int | float | bool] = {
        "surrogates": len(items),
        "shifted_dates": len(shifted_dates),
        "generalised": len(generalised),
        **problems,
    }
    return CheckResult(VerificationCheck.SURROGATE, total == 0, metrics)
