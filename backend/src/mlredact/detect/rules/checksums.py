"""Check-digit validators for Australian identifiers.

A *failed* check never suppresses a labelled candidate (OCR damages check digits); a *passed*
check upgrades unlabelled candidates.  Surrogate generation uses these to guarantee surrogates
deliberately **fail** (so a surrogate can never be a real person's identifier).
"""

from __future__ import annotations

_MEDICARE_WEIGHTS = (1, 3, 7, 9, 1, 3, 7, 9)
_TFN_WEIGHTS_9 = (1, 4, 3, 7, 5, 8, 6, 9, 10)
_TFN_WEIGHTS_8 = (10, 7, 8, 4, 6, 3, 5, 1)
_ABN_WEIGHTS = (10, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19)
_ACN_WEIGHTS = (8, 7, 6, 5, 4, 3, 2, 1)
_PROVIDER_WEIGHTS = (3, 5, 8, 4, 2, 1)
_PROVIDER_CHECK = "YXWTLKJHFBA"
_PROVIDER_PLV = {
    **{str(i): i for i in range(10)},
    **dict(zip("ABCDEFGHJKLMNPQRTUVWXY", range(10, 32), strict=True)),
}


def _digits(s: str) -> list[int] | None:
    return [int(c) for c in s] if s.isdigit() else None


def medicare_valid(number: str) -> bool:
    """10-digit card number (8-digit id, check digit, issue number); IRN not included."""
    d = _digits(number)
    if d is None or len(d) != 10 or d[0] < 2 or d[0] > 6 or d[9] == 0:
        return False
    return sum(a * w for a, w in zip(d[:8], _MEDICARE_WEIGHTS, strict=True)) % 10 == d[8]


def luhn_valid(number: str) -> bool:
    d = _digits(number)
    if d is None or len(d) < 2:
        return False
    total = 0
    for i, digit in enumerate(reversed(d)):
        if i % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def ihi_valid(number: str) -> bool:
    """Individual / provider Healthcare Identifiers: 16 digits, prefix 80036[0-2], Luhn."""
    return len(number) == 16 and number.startswith(("800360", "800361", "800362")) and luhn_valid(number)


def tfn_valid(number: str) -> bool:
    d = _digits(number)
    if d is None:
        return False
    if len(d) == 9:
        return sum(a * w for a, w in zip(d, _TFN_WEIGHTS_9, strict=True)) % 11 == 0
    if len(d) == 8:
        return sum(a * w for a, w in zip(d, _TFN_WEIGHTS_8, strict=True)) % 11 == 0
    return False


def abn_valid(number: str) -> bool:
    d = _digits(number)
    if d is None or len(d) != 11 or d[0] == 0:
        return False
    d = [d[0] - 1, *d[1:]]
    return sum(a * w for a, w in zip(d, _ABN_WEIGHTS, strict=True)) % 89 == 0


def acn_valid(number: str) -> bool:
    d = _digits(number)
    if d is None or len(d) != 9:
        return False
    check = (10 - sum(a * w for a, w in zip(d[:8], _ACN_WEIGHTS, strict=True)) % 10) % 10
    return check == d[8]


def provider_number_valid(value: str) -> bool:
    """Medicare provider number: 6-digit stem (leading zero may be omitted) + location + check."""
    v = value.upper()
    if len(v) == 7:
        v = "0" + v
    if len(v) != 8 or not v[:6].isdigit():
        return False
    plv = _PROVIDER_PLV.get(v[6])
    if plv is None:
        return False
    total = sum(int(c) * w for c, w in zip(v[:6], _PROVIDER_WEIGHTS, strict=True)) + plv * 6
    return _PROVIDER_CHECK[total % 11] == v[7]
