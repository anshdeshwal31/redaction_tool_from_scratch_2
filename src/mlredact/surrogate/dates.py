"""Date shifting and age generalisation (Maximal profile, plan §7.1 / §10).

* One keyed offset (±30..365 days) per scope moves every shifted date, so intervals between events
  are preserved.  Each date keeps its own layout ("14/03/2023", "14th March 2023", "Mar 2023").
* The offset is drawn by keyed rejection sampling: the first candidate under which no shifted date
  contains any removed value of the document the way V2 searches for leaks (a collision would look
  exactly like a leak).
* Month-year dates ("March 2023") move by the offset in whole months; a bare year is returned
  unchanged (the caller keeps it); a date that cannot be parsed becomes a neutral placeholder.
* Ages of 90 and over become "90+" (HIPAA-style top coding); younger ages are unchanged.
"""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from mlredact.surrogate.drbg import Chooser
from mlredact.surrogate.generators import apply_case
from mlredact.text.normalize import digits_only, match_key

_MIN_SHIFT, _MAX_SHIFT = 30, 365
_MAX_ATTEMPTS = 256
UNPARSEABLE_DATE = "[date]"
AGE_CAP = 90

_MONTHS = (
    "January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
    "November", "December",
)  # fmt: skip
_NUMERIC = re.compile(r"^(\d{1,2})([ \t]?[/.\-][ \t]?)(\d{1,2})([ \t]?[/.\-][ \t]?)(\d{2}|\d{4})$")
_ISO = re.compile(r"^((?:18|19|20)\d{2})-(\d{2})-(\d{2})$")
_ORDINAL = re.compile(r"^(\d{1,2})(st|nd|rd|th)$", re.IGNORECASE)
_WORD = re.compile(r"[A-Za-z]+|\d+(?:st|nd|rd|th)?|[^A-Za-z\d]+", re.IGNORECASE)
_YEAR_ONLY = re.compile(r"^(?:18|19|20)\d{2}$")
_EDGES = re.compile(r"^(\s*[(\[{\"'“‘]*)(.*?)([)\]}\"'”’.,;:!?]*\s*)$", re.DOTALL)
_AGE_NUMBER = re.compile(r"(?<!\d)(\d{1,3})(?!\d)")


def _month_index(word: str) -> int | None:
    """1-12 for a month name or a prefix of at least three letters ("Mar", "Sept"), else None."""
    w = word.lower()
    if len(w) < 3 or not w.isalpha():
        return None
    for i, name in enumerate(_MONTHS):
        if name.lower().startswith(w):
            return i + 1
    return None


def _ordinal(day: int) -> str:
    if 11 <= day % 100 <= 13:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")


def _add_months(year: int, month: int, months: int) -> tuple[int, int]:
    total = year * 12 + (month - 1) + months
    return total // 12, total % 12 + 1


def _year_text(original: str, year: int) -> str:
    return f"{year % 100:02d}" if len(original) == 2 else f"{year:04d}"


def _full_year(text: str) -> int:
    year = int(text)
    if len(text) == 2:
        year += 2000 if year < 30 else 1900
    return year


def shift_date(value: str, days: int) -> str:
    """``value`` moved by ``days``, in the same layout; ``UNPARSEABLE_DATE`` if it is not a date."""
    # Token text carries surrounding punctuation ("(14/03/2023);"): shift the core, keep the rest.
    edges = _EDGES.match(value)
    assert edges is not None  # the pattern matches any string
    lead, v, trail = edges.group(1), edges.group(2), edges.group(3)
    if not v:
        return UNPARSEABLE_DATE
    iso = _ISO.match(v)
    if iso:
        try:
            moved = dt.date(int(iso.group(1)), int(iso.group(2)), int(iso.group(3))) + dt.timedelta(days=days)
        except ValueError:
            return UNPARSEABLE_DATE
        return f"{lead}{moved.isoformat()}{trail}"
    m = _NUMERIC.match(v)
    if m:
        try:
            date = dt.date(_full_year(m.group(5)), int(m.group(3)), int(m.group(1)))
        except ValueError:
            return UNPARSEABLE_DATE
        new = date + dt.timedelta(days=days)
        d = f"{new.day:02d}" if len(m.group(1)) == 2 else str(new.day)
        mo = f"{new.month:02d}" if len(m.group(3)) == 2 else str(new.month)
        return f"{lead}{d}{m.group(2)}{mo}{m.group(4)}{_year_text(m.group(5), new.year)}{trail}"
    parts = _WORD.findall(v)
    day_i = month_i = year_i = None
    for i, p in enumerate(parts):
        if _ORDINAL.match(p) or (p.isdigit() and len(p) <= 2 and day_i is None and month_i is None):
            day_i = i if day_i is None else day_i
        elif p.isdigit() and len(p) == 4:
            year_i = i
        elif _month_index(p) is not None and month_i is None:
            month_i = i
        elif p.isdigit() and len(p) <= 2 and month_i is not None and day_i is None:
            day_i = i  # "March 14, 2023"
    if year_i is not None and month_i is None and day_i is None and _YEAR_ONLY.match(parts[year_i]):
        return value  # a bare year: an offset under one year cannot move it meaningfully (kept)
    if month_i is None or year_i is None:
        return UNPARSEABLE_DATE
    month = _month_index(parts[month_i])
    assert month is not None
    year = int(parts[year_i])
    if day_i is None:  # month + year: move by whole months
        new_year, new_month = _add_months(year, month, round(days / 30.4375))
        new_day = None
    else:
        ordinal = _ORDINAL.match(parts[day_i])
        day = int(ordinal.group(1)) if ordinal else int(parts[day_i])
        try:
            new = dt.date(year, month, day) + dt.timedelta(days=days)
        except ValueError:
            return UNPARSEABLE_DATE
        new_year, new_month, new_day = new.year, new.month, new.day
    word = parts[month_i]
    name = _MONTHS[new_month - 1]
    abbreviated = len(word) < len(_MONTHS[month - 1])  # "Mar", "Sept" stay abbreviated
    parts[month_i] = apply_case(word, name[:3] if abbreviated else name)
    parts[year_i] = f"{new_year:04d}"
    if day_i is not None and new_day is not None:
        ordinal = _ORDINAL.match(parts[day_i])
        if ordinal:
            parts[day_i] = f"{new_day}{apply_case(ordinal.group(2), _ordinal(new_day))}"
        else:
            parts[day_i] = f"{new_day:02d}" if len(parts[day_i]) == 2 else str(new_day)
    return lead + "".join(parts) + trail


def date_delta(original: str, shifted: str) -> int | None:
    """Days between two full dates in the same layout (independent re-parse, used by V5)."""
    a, b = _parse_full(original), _parse_full(shifted)
    if a is None or b is None:
        return None
    return (b - a).days


def _parse_full(value: str) -> dt.date | None:
    v = value.strip().rstrip(".,")
    m = _NUMERIC.match(v)
    iso = _ISO.match(v)
    try:
        if iso:
            return dt.date(int(iso.group(1)), int(iso.group(2)), int(iso.group(3)))
        if m:
            return dt.date(_full_year(m.group(5)), int(m.group(3)), int(m.group(1)))
        parts = _WORD.findall(v)
        day = month = year = None
        for p in parts:
            o = _ORDINAL.match(p)
            if o:
                day = int(o.group(1))
            elif p.isdigit() and len(p) == 4:
                year = int(p)
            elif p.isdigit() and len(p) <= 2 and day is None:
                day = int(p)
            elif month is None and _month_index(p) is not None:
                month = _month_index(p)
        if day is None or month is None or year is None:
            return None
        return dt.date(year, month, day)
    except ValueError:
        return None


def leak_needles(values: Iterable[str], min_len: int = 4) -> frozenset[str]:
    """The strings V2 looks for: each value's match key and (6+) digit string."""
    out: set[str] = set()
    for v in values:
        key = match_key(v)
        if len(key) >= min_len:
            out.add(key)
        digits = digits_only(v)
        if len(digits) >= max(min_len, 6):
            out.add(digits)
    return frozenset(out)


def _collides(text: str, needles: frozenset[str]) -> bool:
    key, digits = match_key(text), digits_only(text)
    return any(n in key or n in digits for n in needles)


@dataclass(frozen=True, slots=True)
class DateShifter:
    days: int

    @classmethod
    def choose(cls, chooser: Chooser, values: Sequence[str], needles: frozenset[str]) -> DateShifter:
        """Keyed offset under which no shifted value contains a removed value (see ``leak_needles``)."""
        span = _MAX_SHIFT - _MIN_SHIFT + 1
        for attempt in range(_MAX_ATTEMPTS):
            magnitude = _MIN_SHIFT + chooser.index("date_shift", "magnitude", span, attempt)
            sign = 1 if chooser.index("date_shift", "sign", 2, attempt) else -1
            days = sign * magnitude
            shifted = (s for v in values if (s := shift_date(v, days)) not in (v, UNPARSEABLE_DATE))
            if not any(_collides(s, needles) for s in shifted):
                return cls(days)
        # Pathological documents: every candidate collides somewhere.  Fail safe: the caller renders
        # every date as the placeholder (nothing original survives, nothing collides).
        return cls(0)

    def shift(self, value: str) -> str:
        if self.days == 0:
            return UNPARSEABLE_DATE
        return shift_date(value, self.days)


def generalise_age(value: str) -> str:
    """Top-code ages: every number >= 90 becomes "90+"; anything else is returned unchanged."""

    def repl(m: re.Match[str]) -> str:
        return f"{AGE_CAP}+" if int(m.group(1)) >= AGE_CAP else m.group(1)

    return _AGE_NUMBER.sub(repl, value)
