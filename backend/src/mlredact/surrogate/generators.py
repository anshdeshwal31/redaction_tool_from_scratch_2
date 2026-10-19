"""Surrogate generators (plan §10).

Every generator is a pure function of (scope key, entity type, canonical original value) and is
*safe by construction*:

* phone numbers come only from ACMA's ranges reserved for fiction;
* e-mail addresses / URLs use RFC 2606 reserved domains;
* identifiers keep their format but **fail** their check digit, so a surrogate can never be a
  real person's identifier;
* no surrogate equals (by comparison key) any original value in the document.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence

from mlredact.core.types import EntityType
from mlredact.detect.rules.checksums import (
    abn_valid,
    acn_valid,
    ihi_valid,
    luhn_valid,
    medicare_valid,
    provider_number_valid,
    tfn_valid,
)
from mlredact.detect.rules.lexicon import STATES, STREET_TYPE_WORDS
from mlredact.surrogate.drbg import Chooser
from mlredact.surrogate.pools import Locality, Pools, load_pools
from mlredact.text.normalize import digits_only, match_key

ACMA_MOBILES = (
    "0491 570 006", "0491 570 156", "0491 570 157", "0491 570 158", "0491 570 159", "0491 570 110",
    "0491 570 313", "0491 570 737", "0491 571 266", "0491 571 491", "0491 571 804", "0491 572 549",
    "0491 572 665", "0491 572 983", "0491 573 770", "0491 573 087", "0491 574 118", "0491 574 632",
    "0491 575 254", "0491 575 789", "0491 576 398", "0491 576 801", "0491 577 426", "0491 577 644",
    "0491 578 957", "0491 578 148", "0491 578 888", "0491 579 212", "0491 579 760", "0491 579 455",
)  # fmt: skip
ACMA_LANDLINE_PREFIXES = ("5550", "7010")  # (0x) 5550 xxxx / (0x) 7010 xxxx
ACMA_FREEPHONE = ("1800 160 401", "1800 975 707", "1800 975 708", "1800 975 709", "1800 975 710", "1800 975 711")
ACMA_LOCALRATE = ("1300 975 707", "1300 975 708", "1300 975 709", "1300 975 710", "1300 975 711")
RESERVED_EMAIL_DOMAINS = ("example.com", "example.org", "example.net")

_STATE_RE = re.compile(rf"(?i)^{STATES}$")
_STATE_CANON = {
    "nsw": "NSW", "n.s.w": "NSW", "n.s.w.": "NSW", "new south wales": "NSW",
    "vic": "VIC", "vic.": "VIC", "victoria": "VIC",
    "qld": "QLD", "qld.": "QLD", "queensland": "QLD",
    "sa": "SA", "south australia": "SA",
    "wa": "WA", "western australia": "WA",
    "tas": "TAS", "tas.": "TAS", "tasmania": "TAS",
    "nt": "NT", "northern territory": "NT",
    "act": "ACT", "australian capital territory": "ACT",
}  # fmt: skip
_STATE_POSTCODE_LEAD = {"NSW": "2", "ACT": "2", "VIC": "3", "QLD": "4", "SA": "5", "WA": "6", "TAS": "7", "NT": "0"}
_ORG_GENERIC = frozenset(
    """pty ltd limited p/l inc co corporation corp group holdings services service solutions lawyers solicitors
    legal partners associates hospital hospitals private public clinic clinics medical centre center health
    healthcare practice surgery rehabilitation rehab physiotherapy physio radiology imaging pathology insurance
    constructions construction building builders logistics transport warehouse supermarket store stores
    school college university council club foundation trust &""".split()
)
_MAX_ATTEMPTS = 64


def canonical_state(text: str | None) -> str | None:
    if not text:
        return None
    return _STATE_CANON.get(text.strip(" ,.").lower())


def apply_case(template: str, value: str) -> str:
    """Give ``value`` the letter case style of ``template`` (UPPER, lower or as-is)."""
    letters = [c for c in template if c.isalpha()]
    if letters and all(c.isupper() for c in letters) and len(letters) > 1:
        return value.upper()
    if letters and all(c.islower() for c in letters):
        return value.lower()
    return value


class SurrogateFactory:
    def __init__(self, chooser: Chooser, forbidden_keys: Iterable[str], pools: Pools | None = None) -> None:
        self._c = chooser
        self._forbidden = {k for k in forbidden_keys if k}
        self._forbidden_sorted = sorted(self._forbidden)
        self._pools = pools or load_pools()
        self._cache: dict[tuple[str, str], str] = {}
        self._issued: set[str] = set()
        self._dates: dict[str, int] = {}

    @property
    def chooser(self) -> Chooser:
        """The scope-keyed DRBG (shared with other keyed transformations such as date shifting)."""
        return self._c

    # ----------------------------------------------------------------------------------- helpers
    def _acceptable(self, candidate: str) -> bool:
        """Not equal to, nor (for >= 5 chars) within one edit of, any original value or component,
        and not already issued for a *different* original (two people must not merge into one).

        Uniqueness makes a choice depend on the (canonical, deterministic) processing order; with
        large pools cross-document consistency within a scope is high but not guaranteed."""
        from mlredact.verify.surrogates import near_any

        key = match_key(candidate)
        return bool(key) and key not in self._issued and not near_any(key, self._forbidden_sorted)

    def _unique(self, domain: str, value_key: str, make: Callable[[int], str]) -> str:
        cache_key = (domain, value_key)
        hit = self._cache.get(cache_key)
        if hit is not None:
            return hit
        for attempt in range(_MAX_ATTEMPTS):
            candidate = make(attempt)
            if self._acceptable(candidate):
                self._cache[cache_key] = candidate
                self._issued.add(match_key(candidate))
                return candidate
        raise RuntimeError("surrogate generation exhausted its attempts")  # pragma: no cover

    def _digits_like(self, domain: str, value_key: str, template: str, attempt: int) -> str:
        """Format-preserving: digits -> digits, letters -> letters (same case), others kept.

        A digit run keeps a non-zero leading digit when the original's was non-zero ("123" never
        becomes "012")."""
        stream = self._c.integers(domain, f"{value_key}\x03{attempt}")
        out = []
        prev_digit = False
        for ch in template:
            if ch.isdigit():
                if not prev_digit and ch != "0":
                    out.append(str(1 + next(stream) % 9))
                else:
                    out.append(str(next(stream) % 10))
                prev_digit = True
                continue
            prev_digit = False
            if ch.isalpha():
                letter = chr(ord("A") + next(stream) % 26)
                out.append(letter if ch.isupper() else letter.lower())
            else:
                out.append(ch)
        return "".join(out)

    # ------------------------------------------------------------------------------------ people
    def _unique_pool(self, domain: str, value_key: str, pool: Sequence[str]) -> str:
        """Pick from a pool: a few keyed random draws, then a keyed deterministic sweep of the whole
        pool (so heavy exclusion can never exhaust the attempts while an acceptable entry exists)."""
        cache_key = (domain, value_key)
        hit = self._cache.get(cache_key)
        if hit is not None:
            return hit
        candidates = [self._c.pick(domain, value_key, pool, a) for a in range(8)]
        start = self._c.index(domain + ":sweep", value_key, len(pool))
        candidates.extend(pool[(start + i) % len(pool)] for i in range(len(pool)))
        for candidate in candidates:
            if self._acceptable(candidate):
                self._cache[cache_key] = candidate
                self._issued.add(match_key(candidate))
                return candidate
        raise RuntimeError("surrogate pool exhausted")

    def given_name(self, key: str, gender: str, template: str) -> str:
        pool = {
            "m": self._pools.given_male,
            "f": self._pools.given_female,
        }.get(gender, self._pools.given_male + self._pools.given_female)
        return apply_case(template, self._unique_pool(f"given:{gender}", key, pool))

    def surname(self, key: str, template: str) -> str:
        return apply_case(template, self._unique_pool("surname", key, self._pools.surnames))

    def initial(self, key: str) -> str:
        return chr(ord("A") + self._c.index("initial", key, 26))

    # ----------------------------------------------------------------------------- identifiers
    def identifier(self, etype: EntityType, value: str) -> str:
        key = match_key(value)
        domain = f"id:{etype.value}"

        def make(attempt: int) -> str:
            s = self._digits_like(domain, key, value, attempt)
            if etype in (EntityType.IHI, EntityType.HPI):
                s = _keep_digit_prefix(value, s, 6)  # 800360/1/2 identifies the identifier kind
            return self._invalidate(etype, s, attempt)

        # The surrogate's characters are fixed per value; each occurrence keeps its own layout
        # ("WC 1234567" and "WC1234567," map to the same characters in their own formats).
        return _project(value, self._unique(domain, key, make))

    def _invalidate(self, etype: EntityType, s: str, attempt: int) -> str:
        """Force the check digit to be wrong for identifier types that have one."""
        d = digits_only(s)
        checks: dict[EntityType, Callable[[str], bool]] = {
            EntityType.MEDICARE: lambda x: medicare_valid(x[:10]),
            EntityType.IHI: ihi_valid,
            EntityType.HPI: ihi_valid,
            EntityType.TFN: tfn_valid,
            EntityType.ABN: abn_valid,
            EntityType.ACN: acn_valid,
            EntityType.CARD_NUMBER: luhn_valid,
        }
        if etype is EntityType.MEDICARE and d and d[0] not in "23456":
            s = s.replace(d[0], str(2 + (int(d[0]) % 5)), 1)
            d = digits_only(s)
        if etype is EntityType.PROVIDER_NUMBER:
            compact = re.sub(r"[^0-9A-Za-z]", "", s).upper()
            while provider_number_valid(compact):
                s = s[:-1] + ("A" if s[-1].upper() != "A" else "B")
                compact = re.sub(r"[^0-9A-Za-z]", "", s).upper()
            return s
        check = checks.get(etype)
        if check is None or not d:
            return s
        # Change the check digit until the identifier no longer validates.  For Medicare the check
        # digit is the 9th digit (the 10th is the issue number and is not part of the checksum).
        positions = [i for i, c in enumerate(s) if c.isdigit()]
        target = positions[8] if etype is EntityType.MEDICARE and len(positions) >= 9 else positions[-1]
        original = int(s[target])
        for bump in range(1, 10):
            if not check(digits_only(s)):
                break
            s = s[:target] + str((original + bump) % 10) + s[target + 1 :]
        if check(digits_only(s)):  # pragma: no cover - every checksum here has a single valid digit
            raise RuntimeError("could not invalidate the check digit")
        return s

    # ---------------------------------------------------------------------------------- contact
    def phone(self, value: str) -> str:
        key = digits_only(value)
        d = key
        if d.startswith("61"):
            d = "0" + d[2:]

        def make(attempt: int) -> str:
            if d.startswith("04") or d.startswith("4"):
                base = self._c.pick("phone.mobile", key, ACMA_MOBILES, attempt)
            elif d.startswith("1800"):
                base = self._c.pick("phone.1800", key, ACMA_FREEPHONE, attempt)
            elif d.startswith("13"):
                base = self._c.pick("phone.1300", key, ACMA_LOCALRATE, attempt)
            else:
                area = d[1] if len(d) >= 10 and d[0] == "0" and d[1] in "2378" else "2"
                prefix = self._c.pick("phone.prefix", key, ACMA_LANDLINE_PREFIXES, attempt)
                tail = self._c.digit_string("phone.tail", key, 4, attempt)
                base = f"0{area} {prefix} {tail}"
            return _reformat_like(value, base)

        return self._unique("phone", key, make)

    def email(self, value: str, local_hint: str | None = None) -> str:
        key = match_key(value)
        original_local = value.split("@", 1)[0].strip()

        def make(attempt: int) -> str:
            domain = self._c.pick("email.domain", key, RESERVED_EMAIL_DOMAINS, attempt)
            if original_local.lower() in GENERIC_EMAIL_LOCALS and not attempt:
                return f"{original_local}@{domain}"  # role mailbox: only the domain identifies
            if local_hint:
                local = local_hint.lower().replace(" ", ".")
            else:
                given = self._c.pick("email.given", key, self._pools.given_male + self._pools.given_female, attempt)
                sur = self._c.pick("email.surname", key, self._pools.surnames, attempt)
                local = f"{given}.{sur}".lower().replace("'", "")
            if attempt:
                local += str(attempt)
            return f"{local}@{domain}"

        return self._unique("email", key, make)

    def url(self, value: str) -> str:
        key = match_key(value)
        return self._unique("url", key, lambda a: f"www.{self._c.pick('url.domain', key, RESERVED_EMAIL_DOMAINS, a)}")

    # ----------------------------------------------------------------------------------- dates
    def date_of_birth(self, value: str) -> str:
        """Same year (ages stay consistent), keyed day/month, original format preserved.

        The choice is keyed by the *parsed* date, so "14/03/1978" and "14 March 1978" receive the
        same surrogate day and month, each in its own format."""
        canon = canonical_date(value) or match_key(value)
        day_of_year = self._dates.get(canon)
        if day_of_year is None:
            for attempt in range(_MAX_ATTEMPTS):
                candidate_doy = self._c.index("dob", canon, 365, attempt)
                if self._acceptable(_shift_day_month(value, candidate_doy)):
                    day_of_year = candidate_doy
                    break
            else:  # pragma: no cover
                raise RuntimeError("surrogate generation exhausted its attempts")
            self._dates[canon] = day_of_year
        result = _shift_day_month(value, day_of_year)
        self._issued.add(match_key(result))
        return result

    # -------------------------------------------------------------------------------- locations
    def locality(self, value: str, state: str | None) -> Locality:
        key = match_key(value)
        pool = self._pools.localities_in(state)
        name = self._unique_pool(f"locality:{state or 'any'}", key, [loc.name for loc in pool])
        return next(loc for loc in pool if loc.name == name)

    def postcode(self, value: str, state: str | None) -> str:
        key = match_key(value)
        lead = _STATE_POSTCODE_LEAD.get(state or "", value[:1] if value[:1].isdigit() else "2")
        return self._unique("postcode", key, lambda a: lead + self._c.digit_string("postcode", key, 3, a))

    def street_name(self, key: str) -> str:
        return self._unique_pool("street", key, self._pools.street_names)

    def number_like(self, domain: str, value: str) -> str:
        key = match_key(value)
        return _project(value, self._unique(domain, key, lambda a: self._digits_like(domain, key, value, a)))

    def organisation(self, value: str) -> str:
        key = match_key(value)
        words = value.split()
        generic_tail: list[str] = []
        for w in reversed(words):
            if w.strip(",.()").lower() in _ORG_GENERIC:
                generic_tail.insert(0, w)
            else:
                break
        if len(generic_tail) == len(words):
            generic_tail = generic_tail[1:]

        def make(attempt: int) -> str:
            stem = self._c.pick("org", key, self._pools.org_stems, attempt)
            stem = apply_case(words[0] if words else stem, stem)
            return " ".join([stem, *generic_tail])

        return self._unique("org", key, make)


GENERIC_EMAIL_LOCALS = frozenset(
    """admin info enquiries enquiry reception office accounts contact contactus mail hello support claims
    referrals bookings appointments records medicalrecords reports legal noreply no-reply""".split()
)


def _project(template: str, surrogate: str) -> str:
    """Write the surrogate's alphanumerics into the template's alphanumeric slots (keeping the
    template's separators, punctuation and letter case)."""
    chars = [c for c in surrogate if c.isalnum()]
    out = []
    k = 0
    for ch in template:
        if ch.isalnum() and k < len(chars):
            c = chars[k]
            k += 1
            out.append(c.upper() if ch.isupper() else c.lower() if ch.islower() else c)
        else:
            out.append(ch)
    return "".join(out)


_NUMERIC_DMY = re.compile(r"^(\d{1,2})[ \t]?[/.\-][ \t]?(\d{1,2})[ \t]?[/.\-][ \t]?(\d{2,4})$")


def canonical_date(value: str) -> str | None:
    """Parse a day-first date into ``YYYY-MM-DD`` (two-digit years pivot at 30), else None."""
    v = value.strip().rstrip(".,")
    day = month = year = None
    m = _NUMERIC_DMY.match(v)
    if m:
        day, month, yr = int(m.group(1)), int(m.group(2)), m.group(3)
        year = int(yr)
        if len(yr) == 2:
            year += 2000 if year < 30 else 1900
    else:
        for word in v.replace(",", " ").split():
            w = word.strip(".").lower()
            ordinal = _ORDINAL_DAY.match(w)
            if ordinal:
                day = int(ordinal.group(1))
            elif w.isdigit() and len(w) <= 2 and day is None:
                day = int(w)
            elif w.isdigit() and len(w) == 4:
                year = int(w)
            elif w[:3] in _MONTH_PREFIXES and w.isalpha():
                month = [mn[:3].lower() for mn in _MONTH_NAMES].index(w[:3]) + 1
    if day is None or month is None or year is None or not (1 <= day <= 31 and 1 <= month <= 12):
        return None
    return f"{year:04d}-{month:02d}-{day:02d}"


def _reformat_like(original: str, digits_spaced: str) -> str:
    """Keep the original's international prefix / brackets style where possible."""
    o = original.strip()
    if o.startswith("+61"):
        d = digits_only(digits_spaced)
        if d.startswith("04") and len(d) == 10:
            return f"+61 {d[1:4]} {d[4:7]} {d[7:]}"
        if d.startswith("0") and len(d) == 10:
            return f"+61 {d[1]} {d[2:6]} {d[6:]}"
        return digits_spaced
    if o.startswith("(") and digits_spaced.startswith("0") and digits_spaced[1:2] in "2378":
        return f"({digits_spaced[:2]}) {digits_spaced[3:]}"
    return digits_spaced


_MONTH_NAMES = (
    "January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
    "November", "December",
)  # fmt: skip
_MONTH_DAYS = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
_NUMERIC_DATE = re.compile(r"^(\d{1,2})([ \t]?[/.\-][ \t]?)(\d{1,2})([ \t]?[/.\-][ \t]?)(\d{2,4})$")


def _day_month(day_of_year: int) -> tuple[int, int]:
    for m, days in enumerate(_MONTH_DAYS, start=1):
        if day_of_year < days:
            return day_of_year + 1, m
        day_of_year -= days
    return 31, 12  # pragma: no cover


def _ordinal(day: int) -> str:
    if 11 <= day % 100 <= 13:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")


_ORDINAL_DAY = re.compile(r"^(\d{1,2})(st|nd|rd|th)$", re.IGNORECASE)
_YEAR = re.compile(r"(?<![0-9])((?:18|19|20)\d{2})(?![0-9])")
_MONTH_PREFIXES = frozenset(mn[:3].lower() for mn in _MONTH_NAMES)


def _shift_day_month(value: str, day_of_year: int) -> str:
    """Replace day and month, keep the year and the original layout.  Never returns ``value``."""
    day, month = _day_month(day_of_year % 365)
    v = value.strip()
    m = _NUMERIC_DATE.match(v)
    if m:
        d_fmt = f"{day:02d}" if len(m.group(1)) == 2 else str(day)
        m_fmt = f"{month:02d}" if len(m.group(3)) == 2 else str(month)
        return f"{d_fmt}{m.group(2)}{m_fmt}{m.group(4)}{m.group(5)}"
    # Textual: replace the day number (incl. ordinals) and the month word, keep everything else.
    out = []
    changed = False
    for w in v.split():
        core = w.strip(",.")
        ordinal = _ORDINAL_DAY.match(core)
        if ordinal:
            out.append(w.replace(core, f"{day}{apply_case(ordinal.group(2), _ordinal(day))}", 1))
            changed = True
        elif core.isdigit() and len(core) <= 2:
            out.append(w.replace(core, str(day), 1))
            changed = True
        elif core[:3].lower() in _MONTH_PREFIXES and core.isalpha():
            name = _MONTH_NAMES[month - 1]
            name = name[: len(core)] if len(core) <= 3 else name
            out.append(w.replace(core, apply_case(core, name), 1))
            changed = True
        else:
            out.append(w)
    shifted = " ".join(out)
    if changed and shifted != v:
        return shifted
    # Unrecognised layout: a plain day-first date that keeps the original year (if any).
    year = _YEAR.search(v)
    return f"{day:02d}/{month:02d}/{year.group(1) if year else '1970'}"


def _keep_digit_prefix(original: str, generated: str, n: int) -> str:
    """Copy the first ``n`` digits of ``original`` into the digit positions of ``generated``."""
    src = [c for c in original if c.isdigit()][:n]
    out = list(generated)
    k = 0
    for i, c in enumerate(out):
        if k >= len(src):
            break
        if c.isdigit():
            out[i] = src[k]
            k += 1
    return "".join(out)


def is_state_token(text: str) -> bool:
    return bool(_STATE_RE.match(text.strip(" ,.")))


def is_street_type(text: str) -> bool:
    return text.strip(" ,.").lower() in STREET_TYPE_WORDS
