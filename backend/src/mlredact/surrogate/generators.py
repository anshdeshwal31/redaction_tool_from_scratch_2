"""Surrogate generators (plan §10).

Every generator is a pure function of (scope key, entity type, canonical original value) and is
*safe by construction*:

* phone numbers come only from ACMA's ranges reserved for fiction;
* e-mail addresses / URLs use RFC 2606 reserved domains;
* identifiers keep their format but **fail** their check digit, so a surrogate can never be a
  real person's identifier;
* file paths keep their shape (drive, separators, folder words, extension) with new names;
* no surrogate equals (by comparison key) any original value in the document.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import TypeVar

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
from mlredact.detect.rules.files import EXTENSIONS
from mlredact.detect.rules.lexicon import STATES, STREET_TYPE_WORDS
from mlredact.surrogate.drbg import Chooser
from mlredact.surrogate.exclusions import NO_EXCLUSIONS, Exclusions
from mlredact.surrogate.pools import Locality, Pools, load_pools
from mlredact.text.address import TYPE_FORMS, canonical_street_type, is_abbreviated_type
from mlredact.text.address.tables import BUILDING_WORDS, PLURAL_TYPES, PROPERTY_WORDS
from mlredact.text.normalize import digits_only, match_key

T = TypeVar("T")

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
ORG_GENERIC = frozenset(
    """pty ltd limited p/l inc co corporation corp group holdings services service solutions lawyers solicitors
    legal partners associates hospital hospitals private public clinic clinics medical centre center health
    healthcare practice surgery rehabilitation rehab physiotherapy physio radiology imaging pathology insurance
    constructions construction building builders logistics transport warehouse supermarket store stores
    school college university council club foundation trust &""".split()
)
# Folder and file-name words that identify no one: kept in file-path surrogates.
_PATH_GENERIC = frozenset(
    """users user documents document docs desktop downloads onedrive sharepoint sites shared share shares data
    home mnt volumes appdata local temp tmp program files file folder clients client matters matter cases case
    claims claim records correspondence letters letter reports report medical legal medicolegal scans scan
    scanned images image photos photo attachments email emails outlook archive admin misc general final draft
    copy signed redacted version new old notes invoice invoices statement statements affidavit brief court
    hearing notice damages plaintiff defendant transcript training independent impairment assessment ime dpi
    psychiatrist psychiatric psychology psychologist dr of and the re v for to""".split()
)
_PATH_SPLIT = re.compile(r"([\\/ _\-.()]+)")
_PATH_EXTENSION = re.compile(rf"(?i)^(?:{EXTENSIONS})$")
_DRIVE = re.compile(r"^[A-Za-z]:$")
# Years and day/month numbers in file names are dates, which the policies keep.
_PATH_DATE_PART = re.compile(r"^(?:(?:19|20)\d\d|\d{1,2})$")
_MAX_ATTEMPTS = 64
#: An alphabetic original key this long or longer must not occur inside a person-name fake (V2's ``min_len``).
NAME_CONTAINMENT_MIN = 4
_ADDRESS_ATTEMPTS = 48  # keyed draws of an address number before the first one is used regardless
_STRICT_ATTEMPTS = 24  # of them, the draws that must keep a range's digit counts and gap
_STRICT, _RELAXED, _FORCED = 0, 1, 2  # how a range is faked (see _fake_number)
STATE_NAMES = {
    "NSW": "New South Wales",
    "VIC": "Victoria",
    "QLD": "Queensland",
    "SA": "South Australia",
    "WA": "Western Australia",
    "TAS": "Tasmania",
    "NT": "Northern Territory",
    "ACT": "Australian Capital Territory",
}
_STATE_NAME_WORDS = frozenset(n.lower() for n in STATE_NAMES.values())
_TITLE_CASE_CODES = frozenset({"VIC", "QLD", "TAS"})  # the codes written in mixed case; NSW, SA, WA, NT, ACT never are
# Directionals as (code, abbreviation, word), the three styles of a street-type suffix.
_DIRECTIONS = {
    "N": ("N", "Nth", "North"),
    "S": ("S", "Sth", "South"),
    "E": ("E", "E", "East"),
    "W": ("W", "W", "West"),
}
_DIRECTION_OF = {
    "n": "N", "nth": "N", "north": "N", "s": "S", "sth": "S", "south": "S",
    "e": "E", "east": "E", "w": "W", "west": "W",
}  # fmt: skip
# Canonical street types a fake may take (the residential and common ones), and the ones an untyped street gets.
_FAKE_TYPES = (
    "STREET", "ROAD", "AVENUE", "DRIVE", "COURT", "PLACE", "CRESCENT", "LANE", "WAY", "PARADE", "TERRACE",
    "BOULEVARD", "CLOSE", "CIRCUIT", "GROVE", "SQUARE", "ESPLANADE", "RISE", "MEWS", "WALK", "LOOP", "LINK",
    "VIEW", "GLADE", "GARDENS", "HEIGHTS", "RIDGE", "PROMENADE", "QUAY", "COVE", "CREST", "GREEN", "HILL",
    "VALE", "VISTA", "CHASE",
)  # fmt: skip
_UNCOMMON_ABBREVIATIONS = frozenset({"RIDGE", "QUAY", "CREST", "GREEN", "VISTA"})  # written in full instead
_COMMON_TYPES = ("STREET", "ROAD", "AVENUE", "DRIVE", "COURT", "PLACE", "CRESCENT", "LANE")
_BUILDING_FAKE = ("Tower", "House", "Chambers", "Centre", "Building", "Complex", "Arcade", "Plaza")
_PROPERTY_FAKE = ("Station", "Farm", "Homestead", "Estate", "Vineyard", "Orchard", "Downs", "Park")
_DIGIT_RUN = re.compile(r"\d+")
_LEVEL = re.compile(r"(\d{1,2})(st|nd|rd|th)?", re.IGNORECASE)
_NUMBER_RANGE = re.compile(r"^(\d+)([A-Za-z]?)([-–—])(\d+)([A-Za-z]?)$")
_NUMBER_ITEM = re.compile(r"\d+|(?<![A-Za-z])[A-Za-z]{1,2}(?![A-Za-z])")


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
        # A person-name fake must not contain an original word: V2 reads a removed value of 4+ letters inside any
        # word as a leak, and the benchmark grades an original name inside its in-place fake as surviving.
        self._contained = sorted(k for k in self._forbidden if k.isalpha() and len(k) >= NAME_CONTAINMENT_MIN)
        self._name_parts: set[str] = set()
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
    def add_name_parts(self, keys: Iterable[str]) -> None:
        """Register the given-name and surname keys of the document's people: no person-name fake contains one of
        3 or more letters ('Ann' never becomes 'Joanne', 'Ray' never 'Murray')."""
        self._name_parts.update(k for k in keys if k.isalpha() and len(k) >= 3)

    def _name_ok(self, candidate: str) -> bool:
        """A person-name candidate contains no alphabetic original key of 4+ letters (V2's rule) and no registered
        name part of the document's people."""
        key = match_key(candidate)
        return self.not_containing(candidate) and not any(k in key for k in self._name_parts)

    def not_containing(self, text: str) -> bool:
        """True when ``text`` contains no alphabetic original key of 4+ letters: V2 reads such a key inside any word as
        a leak ('Hill' in 'Hillsdale', 'Orange' in 'Orangeville')."""
        key = match_key(text)
        return not any(k in key for k in self._contained)

    def _unique_pool(self, domain: str, value_key: str, pool: Sequence[str], *, names: bool = False) -> str:
        """Pick from a pool: a few keyed random draws, then a keyed deterministic sweep of the whole
        pool (so heavy exclusion can never exhaust the attempts while an acceptable entry exists).
        ``names``: the pool holds person names, which also must not contain an original word (:meth:`_name_ok`)."""
        cache_key = (domain, value_key)
        hit = self._cache.get(cache_key)
        if hit is not None:
            return hit
        candidates = [self._c.pick(domain, value_key, pool, a) for a in range(8)]
        start = self._c.index(domain + ":sweep", value_key, len(pool))
        candidates.extend(pool[(start + i) % len(pool)] for i in range(len(pool)))
        for candidate in candidates:
            if self._acceptable(candidate) and (not names or self._name_ok(candidate)):
                self._cache[cache_key] = candidate
                self._issued.add(match_key(candidate))
                return candidate
        raise RuntimeError("surrogate pool exhausted")

    def given_name(self, key: str, gender: str, template: str) -> str:
        pool = {
            "m": self._pools.given_male,
            "f": self._pools.given_female,
        }.get(gender, self._pools.given_male + self._pools.given_female)
        return apply_case(template, self._unique_pool(f"given:{gender}", key, pool, names=True))

    def surname(self, key: str, template: str) -> str:
        return apply_case(template, self._unique_pool("surname", key, self._pools.surnames, names=True))

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

    def file_path(self, value: str) -> str:
        """Same shape: drive letter, separators, extension and generic folder words stay; other words
        become pool surnames, and digits and short codes become other digits and letters."""
        key = match_key(value)
        parts = _PATH_SPLIT.split(value)  # even indices: words; odd: separators
        words = [i for i in range(0, len(parts), 2) if parts[i]]
        fixed = {0} if _DRIVE.match(parts[0]) else set()
        if len(words) > 1 and parts[words[-1] - 1].endswith(".") and _PATH_EXTENSION.match(parts[words[-1]]):
            fixed.add(words[-1])
        changed = [
            i
            for i in words
            if i not in fixed and parts[i].lower() not in _PATH_GENERIC and not _PATH_DATE_PART.match(parts[i])
        ]
        if not changed:  # nothing identifying: still never equal to the original
            changed = [i for i in words if i not in fixed][-1:] or words[-1:]
        # One changeable character ("scan 3.pdf") cannot move the result far enough from the
        # original: that part becomes a pool word instead.
        renamed = set(changed) if sum(len(parts[i]) for i in changed) < 2 else set()

        def make(attempt: int) -> str:
            out = list(parts)
            for i in changed:
                word, slot = parts[i], f"{key}\x03{i}"
                if (word.isalpha() and len(word) >= 3) or i in renamed:
                    out[i] = apply_case(word, self._c.pick("file.word", slot, self._pools.surnames, attempt))
                else:
                    out[i] = self._digits_like("file", slot, word, attempt)
            return "".join(out)

        return self._unique("file", key, make)

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
    def not_original(self, text: str) -> bool:
        """True when neither ``text`` nor any of its words equals (or, from 5 letters, is one edit from) an original
        value of the document: the check V5 makes of every replaced address word."""
        from mlredact.verify.surrogates import near_any

        for part in (text, *text.split()):
            key = match_key(part)
            if key and near_any(key, self._forbidden_sorted):
                return False
        return True

    def _pick(self, domain: str, key: str, pool: Sequence[T], accept: Callable[[T], bool]) -> T | None:
        """The first acceptable entry of a keyed order of ``pool``: a few keyed random draws, then a keyed sweep of the
        whole pool.  The first draw does not depend on ``accept``, so a filter that rejects other entries leaves the
        choice as it was (one address keeps its fake across the documents of a matter).  No uniqueness across
        originals: None only when nothing is acceptable."""
        n = len(pool)
        if not n:
            return None
        for attempt in range(8):
            candidate = pool[self._c.index(domain, key, n, attempt)]
            if accept(candidate):
                return candidate
        start = self._c.index(domain + ":sweep", key, n)
        for i in range(n):
            candidate = pool[(start + i) % n]
            if accept(candidate):
                return candidate
        return None

    def locality(
        self,
        value: str,
        state: str | None,
        klass: str | None = None,
        *,
        key: str | None = None,
        accept: Callable[[Locality], bool] | None = None,
    ) -> Locality:
        """A fake locality of ``state`` and ``klass`` ('metro' or 'regional'; None for any) from the suburb list's
        ``fake_pool`` rows, chosen from ``key`` (default: the original ``value``).  ``accept`` filters rows (the fake
        rules); the original values of the document, and names that contain one, are never offered.  An empty choice
        widens (the class, then the state) and never raises."""
        k = key if key is not None else match_key(value)

        def ok(loc: Locality) -> bool:
            return self.not_original(loc.name) and self.not_containing(loc.name) and (accept is None or accept(loc))

        for st, kl in ((state, klass), (state, None), (None, None)):
            hit = self._pick(f"locality:{st or 'any'}:{kl or 'any'}", k, self._pools.localities_in(st, kl), ok)
            if hit is not None:
                return hit
        pool = self._pools.localities_in(state) or self._pools.localities_in()
        return pool[self._c.index("locality:last", k, len(pool))]

    def postcode(self, value: str, state: str | None) -> str:
        key = match_key(value)
        lead = _STATE_POSTCODE_LEAD.get(state or "", value[:1] if value[:1].isdigit() else "2")
        return self._unique("postcode", key, lambda a: lead + self._c.digit_string("postcode", key, 3, a))

    def street_name(self, key: str, accept: Callable[[str], bool] | None = None) -> str:
        """A street name from the pool, chosen from ``key``: no uniqueness across originals, none that contains an
        original word, and never an error."""
        pool = self._pools.street_names

        def ok(w: str) -> bool:
            return self.not_original(w) and self.not_containing(w) and (accept is None or accept(w))

        hit = self._pick("street", key, pool, ok)
        return hit if hit is not None else pool[self._c.index("street:last", key, len(pool))]

    def number_like(self, domain: str, value: str) -> str:
        key = match_key(value)
        return _project(value, self._unique(domain, key, lambda a: self._digits_like(domain, key, value, a)))

    # ------------------------------------------------------------------------------- address parts
    # The parts of a whole-address fake (FINAL_PLAN TOOL2 section 5, amendment D).  Every part is chosen from a key the
    # caller builds from the address's canonical key under the scope key, never from an original part alone.  None of
    # them uses the per-document uniqueness pool, and none raises when a filter leaves nothing: it widens.
    def street_type(self, original: str | None, key: str, ex: Exclusions | None = None) -> str:
        """Another canonical street type, written in the original's style: an abbreviation stays an abbreviation, a
        plural stays plural, the case is the original's.  An untyped street (``original`` None) gets a common type
        written in full, or any fake type when the document's own types leave no common one (amendment D1)."""
        ex = ex or NO_EXCLUSIONS
        old = canonical_street_type(original) if original else None
        pool = _FAKE_TYPES if original else _COMMON_TYPES

        def ok(canon: str) -> bool:
            forms = TYPE_FORMS[canon]
            return canon != old and all(ex.word_ok(f) for f in forms) and self.not_original(forms[0])

        canon = self._pick("addr.type", key, pool, ok)
        if canon is None and not original:
            canon = self._pick("addr.type:wide", key, _FAKE_TYPES, ok)
        canon = canon or self._pick("addr.type:any", key, pool, lambda c: c != old)
        canon = canon or pool[self._c.index("addr.type:last", key, len(pool))]
        forms = TYPE_FORMS[canon]
        written = forms[0]
        if original and is_abbreviated_type(original) and canon not in _UNCOMMON_ABBREVIATIONS:
            written = next((f for f in forms if is_abbreviated_type(f)), forms[0])
        if original and original.strip(" ,.;:()").lower() in PLURAL_TYPES and not written.endswith("s"):
            written += "es" if written.endswith(("x", "ch")) else "s"
        return apply_case(original.strip(" ,.;:()"), written.title()) if original else written.title()

    def street_suffix(self, original: str, key: str, ex: Exclusions | None = None) -> str:
        """Another directional in the original's style: a code ('N'), an abbreviation ('Nth') or the word ('North')."""
        ex = ex or NO_EXCLUSIONS
        core = original.strip(" ,.;:()")
        old = _DIRECTION_OF.get(core.lower())
        style = 0 if len(core) <= 2 else 1 if core.lower() in ("nth", "sth") else 2
        codes = tuple(_DIRECTIONS)

        def ok(code: str) -> bool:
            words = _DIRECTIONS[code]
            return code != old and all(ex.word_ok(w) for w in words) and self.not_original(words[2])

        code = self._pick("addr.suffix", key, codes, ok)
        code = code or self._pick("addr.suffix:any", key, codes, lambda c: c != old)
        written = _DIRECTIONS[code or "N"][style]
        return apply_case(core, written.title() if style else written)

    def building_name(
        self, words: Sequence[str], key: str, ex: Exclusions | None = None, used: set[str] | None = None
    ) -> list[str]:
        """The words of a building or property name ('Westpoint Tower'): each name word becomes a distinct pool word
        and a building word ('Tower', 'House', 'Station', 'Farm') becomes another, in the original's case.  ``used``
        holds the (lower-case) words the unit's fake already has; the new ones are added to it."""
        ex = ex or NO_EXCLUSIONS
        names = self._pools.street_names
        out: list[str] = []
        chosen = used if used is not None else set()
        for i, word in enumerate(words):
            low = word.lower()
            slot = f"{key}\x1f{i}"
            if i == len(words) - 1 and len(words) > 1 and low in BUILDING_WORDS | PROPERTY_WORDS:
                lexicon = _PROPERTY_FAKE if low in PROPERTY_WORDS else _BUILDING_FAKE
                new = (
                    self._pick(
                        "addr.building.word",
                        slot,
                        lexicon,
                        lambda w, low=low: w.lower() != low and ex.word_ok(w) and self.not_original(w),  # type: ignore[misc]
                    )
                    or lexicon[self._c.index("addr.building.word:last", slot, len(lexicon))]
                )
            else:
                new = (
                    self._pick(
                        "addr.building.name",
                        slot,
                        names,
                        lambda w: w.lower() not in chosen and ex.word_ok(w) and self.not_original(w),
                    )
                    or names[self._c.index("addr.building.name:last", slot, len(names))]
                )
            chosen.add(new.lower())
            out.append(apply_case(word, new))
        return out

    def pool_word(self, domain: str, key: str, template: str, ex: Exclusions | None = None) -> str:
        """A pool word for an identifying word with no other rule (a box extra, a town, an unread word), in the case
        of ``template``."""
        ex = ex or NO_EXCLUSIONS
        return apply_case(template, self.street_name(f"{domain}\x1f{key}", ex.word_ok))

    def address_number(self, value: str, key: str, ex: Exclusions | None = None) -> str:
        """A house, unit, box or DX number of the same shape: the digit count kept, every digit different at its
        position and a non-zero lead, a letter suffix kept as a letter suffix but another letter, separators kept.  A
        range ('12-14') stays ascending with the original gap; when no fake keeps both digit counts and the gap
        ('8-10', '98-104'), each end keeps its own digit count (every digit still differs) and the range stays
        ascending."""
        ex = ex or NO_EXCLUSIONS
        first: str | None = None
        for attempt in range(_ADDRESS_ATTEMPTS):
            stream = self._c.integers("addr.number", f"{key}\x03{attempt}")
            candidate = _fake_number(value, stream, _STRICT if attempt < _STRICT_ATTEMPTS else _RELAXED)
            if candidate is None:
                continue
            first = first or candidate
            if ex.number_ok(candidate) and all(self.not_original(r) for r in _DIGIT_RUN.findall(candidate)):
                return candidate
        if first is None:  # a range that no keyed draw could fake: ascending is all that is left to keep
            first = _fake_number(value, self._c.integers("addr.number", f"{key}\x03force"), _FORCED)
        return first or value

    def level_number(self, value: str, key: str, ex: Exclusions | None = None) -> str:
        """Another realistic level (1-5; 1-25 when nothing in 1-5 qualifies) with no digit equal to the original's at
        its position ('Level 2' never becomes 20-25), an ordinal kept as an ordinal ('2nd' becomes '4th')."""
        ex = ex or NO_EXCLUSIONS
        m = _LEVEL.fullmatch(value)
        if m is None:
            return self.address_number(value, key, ex)
        digits, suffix = m.group(1), m.group(2)

        def ok(n: int) -> bool:
            text = str(n)
            same = any(a == b for a, b in zip(text, digits, strict=False))
            return n != int(digits) and not same and ex.number_ok(text) and self.not_original(text)

        pick = self._pick("addr.level:5", key, range(1, 6), ok)
        if pick is None:
            pick = self._pick("addr.level:25", key, range(1, 26), ok)
        if pick is None:
            return self.address_number(value, key, ex)
        return f"{pick}{apply_case(suffix, _ordinal(pick))}" if suffix else str(pick)

    def overseas_postcode(self, value: str, key: str, ex: Exclusions | None = None) -> str:
        """An overseas postcode of the same shape: every character changes, digit for digit and letter for letter, in
        the original's case ('OX1 4AA' becomes, say, 'KR7 2TB')."""
        ex = ex or NO_EXCLUSIONS
        first: str | None = None
        for attempt in range(_ADDRESS_ATTEMPTS):
            stream = self._c.integers("addr.postcode.overseas", f"{key}\x03{attempt}")
            out = []
            for i, ch in enumerate(value):
                if ch.isdigit():
                    out.append(_other_digit(ch, next(stream), nonzero=i == 0))
                elif ch.isalpha():
                    out.append(_other_letter(ch, next(stream)))
                else:
                    out.append(ch)
            candidate = "".join(out)
            first = first or candidate
            if ex.number_ok(candidate):
                return candidate
        return first or value

    @staticmethod
    def state_form(original: str, code: str) -> str:
        """The state ``code`` written in the style of ``original``: a spelled-out name ('New South Wales'), a dotted
        form ('N.S.W.'), a mixed-case abbreviation ('Vic.') or upper case; the case and a final dot of the original
        are kept."""
        core = original.strip(" ,;:()")
        dot = "." if core.endswith(".") else ""
        body = core.rstrip(".")
        if "." in body:
            dotted = ".".join(code)
            return (dotted.lower() if body.islower() else dotted) + dot
        if " " in body or len(body) > 4 or body.lower() in _STATE_NAME_WORDS:
            name = STATE_NAMES[code]
            return (name.upper() if body.isupper() else name.lower() if body.islower() else name) + dot
        if body.isupper() or not body.isalpha():
            return code + dot
        if body.islower():
            return code.lower() + dot
        return (code.title() if code in _TITLE_CASE_CODES else code) + dot  # 'Vic', 'Qld', 'Tas'; never 'Nsw'

    def organisation(self, value: str) -> str:
        key = match_key(value)
        words = value.split()
        generic_tail: list[str] = []
        for w in reversed(words):
            if w.strip(",.()").lower() in ORG_GENERIC:
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


def _other_digit(ch: str, draw: int, *, nonzero: bool = False) -> str:
    """A digit that is not ``ch`` (and not 0 when ``nonzero``), chosen by the integer ``draw``."""
    options = [c for c in "0123456789" if c != ch and not (nonzero and c == "0")]
    return options[draw % len(options)]


def _other_letter(ch: str, draw: int) -> str:
    """A letter of the case of ``ch`` that is not ``ch``, chosen by the integer ``draw``."""
    base = "A" if ch.isupper() else "a"
    options = [c for c in (chr(ord(base) + i) for i in range(26)) if c.lower() != ch.lower()]
    return options[draw % len(options)]


def _other_digits(digits: str, stream: Iterator[int]) -> str:
    return "".join(_other_digit(ch, next(stream), nonzero=i == 0) for i, ch in enumerate(digits))


def _fake_number(value: str, stream: Iterator[int], mode: int) -> str | None:
    """One keyed candidate for an address number: every digit different at its position (a non-zero lead), a letter
    suffix another letter.  A range '12-14' is faked together: ``_STRICT`` keeps both digit counts and the gap (None
    when this draw cannot), ``_RELAXED`` lets each end keep its own digit count and the range stays ascending (None when
    this draw does not), ``_FORCED`` keeps only the gap.  Everything else is faked item by item."""
    r = _NUMBER_RANGE.match(value)
    if r is None:
        return _NUMBER_ITEM.sub(
            lambda m: (
                _other_digits(m.group(), stream) if m.group().isdigit() else _other_letter(m.group(), next(stream))
            ),
            value,
        )
    a, sa, dash, b, sb = r.groups()
    fa = _other_digits(a, stream)
    gap = int(b) - int(a)
    if mode == _FORCED:
        fb = str(int(fa) + max(gap, 1))
    elif mode == _STRICT:
        fb = str(int(fa) + gap)
        if gap <= 0 or len(a) != len(b) or len(fb) != len(b) or any(x == y for x, y in zip(fb, b, strict=True)):
            return None
    else:
        fb = _other_digits(b, stream)
        if gap > 0 and int(fb) <= int(fa):
            return None
    la = _other_letter(sa, next(stream)) if sa else ""
    lb = _other_letter(sb, next(stream)) if sb else ""
    return f"{fa}{la}{dash}{fb}{lb}"
