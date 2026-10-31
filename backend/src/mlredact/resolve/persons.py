"""Person-name parsing, clustering and component-level identity (plan §8, §9).

Each PERSON mention is parsed into tokens with roles (given / surname / initial / particle).
Clustering is deterministic and rule-based:

* full names seed clusters keyed by (surname, first given name);
* "J. Smith" attaches to the unique same-surname cluster whose given name starts with J;
* surname-only ("Mr Smith") and given-only ("John") mentions attach to the unique compatible
  cluster, otherwise to the surname/given *group* only.

Surrogates are assigned per *component* (surname key, given key), so every variant of a name is
replaced consistently and family members share a surrogate surname.  A wrong merge can only make
surrogates less consistent; it never causes a miss, because every mention is still redacted.
"""

from __future__ import annotations

import csv
import hashlib
import io
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from functools import cache

from mlredact.core.errors import EnvironmentErrorMl, ReasonCode
from mlredact.core.model import Candidate, Evidence, Mention, Token
from mlredact.core.types import EntityType, EvidenceStrength, ViewKind
from mlredact.detect.rules.lexicon import CLINICAL_NOUNS, MEDICAL_ABBREVIATIONS, NON_NAME_WORDS, PARTICLE, TITLES
from mlredact.resources.loader import _parse_sums, resource_root
from mlredact.surrogate.pools import load_pools
from mlredact.text.normalize import match_key
from mlredact.text.views import TextView


class Role(StrEnum):
    GIVEN = "given"
    SURNAME = "surname"
    INITIAL = "initial"
    PARTICLE = "particle"
    OTHER = "other"


class Gender(StrEnum):
    MALE = "m"
    FEMALE = "f"
    UNKNOWN = "u"


@dataclass(frozen=True, slots=True)
class NameToken:
    prefix: str = field(repr=False)
    core: str = field(repr=False)
    suffix: str = field(repr=False)
    role: Role

    @property
    def key(self) -> str:
        return match_key(self.core)


@dataclass(frozen=True, slots=True)
class ParsedMention:
    mention_id: str
    tokens: tuple[NameToken, ...]
    title: str | None = field(repr=False)
    gender: Gender  # what the document states (title, kinship word), else the name list: its fakes keep it
    cluster: str | None  # cluster id, if resolved to an individual


@dataclass(frozen=True, slots=True)
class PersonResolution:
    mentions: dict[str, ParsedMention]
    given_gender: dict[str, Gender]  # given-name key -> gender used for its surrogate
    clusters: dict[str, tuple[str, ...]]  # cluster id -> sorted mention ids


_TITLE_RE = re.compile(rf"(?i)^{TITLES}\.?$")
_PARTICLE_RE = re.compile(rf"^{PARTICLE}$")
_INITIALS_RE = re.compile(r"^(?:[A-Z]\.?){1,3}$")
_POSSESSIVE = re.compile(r"(['’]s|['’])$")
_EDGE = ',;:()[]{}"“”'
_MALE_TITLES = {"mr", "master", "mstr", "sir", "fr", "father", "br", "brother"}
_FEMALE_TITLES = {"mrs", "ms", "miss", "dame", "sr", "sister"}
# Kinship words that state the gender of the person they name, when written right next to the name.
_MALE_ROLES = frozenset(
    """husband father son brother stepfather stepson stepbrother grandfather grandson uncle nephew boyfriend widower
    fiance""".split()
)
_FEMALE_ROLES = frozenset(
    """wife mother daughter sister stepmother stepdaughter stepsister grandmother granddaughter aunt niece girlfriend
    widow fiancee""".split()
)
# Pronoun counts after a name: a rough cue that only steers which cluster a partial mention joins (as before).  It is
# not used for the gender of a fake given name: nothing here can tell reliably whom a "he" or "she" refers to.
_HE = re.compile(r"(?i)\b(he|him|his|himself)\b")
_SHE = re.compile(r"(?i)\b(she|her|hers|herself)\b")
_ROLE_GENDER = {**dict.fromkeys(_MALE_ROLES, "m"), **dict.fromkeys(_FEMALE_ROLES, "f")}
_DETERMINERS = frozenset("his her their my our your the".split())
_CONJUNCTIONS = frozenset({"and", "or", "&"})
_LIST_WORDS = _CONJUNCTIONS | _DETERMINERS | frozenset(_ROLE_GENDER)  # what starts the next item of a list
_NAMES_DIR = "names"


def split_token(text: str) -> tuple[str, str, str]:
    """``'(Smith's,'`` -> ``('(', 'Smith', "'s,")``."""
    lead = len(text) - len(text.lstrip(_EDGE))
    prefix, rest = text[:lead], text[lead:]
    core = rest.rstrip(_EDGE)
    suffix = rest[len(core) :]
    m = _POSSESSIVE.search(core)
    if m and len(core) > len(m.group(0)) + 1:
        suffix = m.group(0) + suffix
        core = core[: m.start()]
    if core.endswith(".") and not _INITIALS_RE.fullmatch(core):
        suffix = "." + suffix
        core = core[:-1]
    return prefix, core, suffix


def parse_name(raw_tokens: Sequence[str], *, title: str | None = None) -> list[NameToken]:
    parts = [split_token(t) for t in raw_tokens]
    roles: list[Role] = [Role.OTHER] * len(parts)
    word_idx: list[int] = []
    for i, (_p, core, _s) in enumerate(parts):
        if not core:
            continue
        if _TITLE_RE.fullmatch(core) and not word_idx:
            continue
        if _INITIALS_RE.fullmatch(core) and (
            core.endswith(".") or len(core) == 1 or (len(parts) > 1 and len(core) <= 2)
        ):
            roles[i] = Role.INITIAL
        elif _PARTICLE_RE.fullmatch(core):
            roles[i] = Role.PARTICLE
        else:
            word_idx.append(i)
    if word_idx:
        first = word_idx[0]
        inverted = "," in parts[first][2] and len(word_idx) > 1
        caps = [i for i in word_idx if parts[i][1].isupper() and len(parts[i][1]) > 1]
        mixed = [i for i in word_idx if not (parts[i][1].isupper() and len(parts[i][1]) > 1)]
        if inverted:
            surname_idx = {first}
        elif caps and mixed:
            surname_idx = set(caps)
        elif len(word_idx) == 1:
            # A lone word: a surname after a title ("Mr Smith") or next to initials ("J. Smith"),
            # otherwise a given name ("John").
            has_initials = any(r is Role.INITIAL for r in roles)
            surname_idx = {first} if (title or has_initials) else set()
        else:
            surname_idx = {word_idx[-1]}
        for i in word_idx:
            roles[i] = Role.SURNAME if i in surname_idx else Role.GIVEN
        # Particles directly before a surname belong to it ("van der Berg": right to left).
        for i in range(len(roles) - 2, -1, -1):
            if roles[i] is Role.PARTICLE and roles[i + 1] is Role.SURNAME:
                roles[i] = Role.SURNAME
    return [NameToken(p, c, s, r) for (p, c, s), r in zip(parts, roles, strict=True)]


def _preceding_title(view: TextView, first_token: int) -> str | None:
    if first_token == 0:
        return None
    prev, cur = view.tokens[first_token - 1], view.tokens[first_token]
    if (prev.page, prev.line_no) != (cur.page, cur.line_no):
        return None
    core = prev.text.strip(_EDGE)
    return core if _TITLE_RE.fullmatch(core) else None


def _title_gender(title: str | None) -> Gender:
    if title:
        t = title.lower().rstrip(".")
        if t in _MALE_TITLES:
            return Gender.MALE
        if t in _FEMALE_TITLES:
            return Gender.FEMALE
    return Gender.UNKNOWN


def _cluster_cue(title: str | None, view: TextView, last_token: int) -> Gender:
    """The gender cue that steers clustering (unchanged): the title, else the pronouns in the 240 characters after
    the name (the surrogate pools' names are added in :func:`resolve_persons`)."""
    gender = _title_gender(title)
    if gender is not Gender.UNKNOWN:
        return gender
    end = view.ends[last_token]
    window = view.text[end : end + 240]
    he, she = len(_HE.findall(window)), len(_SHE.findall(window))
    if he > she:
        return Gender.MALE
    if she > he:
        return Gender.FEMALE
    return Gender.UNKNOWN


def _kinship(view: TextView, first_token: int, last_token: int) -> Gender:
    """The gender a kinship word written right next to the name states ("her husband Tom Smithers", "Tom Smithers,
    her husband,", "Tom Smithers (husband)"), or UNKNOWN.  Pronouns are not read (see ``_HE``)."""
    role = _role_before(view, first_token) or _role_after(view, last_token, first_token)
    return Gender(role) if role else Gender.UNKNOWN


def _same_line(a: Token, b: Token) -> bool:
    return (a.page, a.line_no) == (b.page, b.line_no)


def _neighbour(view: TextView, i: int, of: int) -> tuple[str, str, str] | None:
    """The split token ``i`` when it exists and is on the line of token ``of``, else None."""
    tokens = view.tokens
    if 0 <= i < len(tokens) and _same_line(tokens[i], tokens[of]):
        return split_token(tokens[i].text)
    return None


def _role_before(view: TextView, first: int) -> str | None:
    """'m' or 'f' for a kinship word directly before the name, or before its ungendered title ("her husband Dr Tom
    Smithers").  A kinship word in a list of them ("son and daughter, Tom and Jane", "son and daughter Tom and
    Jane", "his son, daughter, Tom ...") names no one in particular."""
    tokens = view.tokens
    i = first - 1
    if i >= 0 and _same_line(tokens[i], tokens[first]) and _TITLE_RE.fullmatch(tokens[i].text.strip(_EDGE)):
        i -= 1
    if i < 0 or not _same_line(tokens[i], tokens[first]):
        return None
    _p, core, suffix = split_token(tokens[i].text)
    role = _ROLE_GENDER.get(core.lower())
    if role is None or suffix.startswith(("'", "’")):  # "her husband's employer Tom" is not the husband
        return None
    if "." in suffix or ";" in suffix:  # "... with her son. Tom Bloggs said": another sentence
        return None
    prev = _neighbour(view, i - 1, i)
    if prev is not None:
        word = prev[1].lower()
        if word in _ROLE_GENDER and "," in prev[2]:  # "son, daughter, Tom"
            return None
        if word in _CONJUNCTIONS:
            if "," in suffix:  # "son and daughter, Tom and Jane"
                return None
            before = _neighbour(view, i - 2, i)
            if before is not None and before[1].lower() in _ROLE_GENDER:  # "son and daughter Tom and Jane"
                return None
    return role


def _role_after(view: TextView, last: int, first: int | None = None) -> str | None:
    """'m' or 'f' for an appositive kinship word after the name: "Tom Smithers (husband)" or "Tom Smithers, her
    husband," with the word ending the phrase (so "Jane Smithers, her husband Tom ..." says nothing about Jane).
    The comma form is not read in a list of people: after a label ("Attendees: Jane Smithers, her husband, ..."),
    after another item ("Pat Doe, Jane Smithers, her husband"), or when the kinship word is itself followed by
    another item ("Jane Smithers, her husband, and Dr Lou Brown", "Robyn Quist, her son, her daughter")."""
    tokens = view.tokens
    j = last + 1
    if j >= len(tokens) or not _same_line(tokens[j], tokens[last]):
        return None
    _lp, _lcore, last_suffix = split_token(tokens[last].text)
    prefix, core, suffix = split_token(tokens[j].text)
    if prefix == "(" and suffix.startswith(")"):
        return _ROLE_GENDER.get(core.lower())
    if prefix or not last_suffix.startswith(","):
        return None
    if core.lower() in _DETERMINERS or suffix in ("'s", "’s", "'", "’"):
        j += 1
        if j >= len(tokens) or not _same_line(tokens[j], tokens[last]):
            return None
        prefix, core, suffix = split_token(tokens[j].text)
    role = _ROLE_GENDER.get(core.lower())
    if role is None or prefix:
        return None
    ends_phrase = (
        suffix[:1] in (",", ".", ";", ":", ")") or j + 1 >= len(tokens) or not _same_line(tokens[j + 1], tokens[j])
    )
    if not ends_phrase:
        return None
    after = _neighbour(view, j + 1, j)
    if after is not None and not suffix.startswith(".") and after[1].lower() in _LIST_WORDS:
        return None  # "her husband, and Dr Lou Brown" / "her son, her daughter"
    if first is not None:
        start = first - 1
        title = _neighbour(view, start, first)
        if title is not None and _TITLE_RE.fullmatch(title[1]):
            start -= 1
        before = _neighbour(view, start, first)
        if before is not None and (before[2].endswith((",", ":")) or before[1].lower() in _CONJUNCTIONS):
            return None  # "Attendees: Jane Smithers, her husband," / "Pat Doe, Jane Smithers, her husband,"
    return role


@cache
def _listed_genders() -> dict[str, Gender]:
    """The given-name gender list (``resources/names/given_gender.csv``, NSW popular baby names 1952-2025, CC BY 4.0),
    verified against its ``SHA256SUMS``: comparison key -> MALE, FEMALE or UNKNOWN (a name used for both)."""
    root = resource_root() / _NAMES_DIR
    try:
        sums = _parse_sums((root / "SHA256SUMS").read_text("utf-8"))
        data = (root / "given_gender.csv").read_bytes()
    except OSError as exc:  # the folder or a file is missing: the same error as a changed file
        raise EnvironmentErrorMl(ReasonCode.RESOURCE_HASH_MISMATCH) from exc
    if hashlib.sha256(data).hexdigest() != sums.get("given_gender.csv"):
        raise EnvironmentErrorMl(ReasonCode.RESOURCE_HASH_MISMATCH)
    return {r["name"]: Gender(r["gender"]) for r in csv.DictReader(io.StringIO(data.decode("utf-8")))}


@cache
def _pool_names() -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    """The comparison keys of the male, female and any-gender surrogate pools."""
    pools = load_pools()
    male, female, neutral = (
        frozenset(match_key(n) for n in p) for p in (pools.given_male, pools.given_female, pools.given_neutral)
    )
    return male, female, neutral


def _key_gender(key: str) -> Gender:
    male, female, neutral = _pool_names()
    if key in male and key not in female:
        return Gender.MALE
    if key in female and key not in male:
        return Gender.FEMALE
    if key in male or key in neutral:  # in both gendered pools, or a name for any gender
        return Gender.UNKNOWN
    return _listed_genders().get(key, Gender.UNKNOWN)


def _plain_key(text: str) -> str:
    """The comparison key without accents ("Zoë" -> "zoe"): the name lists hold plain letters."""
    decomposed = unicodedata.normalize("NFKD", match_key(text))
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def name_gender(given: str) -> Gender:
    """The gender a given name usually has: the surrogate pools first (as before), then the NSW list; UNKNOWN for a
    name of either gender or one in neither.  A hyphenated name ("Kellie-Ann") takes its parts' gender when every
    part has the same one."""
    gender = _key_gender(_plain_key(given))
    if gender is Gender.UNKNOWN and "-" in given:
        parts = {_key_gender(_plain_key(p)) for p in given.split("-") if _plain_key(p)}
        if len(parts) == 1:
            gender = parts.pop()
    return gender


def resolve_persons(mentions: Sequence[Mention], view: TextView) -> PersonResolution:
    index = {t.token_id: i for i, t in enumerate(view.tokens)}
    parsed: dict[str, tuple[list[NameToken], str | None, Gender]] = {}  # gender: the clustering cue
    stated: dict[str, Gender] = {}  # the gender each mention's fake given names take
    kinship: dict[str, Gender] = {}  # a kinship word's gender, for a mention without a gendered title
    for m in mentions:
        if m.entity_type is not EntityType.PERSON:
            continue
        first, last = index[m.token_ids[0]], index[m.token_ids[-1]]
        title = _preceding_title(view, first)
        tokens = parse_name([view.tokens[i].text for i in range(first, last + 1)], title=title)
        parsed[m.mention_id] = (tokens, title, _cluster_cue(title, view, last))
        stated[m.mention_id] = _title_gender(title)
        if stated[m.mention_id] is Gender.UNKNOWN:
            kinship[m.mention_id] = _kinship(view, first, last)

    def keys(tokens: list[NameToken], role: Role) -> list[str]:
        return [t.key for t in tokens if t.role is role and t.key]

    # A lone word ("Smith" found by propagation) takes the role its key has in full names.
    known_surnames = {
        k for toks, _t, _g in parsed.values() if len(keys(toks, Role.GIVEN)) for k in keys(toks, Role.SURNAME)
    }
    known_givens = {
        k for toks, _t, _g in parsed.values() if len(keys(toks, Role.SURNAME)) for k in keys(toks, Role.GIVEN)
    }
    male_names, female_names, _neutral = _pool_names()
    for mid, (toks, title, gender) in list(parsed.items()):
        words = [t for t in toks if t.role in (Role.GIVEN, Role.SURNAME)]
        if len(words) == 1 and words[0].role is Role.GIVEN:
            k = words[0].key
            if k in known_surnames and k not in known_givens:
                toks = [NameToken(t.prefix, t.core, t.suffix, Role.SURNAME) if t is words[0] else t for t in toks]
        given = [t for t in toks if t.role is Role.GIVEN and t.key]
        if gender is Gender.UNKNOWN and given:
            if given[0].key in male_names and given[0].key not in female_names:
                gender = Gender.MALE
            elif given[0].key in female_names and given[0].key not in male_names:
                gender = Gender.FEMALE
        if stated[mid] is Gender.UNKNOWN:
            # The name list first: a kinship word only decides a name the list does not gender, so a misread
            # kinship word (one in a list of people, say) never gives a listed name the other sex.
            listed = name_gender(given[0].core) if given else Gender.UNKNOWN
            stated[mid] = listed if listed is not Gender.UNKNOWN else kinship.get(mid, Gender.UNKNOWN)
        parsed[mid] = (toks, title, gender)

    # 1. Full names seed clusters keyed by (surname, first given).
    clusters: dict[tuple[str, str], list[str]] = defaultdict(list)
    for mid in sorted(parsed):
        tokens = parsed[mid][0]
        sur, giv = keys(tokens, Role.SURNAME), keys(tokens, Role.GIVEN)
        if sur and giv:
            clusters[("".join(sur), giv[0])].append(mid)
    by_surname: dict[str, list[tuple[str, str]]] = defaultdict(list)
    by_given: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for ck in sorted(clusters):
        by_surname[ck[0]].append(ck)
        by_given[ck[1]].append(ck)

    assignment: dict[str, tuple[str, str] | None] = {}
    for ck, mids in clusters.items():
        for mid in mids:
            assignment[mid] = ck
    # 2. Partial mentions attach to a unique compatible cluster.
    for mid in sorted(parsed):
        if mid in assignment:
            continue
        tokens, _title, gender = parsed[mid]
        sur, giv = keys(tokens, Role.SURNAME), keys(tokens, Role.GIVEN)
        initials = [t.core[0].lower() for t in tokens if t.role is Role.INITIAL]
        candidates: list[tuple[str, str]] = []
        if sur:
            candidates = list(by_surname.get("".join(sur), []))
            if initials:
                candidates = [c for c in candidates if c[1].startswith(initials[0])]
        elif giv:
            candidates = list(by_given.get(giv[0], []))
        if gender is not Gender.UNKNOWN and len(candidates) > 1:
            compatible = [c for c in candidates if _cluster_gender(clusters[c], parsed) in (gender, Gender.UNKNOWN)]
            candidates = compatible or candidates
        assignment[mid] = candidates[0] if len(candidates) == 1 else None

    cluster_ids = {ck: f"P{i:03d}" for i, ck in enumerate(sorted(clusters))}
    members: dict[str, list[str]] = defaultdict(list)
    out: dict[str, ParsedMention] = {}
    for mid in sorted(parsed):
        tokens, title, _cue = parsed[mid]
        assigned = assignment.get(mid)
        cid = cluster_ids[assigned] if assigned else None
        if cid:
            members[cid].append(mid)
        out[mid] = ParsedMention(mid, tuple(tokens), title, stated[mid], cid)
    # Gender per cluster (majority of the stated genders), propagated to its members' given names.
    given_votes: dict[str, Counter[Gender]] = defaultdict(Counter)
    for mids in members.values():
        g = _majority([out[member].gender for member in mids])
        for member in mids:
            for t in out[member].tokens:
                if t.role is Role.GIVEN:
                    given_votes[t.key][g] += 1
    for pm in out.values():
        if pm.cluster is None:
            for t in pm.tokens:
                if t.role is Role.GIVEN:
                    given_votes[t.key][pm.gender] += 1
    given_gender = {k: _majority(list(v.elements())) for k, v in sorted(given_votes.items())}
    return PersonResolution(out, given_gender, {c: tuple(sorted(v)) for c, v in sorted(members.items())})


def with_clusters(resolution: PersonResolution, clusters: Mapping[str, str]) -> PersonResolution:
    """``resolution`` with given clusters instead of the resolver's own (the benchmark's injected clusters):
    ``clusters`` maps a person mention id to its cluster id; a mention it does not name belongs to no cluster.
    Names, roles and each mention's own gender cue are kept, and the gender of each given name is voted again as
    :func:`resolve_persons` votes it, over the given clusters."""
    out = {mid: replace(pm, cluster=clusters.get(mid) or None) for mid, pm in sorted(resolution.mentions.items())}
    members: dict[str, list[str]] = defaultdict(list)
    for mid, pm in out.items():
        if pm.cluster:
            members[pm.cluster].append(mid)
    given_votes: dict[str, Counter[Gender]] = defaultdict(Counter)
    for mids in members.values():
        g = _majority([out[member].gender for member in mids])
        for member in mids:
            for t in out[member].tokens:
                if t.role is Role.GIVEN:
                    given_votes[t.key][g] += 1
    for pm in out.values():
        if pm.cluster is None:
            for t in pm.tokens:
                if t.role is Role.GIVEN:
                    given_votes[t.key][pm.gender] += 1
    given_gender = {k: _majority(list(v.elements())) for k, v in sorted(given_votes.items())}
    return PersonResolution(out, given_gender, {c: tuple(sorted(v)) for c, v in sorted(members.items())})


def _cluster_gender(mids: Sequence[str], parsed: Mapping[str, tuple[list[NameToken], str | None, Gender]]) -> Gender:
    return _majority([parsed[m][2] for m in mids])


def _majority(genders: Sequence[Gender]) -> Gender:
    counts = Counter(g for g in genders if g is not Gender.UNKNOWN)
    if not counts:
        return Gender.UNKNOWN
    (top, n), *rest = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0].value))
    if rest and rest[0][1] == n:
        return Gender.UNKNOWN
    return top


# ---------------------------------------------------------------------------------- propagation
def _is_eponym_context(view: TextView, i: int, eponyms: frozenset[str]) -> bool:
    tok = view.tokens[i]
    _p, core, _suffix = split_token(tok.text)
    if core.lower() not in eponyms:
        return False
    if i + 1 >= len(view.tokens):
        return False
    nxt = view.tokens[i + 1]
    if (nxt.page, nxt.line_no) != (tok.page, tok.line_no):
        return False
    return nxt.text.strip(_EDGE + ".").lower() in CLINICAL_NOUNS


def component_candidates(resolution: PersonResolution, view: TextView, min_len: int) -> list[Candidate]:
    """Find standalone occurrences of known surnames / given names (capitalised or ALL CAPS)."""
    surnames: set[str] = set()
    givens: set[str] = set()
    for pm in resolution.mentions.values():
        for t in pm.tokens:
            if len(t.key) < min_len or t.core.lower() in NON_NAME_WORDS:
                continue
            if t.role is Role.SURNAME:
                surnames.add(t.key)
            elif t.role is Role.GIVEN:
                givens.add(t.key)
    eponyms = load_pools().eponyms
    out: list[Candidate] = []
    tokens = view.tokens
    for i, tok in enumerate(tokens):
        _p, core, _s = split_token(tok.text)
        if not core or not core[0].isupper():
            continue
        key = match_key(core)
        role = "surname" if key in surnames else "given" if key in givens else None
        if role is None or _is_eponym_context(view, i, eponyms):
            continue
        start = i
        if role == "surname":
            # Absorb adjacent initials / known given names on the same line: "J. Smith", "John S. Smith".
            while start > 0:
                prev = tokens[start - 1]
                if (prev.page, prev.line_no) != (tok.page, tok.line_no):
                    break
                _pp, prev_core, _ps = split_token(prev.text)
                bare = prev_core.rstrip(".").lower()
                if _TITLE_RE.fullmatch(prev_core) or bare in MEDICAL_ABBREVIATIONS:
                    break  # "DR SMITH", "GP Smith": titles and abbreviations stay visible
                if _INITIALS_RE.fullmatch(prev_core) or match_key(prev_core) in givens:
                    start -= 1
                else:
                    break
        out.append(
            Candidate(
                entity_type=EntityType.PERSON,
                view=ViewKind.LINE,
                start=view.starts[start],
                end=view.ends[i],
                evidence=Evidence("resolve.persons", "1", f"component.{role}", EvidenceStrength.STRONG, 1.0),
            )
        )
    return out


def tokens_of(mention: Mention, index: Mapping[str, Token]) -> list[Token]:
    return [index[t] for t in mention.token_ids]
