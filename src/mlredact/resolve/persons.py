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

import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from mlredact.core.model import Candidate, Evidence, Mention, Token
from mlredact.core.types import EntityType, EvidenceStrength, ViewKind
from mlredact.detect.rules.lexicon import CLINICAL_NOUNS, MEDICAL_ABBREVIATIONS, NON_NAME_WORDS, PARTICLE, TITLES
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
    gender: Gender
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
_HE = re.compile(r"(?i)\b(he|him|his|himself)\b")
_SHE = re.compile(r"(?i)\b(she|her|hers|herself)\b")


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


def _gender(title: str | None, view: TextView, last_token: int) -> Gender:
    if title:
        t = title.lower().rstrip(".")
        if t in _MALE_TITLES:
            return Gender.MALE
        if t in _FEMALE_TITLES:
            return Gender.FEMALE
    end = view.ends[last_token]
    window = view.text[end : end + 240]
    he, she = len(_HE.findall(window)), len(_SHE.findall(window))
    if he > she:
        return Gender.MALE
    if she > he:
        return Gender.FEMALE
    return Gender.UNKNOWN


def resolve_persons(mentions: Sequence[Mention], view: TextView) -> PersonResolution:
    index = {t.token_id: i for i, t in enumerate(view.tokens)}
    parsed: dict[str, tuple[list[NameToken], str | None, Gender]] = {}
    for m in mentions:
        if m.entity_type is not EntityType.PERSON:
            continue
        first, last = index[m.token_ids[0]], index[m.token_ids[-1]]
        title = _preceding_title(view, first)
        tokens = parse_name([view.tokens[i].text for i in range(first, last + 1)], title=title)
        parsed[m.mention_id] = (tokens, title, _gender(title, view, last))

    def keys(tokens: list[NameToken], role: Role) -> list[str]:
        return [t.key for t in tokens if t.role is role and t.key]

    # A lone word ("Smith" found by propagation) takes the role its key has in full names.
    known_surnames = {
        k for toks, _t, _g in parsed.values() if len(keys(toks, Role.GIVEN)) for k in keys(toks, Role.SURNAME)
    }
    known_givens = {
        k for toks, _t, _g in parsed.values() if len(keys(toks, Role.SURNAME)) for k in keys(toks, Role.GIVEN)
    }
    pools = load_pools()
    male_names = {match_key(n) for n in pools.given_male}
    female_names = {match_key(n) for n in pools.given_female}
    for mid, (toks, title, gender) in list(parsed.items()):
        words = [t for t in toks if t.role in (Role.GIVEN, Role.SURNAME)]
        if len(words) == 1 and words[0].role is Role.GIVEN:
            k = words[0].key
            if k in known_surnames and k not in known_givens:
                toks = [NameToken(t.prefix, t.core, t.suffix, Role.SURNAME) if t is words[0] else t for t in toks]
        if gender is Gender.UNKNOWN:
            given = keys(toks, Role.GIVEN)
            if given and given[0] in male_names and given[0] not in female_names:
                gender = Gender.MALE
            elif given and given[0] in female_names and given[0] not in male_names:
                gender = Gender.FEMALE
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
        tokens, title, gender = parsed[mid]
        assigned = assignment.get(mid)
        cid = cluster_ids[assigned] if assigned else None
        if cid:
            members[cid].append(mid)
        out[mid] = ParsedMention(mid, tuple(tokens), title, gender, cid)
    # Gender per cluster (majority of explicit cues), propagated to its members' given names.
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
