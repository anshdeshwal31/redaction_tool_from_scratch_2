"""Person roles for the claimant-focused profile (plan §7.1, §8).

Only one decision is made here: which resolved people are *professionals* whose names may stay
visible (treating clinicians, lawyers, case managers).  The rule is deliberately one-sided, because
a wrong "professional" is a leak while a wrong "subject" only costs utility.  A cluster is
professional only if

* at least one of its mentions carries explicit professional evidence: a Dr/Prof title, an adjacent
  post-nominal ("FRACS"), or a professional role word right next to the name ("Peter Brown,
  Orthopaedic Surgeon"; "treating surgeon, Dr Brown") - a role word elsewhere on the line is not
  enough ("Mr Smith was examined by his treating surgeon");
* none of its mentions carries any subject evidence on its line: "Re:", patient/claimant/worker
  labels, kinship words, a date of birth, a home address;
* its surname is not shared with any non-professional cluster (family members share surnames).

Unclustered mentions are never professional.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping, Sequence

from mlredact.core.model import Mention
from mlredact.core.types import EntityType
from mlredact.detect.rules.lexicon import POST_NOMINALS
from mlredact.resolve.persons import PersonResolution, Role, split_token
from mlredact.text.views import TextView

_PROFESSIONAL_TITLES = frozenset({"dr", "doctor", "prof", "professor", "a/prof", "assoc", "associate"})
_PROFESSIONAL_WORDS = re.compile(
    r"(?i)\b(surgeon|orthopaedic|orthopedic|psychiatrist|psychologist|physiotherapist|physio|practitioner|"
    r"radiologist|neurologist|neurosurgeon|rheumatologist|anaesthetist|physician|specialist|consultant|"
    r"registrar|occupational[ \t]+therapist|nurse|clinician|solicitor|lawyer|barrister|counsel|paralegal|"
    r"case[ \t]+manager|claims?[ \t]+(?:officer|manager|consultant)|rehabilitation[ \t]+(?:provider|consultant)|"
    r"assessor|examiner|treating)\b"
)
_SUBJECT_WORDS = re.compile(
    r"(?i)(\bre:|\bre\b[ \t]*-|\bpatient\b|\bclaimant\b|\bworker\b|\binjured\b|\bapplicant\b|\bplaintiff\b|"
    r"\bdeceased\b|\bclient\b|\bour[ \t]+client\b|\bwife\b|\bhusband\b|\bpartner\b|\bspouse\b|\bson\b|"
    r"\bdaughter\b|\bmother\b|\bfather\b|\bbrother\b|\bsister\b|\bchild(?:ren)?\b|\bde[ \t]+facto\b|"
    r"\bfianc[eé]e?\b|\bgrand(?:son|daughter|mother|father|child)\b|\bnephew\b|\bniece\b|\bfriend\b|"
    r"\bwitness\b|\bd\.?o\.?b\b|date[ \t]+of[ \t]+birth|\bborn\b|\baddress\b|\bresides\b|\blives\b)"
)
_SUBJECT_TYPES = frozenset({EntityType.DATE_OF_BIRTH, EntityType.STREET_ADDRESS, EntityType.MEDICARE, EntityType.IHI})


_AFTER, _BEFORE = 4, 3


def _line_text(view: TextView, page: int, line_no: int) -> str:
    return " ".join(t.text for t in view.tokens if t.page == page and t.line_no == line_no)


def _adjacent_text(view: TextView, first: int, last: int) -> tuple[str, str]:
    """Same-line text just before and just after the token range ``first..last``."""
    key = (view.tokens[first].page, view.tokens[first].line_no)
    before = [t.text for t in view.tokens[max(0, first - _BEFORE) : first] if (t.page, t.line_no) == key]
    after = [t.text for t in view.tokens[last + 1 : last + 1 + _AFTER] if (t.page, t.line_no) == key]
    return " ".join(before), " ".join(after)


def professional_mentions(resolution: PersonResolution, mentions: Sequence[Mention], view: TextView) -> frozenset[str]:
    """Mention ids of people who are professionals by the conservative rule above."""
    index = {t.token_id: i for i, t in enumerate(view.tokens)}
    lines: dict[tuple[int, int], str] = {}
    subject_lines = {
        (view.tokens[index[m.token_ids[0]]].page, view.tokens[index[m.token_ids[0]]].line_no)
        for m in mentions
        if m.entity_type in _SUBJECT_TYPES
    }
    by_id: Mapping[str, Mention] = {m.mention_id: m for m in mentions}
    professional: dict[str, bool] = defaultdict(bool)
    subject: dict[str, bool] = defaultdict(bool)
    for mid, pm in resolution.mentions.items():
        if pm.cluster is None or mid not in by_id:
            continue
        m = by_id[mid]
        first, last = index[m.token_ids[0]], index[m.token_ids[-1]]
        key = (view.tokens[first].page, view.tokens[first].line_no)
        line = lines.setdefault(key, _line_text(view, *key))
        title = (pm.title or "").lower().rstrip(".")
        before, after = _adjacent_text(view, first, last)
        next_word = split_token(after.split()[0])[1].lower() if after else ""
        if (
            title in _PROFESSIONAL_TITLES
            or next_word in POST_NOMINALS
            or _PROFESSIONAL_WORDS.search(before)
            or _PROFESSIONAL_WORDS.search(after)
        ):
            professional[pm.cluster] = True
        if _SUBJECT_WORDS.search(line) or key in subject_lines:
            subject[pm.cluster] = True
    surnames: dict[str, set[str]] = defaultdict(set)  # surname key -> clusters
    for pm in resolution.mentions.values():
        if pm.cluster is not None:
            for t in pm.tokens:
                if t.role is Role.SURNAME:
                    surnames[t.key].add(pm.cluster)
    candidates = {c for c, is_pro in professional.items() if is_pro and not subject[c]}
    family = {c for clusters in surnames.values() if clusters - candidates for c in clusters}
    keep = candidates - family
    return frozenset(mid for mid, pm in resolution.mentions.items() if pm.cluster in keep)
