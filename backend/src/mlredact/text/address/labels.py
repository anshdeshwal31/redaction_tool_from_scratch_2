"""Labels: the form and field labels that precede an address, the cue before a word, and the place-kind cue.

A label is read from a lexicon (``tables.LABELS``) and from compound labels ('Unit/building no.',
'Suburb/town/locality') whose pieces all come from one address-part family.  Colon-less labels count only before a
value of the right shape.
"""

from __future__ import annotations

import re
from typing import Final

from mlredact.text.address.internal import Label, LabelRole
from mlredact.text.address.model import BoxRole
from mlredact.text.address.state import ParseState
from mlredact.text.address.tables import (
    BOXNUM,
    LABEL_FAMILY,
    LABEL_MODS,
    LABEL_QUALS,
    LABELS,
    NUMBERING,
    OTHER_LABELS,
    PHRASE_CUES,
    PLACE_KIND_CUES,
    RESIDENCE_CUES,
)
from mlredact.text.address.words import Line, Word, pc_value, state_at

_UNREADABLE: Final = re.compile(r"[A-Za-z][#@0-9$][A-Za-z]|[#@]")
_LETTERS: Final = re.compile(r"[^a-z]")
_NON_LETTER_SPACE: Final = re.compile(r"[^a-z ]")

_KIND, _NONE, _BREAK = 0, 1, 2  # outcomes of scanning back from a word for a place-kind cue


class LabelStage(ParseState):
    # ------------------------------------------------------------------ labels
    def role_of(self, ws: list[Word]) -> LabelRole | None:
        """The label role of a short phrase: the lexicon, then compound labels ('Suburb/town/locality')."""
        if not ws or len(ws) > 5:
            return None
        key = tuple(w.lkey for w in ws)
        hit = LABELS.get(key)
        if hit is not None:
            return LabelRole(hit)
        if key in OTHER_LABELS:
            return LabelRole.OTHER
        fams: set[str] = set()
        mods: set[str] = set()
        pieces = [x for k in key for x in k.split("/") if x]
        if "post" in pieces and "code" in pieces:
            pieces = [x for x in pieces if x not in ("post", "code")] + ["postcode"]
        for x in pieces:
            x = x.strip(".,:;()")
            if x in LABEL_FAMILY:
                fams.add(LABEL_FAMILY[x])
            elif x in LABEL_MODS:
                mods.add(LABEL_MODS[x])
            elif x in LABEL_QUALS or not x:
                continue
            else:
                return None
        if not fams:
            return None
        if "address" in fams:
            last = pieces[-1].strip(".,:;()")
            if LABEL_FAMILY.get(last) not in (None, "address"):
                return None
            return LabelRole.ADDRESS
        if fams == {"street"}:
            return (
                LabelRole.STREET_NAME if "name" in mods else (LabelRole.STREET_NO if "no" in mods else LabelRole.STREET)
            )
        if fams == {"unit"}:
            return LabelRole.UNIT
        if len(fams) == 1:
            return LabelRole(next(iter(fams)))
        return None

    def labels_in(self, ln: Line) -> list[Label]:
        """The label spans on a line."""
        ws = ln.words
        out: list[Label] = []
        if not ws:
            return out
        lead = 1 if len(ws) > 1 and NUMBERING.fullmatch(ws[0].text.strip().lower()) else 0
        whole = self.role_of(ws[lead:])
        if ln.box is BoxRole.LABEL or (len(ws) - lead <= 5 and whole is not None):
            role = whole
            if role is None:
                txt = " ".join(w.text for w in ws)
                role = LabelRole.UNREADABLE if _UNREADABLE.search(txt) else LabelRole.OTHER
            out.append(Label(0, len(ws) - 1, role, ws[-1].label, True))
            return out
        # colon labels anywhere; colon-less lexicon labels before a value of the right shape
        i = lead
        prev_end = lead - 1
        n = len(ws)
        while i < n:
            found: tuple[int, int, LabelRole, bool] | None = None
            for j in range(min(n - 1, i + 3), i - 1, -1):
                role = self.role_of(ws[i : j + 1])
                if role is None:
                    continue
                if ws[j].label:
                    found = (i, j, role, True)
                    break
                if ws[j].desig and j + 1 < n and BOXNUM.fullmatch(ws[j + 1].b):
                    continue  # 'Suite 3', 'Level 2': a designator pair, not (the end of) a colon-less label
                if (
                    j + 1 < n
                    and role is not LabelRole.OTHER
                    and self.value_shape(role, ws[j + 1 :])
                    and not (i > 0 and not (ws[i - 1].label or ws[i - 1].sep or i - 1 == prev_end))
                ):
                    found = (i, j, role, False)
                    break
            if found is None and ws[i].label and not any(c.isdigit() for c in ws[i].b):
                # a colon phrase not in the lexicon: an 'other' label (back to 4 words, no digits)
                st = i
                while st - 1 > prev_end and i - st < 3:
                    q = ws[st - 1]
                    if q.sep or q.digits or q.comma or q.stop or q.label:
                        break
                    st -= 1
                found = (st, i, self.role_of(ws[st : i + 1]) or LabelRole.OTHER, True)
            if found is not None:
                st, j, role, colon = found
                if st == lead and lead:
                    st = 0
                out.append(Label(st, j, role, colon, False))
                prev_end = j
                i = j + 1
            else:
                i += 1
        return out

    def value_shape(self, role: LabelRole, rest: list[Word]) -> bool:
        """Whether the words after a colon-less label look like a value of that label's kind."""
        if not rest:
            return False
        w = rest[0]
        if role is LabelRole.STATE:
            for n in (1, 2, 3):
                st = state_at(rest, n - 1)
                if st is not None and st[0] == 0:
                    return True
            return False
        if role is LabelRole.POSTCODE:
            return pc_value(w.b) is not None
        if role is LabelRole.STREET_NO:
            return w.numw
        if role is LabelRole.UNIT:
            return (
                w.numw
                or bool(BOXNUM.fullmatch(w.b))
                or (w.desig and len(rest) > 1 and bool(BOXNUM.fullmatch(rest[1].b)))
            )
        if role is LabelRole.SUBURB:
            run: list[str] = []
            for x in rest:
                if not x.name or x.stopw or len(run) >= 4:
                    break
                run.append(x.b)
                if x.comma:
                    break
            return bool(run) and bool(self.gaz.states_of(run))
        if role is LabelRole.STREET_NAME:
            return w.name
        if role in (LabelRole.ADDRESS, LabelRole.STREET):
            return w.numw or w.name or w.desig
        return False

    def label_ending_before(self, li: int, i: int) -> Label | None:
        for lab in self.labels[li]:
            if lab.end == i - 1:
                return lab
        return None

    def address_cue_before(self, li: int, start: int) -> str | None:
        """The cue before a word: an 'address_label' (an address or street label right before it), another 'label', a
        residence 'phrase' ('lives at'), or None."""
        ws = self.L[li].words
        lab = self.label_ending_before(li, start)
        if lab is not None:
            return "address_label" if lab.role in (LabelRole.ADDRESS, LabelRole.STREET) else "label"
        if start >= 2 and ws[start - 1].lo in ("at", "in") and ws[start - 2].lo in PHRASE_CUES:
            return "phrase"
        # a label box or label cell to the left on the same row
        if start == 0:
            ln = self.L[li]
            for o in self.geo.row_mates(ln):
                if o.x1 > ln.x0 + 1:
                    continue
                labs = self.labels[o.li]
                if labs and labs[-1].end == len(o.words) - 1 and labs[-1].whole:
                    return "address_label" if labs[-1].role in (LabelRole.ADDRESS, LabelRole.STREET) else "label"
        return None

    # ------------------------------------------------------------------ place-kind cue (owner ruling, question 2)
    def place_kind(self, ln: Line, start: int) -> bool:
        """A place right after a label or phrase that names another kind of place (birth, injury, accident, incident,
        declaration) stays kept.  A residence word after the cue cancels it."""
        scan = self._cue_scans.get(ln.li)
        if scan is None:
            scan = self._cue_scan(ln)
            self._cue_scans[ln.li] = scan
        got = scan[start] if start < len(scan) else _BREAK
        if got == _KIND:
            return True
        if got == _NONE:
            return False
        return self._cell_cue(ln)

    def _cue_scan(self, ln: Line) -> list[int]:
        """For each word, the outcome of scanning backwards from it for a place-kind cue, as the grammar defines it:
        the word before is checked for a cue; every earlier word first for a sentence stop."""
        ws = ln.words
        n = len(ws)
        letters = [_LETTERS.sub("", w.text.lower()) for w in ws]
        kind = [lo in PLACE_KIND_CUES for lo in letters]
        resid = [lo in RESIDENCE_CUES for lo in letters]
        stop = [w.sbreak and not w.label for w in ws]
        # g[k]: the outcome of scanning from word k downwards (k is not the word right before the start)
        g = [_BREAK] * (n + 1)  # g[k + 1] holds the outcome for word k; g[0] is word -1
        for k in range(n):
            if stop[k]:
                g[k + 1] = _BREAK
            elif kind[k]:
                g[k + 1] = _KIND
            elif resid[k]:
                g[k + 1] = _NONE
            else:
                g[k + 1] = g[k]
        out = [_BREAK] * n
        for start in range(n):
            k = start - 1
            if k < 0 or stop[k]:
                out[start] = _BREAK  # the sentence ended right before the place
            elif kind[k]:
                out[start] = _KIND
            elif resid[k]:
                out[start] = _NONE
            else:
                out[start] = g[k]  # words k - 1 and below: g[(k - 1) + 1]
        return out

    def _cell_cue(self, ln: Line) -> bool:
        """The label cell to the left on the same row, or a label line directly above, names another kind of place."""
        got = self._cell_cues.get(ln.li)
        if got is not None:
            return got
        cells = [o for o in self.geo.row_mates(ln) if o.x1 <= ln.x0 + 1]
        up = self.geo.above(ln)
        result = False
        for o in sorted(cells, key=lambda o: -o.x1) + ([up] if up is not None else []):
            if o.words[-1].sbreak:
                continue  # a finished sentence is no label ('She was injured on Victoria Road.')
            txt = _NON_LETTER_SPACE.sub(" ", o.text.lower()).split()
            if any(t in PLACE_KIND_CUES for t in txt) and len(txt) <= 6:
                result = True
                break
            if any(t in RESIDENCE_CUES for t in txt) and len(txt) <= 6:
                break
        self._cell_cues[ln.li] = result
        return result
