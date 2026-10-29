"""Places, read from the end: the anchors of a line (country, postcode, state), the suburb that goes with each, and
where a street line meets its place on one line or at the start of the line below."""

from __future__ import annotations

from collections.abc import Set as AbstractSet
from dataclasses import dataclass

from mlredact.text.address.internal import Anchor, AnchorKind, Flag, Street
from mlredact.text.address.labels import LabelStage
from mlredact.text.address.model import Pos, Role, UnitKind
from mlredact.text.address.tables import (
    COUNTRY,
    GLUED,
    OTHER_COUNTRIES,
    POSTCODE_CA_IN,
    POSTCODE_CA_OUT,
    POSTCODE_ONE_WORD,
    POSTCODE_UK_IN,
    POSTCODE_UK_OUT,
)
from mlredact.text.address.words import Line, Word, pc_value, state_at
from mlredact.text.gazetteer import range_states

_PlaceRoles = dict[Pos, Role]


@dataclass(slots=True)
class LineStart:
    """A place that starts a line: its roles, the line holding the country alone below it (if any), and whether it is
    an overseas place."""

    roles: _PlaceRoles
    country_line: Line | None
    overseas: bool = False


def foreign_country_at(ws: list[Word], k: int) -> int | None:
    """The last word of a foreign country name starting at word ``k`` (title or upper case), or None."""
    for m in (2, 1):
        run = ws[k : k + m]
        if len(run) == m and tuple(w.lo for w in run) in OTHER_COUNTRIES and all(w.cap for w in run):
            return k + m - 1
    return None


def postcode_before(ws: list[Word], p: int) -> tuple[int, int] | None:
    """The overseas postcode ending at word ``p``: (first word, last word).  A four- to six-digit word, or a UK or
    Canadian code of two words ('OX1 4AA', 'K1A 0B1')."""
    if p < 0:
        return None
    b = ws[p].b
    if POSTCODE_ONE_WORD.fullmatch(b):
        return p, p
    if p >= 1:
        out_code = ws[p - 1].b
        if (POSTCODE_UK_OUT.fullmatch(out_code) and POSTCODE_UK_IN.fullmatch(b)) or (
            POSTCODE_CA_OUT.fullmatch(out_code) and POSTCODE_CA_IN.fullmatch(b)
        ):
            return p - 1, p
    return None


class PlaceStage(LabelStage):
    # ------------------------------------------------------------------ anchors
    def anchors_in(self, ln: Line) -> list[Anchor]:
        ws = ln.words
        out: list[Anchor] = []
        n = len(ws)
        # overseas places: a postcode shape right before a foreign country (owner ruling on overseas addresses)
        claimed: set[int] = set()
        for k in range(n):
            last = foreign_country_at(ws, k)
            if last is None or k == 0:
                continue
            pcspan = postcode_before(ws, k - 1)
            if pcspan is None or state_at(ws, pcspan[0] - 1, ocr=True) is not None:
                continue
            out.append(Anchor(ln.li, AnchorKind.OVERSEAS, None, pcspan[1], False, None, None, (k, last), pcspan[0]))
            claimed.add(pcspan[1])
        for k, w in enumerate(ws):
            g = GLUED.fullmatch(w.b)
            if g:
                gcode, gpc = g.group(1), int(g.group(2))
                if self.gaz.ok_pair(gcode, gpc):
                    out.append(Anchor(ln.li, AnchorKind.STRONG, (k, k), k, True, gcode, gpc))
                continue
            pcv = pc_value(w.b)
            if pcv is not None and not w.text.strip().startswith("$"):
                st = state_at(ws, k - 1, ocr=True)
                if st is not None:
                    s0, code = st
                    if s0 == k - 1 and s0 >= 1 and ws[s0 - 1].b == ws[s0].b:
                        s0 -= 1  # the state written twice ('Sydney NSW NSW 2000'): both are the state
                    if self.gaz.ok_pair(code, pcv):
                        out.append(Anchor(ln.li, AnchorKind.STRONG, (s0, k - 1), k, False, code, pcv))
                    else:
                        self.notes.append(f"range-gate:{ln.li}:{k}")
                    continue
                if w.b.isascii() and len(w.b) == 4 and w.b.isdigit() and k not in claimed:
                    out.append(Anchor(ln.li, AnchorKind.STATELESS, None, k, False, None, pcv))
                continue
            # weak anchor: a state form with no postcode after it, ending a phrase
            st = state_at(ws, k)
            if st is not None:
                s0, code = st
                nxt = ws[k + 1] if k + 1 < n else None
                if nxt is not None and (pc_value(nxt.b) is not None or GLUED.fullmatch(nxt.b)):
                    continue
                if nxt is None or nxt.sep or nxt.lowr or nxt.label or nxt.lo in COUNTRY or w.stop or w.comma:
                    out.append(Anchor(ln.li, AnchorKind.WEAK, (s0, k), None, False, code, None))
        # an overseas postcode that ends a line, its country on the line below (checked at the join)
        if n >= 2 and not any(a.kind is AnchorKind.OVERSEAS for a in out):
            span = postcode_before(ws, n - 1)
            if (
                span is not None
                and span[0] >= 1
                and (span[0] != span[1] or len(ws[n - 1].b) >= 5)
                and ws[span[0] - 1].name
                and state_at(ws, span[0] - 1, ocr=True) is None
            ):
                out.append(Anchor(ln.li, AnchorKind.OVERSEAS, None, span[1], False, None, None, None, span[0]))
        for a in out:
            if a.kind is AnchorKind.OVERSEAS:
                continue
            end = a.pc if a.pc is not None else a.state[1] if a.state is not None else None
            if end is not None and end + 1 < n and ws[end + 1].lo in COUNTRY and ws[end + 1].cap:
                a.country = (end + 1, end + 1)
        out.sort(key=lambda a: (a.start, a.end))
        return out

    def stops(self, li: int) -> set[int]:
        """The word indexes where a place starts on the line: a name run stops there."""
        got = self._stops.get(li)
        if got is None:
            got = self._stops[li] = {a.start for a in self.anchors[li]}
        return got

    # ------------------------------------------------------------------ words that can sit in a place
    def words_ok(self, ws: list[Word], idx: list[int]) -> bool:
        """Words that can sit between a street line and its state (a suburb, words after a box number)."""
        return all(not (ws[x].stopw or ws[x].lowr or not ws[x].alpha) for x in idx)

    def gaz_ok(self, words: list[str], code: str | None) -> bool:
        return bool(self.gaz.states_of(words, state=code) if code else self.gaz.states_of(words))

    def st_suburb(self, ws: list[Word], k: int) -> bool:
        """'St'/'Mt' + a name that starts a real locality ('St Ives', 'St Marys'): not a street type."""
        w = ws[k]
        if w.lo not in ("st", "mt", "saint", "mount") or w.comma or k + 1 >= len(ws):
            return False
        for n in (2, 1):
            run = ws[k + 1 : k + 1 + n]
            if (
                len(run) == n
                and all(x.name for x in run)
                and not any(x.comma for x in run[:-1])
                and self.gaz.states_of([w.b] + [x.b for x in run])
            ):
                return True
        return False

    # ------------------------------------------------------------------ the place after a street line
    def same_line_place(self, s: Street, a: Anchor) -> tuple[list[int], list[int], int | None] | None:
        """(suburb word indexes, extra street-line indexes, new street end or None), or None when the anchor is not the
        place of this street line.

        A numberless 'Name Type' street joins only a place that directly follows it: an optional comma (or a cell
        separator), then the suburb, state and postcode.  It never joins across other words ('walked along Smith Street
        before driving to Penrith NSW 2750' keeps the street; the place follows decision 2 on its own)."""
        ws = self.L[s.li].words
        kind, flags, s_end, name_start = s.kind, s.flags, s.end, s.name_start
        st0 = a.start
        if any(ws[x].sbreak for x in range(s_end, st0)):
            return None  # never join across a sentence break ('Smith Street. Penrith NSW 2750 is where she lives')
        lo, hi = s_end + 1, st0 - 1
        sep_skip = lo <= hi and ws[lo].sep  # a cell separator: '12 Smith St | Penrith NSW 2750'
        if sep_skip:
            lo += 1
        tail = list(range(lo, hi + 1))
        code = a.code
        sub: list[int] | None = None
        extra: list[int] = []
        new_end: int | None = None
        numberless = kind is UnitKind.STREET_NUMBERLESS and Flag.VIA not in flags
        overseas = a.kind is AnchorKind.OVERSEAS
        commas = [x for x in tail[:-1] if ws[x].comma]
        if commas:
            # 'PO Box 123 City East, Brisbane': the suburb starts after the last comma
            c = commas[-1]
            sub, extra = [x for x in tail if x > c], [x for x in tail if x <= c]
            if numberless and extra:
                return None  # words between the numberless street and the place's comma
        elif ws[s_end].comma or kind is UnitKind.DX or sep_skip:
            sub = tail
        elif kind is UnitKind.BOX:
            sub = tail
            if not overseas:
                for n in range(min(4, len(tail)), 0, -1):
                    if self.gaz_ok([ws[x].b for x in tail[-n:]], code):
                        sub, extra = tail[-n:], tail[:-n]
                        break
        elif name_start is not None and not overseas:
            # no comma between street line and place: the split whose suburb is a real locality.  Several can be
            # ('Smith Street North Ryde'): prefer the longest whose locality owns the written postcode, else the
            # longest.  'Dr' before the suburb of an anchored place is a street type (amendment A5).
            combined = list(range(name_start, hi + 1))
            cands: list[list[int]] = []
            for n in range(min(4, len(combined) - 1), 0, -1):
                cand = combined[-n:]
                if all(ws[x].name for x in cand) and self.gaz_ok([ws[x].b for x in cand], code):
                    cands.append(cand)
            if cands:
                # a split may make only capitalised words street names: 'George Street to Parramatta' keeps 'to' out
                cands = [c for c in cands if all(ws[x].name for x in range(s_end + 1, combined[-len(c) - 1] + 1))]
                if not cands:
                    return None
            if cands and numberless:
                # the split may give the street's own words to the suburb ('Smith Road Lane Cove') or take a
                # street-type suffix right after the type ('Smith Street North Ryde NSW 2112'), but it never makes
                # other words street names ('Smith Street before driving to Penrith NSW 2750'), and the street keeps
                # its type ('Smith Lane Cove NSW 2066' has no street line: 'Lane Cove' is the place)
                cands = [
                    c
                    for c in cands
                    if self.direct_end(s, combined[-len(c) - 1]) and self.keeps_type(s, combined[-len(c) - 1])
                ]
                if not cands:
                    return None
            if cands:
                pick = cands[0]
                if a.pcv is not None:
                    codes = [code] if code else sorted(range_states(a.pcv))
                    owned = [
                        c
                        for c in cands
                        if any(a.pcv in self.gaz.postcodes_of([ws[x].b for x in c], st) for st in codes)
                    ]
                    if owned:
                        pick = owned[0]
                sub, new_end = pick, combined[-len(pick) - 1]
            if sub is None:
                if Flag.UNTYPED in flags or Flag.DR_NAME in flags:
                    return None
                sub = tail
        elif overseas and name_start is not None and Flag.UNTYPED in flags:
            return None
        else:
            sub = tail
        if numberless and new_end is None and len(sub) > 1 and not overseas:
            # the suburb taken by position must be the place itself: when a shorter run ending at the state is a
            # real locality and the whole run is not, other words stand between ('Smith Street, Westfield Penrith')
            words = [ws[x].b for x in sub]
            if not self.gaz_ok(words, code) and any(self.gaz_ok(words[n:], code) for n in range(1, len(words))):
                return None
        if len(sub) > 4 or len(extra) > 3 or not self.words_ok(ws, sub + extra):
            return None
        if a.kind is AnchorKind.WEAK:
            # 'Mount Victoria': the state word is part of the suburb when together they name a locality
            if not sub:
                return None
            sw = [ws[x].b for x in range(a.state[0], a.state[1] + 1)] if a.state else []
            for n in range(1, min(3, len(sub)) + 1):
                if self.gaz.states_of([ws[x].b for x in sub[-n:]] + sw):
                    return None
        if a.kind is AnchorKind.STATELESS:
            assert a.pcv is not None
            if not sub:
                return None
            sts = self.gaz.states_of([ws[x].b for x in sub])
            if not (sts & range_states(a.pcv)) and not (ws[s_end].comma or commas):
                return None
        return sub, extra, new_end

    def keeps_type(self, s: Street, new_end: int) -> bool:
        """A street cut back to ``new_end`` still ends on a street type (or a type and its suffix)."""
        ws = self.L[s.li].words
        assert s.name_start is not None
        if ws[new_end].suffix and new_end - 1 > s.name_start and ws[new_end - 1].typ:
            return True
        return ws[new_end].typ and new_end > s.name_start

    def direct_end(self, s: Street, new_end: int) -> bool:
        """A no-comma split that leaves a numberless street directly before its suburb: it ends within the street's own
        words, or on a street-type suffix right after its type."""
        ws = self.L[s.li].words
        if new_end <= s.end:
            return True
        return new_end == s.end + 1 and ws[new_end].suffix and ws[s.end].typ

    def place_roles(self, li: int, a: Anchor, sub_idx: list[int]) -> _PlaceRoles:
        out: _PlaceRoles = {}
        for x in sub_idx:
            out[(li, x)] = Role.SUBURB
        if a.glued:
            assert a.pc is not None
            out[(li, a.pc)] = Role.STATE_AND_POSTCODE
        else:
            if a.state is not None:
                for x in range(a.state[0], a.state[1] + 1):
                    out[(li, x)] = Role.STATE
            if a.pc is not None:
                for x in range(a.pc_first if a.pc_first is not None else a.pc, a.pc + 1):
                    out[(li, x)] = Role.POSTCODE
        if a.country is not None:
            for x in range(a.country[0], a.country[1] + 1):
                out[(li, x)] = Role.COUNTRY
        return out

    def suburb_tail(self, ws: list[Word], j: int, label_starts: AbstractSet[int] = frozenset()) -> list[int]:
        """The suburb-only tail after a street line ('lives at 4 Rose Lane, Katoomba with ...'): 1-3 capitalised words
        that name a real locality, then the line end, a lower-case word, a separator or a label (``label_starts``:
        the words that start one)."""
        for n in (3, 2, 1):
            run = list(range(j, j + n))
            if run[-1] >= len(ws):
                continue
            if not all(ws[x].name and not ws[x].stopw for x in run):
                continue
            if any(ws[x].comma or ws[x].stop for x in run[:-1]):
                continue
            nxt = ws[run[-1] + 1] if run[-1] + 1 < len(ws) else None
            last = ws[run[-1]]
            if nxt is not None and not (nxt.lowr or nxt.sep or last.stop or last.comma or run[-1] + 1 in label_starts):
                continue
            if self.gaz.states_of([ws[x].b for x in run]):
                return run
        return []

    def country_line_below(self, ln: Line, *, foreign: bool) -> Line | None:
        """The line below ``ln`` holding only a country: 'Australia', or (``foreign``) a foreign country name."""
        nb = self.geo.below(ln)
        if nb is None or not nb.words:
            return None
        if not foreign:
            return nb if len(nb.words) == 1 and nb.words[0].lo in COUNTRY else None
        last = foreign_country_at(nb.words, 0)
        return nb if last == len(nb.words) - 1 else None

    def line_start_place(self, ln: Line) -> LineStart | None:
        """A place starting the line (the suburb from the line start)."""
        ws = ln.words
        for a in self.anchors[ln.li]:
            st0 = a.start
            if a.kind is AnchorKind.OVERSEAS:
                got = self._overseas_line_start(ln, a)
                if got is not None:
                    return got
                continue
            if a.kind is AnchorKind.STATELESS:
                if st0 == 0:
                    continue
                sub = list(range(0, st0))
                assert a.pcv is not None
                if not self.gaz.states_of([ws[x].b for x in sub]) & range_states(a.pcv):
                    got = self._overseas_line_start(ln, a)
                    if got is not None:
                        return got
                    continue
            else:
                sub = list(range(0, st0))
            if len(sub) > 4 or any(not ws[x].name or ws[x].stopw for x in sub):
                continue
            if any(ws[x].comma for x in sub[:-1]):
                continue
            if a.kind is AnchorKind.WEAK:
                sw = [ws[x].b for x in range(a.state[0], a.state[1] + 1)] if a.state else []
                if sub and self.gaz.states_of([ws[x].b for x in sub] + sw):
                    continue
                if not sub:
                    continue
            roles = self.place_roles(ln.li, a, sub)
            country_line = None
            if a.country is None and a.end == len(ws) - 1:
                country_line = self.country_line_below(ln, foreign=False)
            return LineStart(roles, country_line)
        return None

    def _overseas_line_start(self, ln: Line, a: Anchor) -> LineStart | None:
        """An overseas place at the start of a line ('AUCKLAND 1010' over 'NEW ZEALAND', 'Oxford OX1 4AA')."""
        ws = ln.words
        st0 = a.start
        sub = list(range(0, st0))
        if not sub or len(sub) > 4 or any(not ws[x].name or ws[x].stopw for x in sub):
            return None
        if any(ws[x].comma for x in sub[:-1]):
            return None
        if a.end != len(ws) - 1 and a.country is None:
            return None
        country_line = None
        if a.country is None:
            country_line = self.country_line_below(ln, foreign=True)
        pc_first = a.pc_first if a.pc_first is not None else a.pc
        assert pc_first is not None and a.pc is not None
        uk_ca = pc_first != a.pc
        if a.country is None and country_line is None and not uk_ca:
            return None  # a plain number needs its country
        roles: _PlaceRoles = {(ln.li, x): Role.SUBURB for x in sub}
        for x in range(pc_first, a.pc + 1):
            roles[(ln.li, x)] = Role.POSTCODE
        if a.country is not None:
            for x in range(a.country[0], a.country[1] + 1):
                roles[(ln.li, x)] = Role.COUNTRY
        return LineStart(roles, country_line, overseas=True)
