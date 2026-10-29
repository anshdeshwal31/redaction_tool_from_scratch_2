"""The join: a street line meets its place (on its line, in the suburb-only tail, on the line below or across a page
break), the words in front of it (the lines above), and the unit is emitted."""

from __future__ import annotations

from mlredact.text.address.geometry import Span
from mlredact.text.address.internal import Anchor, AnchorKind, Flag, Street, UnitB
from mlredact.text.address.model import BoxRole, PartialWord, Pos, Role, UnitKind
from mlredact.text.address.streets import StreetStage
from mlredact.text.address.tables import (
    BOXNUM,
    BUILDING_DESIG,
    BUILDING_WORDS,
    CAPITAL_CITY_PHRASE,
    CONNECTORS,
    DESIGNATORS,
    GLUED,
    LETTER_ID,
    LEVEL_CODE,
    OCCUPATION_WORDS,
    ORDINAL,
    ORG_NOT_BUILDING,
    PUBLIC_BODY_WORDS,
    RURAL_BOXES,
    VETO_NOUNS,
)
from mlredact.text.address.words import Line, Word

_STATE_OR_POSTCODE = frozenset({Role.STATE, Role.POSTCODE, Role.STATE_AND_POSTCODE})
_POSTCODE_ROLES = frozenset({Role.POSTCODE, Role.STATE_AND_POSTCODE})
_LEVEL_DESIG = frozenset({"level", "floor"})
_AND = ("and", "&")
_ROAD_FLOOR_WORDS = frozenset({"floor", "flr", "level"})

# Words that start the next field of a line after an address ('... Penrith Claim no. 123456', '... Penrith Ph 02 ...').
_ID_LABEL_WORDS = frozenset({"claim", "ref", "reference", "file", "policy", "ph", "phone", "tel", "mobile", "fax"})


class JoinStage(StreetStage):
    def join_street(self, s: Street) -> None:
        ln = self.L[s.li]
        ws = ln.words
        unit_roles: dict[Pos, Role] = {}
        how: list[str] = []
        # street words on the lines above (line-above join, split street line)
        for li2, rr in s.above:
            for wi, r in sorted(rr.items()):
                unit_roles[(li2, wi)] = r
        # ---- same line: the first anchor after the street line, before the next street line
        nxt_street_start = len(ws)
        for o in self.line_streets.get(s.li, []):
            if o is not s and o.start > s.end:
                nxt_street_start = min(nxt_street_start, o.start)
        cand = [a for a in self.anchors[s.li] if s.end < a.start <= nxt_street_start]
        cand.sort(key=lambda a: a.start)
        for a in cand:
            res = self.same_line_place(s, a)
            if res is None:
                continue
            sub_idx, extra_idx, new_end = res
            if new_end is not None:
                self._resplit(s, new_end)
            for x in extra_idx:
                s.roles[x] = Role.BOX_EXTRA if s.kind is UnitKind.BOX else Role.STREET_NAME
            if (
                s.kind is UnitKind.BOX
                and len(extra_idx) >= 2
                and ws[extra_idx[-1]].typ
                and all(ws[x].name for x in extra_idx)
            ):
                for x in extra_idx:  # a road after the box number ('RMB 4600 Lakes Road,')
                    s.roles[x] = Role.STREET_NAME
                s.roles[extra_idx[-1]] = Role.STREET_TYPE
            if extra_idx:
                s.end = max(s.end, max(extra_idx))
            loc = self.place_roles(s.li, a, sub_idx)
            overseas = a.kind is AnchorKind.OVERSEAS
            if overseas and a.country is None and not self._overseas_pc_ok(ln, a, loc):
                continue
            self._emit(s, loc, [*how, "same_line:" + a.kind.value], unit_roles, overseas=overseas)
            return
        # ---- 'IN YOUR CAPITAL CITY' in place of a locality, after a box (same line or the line below)
        if s.kind is UnitKind.BOX:
            got = self._capital_city_phrase(s)
            if got is not None:
                loc_roles, via = got
                self._emit(s, loc_roles, [*how, via], unit_roles)
                return
        # ---- suburb-only tail after the street line's comma ('lives at 4 Rose Lane, Katoomba with ...')
        if ws[s.end].comma and s.end + 1 < len(ws):
            tail = self.suburb_tail(ws, s.end + 1)
            if tail:
                tail_loc = {(s.li, x): Role.SUBURB for x in tail}
                route = "suburb_tail"
                if tail[-1] == len(ws) - 1:  # the state and postcode wrapped onto the next line
                    cont = self._tail_continuation(ln, ws, tail)
                    if cont is not None:
                        tail_loc.update(cont)
                        route = "suburb_tail_next_line"
                self._emit(s, tail_loc, [*how, route], unit_roles)
                return
        # ---- a suburb after the street line with no comma, in an address position: after an address label or a
        # residence phrase, or in a name or address block ('Address: 12 Smith Street Penrith', 'lives at 12 Smith
        # Street Penrith with ...', 'Ms Jane Citizen' over '12 Smith Street Penrith')
        if (
            s.kind is UnitKind.STREET
            and Flag.TYPED in s.flags
            and not (ws[s.end].comma or ws[s.end].sbreak)
            and s.end + 1 < len(ws)
            and (self.address_cue_before(s.li, s.start) in ("address_label", "phrase") or self._block_position(s))
        ):
            starts = {lab.start for lab in self.labels[s.li]} | {
                x for x in range(s.end + 1, len(ws)) if ws[x].lo in _ID_LABEL_WORDS
            }
            tail = self.suburb_tail(ws, s.end + 1, starts)
            if tail:
                tail_loc = {(s.li, x): Role.SUBURB for x in tail}
                if tail[-1] == len(ws) - 1:  # the state and postcode on the line below
                    tail_loc.update(self._tail_continuation(ln, ws, tail) or {})
                self._emit(s, tail_loc, [*how, "suburb_tail_no_comma"], unit_roles)
                return
        # ---- the line below in the block, or the first line of the next page
        rest = [w for w in ws[s.end + 1 :] if w.alnum_text]
        if not rest and not ws[s.end].sbreak:
            span = self.span(s.li, s.start, s.end)
            below = self.geo.below(ln, span, skip_furniture=True)
            via = "next_line"
            if below is None and self.is_prose(ln):
                below = self.geo.below(ln, (ln.x0, ln.x1), skip_furniture=True)  # wrapped running text
            if below is None:
                below = self.geo.next_page_first(ln)
                via = "page_break"
            if below is not None:
                start = self.line_start_place(below)
                if start is not None:
                    loc_roles = start.roles
                    if start.country_line is not None:
                        for x in range(len(start.country_line.words)):
                            loc_roles[(start.country_line.li, x)] = Role.COUNTRY
                    self._emit(s, loc_roles, [*how, via], unit_roles, overseas=start.overseas)
                    return
        # ---- amendments A1 and A3: a place line after a paragraph gap, or a bare locality line under the street line
        if not rest and not ws[s.end].sbreak:
            got2 = self._gap_or_bare_below(ln, self.span(s.li, s.start, s.end))
            if got2 is not None:
                loc_roles, via = got2
                self._emit(s, loc_roles, [*how, via], unit_roles)
                return
        # ---- no place: the street line alone, if the grammar accepts it
        if self._alone_ok(s):
            self._emit(s, {}, [*how, "alone"], unit_roles)

    def _tail_continuation(self, ln: Line, ws: list[Word], tail: list[int]) -> dict[Pos, Role] | None:
        """A suburb-only tail that ends its line, whose state and postcode start the line below
        ('... Victoria Road, Ryde' over 'NSW 2112 in May.'): the roles of that state, postcode and country."""
        if ws[tail[-1]].sbreak:
            return None
        below = self.geo.below(ln, (ln.x0, ln.x1), skip_furniture=True)
        if below is None:
            return None
        for a in self.anchors[below.li]:
            if a.kind is AnchorKind.STRONG and a.start == 0:
                if not self.gaz_ok([ws[x].b for x in tail], a.code):
                    return None
                return self.place_roles(below.li, a, [])
        return None

    def _overseas_pc_ok(self, ln: Line, a: Anchor, loc: dict[Pos, Role]) -> bool:
        """An overseas postcode with no country on its line is an address only as a UK or Canadian code, or when the
        country stands alone on the line below (it then joins the unit)."""
        if a.pc_first is not None and a.pc_first != a.pc:
            return True
        country = self.country_line_below(ln, foreign=True)
        if country is None or a.end != len(ln.words) - 1:
            return False
        for x in range(len(country.words)):
            loc[(country.li, x)] = Role.COUNTRY
        return True

    def _resplit(self, s: Street, new_end: int) -> None:
        ws = self.L[s.li].words
        assert s.name_start is not None
        for x in list(s.roles):
            if x > new_end:
                del s.roles[x]
        for x in range(s.name_start, new_end + 1):
            s.roles[x] = Role.STREET_NAME
        if ws[new_end].suffix and new_end - 1 > s.name_start and ws[new_end - 1].typ:
            s.roles[new_end - 1] = Role.STREET_TYPE  # 'Smith Street North' before 'Ryde NSW 2112'
            s.roles[new_end] = Role.STREET_SUFFIX
            s.flags.add(Flag.TYPED)
            s.flags.discard(Flag.UNTYPED)
        elif ws[new_end].typ and new_end > s.name_start:
            s.roles[new_end] = Role.STREET_TYPE
            s.flags.add(Flag.TYPED)
            s.flags.discard(Flag.UNTYPED)
        else:
            s.flags.discard(Flag.TYPED)
            s.flags.add(Flag.UNTYPED)
        s.end = new_end

    # ------------------------------------------------------------------ emitting a unit
    def _alone_ok(self, s: Street) -> bool:
        ws = self.L[s.li].words
        if Flag.NUMBERLESS in s.flags:
            return False  # a numberless street, box kind or 'via' line needs its place
        if s.kind in (UnitKind.BOX, UnitKind.DX):
            if Flag.NEEDS_CTX in s.flags:
                return self.address_cue_before(s.li, s.start) == "address_label"
            return True
        if Flag.UNTYPED in s.flags or Flag.DR_NAME in s.flags or s.kind is UnitKind.STREET_NUMBERLESS:
            return False
        if Flag.MONTH in s.flags:
            return self.address_cue_before(s.li, s.start) == "address_label"
        last = ws[s.end]
        nxt = ws[s.end + 1] if s.end + 1 < len(ws) else None
        if nxt is not None and nxt.lo in VETO_NOUNS:
            return False
        if last.comma or last.sbreak or nxt is None or nxt.sep or nxt.label:
            return True
        if not any(w.alnum_text for w in ws[s.end + 1 :]):
            return True
        if self.address_cue_before(s.li, s.start) is not None or self._block_position(s):
            return True
        # a street, lower-case words, then a place of its own ('walked from 12 George Street to Parramatta NSW 2150')
        if nxt.lowr and any(
            a.start > s.end and any(ws[x].lowr for x in range(s.end + 1, a.start)) for a in self.anchors[s.li]
        ):
            return True
        # a list of street lines: '12-14 Smith Street and 12A Smith Street, ...' (amendment A2)
        if nxt.text.strip() in _AND and s.end + 2 < len(ws):
            o = [x for x in self.line_streets.get(s.li, []) if x.start == s.end + 2]
            if o and Flag.TYPED in o[0].flags:
                return True
        return False

    def _block_position(self, s: Street) -> bool:
        """A number-led street line that starts its line in a name or address block: the line right above it in its
        column holds a name or an address line (capitalised words and numbers, no sentence), as 'Ms Jane Citizen' over
        '12 Smith Street Penrith' (FINAL_PLAN TOOL2 section 2: a block position is an address cue)."""
        if s.start != 0 or s.kind is not UnitKind.STREET or Flag.NUMBERLESS in s.flags:
            return False
        up = self.geo.above(self.L[s.li])
        if up is None or self.is_prose(up) or not up.words or up.words[-1].sbreak:
            return False
        words = [w for w in up.words if w.alnum_text]
        return 0 < len(words) <= 8 and all(
            w.cap or w.upper or w.digits or w.lo in ("&", "c/-", "c/o", "of", "and") for w in words
        )

    def _emit(
        self, s: Street, loc: dict[Pos, Role], how: list[str], unit_roles: dict[Pos, Role], *, overseas: bool = False
    ) -> None:
        ws = self.L[s.li].words
        flags = s.flags
        has_state_or_pc = any(r in _STATE_OR_POSTCODE for r in loc.values())
        if Flag.NEEDS_CTX in flags and not loc and self.address_cue_before(s.li, s.start) != "address_label":
            return
        if Flag.UNTYPED in flags and not has_state_or_pc:
            return
        if Flag.MONTH in flags and not loc and self.address_cue_before(s.li, s.start) != "address_label":
            return
        if Flag.MONTH in flags and Flag.TYPED not in flags:
            return
        if s.kind is UnitKind.STREET_NUMBERLESS and not has_state_or_pc:
            return
        if Flag.NUMBERLESS in flags and not has_state_or_pc:
            return  # 'CARE PO', 'CMA', 'CPA' with no number: only before a place
        if Flag.PROSE in flags and not any(r in _POSTCODE_ROLES for r in loc.values()):
            return  # a numberless street in prose is an address only before a suburb, state and postcode
        # the line(s) above: designators and building names; above an RMB, RSD, MS or RMS line also a road line
        if s.start == 0 and not s.above:
            rural = s.kind is UnitKind.BOX and ws[s.start].lo.replace(".", "") in RURAL_BOXES
            for li2, rr in self._lines_above(self.L[s.li], road_ok=rural):
                for wi, r in sorted(rr.items()):
                    unit_roles[(li2, wi)] = r
                how = [*how, "line_above"]
        for wi in range(s.start, s.end + 1):
            role = s.roles.get(wi)
            if role is None:
                role = Role.BOX_EXTRA if (s.kind is UnitKind.BOX and not ws[wi].name) else Role.STREET_NAME
            unit_roles[(s.li, wi)] = role
        for p, r in loc.items():
            unit_roles[p] = r
        if any(p in self.used for p in unit_roles):
            return
        u = UnitB(s.kind, unit_roles, how=how, overseas=overseas)
        u.partial.extend(self._partials(u))
        self.add_unit(u)

    def _partials(self, u: UnitB) -> list[PartialWord]:
        out: list[PartialWord] = []
        for p, r in sorted(u.roles.items()):
            w = self.word(p)
            if r is Role.STATE_AND_POSTCODE:
                g = GLUED.fullmatch(w.b)
                assert g is not None
                out.append(PartialWord(p, g.group(1), g.group(2)))
            elif r in (Role.UNIT_NUMBER_AND_NUMBER, Role.UNIT_NUMBER):
                m = LEVEL_CODE.fullmatch(w.b)  # 'B02/53': the level code stays, the digits change
                if m is not None:
                    out.append(PartialWord(p, m.group(1), m.group(2)))
        return out

    # ------------------------------------------------------------------ what stands above a street line
    def _lines_above(self, ln: Line, road_ok: bool = False) -> list[tuple[int, dict[int, Role]]]:
        out: list[tuple[int, dict[int, Role]]] = []
        cur = ln
        for n in range(2):
            up = self.geo.above(cur)
            if up is None:
                break
            rr = self._road_roles(up) if road_ok and n == 0 else None
            if rr is None:
                rr = self._above_roles(up)
            if rr is None:
                break
            out.insert(0, (up.li, rr))
            cur = up
        return out

    def _road_roles(self, ln: Line) -> dict[int, Role] | None:
        """A numberless road line ('Lakes Road') directly above an RMB, RSD, MS or RMS line (Australia Post: road names
        for these must appear above the address lines)."""
        ws = ln.words
        if not 2 <= len(ws) <= 4 or self.anchors[ln.li] or self.labels[ln.li] or ln.box is not None:
            return None
        if not all(w.name and not w.stopw for w in ws) or not ws[-1].typ:
            return None
        if any(w.comma for w in ws[:-1]):
            return None
        roles = {k: Role.STREET_NAME for k in range(len(ws) - 1)}
        roles[len(ws) - 1] = Role.STREET_TYPE
        return roles

    def _above_roles(self, ln: Line) -> dict[int, Role] | None:
        ws = ln.words
        if not ws or len(ws) > 8 or self.anchors[ln.li] or self.labels[ln.li]:
            return None
        roles: dict[int, Role] = {}
        has_key = False
        k = 0
        while k < len(ws):
            w = ws[k]
            if (
                w.title
                or w.blocked
                or w.lo in ORG_NOT_BUILDING
                or w.lo in ("centre", "center")
                or w.lo in OCCUPATION_WORDS
                or w.lo in PUBLIC_BODY_WORDS
                or w.sep
                or "@" in w.text
                or (w.lowr and w.lo not in CONNECTORS)
            ):
                return None
            if w.desig and k + 1 < len(ws) and BOXNUM.fullmatch(ws[k + 1].b):
                kind = DESIGNATORS[w.lo]
                roles[k] = Role.DESIGNATOR
                roles[k + 1] = Role.LEVEL_NUMBER if kind in _LEVEL_DESIG else Role.UNIT_NUMBER
                has_key = True
                k += 2
                continue
            if ORDINAL.fullmatch(w.b) and k + 1 < len(ws) and ws[k + 1].lo in _ROAD_FLOOR_WORDS:
                roles[k] = Role.LEVEL_NUMBER
                roles[k + 1] = Role.DESIGNATOR
                has_key = True
                k += 2
                continue
            # 'Building C,', 'Tower 2': a designator pair (the word stays, the designator changes)
            if (
                w.lo in BUILDING_DESIG
                and w.cap
                and not w.comma
                and k + 1 < len(ws)
                and LETTER_ID.fullmatch(ws[k + 1].b)
            ):
                roles[k] = Role.DESIGNATOR
                roles[k + 1] = Role.UNIT_NUMBER
                has_key = True
                k += 2
                continue
            if w.level_type:  # 'Ground Floor', 'Mezzanine'
                roles[k] = Role.DESIGNATOR
                if k + 1 < len(ws) and ws[k + 1].lo in ("floor", "flr", "fl", "level"):
                    roles[k + 1] = Role.DESIGNATOR
                    k += 1
                has_key = True
                k += 1
                continue
            if w.lo in BUILDING_WORDS - {"centre", "center", "court", "plaza", "square", "park"} and w.cap:
                roles[k] = Role.BUILDING
                has_key = True
                k += 1
                continue
            if w.name or LETTER_ID.fullmatch(w.b):
                roles[k] = Role.BUILDING
                k += 1
                continue
            return None
        return roles if has_key else None

    # ------------------------------------------------------------------ places on other lines
    def _capital_city_phrase(self, s: Street) -> tuple[dict[Pos, Role], str] | None:
        """'IN YOUR CAPITAL CITY' after a box, on its line (after a comma) or as the line below: a locality that
        names no place."""
        ws = self.L[s.li].words
        rest = [x for x in range(s.end + 1, len(ws)) if ws[x].alnum_text]
        if rest:
            if ws[s.end].comma and len(rest) >= 4 and tuple(ws[x].lo for x in rest[:4]) == CAPITAL_CITY_PHRASE:
                return {(s.li, x): Role.CAPITAL_CITY_PHRASE for x in rest[:4]}, "capital_city_phrase"
            return None
        below = self.geo.below(self.L[s.li], self.span(s.li, s.start, s.end), skip_furniture=True)
        if below is not None and tuple(w.lo for w in below.words) == CAPITAL_CITY_PHRASE:
            return {(below.li, x): Role.CAPITAL_CITY_PHRASE for x in range(4)}, "next_line_capital_city_phrase"
        return None

    def _gap_or_bare_below(self, ln: Line, span: Span) -> tuple[dict[Pos, Role], str] | None:
        near = self.geo.below(ln, span, skip_furniture=True)
        if near is not None:
            ws = near.words
            label_left = any(o.x1 <= near.x0 + 1 and bool(self.labels[o.li]) for o in self.geo.row_mates(near))
            if (
                1 <= len(ws) <= 3
                and all(w.name and not w.stopw for w in ws)
                and self.gaz.states_of([w.b for w in ws])
                and not self.labels[near.li]
                and not label_left
                and near.box not in (BoxRole.LABEL, BoxRole.VALUE)
            ):
                return {(near.li, x): Role.SUBURB for x in range(len(ws))}, "next_line_suburb"
            return None
        far = self.geo.below(ln, span, max_gap=2.0, skip_furniture=True)
        if far is None:
            return None
        got = self.line_start_place(far)
        if got is None or got.overseas:
            return None
        roles = got.roles
        if not any(r in _POSTCODE_ROLES for r in roles.values()):
            return None
        if max(p[1] for p in roles) != len(far.words) - 1:
            return None
        sub = sorted(p for p, r in roles.items() if r is Role.SUBURB)
        if not sub or not self.gaz.states_of([self.word(p).b for p in sub]):
            return None
        return roles, "next_line_gap"
