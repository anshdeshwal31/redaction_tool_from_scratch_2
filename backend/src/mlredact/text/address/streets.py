"""Street lines: the candidates on each line (box kinds, DX, corner forms, number-led streets, 'via' property lines,
numberless streets) and the words that belong in front of a street (designators, floors, building names)."""

from __future__ import annotations

import re

from mlredact.text.address.internal import Flag, Street
from mlredact.text.address.model import Role, UnitKind
from mlredact.text.address.places import PlaceStage
from mlredact.text.address.tables import (
    BOX_CODES,
    BOX_FIRST_WORDS,
    BOX_PHRASES,
    BOXNUM,
    BUILDING_DESIG,
    BUILDING_WORDS,
    CONNECTORS,
    DESIGNATORS,
    HN_SLASH,
    LETTER_ID,
    LEVEL_WORDS,
    MEASURE_WORDS,
    MONTHS,
    NUMBERLESS_BANNED_TYPES,
    NUMBERLESS_BOXES,
    OCCUPATION_WORDS,
    ORDINAL,
    ORG_NOT_BUILDING,
    ORG_WORDS,
    PROPERTY_WORDS,
    PUBLIC_BODY_WORDS,
    TYPE_CANON,
    VETO_NOUNS,
    WEEKDAYS,
)
from mlredact.text.address.words import Line, Word

_DIGITS_1_6 = re.compile(r"\d{1,6}")
_LEVEL_DESIG = frozenset({"level", "floor"})


def _rank(s: Street) -> tuple[bool, int]:
    strong = s.kind in (UnitKind.BOX, UnitKind.DX, UnitKind.CORNER) or Flag.TYPED in s.flags
    return (strong, s.end - s.start)


class StreetStage(PlaceStage):
    def streets(self) -> list[Street]:
        out: list[Street] = []
        for ln in self.L:
            ws = ln.words
            i = 0
            # a street line split over two lines by OCR: '12 Smith' / 'Street, Penrith NSW 2750'
            split = self._split_line_street(ln)
            if split is not None:
                out.append(split)
                i = split.end + 1
            norm = [w.lo.replace(".", "") for w in ws]
            while i < len(ws):
                s = (
                    (self._box_at(ln, norm, i) if norm[i] in BOX_FIRST_WORDS else None)
                    or self._dx_at(ln, i)
                    or self._corner_at(ln, i)
                    or self._number_street_at(ln, i)
                    or self._via_at(ln, i)
                    or self._numberless_at(ln, i)
                )
                if s is None:
                    i += 1
                    continue
                out.append(s)
                i = s.end + 1
        # overlapping candidates on one line: typed, box, DX or corner lines win over untyped ones, then the longer
        by_line: dict[int, list[Street]] = {}
        for s in out:
            by_line.setdefault(s.li, []).append(s)
        kept: list[Street] = []
        for ss in by_line.values():
            ss.sort(key=_rank, reverse=True)
            chosen: list[Street] = []
            for s in ss:
                if any(not (s.end < o.start or o.end < s.start) for o in chosen):
                    continue
                chosen.append(s)
            kept.extend(sorted(chosen, key=lambda x: x.start))
        kept.sort(key=lambda x: (x.li, x.start))
        self.line_streets = {}
        for s in kept:
            self.line_streets.setdefault(s.li, []).append(s)
        return kept

    def _split_line_street(self, ln: Line) -> Street | None:
        ws = ln.words
        if not ws or not ws[0].typ or (ws[0].lo == "dr" and len(ws) > 1 and ws[1].cap and not ws[0].comma):
            return None
        up = self.geo.above(ln)
        if up is None or not up.words:
            return None
        uw = up.words
        # number + 1-3 name words ending the line above
        for k in range(len(uw) - 2, max(-1, len(uw) - 5), -1):
            if (
                uw[k].numw
                and all(x.name and not x.typ for x in uw[k + 1 :])
                and 1 <= len(uw) - k - 1 <= 3
                and (k == 0 or (not uw[k - 1].desig and self.label_ending_before(up.li, k) is not None))
            ):
                roles = {k: Role.UNIT_NUMBER_AND_NUMBER if HN_SLASH.fullmatch(uw[k].b) else Role.NUMBER}
                for x in range(k + 1, len(uw)):
                    roles[x] = Role.STREET_NAME
                s = Street(ln.li, 0, 0, UnitKind.STREET, {0: Role.STREET_TYPE}, {Flag.TYPED, Flag.SPLIT})
                s.above = [(up.li, roles)]
                return s
        return None

    # ------------------------------------------------------------------ postal delivery types, DX, corners, 'via'
    def _box_at(self, ln: Line, norm: list[str], i: int) -> Street | None:
        """Australia Post's postal delivery types.  RMB, RSD, CMB, PMB, MS, RMS, CMA and CPA count only in upper case
        and in an address context; Care PO, CMA, CPA and CMB may stand without a number (then a place must follow)."""
        ws = ln.words
        n = len(ws)
        for ph in BOX_PHRASES:
            j = i + len(ph)
            if j > n or tuple(norm[i:j]) != ph:
                continue
            first = ws[i]
            if ph[0] in BOX_CODES:
                if first.b != first.b.upper():
                    return None
            elif not (first.cap or first.upper):
                return None
            roles = {x: Role.BOX_KIND for x in range(i, j)}
            if j >= n or not BOXNUM.fullmatch(ws[j].b):
                if ph in NUMBERLESS_BOXES and not (j < n and ws[j].digits):
                    return Street(ln.li, i, j - 1, UnitKind.BOX, roles, {Flag.BOX, Flag.NEEDS_CTX, Flag.NUMBERLESS})
                return None
            roles[j] = Role.BOX_NUMBER
            flags = {Flag.BOX}
            if ph[0] in BOX_CODES:
                flags.add(Flag.NEEDS_CTX)
                nxt = ws[j + 1] if j + 1 < n else None
                if nxt is not None and nxt.lo in MEASURE_WORDS:
                    return None
            return Street(ln.li, i, j, UnitKind.BOX, roles, flags)
        return None

    def _via_at(self, ln: Line, i: int) -> Street | None:
        """A property name, 'via' and its place ('Kilbride Station via Berrima NSW 2577'): the name and 'via' form the
        street line; a strong place must follow (checked at the join)."""
        ws = ln.words
        if i > 0 and not (ws[i - 1].sep or ws[i - 1].label):
            return None
        k = i
        names: list[int] = []
        while k < len(ws) and len(names) < 4 and ws[k].lo != "via" and ws[k].name and not ws[k].stopw:
            names.append(k)
            k += 1
        if not names or k >= len(ws) - 1 or ws[k].lo != "via" or ws[names[-1]].lo not in PROPERTY_WORDS:
            return None
        if any(ws[x].comma for x in names):
            return None
        roles = {x: Role.BUILDING for x in names}
        roles[k] = Role.VIA
        return Street(ln.li, i, k, UnitKind.STREET_NUMBERLESS, roles, {Flag.VIA, Flag.NUMBERLESS})

    def _dx_at(self, ln: Line, i: int) -> Street | None:
        ws = ln.words
        if ws[i].b != "DX" or i + 2 >= len(ws) or not _DIGITS_1_6.fullmatch(ws[i + 1].b):
            return None
        for n in (3, 2, 1):
            ex = ws[i + 2 : i + 2 + n]
            if len(ex) != n or not all(x.name for x in ex):
                continue
            if any(x.comma for x in ex[:-1]):
                continue
            if self.gaz.states_of([x.b for x in ex]):
                roles = {i: Role.DX, i + 1: Role.DX_NUMBER}
                for x in range(i + 2, i + 2 + n):
                    roles[x] = Role.DX_EXCHANGE
                return Street(ln.li, i, i + 1 + n, UnitKind.DX, roles, {Flag.DX})
        return None

    def _corner_at(self, ln: Line, i: int) -> Street | None:
        ws = ln.words
        w = ws[i]
        if w.lo not in ("cnr", "corner") or not (w.cap or w.upper):
            return None
        roles = {i: Role.CORNER}
        j = i + 1
        if j < len(ws) and ws[j].lo == "of":
            roles[j] = Role.CORNER
            j += 1

        def name_run(j: int) -> tuple[int, list[int]]:
            got: list[int] = []
            while j < len(ws) and ws[j].name and ws[j].lo != "and" and len(got) < 3 and not ws[j].typ:
                got.append(j)
                j += 1
            return j, got

        j, n1 = name_run(j)
        if not n1:
            return None
        for x in n1:
            roles[x] = Role.STREET_NAME
        if j < len(ws) and ws[j].typ and ws[j].lo != "and":
            roles[j] = Role.STREET_TYPE
            j += 1
        if j >= len(ws) or ws[j].text.strip() not in ("&", "and", "And", "AND"):
            return None
        roles[j] = Role.CORNER
        j += 1
        j, n2 = name_run(j)
        if not n2 or j >= len(ws) or not ws[j].typ:
            return None
        for x in n2:
            roles[x] = Role.STREET_NAME
        roles[j] = Role.STREET_TYPE
        return Street(ln.li, i, j, UnitKind.CORNER, roles, {Flag.TYPED, Flag.CORNER})

    # ------------------------------------------------------------------ name and type
    def _name_type(
        self, ws: list[Word], j: int, stops: set[int] | None = None
    ) -> tuple[list[int], int | None, int, set[Flag]]:
        """Name words from ``j``, then the last type of the first run of types ('Wattle Grove Road').  The run stops
        where a place anchor starts.  Returns (names, type, end, flags)."""
        stops = stops or set()
        names: list[int] = []
        flags: set[Flag] = set()
        k = j
        n = len(ws)
        while k < n and len(names) < 5:
            if k in stops:
                break
            w = ws[k]
            if w.lo in CONNECTORS and w.lowr and names and k + 1 < n and (ws[k + 1].name or ws[k + 1].typ):
                names.append(k)
                k += 1
                continue
            if w.typ and names and not self.st_suburb(ws, k):
                t = k
                while (
                    t + 1 < n
                    and (t + 1) not in stops
                    and ws[t + 1].typ
                    and not ws[t].comma
                    and not self.st_suburb(ws, t + 1)
                ):
                    t += 1
                if ws[t].lo == "dr" and t + 1 < n and ws[t + 1].name and not ws[t].comma and (t + 1) not in stops:
                    flags.add(Flag.DR_NAME)  # 'Dr' before a capitalised word: a title, not a type
                    names.extend(range(k, t))
                    return names, None, (names[-1] if names else j - 1), flags
                names.extend(range(k, t))
                return names, t, t, flags
            if not w.name or (w.title and w.lo not in ("st", "mt", "saint", "mount")):
                break
            if w.desig and k + 1 < n and BOXNUM.fullmatch(ws[k + 1].b):
                break
            if (w.lo in MONTHS or w.lo in WEEKDAYS) and not names:
                flags.add(Flag.MONTH)
            names.append(k)
            if w.comma:
                k += 1
                break
            k += 1
        return names, None, (names[-1] if names else j - 1), flags

    def _suffix_after(self, ln: Line, typ: int) -> int | None:
        """A street-type suffix right after the type ('Smith Street North,'), on numbered and numberless streets alike:
        only when a comma, the line end or a separator follows it; without one the no-comma split decides by the
        postcode."""
        ws = ln.words
        sx = typ + 1
        if (
            sx < len(ws)
            and ws[sx].suffix
            and not ws[typ].comma
            and sx not in self.stops(ln.li)
            and (ws[sx].comma or ws[sx].sbreak or sx == len(ws) - 1 or ws[sx + 1].sep)
        ):
            return sx
        return None

    def _typed_ok(self, s: Street) -> bool:
        ws = self.L[s.li].words
        nxt = ws[s.end + 1] if s.end + 1 < len(ws) else None
        return not (nxt is not None and nxt.lo in VETO_NOUNS and not ws[s.end].comma)

    # ------------------------------------------------------------------ number-led and numberless streets
    def _number_street_at(self, ln: Line, i: int) -> Street | None:
        ws = ln.words
        w = ws[i]
        roles: dict[int, Role] = {}
        flags: set[Flag] = set()
        # designator-led core without a house number: 'Lot 3 Old Bells Line of Road'
        if w.desig and i + 1 < len(ws) and BOXNUM.fullmatch(ws[i + 1].b) and not w.comma:
            if i + 2 < len(ws) and ws[i + 2].name and not ws[i + 2].desig and not ws[i + 2].numw:
                kind = DESIGNATORS[w.lo]
                roles[i] = Role.DESIGNATOR
                roles[i + 1] = Role.LEVEL_NUMBER if kind in _LEVEL_DESIG else Role.UNIT_NUMBER
                names, typ, _end, fl = self._name_type(ws, i + 2, self.stops(ln.li))
                if typ is None or not names:
                    return None
                for x in names:
                    roles[x] = Role.STREET_NAME
                roles[typ] = Role.STREET_TYPE
                s = Street(ln.li, i, typ, UnitKind.STREET, roles, {Flag.TYPED} | fl)
                s.name_start = names[0]
                self._extend_back(s)
                return s if self._typed_ok(s) else None
            return None
        if not w.numw or w.text.strip().startswith("$"):
            return None
        if i + 1 >= len(ws):
            return None
        if w.comma and (
            not self._typed_after(ws, i + 1)
            or (w.b.isdigit() and len(w.b) == 4 and 1900 <= int(w.b) <= 2099)
            or (i > 0 and (ws[i - 1].lo in MONTHS or ws[i - 1].digits))
        ):
            return None  # '12, Smith Street' only when a name and a street type follow, and never a date's year
        if ws[i + 1].lo in MEASURE_WORDS:
            return None
        roles[i] = Role.UNIT_NUMBER_AND_NUMBER if HN_SLASH.fullmatch(w.b) else Role.NUMBER
        names, typ, end, fl = self._name_type(ws, i + 1, self.stops(ln.li))
        flags |= fl
        if not names:
            return None
        for x in names:
            roles[x] = Role.STREET_NAME
        if typ is not None:
            roles[typ] = Role.STREET_TYPE
            flags.add(Flag.TYPED)
            end = typ
            sx = self._suffix_after(ln, typ)
            if sx is not None:
                roles[sx] = Role.STREET_SUFFIX
                end = sx
        else:
            if len(names) > 3:
                return None
            flags.add(Flag.UNTYPED)
        s = Street(ln.li, i, end, UnitKind.STREET, roles, flags)
        s.name_start = names[0]
        self._extend_back(s)
        if Flag.TYPED in flags and not self._typed_ok(s):
            return None
        return s

    def _typed_after(self, ws: list[Word], i: int) -> bool:
        """A street name and a street type start at word ``i``."""
        names, typ, _end, _fl = self._name_type(ws, i, set())
        return bool(names) and typ is not None

    def _numberless_at(self, ln: Line, i: int) -> Street | None:
        """A numberless 'Name Type' street line, with a street-type suffix if one follows the type ('Smith Street
        North,'): at the line start, after a separator or a label, or in prose after a lower-case word ('injured on
        Victoria Road, Ryde NSW 2112').  It becomes a unit only with a place carrying a state or postcode (a postcode in
        prose) that directly follows it, never alone (owner question 3 and the ruling on open point 6)."""
        ws = ln.words
        prose = False
        if i > 0 and not (ws[i - 1].sep or ws[i - 1].label or self.label_ending_before(ln.li, i)):
            if not (ws[i - 1].lowr and ws[i - 1].b.isalpha()):
                return None
            prose = True
        names, typ, _end, fl = self._name_type(ws, i, self.stops(ln.li))
        if typ is None or not names or Flag.MONTH in fl or len(names) > 3:
            return None
        if TYPE_CANON[ws[typ].lo] in NUMBERLESS_BANNED_TYPES:
            return None
        if any(ws[x].lo in PUBLIC_BODY_WORDS or ws[x].lo in ORG_WORDS or ws[x].title for x in names):
            return None
        roles = {x: Role.STREET_NAME for x in names}
        roles[typ] = Role.STREET_TYPE
        end = typ
        sx = self._suffix_after(ln, typ)  # read as on numbered streets (finding F7)
        if sx is not None:
            roles[sx] = Role.STREET_SUFFIX
            end = sx
        flags = {Flag.TYPED, Flag.NUMBERLESS}
        if prose:
            flags.add(Flag.PROSE)
        s = Street(ln.li, i, end, UnitKind.STREET_NUMBERLESS, roles, flags)
        s.name_start = names[0]
        return s

    # ------------------------------------------------------------------ what stands in front of a street
    def _lone_designator_above(self, s: Street) -> Line | None:
        """A line above holding only a unit or level word ('Unit' over '6 10 Smith Street'), not a label box."""
        up = self.geo.above(self.L[s.li])
        if up is None or len(up.words) != 1 or up.box is not None:
            return None
        w = up.words[0]
        return up if w.desig and not w.label else None

    def _extend_back(self, s: Street) -> None:
        """Backward over designator pairs, ordinal floors, 'Building X' and building names on the line."""
        ws = self.L[s.li].words
        j = s.start - 1
        while j >= 0:
            w = ws[j]
            if w.sep or w.label:
                break
            # designator + number ('Level 2,', 'Suite 3')
            if BOXNUM.fullmatch(w.b) and j - 1 >= 0 and ws[j - 1].desig:
                kind = DESIGNATORS[ws[j - 1].lo]
                s.roles[j - 1] = Role.DESIGNATOR
                s.roles[j] = Role.LEVEL_NUMBER if kind in _LEVEL_DESIG else Role.UNIT_NUMBER
                s.start = j - 1
                j -= 2
                continue
            # a unit word before a slash form ('Unit 5/3', 'Shop 1/115')
            if (
                j + 1 == s.start
                and w.desig
                and DESIGNATORS[w.lo] == "unit"
                and not w.comma
                and HN_SLASH.fullmatch(ws[s.start].b)
            ):
                s.roles[j] = Role.DESIGNATOR
                s.start = j
                j -= 1
                continue
            # 'Unit' alone on the line above two bare numbers ('Unit' / '6 10 Smith Street')
            if j == 0 and j + 1 == s.start and BOXNUM.fullmatch(w.b) and not w.comma and not s.above:
                up = self._lone_designator_above(s)
                if up is not None:
                    kind = DESIGNATORS[up.words[0].lo]
                    s.roles[j] = Role.LEVEL_NUMBER if kind in _LEVEL_DESIG else Role.UNIT_NUMBER
                    s.start = j
                    s.above = [(up.li, {0: Role.DESIGNATOR})]
                    break
            # ordinal floor ('1st Floor,')
            if w.lo in LEVEL_WORDS and j - 1 >= 0 and ORDINAL.fullmatch(ws[j - 1].b):
                s.roles[j - 1] = Role.LEVEL_NUMBER
                s.roles[j] = Role.DESIGNATOR
                s.start = j - 1
                j -= 2
                continue
            # floor types with no number ('Ground Floor,', 'Lower Ground Floor', 'Mezzanine,')
            lv: int | None = None
            if w.level_type:
                lv = j
            elif w.lo in LEVEL_WORDS and (w.cap or w.upper) and j - 1 >= 0 and ws[j - 1].level_type:
                lv = j - 1
            if lv is not None:
                if (
                    ws[lv].lo == "ground"
                    and lv - 1 >= 0
                    and ws[lv - 1].lo in ("lower", "upper")
                    and (ws[lv - 1].cap or ws[lv - 1].upper)
                ):
                    lv -= 1
                for x in range(lv, j + 1):
                    s.roles[x] = Role.DESIGNATOR
                s.start = lv
                j = lv - 1
                continue
            # 'Building C,', 'Tower 2': the building word is a designator and the letter or number a unit number
            if (
                LETTER_ID.fullmatch(w.b)
                and j - 1 >= 0
                and ws[j - 1].lo in BUILDING_DESIG
                and ws[j - 1].cap
                and not ws[j - 1].comma
            ):
                s.roles[j - 1] = Role.DESIGNATOR
                s.roles[j] = Role.UNIT_NUMBER
                s.start = j - 1
                j -= 2
                continue
            # a building name: any capitalised run sandwiched between designators and the number is the building's name
            r0 = j
            while (
                r0 - 1 >= 0
                and j - r0 < 3
                and ws[r0 - 1].name
                and not ws[r0 - 1].desig
                and not ws[r0 - 1].title
                and not ws[r0 - 1].comma
                and not ws[r0 - 1].blocked
            ):
                r0 -= 1
            if (
                w.name
                and not w.desig
                and not w.title
                and not w.blocked
                and r0 - 2 >= 0
                and BOXNUM.fullmatch(ws[r0 - 1].b)
                and ws[r0 - 2].desig
            ):
                for x in range(r0, j + 1):
                    s.roles[x] = Role.BUILDING
                s.start = r0
                j = r0 - 1
                continue
            # a capitalised run ending in a building word ('Riverside Medical Chambers'), or after a designator pair
            if (
                w.name
                and not w.title
                and not w.blocked
                and w.lo not in ORG_NOT_BUILDING
                and w.lo not in OCCUPATION_WORDS
                and not w.desig
            ):
                r = j
                while (
                    r - 1 >= 0
                    and r > j - 3
                    and ws[r - 1].name
                    and not ws[r - 1].comma
                    and not ws[r - 1].title
                    and not ws[r - 1].blocked
                    and ws[r - 1].lo not in ORG_NOT_BUILDING
                    and ws[r - 1].lo not in OCCUPATION_WORDS
                    and not ws[r - 1].desig
                ):
                    r -= 1
                before_is_designator = r - 1 >= 1 and bool(BOXNUM.fullmatch(ws[r - 1].b)) and ws[r - 2].desig
                if ws[j].lo in BUILDING_WORDS or before_is_designator:
                    for x in range(r, j + 1):
                        s.roles[x] = Role.BUILDING
                    s.start = r
                    j = r - 1
                    continue
            break
