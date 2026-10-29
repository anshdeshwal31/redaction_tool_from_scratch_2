"""The last stages: place-only units, the repeat sweep, the lower-case whole-line pattern, places kept on their own,
and the canonical key of each unit."""

from __future__ import annotations

from mlredact.text.address.forms import FormStage
from mlredact.text.address.internal import AnchorKind, Flag, KeptB, LabelRole, UnitB
from mlredact.text.address.model import AddressKey, Band, BoxRole, KeptKind, Pos, Role, UnitKind
from mlredact.text.address.tables import (
    BOX_KEY_KINDS,
    CONNECTORS,
    COUNTRY,
    FUNCTION_WORDS,
    GLUED,
    LOCALITY_ROLES,
    OTHER_COUNTRIES,
    STREET_ROLES,
    SUFFIX_CANON,
    TYPE_CANON,
    level_key,
)
from mlredact.text.address.words import Line, Word, pc_value, state_at
from mlredact.text.gazetteer import STATE_CODES, fold_name, range_states

_Sig = tuple[int, tuple[str, ...], tuple[str, ...], str | None, str | None]
_AND_NUMBER = frozenset({Role.NUMBER, Role.UNIT_NUMBER_AND_NUMBER})


class FinalStage(FormStage):
    # ------------------------------------------------------------------ place-only units
    def place_only_units(self) -> None:
        for ln in self.L:
            ws = ln.words
            for a in self.anchors[ln.li]:
                if a.kind is not AnchorKind.STRONG:
                    continue
                st0 = a.start
                if any((ln.li, x) in self.used for x in range(st0, a.end + 1)):
                    continue
                run: list[int] = []
                k = st0 - 1
                while k >= 0 and len(run) < 4:
                    w = ws[k]
                    if (ln.li, k) in self.used or not w.name or w.stopw:
                        break
                    if (w.comma or w.sbreak) and k != st0 - 1:
                        break
                    run.insert(0, k)
                    k -= 1
                if not run:
                    continue
                best: list[int] | None = None
                for n in range(len(run), 0, -1):
                    sub = run[-n:]
                    if self.gaz.states_of([ws[x].b for x in sub], fuzzy=True, state=a.code):
                        best = sub
                        break
                if best is None:
                    self.notes.append(f"no-locality:{ln.li}:{st0}")
                    continue
                own = run[0] == 0 or (run[0] - 1 >= 0 and (ws[run[0] - 1].sep or ws[run[0] - 1].label))
                sub = run if own else best
                roles = self.place_roles(ln.li, a, sub)
                if self.place_kind(ln, sub[0]):
                    self.kept.append(KeptB(KeptKind.PLACE_AFTER_CUE, sorted(roles), after_cue=True))
                    self.used.update(roles)
                    continue
                u = UnitB(UnitKind.PLACE_ONLY, roles, how=["place_only"])
                u.partial.extend(self._partials(u))
                self.add_unit(u)

    # ------------------------------------------------------------------ the lower-case whole-line pattern (A4)
    def lowercase_lines(self) -> None:
        """A wholly lower-case address line: number, name, type, suburb, state code, in-range postcode."""
        for ln in self.L:
            ws = ln.words
            for k in range(4, len(ws)):
                pcv = pc_value(ws[k].b)
                if pcv is None or (ln.li, k) in self.used:
                    continue
                st = ws[k - 1].b
                if st.lower() != st or st.upper() not in STATE_CODES or not self.gaz.ok_pair(st.upper(), pcv):
                    continue
                for n in (3, 2, 1):
                    sub = list(range(k - 1 - n, k - 1))
                    if sub[0] < 2 or not all(ws[x].b.isalpha() and ws[x].b.islower() for x in sub):
                        continue
                    if not self.gaz.states_of([ws[x].b for x in sub], state=st.upper()):
                        continue
                    t = sub[0] - 1
                    if ws[t].lo not in TYPE_CANON:
                        continue
                    names: list[int] = []
                    j = t - 1
                    while (
                        j >= 0
                        and ws[j].b.isalpha()
                        and ws[j].b.islower()
                        and len(names) < 3
                        and ws[j].lo not in CONNECTORS
                    ):
                        names.insert(0, j)
                        j -= 1
                    if not names or j < 0 or not ws[j].numw:
                        continue
                    roles = {
                        (ln.li, j): Role.NUMBER,
                        (ln.li, t): Role.STREET_TYPE,
                        (ln.li, k - 1): Role.STATE,
                        (ln.li, k): Role.POSTCODE,
                    }
                    for x in names:
                        roles[(ln.li, x)] = Role.STREET_NAME
                    for x in sub:
                        roles[(ln.li, x)] = Role.SUBURB
                    if any(p in self.used for p in roles):
                        continue
                    self.add_unit(UnitB(UnitKind.STREET, roles, how=["lowercase_line"]))
                    break

    # ------------------------------------------------------------------ places kept on their own
    def _standalone(self, ws: list[Word], i: int, j: int) -> bool:
        """The words i..j are not part of a longer capitalised name (an organisation or a building)."""
        prev = ws[i - 1] if i > 0 else None
        nxt = ws[j + 1] if j + 1 < len(ws) else None
        if (
            prev is not None
            and prev.name
            and not prev.comma
            and not prev.stop
            and not prev.title
            and prev.lo not in FUNCTION_WORDS
        ):
            return False
        return not (
            nxt is not None
            and not ws[j].comma
            and not ws[j].stop
            and nxt.name
            and state_at(ws, j + 1) is None
            and not (j + 2 < len(ws) and state_at(ws, j + 2))
        )

    def record_kept(self) -> None:
        """Record the positions of places mentioned on their own (verification exempts them).  A place is recorded only
        when it is positively on its own: in running prose or after a place-kind label or phrase, and never in a form
        value cell or a header, footer or letterhead band."""
        for ln in self.L:
            ws = ln.words
            prose = self.is_prose(ln)
            form_value = (
                ln.box is BoxRole.VALUE
                or any(
                    o.x1 <= ln.x0 + 1 and self.labels[o.li] and self.labels[o.li][-1].role is not LabelRole.OTHER
                    for o in self.geo.row_mates(ln)
                )
                or any(x.role is not LabelRole.OTHER for x in self.labels[ln.li])
            )
            excluded = ln.band in (Band.HEADER, Band.FOOTER, Band.LETTERHEAD) or form_value
            k = 0
            while k < len(ws):
                if (ln.li, k) in self.used:
                    k += 1
                    continue
                # positively on its own: after a place-kind label or phrase, or in running prose
                in_prose = prose or (k > 0 and ws[k - 1].lowr and ws[k - 1].b.isalpha())
                if not self.place_kind(ln, k) and (excluded or not in_prose):
                    k += 1
                    continue
                got = self._kept_at(ln, ws, k, in_prose)
                if got is None:
                    k += 1
                    continue
                kind, idx = got
                self.kept.append(KeptB(kind, [(ln.li, x) for x in idx]))
                for x in idx:
                    self.used.add((ln.li, x))
                k = idx[-1] + 1

    def _kept_at(self, ln: Line, ws: list[Word], k: int, in_prose: bool) -> tuple[KeptKind, list[int]] | None:
        w = ws[k]
        # a street named in prose without a number ('walks along Smith Street')
        if in_prose and w.name and not w.title and (k == 0 or not ws[k - 1].numw):
            names, typ, _end, fl = self._name_type(ws, k, self.stops(ln.li))
            if (
                typ is not None
                and names
                and len(names) <= 3
                and Flag.MONTH not in fl
                and not any((ln.li, x) in self.used for x in range(k, typ + 1))
                and not (k > 0 and ws[k - 1].name and not ws[k - 1].comma)
            ):
                sx = self._suffix_after(ln, typ)
                return KeptKind.STREET_IN_PROSE, list(range(k, (sx if sx is not None else typ) + 1))
        # 'postcode 2750'
        if pc_value(w.b) is not None and k > 0 and ws[k - 1].lo in ("postcode", "post"):
            return KeptKind.POSTCODE, [k]
        # countries
        for n in (2, 1):
            key = tuple(x.lo for x in ws[k : k + n])
            is_country = len(key) == n and (key in OTHER_COUNTRIES or (n == 1 and key[0] in COUNTRY))
            if is_country and all(x.cap for x in ws[k : k + n]):
                return KeptKind.COUNTRY, list(range(k, k + n))
        # a state on its own
        for n in (3, 2, 1):
            st = state_at(ws, k + n - 1)
            if st is not None and st[0] == k:
                return KeptKind.STATE, list(range(k, k + n))
        # a real locality on its own (not inside a longer capitalised name), with its state if any
        if w.name and not w.stopw:
            for n in (3, 2, 1):
                run = ws[k : k + n]
                if len(run) < n or not all(x.name and not x.stopw for x in run):
                    continue
                if any(x.comma or x.stop for x in run[:-1]):
                    continue
                if not self.gaz.states_of([x.b for x in run]) or not self._standalone(ws, k, k + n - 1):
                    continue
                idx = list(range(k, k + n))
                st2 = state_at(ws, k + n) if not run[-1].comma else None
                if st2 is not None and st2[0] == k + n:
                    idx.append(k + n)
                return KeptKind.PLACE, idx
        return None

    # ------------------------------------------------------------------ repeat sweep
    def _address_position(self, ln: Line, idx: list[int], what: str) -> bool:
        """The plan's address positions for a repeat: a line of its own, a letterhead or footer band, right after a
        street-like run, or a line of a block that holds a found unit."""
        ws = ln.words
        if what == "street":
            return True
        before = [w for w in ws[: idx[0]] if w.alnum_text]
        after = ws[idx[-1] + 1 :]
        rest_after = [w for w in after if w.alnum_text]
        place_tail = (
            all(
                state_at(after, k) is not None or pc_value(w.b) is not None or w.lo in COUNTRY or w.sep
                for k, w in enumerate(after)
            )
            if after
            else True
        )
        own = not before or (idx[0] - 1 >= 0 and ws[idx[0] - 1].sep)
        if own and (not rest_after or place_tail):
            return True
        if ln.band in (Band.HEADER, Band.FOOTER, Band.LETTERHEAD):
            return True
        if idx[0] >= 2 and ws[idx[0] - 1].typ and any(w.numw for w in ws[max(0, idx[0] - 5) : idx[0]]):
            return True
        for nb in (self.geo.below(ln), self.geo.above(ln)):
            if nb is not None and nb.li in self.unit_lines and own:
                return True
        return False

    def repeat_sweep(self) -> None:
        by_num: dict[str, list[_Sig]] = {}
        by_first: dict[str, list[_Sig]] = {}
        for n, u in enumerate(self.units):
            pos = sorted(u.roles)
            sw = tuple(fold_name(self.word(p).b) for p in pos if u.roles[p] is Role.SUBURB)
            nm = tuple(self.word(p).lo for p in pos if u.roles[p] is Role.STREET_NAME)
            types = [p for p in pos if u.roles[p] is Role.STREET_TYPE]
            nums = [p for p in pos if u.roles[p] in _AND_NUMBER]
            ty = TYPE_CANON.get(self.word(types[-1]).lo) if types else None
            nu = self.word(nums[0]).lo if nums else None
            sig: _Sig = (n, sw, nm, ty, nu)
            if nm and nu is not None and ty is not None:
                by_num.setdefault(nu, []).append(sig)
            if sw:
                by_first.setdefault(sw[0], []).append(sig)
        if not by_num and not by_first:
            return
        for ln in self.L:
            ws = ln.words
            k = 0
            while k < len(ws):
                if (ln.li, k) in self.used:
                    k += 1
                    continue
                hit = self._repeat_hit(ws, k, by_num, by_first)
                if hit is None:
                    k += 1
                    continue
                what, n, idx, sw = hit
                if self.is_prose(ln) or self.place_kind(ln, idx[0]) or not self._address_position(ln, idx, what):
                    self.kept.append(KeptB(KeptKind.REPEAT_IN_PROSE, [(ln.li, x) for x in idx]))
                    for x in idx:
                        self.used.add((ln.li, x))
                    k = idx[-1] + 1
                    continue
                roles: dict[Pos, Role] = {}
                if what == "street":
                    src = self.units[n]
                    names_n = sum(1 for r in src.roles.values() if r is Role.STREET_NAME)
                    roles[(ln.li, idx[0])] = Role.NUMBER
                    for x in idx[1 : 1 + names_n]:
                        roles[(ln.li, x)] = Role.STREET_NAME
                    roles[(ln.li, idx[-1])] = Role.STREET_TYPE
                    j = idx[-1] + 1
                    if sw and j + len(sw) <= len(ws) and tuple(fold_name(w.b) for w in ws[j : j + len(sw)]) == sw:
                        for x in range(j, j + len(sw)):
                            roles[(ln.li, x)] = Role.SUBURB
                else:
                    for x in idx:
                        roles[(ln.li, x)] = Role.SUBURB
                u = UnitB(
                    UnitKind.STREET if what == "street" else UnitKind.PLACE_ONLY,
                    roles,
                    how=["repeat_sweep"],
                    repeat_src=self.units[n],
                )
                self.add_unit(u)
                k = max(p[1] for p in roles) + 1

    def _repeat_hit(
        self, ws: list[Word], k: int, by_num: dict[str, list[_Sig]], by_first: dict[str, list[_Sig]]
    ) -> tuple[str, int, list[int], tuple[str, ...]] | None:
        """A number + street name + type, or a suburb, of a found unit starting at word ``k``."""
        for n, sw, nm, ty, _nu in by_num.get(ws[k].lo, ()):
            span = len(nm) + 2
            if k + span > len(ws):
                continue
            if tuple(w.lo for w in ws[k + 1 : k + 1 + len(nm)]) != nm:
                continue
            if TYPE_CANON.get(ws[k + 1 + len(nm)].lo) != ty:
                continue
            return "street", n, list(range(k, k + span)), sw
        if not ws[k].cap:
            return None  # a suburb is written with a capital
        for n, sw, _nm, _ty, _nu in by_first.get(fold_name(ws[k].b), ()):
            if (
                k + len(sw) <= len(ws)
                and tuple(fold_name(w.b) for w in ws[k : k + len(sw)]) == sw
                and all(w.cap for w in ws[k : k + len(sw)])
                and self._standalone(ws, k, k + len(sw) - 1)
            ):
                return "suburb", n, list(range(k, k + len(sw))), sw
        return None

    # ------------------------------------------------------------------ finish: kinds, keys
    def finish_unit(self, u: UnitB) -> None:
        pos = sorted(u.roles)
        u.street = [p for p in pos if u.roles[p] in STREET_ROLES]
        u.locality = [p for p in pos if u.roles[p] in LOCALITY_ROLES]
        if not u.street and u.kind is not UnitKind.DX:
            u.kind = UnitKind.PLACE_ONLY
        elif u.kind in (UnitKind.STREET, UnitKind.STREET_NUMBERLESS):
            rs = set(u.roles.values())
            if Role.BOX_KIND in rs:
                u.kind = UnitKind.BOX
            elif Role.CORNER in rs:
                u.kind = UnitKind.CORNER
            elif rs & _AND_NUMBER:
                u.kind = UnitKind.STREET
            elif Role.DX in rs:
                u.kind = UnitKind.DX
            else:
                u.kind = UnitKind.STREET_NUMBERLESS if Role.UNIT_NUMBER not in rs else UnitKind.STREET
        u.key = self._key(u)

    def _key(self, u: UnitB) -> AddressKey:
        by: dict[Role, list[Word]] = {}
        for p in sorted(u.roles):
            by.setdefault(u.roles[p], []).append(self.word(p))
        street_kind = ""
        parts: tuple[tuple[str, ...], ...] = ()
        if Role.BOX_KIND in by:
            kind = " ".join(w.lo.replace(".", "") for w in by[Role.BOX_KIND])
            kind = BOX_KEY_KINDS.get(kind, kind)
            number = by[Role.BOX_NUMBER][0].b.upper() if Role.BOX_NUMBER in by else ""
            street_kind, parts = "BOX", ((kind.upper(),), (number,))
        elif Role.DX in by:
            street_kind, parts = "DX", ((by[Role.DX_NUMBER][0].b if Role.DX_NUMBER in by else "",),)
        elif Role.STREET_NAME in by:
            nums = [w.b.upper() for w in by.get(Role.NUMBER, [])] + [
                w.b.upper().split("/")[-1] for w in by.get(Role.UNIT_NUMBER_AND_NUMBER, [])
            ]
            # folded as one string: 'Smith-Jones' and 'Smith Jones' are one name
            names = tuple(fold_name(" ".join(w.b for w in by[Role.STREET_NAME])).upper().split())
            types = tuple(TYPE_CANON.get(w.lo, w.lo.upper()) for w in by.get(Role.STREET_TYPE, []))
            suffix = tuple(SUFFIX_CANON.get(w.lo, w.b.upper()) for w in by.get(Role.STREET_SUFFIX, []))
            street_kind, parts = "ST", (tuple(nums), names, types, suffix)
        elif Role.BUILDING in by:
            # a property line with no street ('Kilbride Station via Berrima'): keyed by the property's name
            street_kind, parts = "PROP", (tuple(fold_name(w.b).upper() for w in by[Role.BUILDING]),)
        subs: list[tuple[str, str]] = []
        for role in (Role.UNIT_NUMBER, Role.LEVEL_NUMBER):
            for w in by.get(role, []):
                # a level by its number: '4th Floor' and 'Floor 4' are one level
                subs.append((role.value, level_key(w.b) if role is Role.LEVEL_NUMBER else w.b.upper()))
        for w in by.get(Role.UNIT_NUMBER_AND_NUMBER, []):
            subs.append((Role.UNIT_NUMBER.value, w.b.upper().split("/")[0]))
        suburb = ""
        state: str | None = None
        if Role.SUBURB in by:  # the list's form: 'North Balwyn' is 'Balwyn North'
            suburb = self.gaz.listed_form(fold_name(" ".join(w.b for w in by[Role.SUBURB]))).upper()
            if not u.overseas:
                state = self._state_of(by)
        return AddressKey(street_kind, parts, tuple(sorted(subs)), suburb, state)

    def _state_of(self, by: dict[Role, list[Word]]) -> str | None:
        code: str | None = None
        st = by.get(Role.STATE, [])
        if st:
            got = state_at(st, len(st) - 1, ocr=True)
            code = got[1] if got else None
        for w in by.get(Role.STATE_AND_POSTCODE, []):
            g = GLUED.fullmatch(w.b)
            if g is not None:
                code = g.group(1)
        if code is None:
            for w in by.get(Role.POSTCODE, []) + by.get(Role.STATE_AND_POSTCODE, []):
                v = pc_value(GLUED.sub(r"\2", w.b))
                if v is not None:
                    states = range_states(v)
                    sts = states & self.gaz.states_of([x.b for x in by[Role.SUBURB]])
                    pick = sorted(sts) or sorted(states)
                    code = pick[0] if pick else None
        return code
