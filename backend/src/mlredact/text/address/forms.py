"""The form layer: address parts that sit in their own boxes (a street box, a suburb box, a state box, a postcode box)
under compound or colon-less labels, read as one unit by geometry."""

from __future__ import annotations

from collections.abc import Iterator

from mlredact.text.address.geometry import Span, x_overlap
from mlredact.text.address.internal import (
    LOCALITY_PART_LABELS,
    STREET_PART_LABELS,
    Field,
    LabelRole,
    UnitB,
)
from mlredact.text.address.join import JoinStage
from mlredact.text.address.model import Pos, Role, UnitKind
from mlredact.text.address.tables import BOXNUM, DESIGNATORS, HN_SLASH, LOCALITY_ROLES, PLACE_KIND_CUES
from mlredact.text.address.words import Line, pc_value, state_at

_Box = tuple[float, float, float, float, tuple[int, int]]
_LEVEL_DESIG = frozenset({"level", "floor"})


class FormStage(JoinStage):
    def _fields(self) -> list[Field]:
        """Every label with the positions of its value, in reading order, with box geometry."""
        out: list[Field] = []
        for ln in self.L:
            labs = self.labels[ln.li]
            for n, lab in enumerate(labs):
                nxt_start = labs[n + 1].start if n + 1 < len(labs) else len(ln.words)
                vals = [(ln.li, x) for x in range(lab.end + 1, nxt_start)]
                vals = [p for p in vals if not self.word(p).sep]
                f = Field(ln.li, lab, lab.role, vals, [(ln.li, lab.start, lab.end)])
                if not vals and lab.end == len(ln.words) - 1:
                    v = self._value_line(ln, lab.start, lab.end, lab.role)
                    if v is not None:
                        f.vals = [(v.li, x) for x in range(len(v.words))]
                        f.boxes.append((v.li, 0, len(v.words) - 1))
                if vals:
                    f.boxes.append((ln.li, vals[0][1], vals[-1][1]))
                out.append(f)
        out.sort(key=lambda f: (self.geo.row_index(self.L[f.li]), self.L[f.li].x0, f.lab.start))
        return out

    def _starts_with_label(self, o: Line) -> bool:
        return bool(self.labels[o.li]) and self.labels[o.li][0].start == 0

    def _value_line(self, ln: Line, start: int, end: int, role: LabelRole) -> Line | None:
        """The value box of a label box: the next box to the right on its row, else the box below.  A box that starts
        with a label is never a value (sub-labels are labels)."""
        span = self.span(ln.li, start, end)
        right = [o for o in self.geo.right_of(ln, span[1]) if o.x0 - span[1] <= max(10 * ln.h, 3 * (span[1] - span[0]))]
        if right:
            o = min(right, key=lambda o: o.x0)
            if not self._starts_with_label(o):
                return o
            if role is LabelRole.ADDRESS:
                return None  # an address label heading a row of part boxes
        below = self.geo.below(ln, span, max_gap=1.2)
        if below is not None and not self._starts_with_label(below):
            return below
        return None

    def _label_continuation(self, f: Field) -> bool:
        ln = self.L[f.li]
        if f.lab.start != 0 or not ln.words[0].lowr or f.vals:
            return False
        up = self.geo.above(ln)
        return up is not None and self._starts_with_label(up) and self.labels[up.li][0].role is not LabelRole.OTHER

    def _field_boxes(self, f: Field) -> Iterator[_Box]:
        for li, a, b in f.boxes:
            x0, x1 = self.span(li, a, b)
            ln = self.L[li]
            yield (x0, ln.y0, x1, ln.y1, (ln.page, ln.frame))

    def _field_x(self, f: Field) -> Span:
        xs = [self.span(li, a, b) for li, a, b in f.boxes]
        return (min(x[0] for x in xs), max(x[1] for x in xs))

    def form_units(self) -> None:
        fields = self._fields()
        groups: list[list[Field]] = []
        cur: list[Field] | None = None
        cur_boxes: list[_Box] = []
        place_kind_ctx = False

        def near(f: Field) -> bool:
            if not cur:
                return False
            fl = self.L[f.li]
            last_y1 = max(b[3] for b in cur_boxes)
            if (fl.page, fl.frame) != cur_boxes[-1][4] or fl.y0 - last_y1 > 5 * fl.h:
                return False
            fx = self._field_x(f)
            return any(x_overlap(fx, (b[0], b[2])) > 0 or abs(b[1] - fl.y0) < 0.5 * fl.h for b in cur_boxes)

        for f in fields:
            role = f.role
            if role is LabelRole.OTHER and self._label_continuation(f):
                continue
            if role is LabelRole.OTHER:
                txt = " ".join(w.lo for w in self.L[f.li].words[f.lab.start : f.lab.end + 1])
                place_kind_ctx = any(c in txt for c in PLACE_KIND_CUES)
                cur = None
                continue
            if role is LabelRole.UNREADABLE:
                if cur is not None and near(f):
                    cur.append(f)
                    cur_boxes.extend(self._field_boxes(f))
                continue
            filled = {g.role for g in cur} if cur else set()
            opens = role is LabelRole.ADDRESS or cur is None or not near(f) or role in filled
            if (
                role is LabelRole.STREET
                and cur is not None
                and near(f)
                and LabelRole.STREET not in filled
                and LabelRole.ADDRESS not in filled
            ):
                opens = False
            if opens:
                if role in LOCALITY_PART_LABELS and place_kind_ctx and cur is None:
                    cur = None
                    continue
                cur = [f]
                groups.append(cur)
                cur_boxes = list(self._field_boxes(f))
                place_kind_ctx = False
            else:
                assert cur is not None
                cur.append(f)
                cur_boxes.extend(self._field_boxes(f))
        for g in groups:
            self._form_unit(g)

    def _form_unit(self, g: list[Field]) -> None:
        roles: dict[Pos, Role] = {}
        loc: list[Pos] = []
        base: UnitB | None = None
        street_vals = False
        pending_unread: list[Field] = []
        for f in g:
            vals = f.vals
            if not vals:
                continue
            hit = [self.unit_at[p] for p in vals if p in self.unit_at]
            if hit:
                if base is None and f.role in STREET_PART_LABELS:
                    base = min(hit, key=lambda u: u.seq)
                continue
            if any(p in self.used for p in vals):
                continue
            r = f.role
            ws = [self.word(p) for p in vals]
            if r is LabelRole.UNIT:
                for n, (p, w) in enumerate(zip(vals, ws, strict=True)):
                    if w.desig and n + 1 < len(ws) and BOXNUM.fullmatch(ws[n + 1].b):
                        kind = DESIGNATORS[w.lo]
                        roles[p] = Role.DESIGNATOR
                        roles[vals[n + 1]] = Role.LEVEL_NUMBER if kind in _LEVEL_DESIG else Role.UNIT_NUMBER
                        street_vals = True
                        break
                    if BOXNUM.fullmatch(w.b) or w.numw:
                        roles[p] = Role.UNIT_NUMBER
                        street_vals = True
                        break
            elif r is LabelRole.STREET_NO:
                for p, w in zip(vals, ws, strict=True):
                    if w.numw:
                        roles[p] = Role.UNIT_NUMBER_AND_NUMBER if HN_SLASH.fullmatch(w.b) else Role.NUMBER
                        street_vals = True
                        break
            elif r in (LabelRole.STREET_NAME, LabelRole.STREET, LabelRole.ADDRESS):
                names = [(p, w) for p, w in zip(vals, ws, strict=True) if w.name or w.numw or w.typ]
                if not names:
                    continue
                for n, (p, w) in enumerate(names):
                    if w.numw and n == 0:
                        roles[p] = Role.NUMBER
                    elif w.typ and n == len(names) - 1 and n > 0:
                        roles[p] = Role.STREET_TYPE
                    else:
                        roles[p] = Role.STREET_NAME
                street_vals = True
            elif r is LabelRole.SUBURB:
                run: list[Pos] = []
                for p, w in zip(vals, ws, strict=True):
                    if not w.name or len(run) >= 4:
                        break
                    run.append(p)
                if run and self.gaz.states_of([self.word(p).b for p in run]):
                    for p in run:
                        roles[p] = Role.SUBURB
                        loc.append(p)
            elif r is LabelRole.STATE:
                for k in range(len(ws)):
                    st = state_at(ws, k)
                    if st is not None and st[0] == 0:
                        for p in vals[: k + 1]:
                            roles[p] = Role.STATE
                            loc.append(p)
                        break
            elif r is LabelRole.POSTCODE:
                for p, w in zip(vals, ws, strict=True):
                    if pc_value(w.b) is not None:
                        roles[p] = Role.POSTCODE
                        loc.append(p)
                        break
            elif r is LabelRole.UNREADABLE:
                pending_unread.append(f)
        # unreadable labels: classify the value boxes by shape (suburb, state, postcode)
        for f in pending_unread:
            vals = f.vals
            if not vals or any(p in self.used or p in roles for p in vals):
                continue
            ws = [self.word(p) for p in vals]
            have = set(roles.values())
            last_state = state_at(ws, len(ws) - 1)
            if Role.SUBURB not in have and all(w.name for w in ws) and self.gaz.states_of([w.b for w in ws]):
                for p in vals:
                    roles[p] = Role.SUBURB
                    loc.append(p)
            elif Role.STATE not in have and last_state is not None and last_state[0] == 0:
                for p in vals:
                    roles[p] = Role.STATE
                    loc.append(p)
            elif Role.POSTCODE not in have and len(ws) == 1 and pc_value(ws[0].b) is not None:
                roles[vals[0]] = Role.POSTCODE
                loc.append(vals[0])
        if base is not None:
            if not loc:
                return
            if any(r in LOCALITY_ROLES for r in base.roles.values()):
                return
            for p in loc:
                self.extend_unit(base, p, roles[p])
            base.how.append("form_locality")
            return
        if not street_vals and not loc:
            return
        rv = set(roles.values())
        if not street_vals and not ({Role.SUBURB, Role.STATE, Role.POSTCODE} <= rv):
            return
        if Role.NUMBER in rv or Role.UNIT_NUMBER_AND_NUMBER in rv:
            kind = UnitKind.STREET
        else:
            kind = UnitKind.STREET_NUMBERLESS if street_vals else UnitKind.PLACE_ONLY
        if kind is UnitKind.STREET_NUMBERLESS and not loc:
            return
        u = UnitB(kind, roles, how=["form"])
        if any(p in self.used for p in roles):
            return
        self.add_unit(u)
