"""Raster editing, typesetting, labels, notice placement, text layer and V1/V2 on built PDFs."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw, ImageFont

from mlredact.config.schema import OutputConfig, RedactionRenderConfig
from mlredact.core.geometry import BBox
from mlredact.core.model import PageGeometry, RenderOp, Token
from mlredact.core.types import ActionKind, EntityType
from mlredact.pdfout.builder import PageSpec, build_pdf
from mlredact.pdfout.encode import encode_raster
from mlredact.pdfout.textlayer import _escape, page_text_layer, text_layer_items
from mlredact.render.label import place_notice
from mlredact.render.raster import apply_ops
from mlredact.render.typeset import erased_cleanly, typeset
from mlredact.resources.loader import font_path
from mlredact.verify.checks import check_structure, check_text_leak, text_segments

FONT = ImageFont.truetype(str(font_path("Sans", "Regular")), 40)


def page_with_words(words: list[str], width: int = 1400, height: int = 400) -> tuple[npt.NDArray[np.uint8], list[BBox]]:
    """White page with one line of words; returns the raster and each word's tight ink box."""
    img = Image.new("RGB", (width, height), (250, 250, 248))
    draw = ImageDraw.Draw(img)
    x, boxes = 60, []
    for w in words:
        draw.text((x, 150), w, fill=(20, 20, 20), font=FONT)
        left, top, right, bottom = draw.textbbox((x, 150), w, font=FONT)
        boxes.append(BBox(int(left), int(top), int(right) + 1, int(bottom) + 1))
        x = int(right) + 22
    return np.asarray(img, dtype=np.uint8).copy(), boxes


def surrogate_op(ink: BBox, text: str, source: str, pad: int = 6) -> RenderOp:
    return RenderOp(
        page=0,
        kind=ActionKind.SURROGATE,
        box=ink.pad(pad, pad, BBox(0, 0, 10_000, 10_000)),
        reason="person",
        ref_id="M-1",
        text=text,
        ink_box=ink,
        source_text=source,
    )


class TestTypeset:
    def test_erases_completely_and_spares_neighbours(self) -> None:
        img, boxes = page_with_words(["Mr", "Smith", "on", "Monday"])
        before = img.copy()
        result = typeset(img, surrogate_op(boxes[1], "Walsh", "Smith"))
        assert erased_cleanly(img, result)
        for neighbour in (boxes[0], boxes[2], boxes[3]):
            region = (slice(neighbour.y0, neighbour.y1), slice(neighbour.x0, neighbour.x1))
            assert np.array_equal(img[region], before[region]), "a neighbouring word was modified"
        # Every pixel of the erased box is paper or part of the new glyphs: nothing original remains.
        b = result.box
        assert np.all(img[b.y0 : b.y1, b.x0 : b.x1][~result.glyph_mask] == np.array(result.background))
        assert result.glyph_mask.any()

    def test_longer_surrogate_uses_blank_space_instead_of_condensing(self) -> None:
        img, boxes = page_with_words(["Ms", "Jane", "Citizen"])
        planned = surrogate_op(boxes[1].union(boxes[2]), "Rebecca Davies-Whitaker", "Jane Citizen")
        result = typeset(img, planned)
        assert erased_cleanly(img, result)
        assert result.box.x1 > planned.box.x1  # grew into blank paper on the right

    def test_blockers_stop_the_extension(self) -> None:
        img, boxes = page_with_words(["Ms", "Jane", "Citizen"])
        planned = surrogate_op(boxes[1].union(boxes[2]), "Rebecca Davies-Whitaker", "Jane Citizen")
        blocker = BBox(planned.box.x1 + 10, planned.box.y0, planned.box.x1 + 200, planned.box.y1)
        result = typeset(img, planned, blockers=[blocker])
        assert result.box.x1 < blocker.x0

    def test_raster_ops_report_verification_and_erased_boxes(self) -> None:
        img, boxes = page_with_words(["Claim", "WC1234567"])
        ops = [
            surrogate_op(boxes[1], "RH8031543", "WC1234567"),
            RenderOp(0, ActionKind.BLACKOUT, boxes[0], "reference", "M-2"),
        ]
        edit = apply_ops(img, ops, RedactionRenderConfig(), notice=None)
        assert edit.verified and len(edit.applied) == 2
        assert np.all(edit.raster[boxes[0].y0 : boxes[0].y1, boxes[0].x0 : boxes[0].x1] == 0)

    def test_labels_fill_and_verify(self) -> None:
        img, boxes = page_with_words(["Mr", "Smith"])
        cfg = RedactionRenderConfig()
        op = RenderOp(0, ActionKind.LABEL, boxes[1].pad(5, 5), "person", "M-1", "[PERSON 1]", boxes[1], False, "Smith")
        edit = apply_ops(img, [op], cfg)
        assert edit.verified
        b = edit.applied[0].erased
        region = edit.raster[b.y0 : b.y1, b.x0 : b.x1]
        assert (region == np.array(cfg.label_fill_rgb, dtype=np.uint8)).all(axis=2).mean() > 0.5


class TestNotice:
    def test_placed_in_blank_bottom_margin(self) -> None:
        img = np.full((3508, 2481, 3), 250, dtype=np.uint8)
        placed = place_notice(img, "De-identified copy", 300)
        assert placed is not None and placed[0].y0 > 3000

    def test_never_drawn_over_content(self) -> None:
        img = np.full((800, 600, 3), 250, dtype=np.uint8)
        img[:60, :, :] = 10  # ink in the top margin band
        img[-80:, :, :] = 10  # and in the bottom one
        assert place_notice(img, "De-identified copy", 300) is None


class TestTextLayer:
    def test_escaping(self) -> None:
        assert _escape(b"(a)" + bytes([92]) + b"\xe9") == "\\(a\\)\\\\\\351"

    def test_items_exclude_covered_tokens_and_include_replacements(self) -> None:
        tok_kept = Token("p0000.l0000.w000", 0, 0, 0, BBox(0, 0, 100, 40), 1.0, 1.0, "t", "Patient")
        tok_gone = Token("p0000.l0000.w001", 0, 0, 1, BBox(110, 0, 200, 40), 1.0, 1.0, "t", "Smith")
        op = RenderOp(0, ActionKind.SURROGATE, BBox(105, 0, 205, 40), "person", "M-1", "Walsh", BBox(110, 0, 200, 40))
        texts = [t for _b, t in text_layer_items([tok_kept, tok_gone], [op])]
        assert texts == ["Patient", "Walsh"]

    def test_built_pdf_with_text_layer_passes_v1_and_v2(self) -> None:
        geom = PageGeometry(0, 200.0, 100.0, 72, 200, 100)
        raster = np.full((100, 200, 3), 255, dtype=np.uint8)
        layer = page_text_layer(geom, [(BBox(10, 10, 120, 30), "Patient Walsh (seen)")])
        pdf = build_pdf([PageSpec(200.0, 100.0, encode_raster(raster, OutputConfig()), layer)], OutputConfig())
        assert check_structure(pdf, 1, allow_fonts=True).passed
        assert not check_structure(pdf, 1, allow_fonts=False).passed
        assert any("Patient Walsh (seen)" in s for s in text_segments(pdf))
        assert check_text_leak(pdf, ["John Smith", "2953 70150 1"]).passed
        assert not check_text_leak(pdf, ["Walsh"]).passed


def test_mentions_on_rotated_lines_are_blacked_out_not_typeset() -> None:
    from mlredact.config.loader import load_config
    from mlredact.core.model import Line, Mention
    from mlredact.policy.engine import Decision
    from mlredact.render.plan import plan_ops, rotated_token_ids
    from mlredact.surrogate.assign import MentionSurrogate

    tok = Token("p0000.l0000.w000", 0, 0, 0, BBox(100, 100, 140, 400), 0.99, 0.99, "t", "Smith")
    upright = Line("p0000.l0001", 0, 1, BBox(0, 0, 10, 10), ((0, 0), (300, 0), (300, 40), (0, 40)), 1.0, "t", ())
    sideways = Line("p0000.l0000", 0, 0, tok.bbox, ((140, 400), (140, 100), (100, 100), (100, 400)), 1.0, "t", (tok,))
    rotated = rotated_token_ids([upright, sideways])
    assert rotated == frozenset({tok.token_id})
    m = Mention("M-1", EntityType.PERSON, (tok.token_id,), (), "Smith")
    geom = {0: PageGeometry(0, 595.0, 842.0, 300, 2481, 3508)}
    cfg = load_config("broad").redaction
    sur = {"M-1": MentionSurrogate("M-1", ("Walsh",))}
    (op,) = plan_ops([Decision(m, ActionKind.SURROGATE)], {tok.token_id: tok}, geom, cfg, sur, rotated)[0]
    assert op.kind is ActionKind.BLACKOUT
    (op,) = plan_ops([Decision(m, ActionKind.SURROGATE)], {tok.token_id: tok}, geom, cfg, sur)[0]
    assert op.kind is ActionKind.SURROGATE


class TestLineRetypeset:
    LINES = (
        "Re: Mr John Andrew SMITH DOB: 14/03/1978",
        "Claim No: WC1234567 Medicare No: 2953 70150 1",
        "Address: 42 Wattle Grove Road, Penrith NSW 2750",
        "",
        "The worker reports constant low back pain radiating to the left leg after lifting boxes at work",
        "and Mr Smith was reviewed again today.",
    )

    def _plan(self, overrides: dict[str, object] | None = None):  # type: ignore[no-untyped-def]
        import hashlib

        from mlredact.config.loader import load_config
        from mlredact.detect.engine import detect_document
        from mlredact.policy.engine import decide
        from mlredact.render.plan import plan_ops
        from mlredact.surrogate.assign import SurrogateAssigner, forbidden_keys
        from mlredact.surrogate.drbg import Chooser
        from mlredact.surrogate.generators import SurrogateFactory
        from support import pages_from_lines

        cfg = load_config("broad", overrides={"redaction": overrides or {}})
        pages = pages_from_lines(self.LINES)
        result = detect_document(pages, cfg, "doc")
        decisions = decide(result.mentions, cfg.policy)
        removed = [d.mention for d in decisions if d.action is not ActionKind.KEEP]
        tokens = {t.token_id: t for p in pages for t in p.tokens()}
        factory = SurrogateFactory(Chooser(hashlib.sha256(b"k").digest()), forbidden_keys(removed, result.persons))
        sur = SurrogateAssigner(factory, result.persons, tokens).assign(decisions)
        geoms = {0: pages[0].geometry}
        lines = list(pages[0].lines)
        return plan_ops(decisions, tokens, geoms, cfg.redaction, sur, frozenset(), lines)[0], lines, decisions

    def test_dense_blocks_are_retypeset_as_whole_lines(self) -> None:
        ops, _lines, decisions = self._plan()
        retyped = [o for o in ops if o.kind is ActionKind.RETYPESET_REGION]
        assert len(retyped) == 3  # the Re / Claim / Address block
        assert all(o.covers and o.text for o in retyped)
        covered = {m for o in retyped for m in o.covers}
        body = [d for d in decisions if d.mention.mention_id not in covered and d.action is not ActionKind.KEEP]
        assert body and all(
            o.kind is ActionKind.SURROGATE for o in ops if o.ref_id in {d.mention.mention_id for d in body}
        )

    def test_positional_hardening_retypesets_every_line_with_a_replacement(self) -> None:
        ops, _lines, _decisions = self._plan({"positional_hardening": "line_retypeset"})
        assert {o.kind for o in ops} == {ActionKind.RETYPESET_REGION}

    def test_blackout_style_is_never_retypeset(self) -> None:
        ops, _lines, _decisions = self._plan({"style": "blackout", "positional_hardening": "line_retypeset"})
        assert ActionKind.RETYPESET_REGION not in {o.kind for o in ops}

    def test_tightly_spaced_lines_keep_each_others_glyphs(self) -> None:
        img, boxes = page_with_words(["Bligh", "Sydney"], height=300)
        first = boxes[0].union(boxes[1])
        second = BBox(first.x0, first.y1 - 12, first.x1, first.y1 + 30)  # overlaps the first line's bottom
        ops = [
            RenderOp(
                0,
                ActionKind.RETYPESET_REGION,
                first.pad(4, 4),
                "retypeset",
                "L-1",
                "Bligh Sydney",
                first,
                True,
                "Bligh Sydney",
            ),
            RenderOp(0, ActionKind.RETYPESET_REGION, second, "retypeset", "L-2", "x", second, False, "x"),
        ]
        edit = apply_ops(img, ops, RedactionRenderConfig())
        assert edit.verified
        # The first line's descenders (drawn first) survive the second line's erase.
        g_rows = edit.raster[second.y0 : first.y1 + 4, first.x0 : first.x1]
        assert (g_rows.mean(axis=2) < 128).any()
