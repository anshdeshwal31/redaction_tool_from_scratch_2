"""Foundation tests: canonical serialisation, IDs, geometry, Sensitive, errors, config."""

from __future__ import annotations

import pickle
from enum import StrEnum

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from mlredact.config.loader import builtin_profiles, config_hash, load_config
from mlredact.config.schema import AppConfig
from mlredact.core.canonical import canonical_json, content_id
from mlredact.core.errors import EnvironmentErrorMl, MlredactError, ReasonCode
from mlredact.core.geometry import Affine, BBox
from mlredact.core.model import Token
from mlredact.core.types import ActionKind, EntityType
from mlredact.security.sensitive import Sensitive


class TestCanonical:
    def test_key_order_does_not_matter(self) -> None:
        assert canonical_json({"b": 1, "a": [1, 2]}) == canonical_json({"a": [1, 2], "b": 1})

    def test_integral_floats_hash_like_ints(self) -> None:
        assert canonical_json({"x": 1.0}) == canonical_json({"x": 1})

    def test_rejects_nan_sets_and_sensitive(self) -> None:
        with pytest.raises(ValueError):
            canonical_json({"x": float("nan")})
        with pytest.raises(TypeError):
            canonical_json({"x": {1, 2}})
        with pytest.raises(TypeError):
            canonical_json({"x": Sensitive("John Smith")})

    def test_enums_serialise_as_values(self) -> None:
        assert canonical_json({"t": EntityType.PERSON}) == b'{"t":"person"}'

    def test_content_id_is_length_prefixed(self) -> None:
        assert content_id("M", "ab", "c") != content_id("M", "a", "bc")
        assert content_id("M", 1) != content_id("M", "1")
        assert content_id("M", True) != content_id("M", 1)
        assert content_id("M", "x") == content_id("M", "x")
        assert content_id("M", "x").startswith("M-")


class TestGeometry:
    def test_outward_rounding_never_shrinks(self) -> None:
        b = BBox.outward(10.2, 5.9, 20.1, 7.01)
        assert b == BBox(10, 5, 21, 8)

    @given(st.floats(-1e4, 1e4), st.floats(-1e4, 1e4), st.floats(-1e4, 1e4), st.floats(-1e4, 1e4))
    def test_outward_contains_float_box(self, a: float, b: float, c: float, d: float) -> None:
        box = BBox.outward(a, b, c, d)
        assert box.x0 <= min(a, c) and box.x1 >= max(a, c)
        assert box.y0 <= min(b, d) and box.y1 >= max(b, d)

    def test_pad_and_clip(self) -> None:
        page = BBox(0, 0, 100, 100)
        assert BBox(1, 1, 10, 10).pad(3, 3, page) == BBox(0, 0, 13, 13)

    def test_intersection_and_iou(self) -> None:
        a, b = BBox(0, 0, 10, 10), BBox(5, 5, 15, 15)
        assert a.intersection(b) == BBox(5, 5, 10, 10)
        assert a.iou(b) == pytest.approx(25 / 175)
        assert a.intersection(BBox(20, 20, 30, 30)) is None

    def test_affine_inverse_roundtrip(self) -> None:
        t = Affine.from_matrix([[0.98, -0.17, 12.0], [0.17, 0.98, -3.0]])
        x, y = t.inverse().apply(*t.apply(123.4, 567.8))
        assert x == pytest.approx(123.4) and y == pytest.approx(567.8)


class TestSensitive:
    def test_never_renders(self) -> None:
        s = Sensitive("John Smith")
        assert "John" not in repr(s)
        assert "John" not in str(s)
        assert "John" not in f"{s}"
        assert "John" not in f"{s!r}"
        assert s.reveal() == "John Smith"

    def test_value_semantics_and_pickle(self) -> None:
        assert Sensitive("a") == Sensitive("a")
        assert len({Sensitive("a"), Sensitive("a")}) == 1
        assert pickle.loads(pickle.dumps(Sensitive("a"))) == Sensitive("a")

    def test_immutable(self) -> None:
        s = Sensitive("a")
        with pytest.raises(AttributeError):
            s._value = "b"  # type: ignore[misc]

    def test_model_reprs_hide_text(self) -> None:
        tok = Token("p0000.l0000.w000", 0, 0, 0, BBox(0, 0, 1, 1), 0.9, 0.8, "e", "Smith", ("5mith",))
        assert "Smith" not in repr(tok) and "5mith" not in repr(tok)


class TestErrors:
    def test_only_scalar_params(self) -> None:
        err = MlredactError(ReasonCode.INPUT_ENCRYPTED, pages=3)
        assert str(err) == "E_INPUT_ENCRYPTED (pages=3)"
        with pytest.raises(TypeError):
            MlredactError(ReasonCode.INTERNAL, detail="John Smith")  # type: ignore[arg-type]

    def test_enum_params_allowed(self) -> None:
        class Kind(StrEnum):
            A = "a"

        assert "kind=a" in str(MlredactError(ReasonCode.INTERNAL, kind=Kind.A))


class TestConfig:
    def test_profiles_load_and_differ(self) -> None:
        assert set(builtin_profiles()) >= {"broad", "claimant_focused", "maximal"}
        broad, claimant, maximal = (load_config(p) for p in ("broad", "claimant_focused", "maximal"))
        assert broad.policy.entity_actions[EntityType.PERSON] is ActionKind.SURROGATE
        assert broad.policy.entity_actions[EntityType.DATE] is ActionKind.KEEP
        assert claimant.policy.entity_actions[EntityType.ORGANISATION] is ActionKind.KEEP
        assert maximal.policy.entity_actions[EntityType.DATE] is ActionKind.DATE_SHIFT
        assert len({config_hash(broad), config_hash(claimant), config_hash(maximal)}) == 3

    def test_hash_is_stable(self) -> None:
        assert config_hash(load_config("broad")) == config_hash(load_config("broad"))

    def test_unknown_keys_rejected_without_echoing_values(self) -> None:
        with pytest.raises(EnvironmentErrorMl) as exc:
            load_config("broad", overrides={"render": {"dpii": "John Smith"}})
        assert "John" not in str(exc.value)

    def test_unknown_profile_rejected(self) -> None:
        with pytest.raises(EnvironmentErrorMl):
            load_config("nope")

    def test_frozen(self) -> None:
        cfg = AppConfig()
        with pytest.raises(ValidationError):
            cfg.render.dpi = 100  # type: ignore[misc]


def test_runtime_profile_records_every_output_relevant_library() -> None:
    from mlredact.runtime.profile import detect_runtime_profile

    libs = detect_runtime_profile().libraries
    for name in ("pdfium", "pypdfium2", "onnxruntime", "tokenizers", "pikepdf", "qpdf", "pillow", "numpy"):
        assert libs.get(name), name
