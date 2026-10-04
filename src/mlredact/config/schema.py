"""Typed configuration.  Unknown keys are rejected; every model is immutable.

The canonical JSON of the validated :class:`AppConfig` is hashed into the job fingerprint, so any
change in configuration is visible in provenance.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mlredact.core.types import ActionKind, EntityType, RegionKind, ViewKind


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ThreadsConfig(_Strict):
    """Fixed thread counts: a determinism requirement, not a tuning knob (plan §13)."""

    ort_intra_op: int = Field(1, ge=1, le=64)
    ort_inter_op: int = Field(1, ge=1, le=8)
    opencv: int = Field(1, ge=1, le=64)
    blas: int = Field(1, ge=1, le=64)


class RuntimeConfig(_Strict):
    # e.g. "linux-x86_64-v3".  None disables the check (development only).
    expected_profile: str | None = None
    threads: ThreadsConfig = ThreadsConfig()
    workers: int = Field(1, ge=1, le=256)
    page_timeout_s: int = Field(900, ge=10, le=86400)
    models_dir: str = ".models"


class LimitsConfig(_Strict):
    max_input_bytes: int = Field(512 * 1024 * 1024, ge=1024)
    max_pages: int = Field(3000, ge=1)
    max_page_pixels: int = Field(80_000_000, ge=1_000_000)


class RenderConfig(_Strict):
    dpi: int = Field(300, ge=150, le=600)
    draw_annotations: bool = True
    draw_forms: bool = True


class OcrEngineAConfig(_Strict):
    # Default: small detector at full resolution (protects small text) + medium recogniser
    # (recognition quality dominates recall).  See ADR in docs/ for the measured trade-off.
    det_model: str = "ppocrv6_det_small"
    rec_model: str = "ppocrv6_rec_medium"
    cls_model: str = "pplcnet_textline_ori_server"
    use_textline_orientation: bool = True
    det_limit_side_len: int = Field(736, ge=32)
    det_thresh: float = Field(0.3, gt=0, lt=1)
    # Lower than the library default (0.5): recall first, a spurious box only costs OCR time.
    det_box_thresh: float = Field(0.4, gt=0, lt=1)
    det_unclip_ratio: float = Field(1.6, gt=1, lt=4)
    alternates_top_k: int = Field(2, ge=1, le=5)
    alternate_min_prob: float = Field(0.05, ge=0, le=1)
    max_alternates_per_token: int = Field(4, ge=0, le=16)


class OcrEngineBConfig(_Strict):
    """Engine B: Tesseract 5 LSTM (an independent second reader, plan §6)."""

    enabled: bool = True
    binary: str = "tesseract"
    model: str = "tessdata_best_eng"
    language: str = "eng"
    page_segmentation: int = Field(3, ge=0, le=13)  # 3 = fully automatic page segmentation
    # Words engine A did not read are added as new tokens only at/above this confidence (0-100):
    # below it Tesseract output on scribbles and noise is mostly garbage.
    min_new_word_confidence: float = Field(70.0, ge=0, le=100)
    timeout_s: int = Field(300, ge=10, le=3600)


class OcrConfig(_Strict):
    engine_a: OcrEngineAConfig = OcrEngineAConfig()
    engine_b: OcrEngineBConfig = OcrEngineBConfig()
    # Tokens whose weakest character is below this are "uncertain": they extend adjacent PII spans.
    uncertain_char_confidence: float = Field(0.55, ge=0, le=1)
    # Engine D: fuse the PDF's own text layer (visible words only).  Where OCR's weakest character is
    # below ``embedded_replace_below`` the embedded text wins; otherwise it is kept as an alternate.
    embedded_text: bool = True
    embedded_replace_below: float = Field(0.9, ge=0, le=1)


def _gliner_labels() -> dict[str, EntityType]:
    """Prompt labels (GLiNER-PII training vocabulary, plus "organization") -> taxonomy.

    There is deliberately no generic "date" label: flat decoding keeps one label per span, so a
    "date" option could only take spans away from "dob" (recall first; the policy keeps dates).
    """
    t = EntityType
    return {
        "name": t.PERSON,
        "first name": t.PERSON,
        "last name": t.PERSON,
        "organization": t.ORGANISATION,
        "dob": t.DATE_OF_BIRTH,
        "age": t.AGE,
        "email address": t.EMAIL,
        "phone number": t.PHONE,
        "url": t.URL,
        "ip address": t.IP_ADDRESS,
        "location address": t.STREET_ADDRESS,
        "location city": t.LOCALITY,
        "location zip": t.POSTCODE,
        "account number": t.BANK_ACCOUNT,
        "credit card": t.CARD_NUMBER,
        "ssn": t.OTHER_ID,
        "passport number": t.PASSPORT,
        "driver license": t.DRIVER_LICENCE,
        "healthcare number": t.CLINICAL_ID,
        "vehicle id": t.VEHICLE_REG,
        "username": t.OTHER_ID,
    }


class GlinerConfig(_Strict):
    """N2: GLiNER-PII zero-shot span NER (plan §7.2)."""

    enabled: bool = True
    model: str = "gliner_pii_base"
    threshold: float = Field(0.3, gt=0, lt=1)
    # Scores at/above this are strong evidence; below it a hit is weak (fusion rule, plan §7.3).
    strong_threshold: float = Field(0.8, gt=0, le=1)
    # Per-type overrides.  Organisations and localities have no other detector, and a single weak hit
    # of those types is not enough for fusion: a moderate organisation score and any locality hit
    # (capital cities and states are allowlisted) count as strong.  Recall first (Broad: suburbs go).
    type_strong_thresholds: dict[EntityType, float] = Field(
        default_factory=lambda: {EntityType.ORGANISATION: 0.5, EntityType.LOCALITY: 0.3}
    )
    # Fixed word windows (batch size 1): results never depend on document length or other inputs.
    window_words: int = Field(300, ge=16, le=1024)
    overlap_words: int = Field(50, ge=0, le=512)
    views: tuple[ViewKind, ...] = (ViewKind.LINE, ViewKind.TITLECASE)
    # The prompt is built in sorted label order, so the configuration hash fully determines it.
    labels: dict[str, EntityType] = Field(default_factory=_gliner_labels)

    @model_validator(mode="after")
    def _check(self) -> GlinerConfig:
        if self.overlap_words >= self.window_words:
            raise ValueError("overlap_words must be smaller than window_words")
        if not self.labels or not self.views:
            raise ValueError("labels and views must not be empty")
        return self


class ViterbiBiases(_Strict):
    """Transition biases of the Privacy Filter's constrained BIOES decoder (names as published).

    All zero is the model's calibrated default operating point.  Raising ``background_to_start`` /
    lowering ``background_stay`` trades precision for recall.
    """

    background_stay: float = 0.0
    background_to_start: float = 0.0
    end_to_background: float = 0.0
    end_to_start: float = 0.0
    inside_to_continue: float = 0.0
    inside_to_end: float = 0.0


def _privacy_filter_categories() -> dict[str, EntityType]:
    t = EntityType
    return {
        "account_number": t.OTHER_ID,
        "private_address": t.STREET_ADDRESS,
        "private_date": t.DATE,
        "private_email": t.EMAIL,
        "private_person": t.PERSON,
        "private_phone": t.PHONE,
        "private_url": t.URL,
        "secret": t.OTHER_ID,
    }


class PrivacyFilterConfig(_Strict):
    """N1: OpenAI Privacy Filter token classifier + our constrained Viterbi decoder (plan §7.2)."""

    enabled: bool = True
    model: str = "privacy_filter_q4"
    # Exact windowing: a token's logits depend only on tokens within ``context_tokens`` (layers x
    # attention band; derived from the model config when None).  Each window contributes the logits of
    # its core (window minus a context margin on each side that is not a document edge), and a single
    # Viterbi pass runs over the stitched document.
    window_tokens: int = Field(3072, ge=256, le=32768)
    context_tokens: int | None = Field(None, ge=0, le=16384)
    biases: ViterbiBiases = ViterbiBiases()
    strong_threshold: float = Field(0.8, gt=0, le=1)
    # Recall sweep: runs of tokens whose posterior P(not background) reaches this value become weak
    # candidates even where the Viterbi path chose background.  1.0 disables the sweep.
    posterior_threshold: float = Field(0.25, gt=0, le=1)
    views: tuple[ViewKind, ...] = (ViewKind.LINE,)
    categories: dict[str, EntityType] = Field(default_factory=_privacy_filter_categories)


class NerConfig(_Strict):
    # NER runs in the parent process while the page workers are idle, so it gets its own (pinned)
    # thread count.  Measured: GLiNER logits are bit-identical for 1-6 intra-op threads.
    threads: ThreadsConfig = ThreadsConfig(ort_intra_op=4)
    gliner: GlinerConfig = GlinerConfig()
    privacy_filter: PrivacyFilterConfig = PrivacyFilterConfig()


class DetectionConfig(_Strict):
    # Values found anywhere are propagated (exact / OCR-tolerant) across the whole document.
    propagate: bool = True
    propagation_min_len: int = Field(4, ge=2)
    propagation_fuzzy_ratio: float = Field(0.88, ge=0.5, le=1.0)
    # Neighbour rule: uncertain tokens adjacent to a PII span join it.
    extend_into_uncertain_neighbours: bool = True
    ner: NerConfig = NerConfig()


class RegionsConfig(_Strict):
    """Non-text regions and ink accounting (plan §6 "ink accounting", §7.2 S4).

    Pixel sizes are given for 300 dpi and scaled with the canonical raster resolution.
    """

    enabled: bool = True
    barcodes: bool = True
    faces: bool = True
    face_model: str = "yunet_face_2023mar"
    # YuNet's own default is 0.9; recall first (a false face only costs a grey box).
    face_score_threshold: float = Field(0.6, gt=0, lt=1)
    ink_accounting: bool = True
    # Darkness = local background minus pixel grey level.  At/above ``ink_darkness``: ink.  Between
    # ``faint_darkness`` and ``ink_darkness`` (after smoothing): faint ink such as show-through.
    ink_darkness: int = Field(60, ge=10, le=200)
    faint_darkness: int = Field(18, ge=4, le=100)
    min_component_px: int = Field(12, ge=1)
    header_footer_band: float = Field(0.10, ge=0, le=0.3)
    side_margin: float = Field(0.07, ge=0, le=0.3)
    # HSV saturation (0-255) above which ink counts as coloured (stamps, pen).
    colour_saturation: int = Field(80, ge=20, le=255)
    # Unexplained text-like ink is re-read; it becomes ordinary tokens only if every re-read line
    # reaches this confidence and the tokens explain most of its ink, otherwise it is removed.
    reread: bool = True
    reread_min_confidence: float = Field(0.85, ge=0, le=1)
    # A token explains (accounts for) the ink under it only if its weakest character reaches this.
    explain_min_confidence: float = Field(0.5, ge=0, le=1)
    # Legibility gate (plan §6): when at least this fraction of a page's ink is unreadable, the page
    # is not trusted at all - "remove_page" removes everything on it, "quarantine" stops the job.
    illegible_fraction: float = Field(0.5, gt=0, le=1)
    illegible_action: Literal["remove_page", "quarantine"] = "remove_page"
    suppress_faint_ink: bool = True

    @model_validator(mode="after")
    def _check(self) -> RegionsConfig:
        if self.faint_darkness >= self.ink_darkness:
            raise ValueError("faint_darkness must be below ink_darkness")
        return self


def _default_entity_actions() -> dict[EntityType, ActionKind]:
    keep = {EntityType.DATE, EntityType.AGE}
    return {t: (ActionKind.KEEP if t in keep else ActionKind.SURROGATE) for t in EntityType}


def _default_region_actions() -> dict[RegionKind, ActionKind]:
    out = {k: ActionKind.REMOVE for k in RegionKind}
    out[RegionKind.FAINT_INK] = ActionKind.SUPPRESS_INK
    return out


class PolicyConfig(_Strict):
    profile_id: str = Field("broad", pattern=r"^[a-z0-9_]{1,40}$")
    entity_actions: dict[EntityType, ActionKind] = Field(default_factory=_default_entity_actions)
    region_actions: dict[RegionKind, ActionKind] = Field(default_factory=_default_region_actions)
    # Claimant-focused: names of people resolved as professionals (treating clinicians, lawyers,
    # case managers) stay visible.  The role rule is one-sided; see ``mlredact.resolve.roles``.
    keep_professionals: bool = False

    @model_validator(mode="after")
    def _complete(self) -> PolicyConfig:
        missing = [t.value for t in EntityType if t not in self.entity_actions]
        missing += [k.value for k in RegionKind if k not in self.region_actions]
        if missing:
            raise ValueError(f"policy has no action for: {missing}")
        return self


class RedactionRenderConfig(_Strict):
    # How text PII is rendered.  "surrogate" requires the surrogate stage (plan §10).
    style: Literal["surrogate", "label", "blackout"] = "surrogate"
    pad_min_px: int = Field(2, ge=0, le=50)
    pad_line_height_frac: float = Field(0.08, ge=0, le=1)
    blackout_rgb: tuple[int, int, int] = (0, 0, 0)
    remove_rgb: tuple[int, int, int] = (96, 96, 96)
    label_fill_rgb: tuple[int, int, int] = (228, 228, 228)
    label_text_rgb: tuple[int, int, int] = (60, 60, 60)
    # Line re-typesetting (plan §10): the whole line is erased and typeset again, so no original pixel
    # and no original word position survives.  "line_retypeset" applies it to every line with a
    # replaced value (positional hardening: gap widths no longer reveal original lengths).
    positional_hardening: Literal["none", "line_retypeset"] = "none"
    # PII-dense blocks (patient-label stickers, address/recipient blocks, fax headers) are always
    # re-typeset: small blocks where >= ``dense_block_pii_fraction`` of tokens are PII, or with at
    # least two mentions and a quarter of their tokens PII.
    dense_blocks: bool = True
    dense_block_max_lines: int = Field(6, ge=1, le=40)
    dense_block_pii_fraction: float = Field(0.4, gt=0, le=1)
    # SUPPRESS_INK: inside the region every pixel lighter than (background - this) becomes background.
    suppress_ink_darkness: int = Field(60, ge=10, le=200)


class SurrogateConfig(_Strict):
    """Keyed surrogate generation (plan §10).  The key itself never appears in configuration."""

    # Hex key file path (e.g. a mounted secret).  ``MLREDACT_SURROGATE_KEY`` (hex) takes precedence.
    key_file: str | None = None
    # Fixed, publicly known development key: tests / local runs only.  Refused when
    # ``runtime.expected_profile`` is set (i.e. in a pinned production deployment).
    allow_development_key: bool = False
    font_family: Literal["Sans", "Serif"] = "Sans"


class OutputConfig(_Strict):
    image_encoding: Literal["auto", "jpeg", "flate"] = "auto"
    jpeg_quality: int = Field(90, ge=50, le=100)
    text_layer: bool = True
    # Small "de-identified copy" banner, drawn only into a blank page margin (never over content).
    notice: bool = True
    producer: str = Field("mlredact", pattern=r"^[A-Za-z0-9 ._\-]{1,40}$")


def _verification_ocr() -> OcrEngineAConfig:
    return OcrEngineAConfig(det_model="ppocrv6_det_small", rec_model="ppocrv6_rec_small")


class VerificationConfig(_Strict):
    re_ocr: bool = True
    # Independent, faster re-read of the output (a check, not the primary reading).
    ocr: OcrEngineAConfig = Field(default_factory=_verification_ocr)
    # Removed values are searched in the re-OCR'd output; similarity at/above this fails the job.
    residual_fuzzy_ratio: float = Field(0.9, ge=0.5, le=1.0)
    residual_min_len: int = Field(5, ge=3)
    max_replans: int = Field(1, ge=0, le=3)


class AppConfig(_Strict):
    schema_version: Literal[1] = 1
    runtime: RuntimeConfig = RuntimeConfig()
    limits: LimitsConfig = LimitsConfig()
    render: RenderConfig = RenderConfig()
    ocr: OcrConfig = OcrConfig()
    regions: RegionsConfig = RegionsConfig()
    detection: DetectionConfig = DetectionConfig()
    policy: PolicyConfig = PolicyConfig()
    redaction: RedactionRenderConfig = RedactionRenderConfig()
    surrogate: SurrogateConfig = SurrogateConfig()
    output: OutputConfig = OutputConfig()
    verification: VerificationConfig = VerificationConfig()
