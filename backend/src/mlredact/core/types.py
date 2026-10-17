"""Shared enumerations.  Values are stable strings: they appear in manifests and configs."""

from __future__ import annotations

from enum import IntEnum, StrEnum


class Stage(StrEnum):
    INTAKE = "intake"
    RENDER = "render"
    NORMALISE = "normalise"
    LAYOUT = "layout"
    REGIONS = "regions"
    OCR = "ocr"
    FUSION = "fusion"
    INK = "ink"
    VIEWS = "views"
    DETECT = "detect"
    RESOLVE = "resolve"
    POLICY = "policy"
    SURROGATE = "surrogate"
    RENDER_OUT = "render_out"
    BUILD = "build"
    VERIFY = "verify"
    EMIT = "emit"


class PageKind(StrEnum):
    IMAGE_ONLY = "image_only"  # scanned page, no usable embedded text
    BORN_DIGITAL = "born_digital"  # embedded text, no full-page image
    MIXED = "mixed"  # embedded text and images
    EMPTY = "empty"


class EntityType(StrEnum):
    """PII taxonomy (plan §7.1).  Policy maps each type to an action per profile."""

    PERSON = "person"
    DATE_OF_BIRTH = "date_of_birth"
    DATE_OF_DEATH = "date_of_death"
    DATE = "date"  # other dates: kept under the Broad profile
    AGE = "age"
    STREET_ADDRESS = "street_address"
    LOCALITY = "locality"
    POSTCODE = "postcode"
    PHONE = "phone"
    EMAIL = "email"
    URL = "url"
    IP_ADDRESS = "ip_address"
    MEDICARE = "medicare"
    IHI = "ihi"
    HPI = "hpi"
    DVA_FILE = "dva_file"
    CRN = "crn"
    TFN = "tfn"
    NDIS = "ndis"
    DRIVER_LICENCE = "driver_licence"
    PASSPORT = "passport"
    CLINICAL_ID = "clinical_id"  # MRN / UR / URN / episode / accession numbers
    PROVIDER_NUMBER = "provider_number"
    AHPRA = "ahpra"
    REFERENCE = "reference"  # claim / policy / insurer / law-firm / matter references
    COURT_FILE = "court_file"
    BANK_ACCOUNT = "bank_account"
    CARD_NUMBER = "card_number"
    ABN = "abn"
    ACN = "acn"
    EMPLOYEE_ID = "employee_id"
    VEHICLE_REG = "vehicle_reg"
    VIN = "vin"
    ORGANISATION = "organisation"
    OTHER_ID = "other_id"


class RegionKind(StrEnum):
    """Non-text (or not-reliably-text) regions that are removed as a whole."""

    SIGNATURE = "signature"
    HANDWRITING = "handwriting"
    FACE = "face"
    PHOTO = "photo"
    BARCODE = "barcode"
    LOGO = "logo"
    STAMP = "stamp"
    ILLEGIBLE = "illegible"
    FAINT_INK = "faint_ink"
    GRAPHIC = "graphic"  # unexplained non-text graphic in the body (figure, diagram, body chart)


class ActionKind(StrEnum):
    SURROGATE = "surrogate"
    RETYPESET_REGION = "retypeset_region"
    REMOVE = "remove"  # opaque neutral fill (optionally labelled)
    BLACKOUT = "blackout"
    LABEL = "label"
    SUPPRESS_INK = "suppress_ink"
    GENERALISE = "generalise"
    DATE_SHIFT = "date_shift"
    KEEP = "keep"


class ViewKind(StrEnum):
    PROSE = "prose"
    LINE = "line"
    TITLECASE = "titlecase"
    OCR_NORMALISED = "ocr_normalised"


class EvidenceStrength(IntEnum):
    WEAK = 1
    STRONG = 2


class VerificationCheck(StrEnum):
    STRUCTURE = "V1_structure"
    TEXT = "V2_text"
    PIXELS = "V3_pixels"
    RESIDUAL = "V4_residual"
    SURROGATE = "V5_surrogate"
    INTEGRITY = "V6_integrity"


class JobStatus(StrEnum):
    RELEASED = "released"
    QUARANTINED = "quarantined"
