/**
 * TypeScript view of the engine's output contract (schemas/manifest.v1.json, run.json).
 *
 * The manifest never contains document text: only geometry, types, actions, evidence, verification
 * and provenance.  Surrogate/label strings are present because they are visible in the released PDF.
 */

export const ENTITY_TYPES = [
  "person",
  "date_of_birth",
  "date_of_death",
  "date",
  "age",
  "street_address",
  "locality",
  "postcode",
  "phone",
  "email",
  "url",
  "ip_address",
  "medicare",
  "ihi",
  "hpi",
  "dva_file",
  "crn",
  "tfn",
  "ndis",
  "driver_licence",
  "passport",
  "clinical_id",
  "provider_number",
  "ahpra",
  "reference",
  "court_file",
  "bank_account",
  "card_number",
  "abn",
  "acn",
  "employee_id",
  "vehicle_reg",
  "vin",
  "organisation",
  "other_id",
  "file_path",
] as const;
export type EntityType = (typeof ENTITY_TYPES)[number];

export const REGION_KINDS = [
  "signature",
  "handwriting",
  "face",
  "photo",
  "barcode",
  "logo",
  "stamp",
  "illegible",
  "faint_ink",
  "graphic",
] as const;
export type RegionKind = (typeof REGION_KINDS)[number];

export const ACTIONS = [
  "surrogate",
  "retypeset_region",
  "remove",
  "blackout",
  "label",
  "suppress_ink",
  "generalise",
  "date_shift",
  "keep",
] as const;
export type Action = (typeof ACTIONS)[number];

export const RENDER_STYLES = ["surrogate", "label", "blackout"] as const;
export type RenderStyle = (typeof RENDER_STYLES)[number];

export const CHECKS = [
  "V1_structure",
  "V2_text",
  "V3_pixels",
  "V4_residual",
  "V5_surrogate",
  "V6_integrity",
] as const;
export type CheckId = (typeof CHECKS)[number];

/** [x0, y0, x1, y1].  Pixels: canonical raster, origin top-left.  Points: PDF, origin bottom-left. */
export type Rect = [number, number, number, number];

export interface Box {
  page: number;
  px: Rect;
  pt: Rect;
  erased_px?: Rect;
}

export interface Evidence {
  detector: string;
  version: string;
  rule: string;
  strength: "strong" | "weak";
  score: number;
}

export interface Mention {
  mention_id: string;
  type: EntityType;
  action: Action;
  render: Action[];
  boxes: Box[];
  token_ids: string[];
  evidence: Evidence[];
  surrogate?: string;
  label?: string;
}

export interface Region {
  region_id: string;
  kind: RegionKind;
  action: Action;
  detector: string;
  rule: string;
  score: number;
  box: Box;
}

export interface ResidualBox {
  page: number;
  px: Rect;
  pt: Rect;
  erased_px?: Rect;
  type: EntityType | RegionKind;
  ref_id: string;
}

export interface LowEvidence {
  type: EntityType;
  token_ids: string[];
  evidence: Evidence[];
}

export interface PageInfo {
  index: number;
  width_pt: number;
  height_pt: number;
  dpi: number;
  width_px: number;
  height_px: number;
  kind: "image_only" | "born_digital" | "mixed" | "empty";
  lines: number;
  tokens: number;
  notice: boolean;
  unexplained_ink?: number;
  mean_ocr_confidence?: number;
}

export interface CheckResult {
  check: CheckId;
  passed: boolean;
  metrics: Record<string, number | boolean>;
}

export interface ModelProvenance {
  id: string;
  version: string;
  license: string;
  files: { name: string; sha256: string }[];
}

export interface InputInspection {
  page_count: number;
  encrypted: boolean;
  has_acroform: boolean;
  has_xfa: boolean;
  has_javascript: boolean;
  attachments_dropped: number;
  annotations_flattened_or_dropped: number;
  has_optional_content: boolean;
  user_unit_pages: number;
}

export interface Manifest {
  schema_version: string;
  status: "released" | "quarantined";
  reasons: string[];
  job: {
    input_sha256: string;
    page_count: number;
    profile: string;
    render_style: RenderStyle;
    source_format?: "pdf" | "tiff" | "png" | "jpeg" | "unknown";
  };
  provenance: {
    mlredact_version: string;
    config_hash: string;
    runtime_profile: { id: string; libraries: Record<string, string> };
    models: ModelProvenance[];
    detectors: { id: string; version: string }[];
    surrogate: null | { key_id: string; scope: "document" | "caller" };
  };
  input: InputInspection | null;
  pages: PageInfo[];
  mentions: Mention[];
  residual_boxes: ResidualBox[];
  low_evidence: LowEvidence[];
  detection: { propagation_rounds: number };
  verification: { passed: boolean; checks: CheckResult[] };
  regions?: Region[];
}

export interface RunRecord {
  job_id: string;
  mlredact_version: string;
  status: "released" | "quarantined";
  reasons: string[];
  workers: number;
  timings_ms: Record<string, number>;
}
