/**
 * Human-readable names for the engine's stable codes.  Codes are shown alongside these names
 * wherever space allows, so operators can match the UI to the manifest and the docs.
 */
import type { Action, CheckId, EntityType, RegionKind, RenderStyle } from "./manifest";

export const ENTITY_LABELS: Record<EntityType, string> = {
  person: "Person",
  date_of_birth: "Date of birth",
  date_of_death: "Date of death",
  date: "Date",
  age: "Age",
  street_address: "Street address",
  locality: "Locality",
  postcode: "Postcode",
  phone: "Phone",
  email: "Email",
  url: "URL",
  ip_address: "IP address",
  medicare: "Medicare number",
  ihi: "IHI",
  hpi: "HPI",
  dva_file: "DVA file number",
  crn: "Centrelink CRN",
  tfn: "Tax file number",
  ndis: "NDIS number",
  driver_licence: "Driver licence",
  passport: "Passport",
  clinical_id: "Clinical ID",
  provider_number: "Provider number",
  ahpra: "AHPRA registration",
  reference: "Reference number",
  court_file: "Court file number",
  bank_account: "Bank account",
  card_number: "Card number",
  abn: "ABN",
  acn: "ACN",
  employee_id: "Employee ID",
  vehicle_reg: "Vehicle registration",
  vin: "VIN",
  organisation: "Organisation",
  other_id: "Other identifier",
  file_path: "File path or name",
};

export const REGION_LABELS: Record<RegionKind, string> = {
  signature: "Signature",
  handwriting: "Handwriting",
  face: "Face",
  photo: "Photo",
  barcode: "Barcode",
  logo: "Logo",
  stamp: "Stamp",
  illegible: "Illegible area",
  faint_ink: "Faint ink",
  graphic: "Graphic",
};

export const ACTION_LABELS: Record<Action, string> = {
  surrogate: "Surrogate",
  retypeset_region: "Line re-typeset",
  remove: "Removed",
  synthesise: "Synthetic stand-in",
  blackout: "Blacked out",
  label: "Label",
  suppress_ink: "Ink suppressed",
  generalise: "Generalised",
  date_shift: "Date shifted",
  keep: "Kept",
};

/**
 * Overlay categories.  The order is the fixed categorical slot order (series-1 ... series-8), so a
 * category always wears the same colour regardless of which ones a document contains.
 */
export const CATEGORIES = [
  { id: "person", label: "People", slot: 1 },
  { id: "location", label: "Addresses & places", slot: 2 },
  { id: "contact", label: "Contact details", slot: 3 },
  { id: "dates", label: "Dates & ages", slot: 4 },
  { id: "health_gov", label: "Health & government IDs", slot: 5 },
  { id: "organisation", label: "Organisations", slot: 6 },
  { id: "other_id", label: "Other identifiers", slot: 7 },
  { id: "region", label: "Visual regions", slot: 8 },
] as const;
export type CategoryId = (typeof CATEGORIES)[number]["id"] | "residual";

const CATEGORY_OF: Record<EntityType, CategoryId> = {
  person: "person",
  date_of_birth: "dates",
  date_of_death: "dates",
  date: "dates",
  age: "dates",
  street_address: "location",
  locality: "location",
  postcode: "location",
  phone: "contact",
  email: "contact",
  url: "contact",
  ip_address: "contact",
  medicare: "health_gov",
  ihi: "health_gov",
  hpi: "health_gov",
  dva_file: "health_gov",
  crn: "health_gov",
  tfn: "health_gov",
  ndis: "health_gov",
  driver_licence: "health_gov",
  passport: "health_gov",
  clinical_id: "health_gov",
  provider_number: "health_gov",
  ahpra: "health_gov",
  reference: "other_id",
  court_file: "other_id",
  bank_account: "other_id",
  card_number: "other_id",
  abn: "other_id",
  acn: "other_id",
  employee_id: "other_id",
  vehicle_reg: "other_id",
  vin: "other_id",
  organisation: "organisation",
  other_id: "other_id",
  file_path: "other_id",
};

export function categoryOf(type: EntityType): CategoryId {
  return CATEGORY_OF[type];
}

export function categoryColor(id: CategoryId): string {
  if (id === "residual") return "var(--residual)";
  const slot = CATEGORIES.find((c) => c.id === id)?.slot ?? 8;
  return `var(--series-${slot})`;
}

export function categoryLabel(id: CategoryId): string {
  if (id === "residual") return "Verifier additions";
  return CATEGORIES.find((c) => c.id === id)?.label ?? id;
}

export function typeLabel(type: string): string {
  return (
    (ENTITY_LABELS as Record<string, string>)[type] ?? (REGION_LABELS as Record<string, string>)[type] ?? humanize(type)
  );
}

export const PROFILE_INFO: Record<string, { title: string; summary: string }> = {
  client: {
    title: "Client policy",
    summary:
      "Replaces every person, organisation, contact detail, identifier, file name and street line. Dates (including dates of birth), ages, suburbs and postcodes stay. Signatures and photos get synthetic stand-ins; figures stay.",
  },
  broad: {
    title: "Broad",
    summary:
      "Replaces every person, identifier, contact detail, date of birth or death, address part and linked organisation. Clinical content and other dates stay.",
  },
  claimant_focused: {
    title: "Claimant-focused",
    summary:
      "As Broad, but professionals' names, provider, AHPRA and HPI numbers and organisations stay visible. Anyone ambiguous is still replaced.",
  },
  maximal: {
    title: "Maximal",
    summary:
      "As Broad, plus every date shifted by one keyed offset, ages of 90 and over top-coded to “90+”, and every line with a replacement re-typeset.",
  },
};

export const STYLE_INFO: Record<RenderStyle, { title: string; summary: string }> = {
  surrogate: {
    title: "Surrogate",
    summary: "Realistic replacements derived from the keyed HMAC. Needs a surrogate key.",
  },
  label: { title: "Label", summary: "Placeholder tags such as [PERSON 1]." },
  blackout: { title: "Blackout", summary: "Solid boxes. No key needed." },
};

export const CHECK_INFO: Record<CheckId, { code: string; title: string; summary: string; independent: string }> = {
  V1_structure: {
    code: "V1",
    title: "Structure",
    summary:
      "Only allowlisted PDF keys and objects, a single revision, the right page and image counts, and the file also opens in PDFium.",
    independent: "Parses the output bytes",
  },
  V2_text: {
    code: "V2",
    title: "Text leak",
    summary:
      "No removed value appears in extracted text, in string operands of any stream, or in document info values.",
    independent: "Parses the output bytes",
  },
  V3_pixels: {
    code: "V3",
    title: "Pixels",
    summary:
      "Fills are uniform, surrogate boxes contain only the drawn glyphs, and suppressed areas hold only paper or strong ink.",
    independent: "Reported by the raster editor while drawing",
  },
  V4_residual: {
    code: "V4",
    title: "Residual re-read",
    summary:
      "The output is re-rendered and re-OCR'd. Outside the erased boxes: no removed value, no new strong rule hit, no barcode or face.",
    independent: "Shares only normalisers and rule detectors",
  },
  V5_surrogate: {
    code: "V5",
    title: "Surrogates",
    summary:
      "No surrogate equals or nearly equals an original. IDs fail their checksums, phones are in ACMA ranges, domains are reserved, dates move by one offset.",
    independent: "Re-checked separately from the generators",
  },
  V6_integrity: {
    code: "V6",
    title: "Integrity",
    summary: "Page count and page sizes match, and every planned operation was applied.",
    independent: "Partly: the applied count comes from the worker",
  },
};

/** Metrics whose non-zero value signals a problem (highlighted when a check fails). */
export const BAD_WHEN_NONZERO = new Set([
  "problems",
  "hits",
  "pages_failed",
  "residual_tokens",
  "from_redetection",
  "from_regions",
  "from_removed_values",
  "age_not_top_coded",
  "checksum_valid",
  "collision",
  "date_collision",
  "date_shift_inconsistent",
  "domain_not_reserved",
  "phone_not_reserved",
]);

const REASONS: Record<string, string> = {
  E_INPUT_UNREADABLE: "The input could not be parsed as a supported document.",
  E_INPUT_ENCRYPTED: "The PDF is encrypted or password-protected. Remove the protection and submit it again.",
  E_INPUT_XFA_ONLY: "The PDF is an XFA-only form with no renderable pages.",
  E_INPUT_TOO_LARGE: "The input is larger than the configured size limit.",
  E_INPUT_TOO_MANY_PAGES: "The input has more pages than the configured limit.",
  E_INPUT_UNSUPPORTED: "The file type is not supported. Use PDF, TIFF, PNG or JPEG.",
  E_INPUT_EMPTY: "The input has no pages.",
  E_RENDER_FAILED: "A page could not be rendered.",
  E_PAGE_TOO_LARGE: "A page exceeds the maximum raster size.",
  E_OCR_FAILED: "OCR failed on a page.",
  E_PAGE_ILLEGIBLE: "A page is mostly unreadable, and this configuration quarantines illegible pages.",
  E_PAGE_ROTATED_SIDEWAYS:
    "Held for review: a page is set to display turned on its side, so its text runs sideways, and this tool does not redact sideways pages. Turn the page upright in a PDF editor, save it and submit it again, or redact it by hand.",
  E_PAGE_CONTENT_SIDEWAYS:
    "Held for review: a page's text was drawn or scanned on its side, and this tool does not redact sideways pages. Turn the page upright in a PDF editor, save it and submit it again, or redact it by hand.",
  E_PAGE_ROTATED_UPSIDE_DOWN:
    "Held for review: a page is set to display upside down, and this tool does not redact upside-down pages. Turn the page upright in a PDF editor, save it and submit it again, or redact it by hand.",
  E_PAGE_CONTENT_UPSIDE_DOWN:
    "Held for review: a page's text was drawn or scanned upside down, and this tool does not redact upside-down pages. Turn the page upright in a PDF editor, save it and submit it again, or redact it by hand.",
  E_WORKER_TIMEOUT: "A page worker exceeded its time limit.",
  E_WORKER_CRASHED: "A page worker process crashed.",
  E_CONFIG_INVALID: "The configuration, or a supplied seed or key file, is invalid.",
  E_MODEL_MISSING:
    "A required model file or engine binary is missing. Run `mlredact models fetch` and check the System page.",
  E_MODEL_HASH_MISMATCH: "A model file does not match its pinned SHA-256.",
  E_RESOURCE_HASH_MISMATCH: "A bundled resource does not match its pinned SHA-256.",
  E_RUNTIME_PROFILE_MISMATCH: "This host does not match the pinned runtime profile.",
  V1_STRUCTURE: "V1 failed: the rebuilt PDF has unexpected structure.",
  V2_TEXT_LEAK: "V2 failed: a removed value was found in the output's text.",
  V3_PIXELS: "V3 failed: an erased area was not clean.",
  V4_RESIDUAL: "V4 failed: re-reading the output still found removable content after the allowed re-plans.",
  V5_SURROGATE: "V5 failed: a surrogate was unsafe (too close to an original, or a valid identifier).",
  V6_INTEGRITY: "V6 failed: page geometry or the applied operation count did not match the plan.",
  E_INTERNAL: "An unexpected internal error. Details are withheld by design.",
  // Raised by the redaction server's API rather than the engine.
  E_API_ENGINE_NOT_FOUND: "The redaction server could not start the engine. Check its MLREDACT_API_ENGINE setting.",
  E_API_ENGINE_FAILED: "The engine exited without a reason code.",
  E_API_TIMEOUT: "The job exceeded the server's time limit and was stopped.",
  E_API_INTERRUPTED: "The redaction server restarted while this job was queued or running.",
  E_API_INTERNAL: "The redaction server hit an unexpected error. Details are withheld by design.",
  E_API_NOT_ACTIVE: "The job was no longer queued or running, so there was nothing to stop.",
  E_API_UPLOAD_IN_PROGRESS: "The document is still being uploaded. Try again when the upload finishes.",
  E_API_SELFCHECK_BUSY: "A self-check for another profile is already running. Try again when it finishes.",
  // Raised by this app.
  E_UI_BACKEND_UNREACHABLE: "The redaction server could not be reached.",
  E_UI_BACKEND_TIMEOUT:
    "The redaction server did not answer in time. It may still be working: a self-check on a cold disk can take several minutes, so try again shortly.",
  E_UI_NETWORK: "This app's server could not be reached.",
  E_UI_BAD_RESPONSE: "This app's server sent a response that could not be read.",
};

export function reasonText(code: string): string {
  return REASONS[code] ?? "No description for this code.";
}

export function humanize(key: string): string {
  const s = key.replace(/_ms(_\d+)?$/, "$1").replace(/_/g, " ").trim();
  return s.charAt(0).toUpperCase() + s.slice(1);
}
