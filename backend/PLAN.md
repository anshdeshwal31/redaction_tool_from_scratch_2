# Plan — `mlredact`: local, deterministic PII redaction for Australian medico-legal PDFs

## Context

**Problem.** Australian medico-legal documents are mostly **scanned PDFs**: IME reports, clinical notes, hospital records, claim forms, legal correspondence and pathology/imaging reports. They are often degraded (skew, noise, faded text, stamps, handwriting, tables, multi-column layouts, mixed scanned/digital pages). They have to be turned into **properly de-identified PDFs**.

**Priorities.**
1. Maximum practical recall. A false negative is far worse than over-redaction.
2. Maximum practical determinism.
3. Fully local, self-hosted processing.
4. Production reliability.

**Scope.** The project directory `redaction_tool_from_scratch_2/` is empty, so this is a greenfield build. Evaluation and benchmarking live in a **separate codebase**. This system only exposes a stable, structured output contract for that evaluator to consume (Appendix A).

**Decisions confirmed with you**

| Topic | Decision | Design consequence |
|---|---|---|
| Hardware | Undecided: support both | CPU-first core path; optional GPU tiers; bit-level determinism is guaranteed **per pinned runtime profile** |
| Default output | **Realistic surrogates** | Pixels are erased and the surrogate is typeset in place; surrogates are derived deterministically from a key; non-text items (signatures, faces, barcodes, illegible text) are still removed with a neutral fill |
| Default policy | **Broad** | All people (including clinicians and lawyers), all identifiers and contact details, DOB/DoD, street + suburb + postcode, employer and claimant-linked organisations. Clinical content and other dates are kept |
| Human review | **None: fully automated** | Verification is fail-closed. Uncertain content is removed rather than flagged. Anything unresolved is quarantined, never released |

**Research findings that drive the design (2026)**

- **No off-the-shelf PII model is enough on clinical or legal text.**
  - OpenAI Privacy Filter (Apache-2.0, Apr 2026; 1.5B-parameter MoE with 50M active; 128k context) reaches F1 0.96 on PII-Masking-300k.
  - An independent benchmark measured only **0.38 recall on EHR notes** out of the box.
  - On SPY (legal/medical), its F1 rose from 0.545 to 0.962 after fine-tuning on 10% of the data.
  - GLiNER-PII models score about 0.81 F1 on general text and about 0.41 on clinical text.
  - A distilled DeBERTa from SHIELD (2026) reaches about 0.88 recall.
  - **Conclusion:** we need heterogeneous redundancy, rules and propagation, and in-domain fine-tuning.
- **OCR.**
  - PP-OCRv6 (Jun 2026, Apache-2.0): the medium tier is about 5 points better than PP-OCRv5-server and improves dot-matrix text.
  - PaddleOCR-VL-1.6 (May 2026, 0.9B, Apache-2.0) is state of the art on OmniDocBench and handles handwriting and stamps, but it is *generative*.
  - Several alternatives are ruled out on licence: Surya/Chandra (weights CC-BY-NC-SA with a revenue cap), MinerU, PyMuPDF and Ultralytics-YOLO (all AGPL), LayoutLMv3 (non-commercial).
- **Determinism.**
  - vLLM (`VLLM_BATCH_INVARIANT=1`) and SGLang (`--enable-deterministic-inference`) now offer batch-invariant LLM inference on NVIDIA sm80+ GPUs, at roughly a 25–45% slowdown.
  - ONNX Runtime has `use_deterministic_compute`. On CUDA, cuDNN algorithm autotuning has to be disabled (`cudnn_conv_algo_search=DEFAULT`).
- **PDF redaction.** Bland et al. (PETS 2023) de-redacted names from 11 redaction tools using glyph-position side channels. **Conclusion:** never edit the original PDF; always rebuild it.
- **Regulation (OAIC).** De-identification means removing direct identifiers *and* dealing with indirect identifiers and context; Australia has no safe-harbour list. **Conclusion:** policy must be profile-driven and configurable.

---

## 1. Proposed architecture

**Principles**

1. **Rebuild, never edit.** The output PDF is built from scratch. Each page is an edited raster, plus an optional invisible text layer generated *only* from content after redaction. No object, stream, font, image or metadata is copied from the input.
2. **Recall by redundancy.** Multiple OCR hypotheses are combined with multiple independent detectors and document-level propagation. Decisions lean towards the union of detector outputs.
3. **Grounded in geometry.** Every decision maps to pixel polygons in a single canonical raster coordinate system. Generative models never supply geometry.
4. **Deterministic core with bounded generative tiers.**
   - The core is pinned encoder models plus rules.
   - The optional vision-language model (VLM) and LLM tiers decode greedily, use grammar-constrained output and batch-invariant serving, only ever *add* detections, and must be grounded in the source text.
5. **Account for every patch of ink.** Ink that recognised text or known graphics don't explain is re-read. If it still can't be read, it is removed.
6. **Fail closed.** An independent verifier gates release. Any failure sends the document to quarantine.

```
 PDF/TIFF ─► [S0 Intake + sandboxed inspection] ─► [S1 Canonical rasterisation: PDFium, 300 dpi]
                                                       │  (+ embedded text layer with char boxes, if any)
 PER PAGE (parallel, pure functions) ──────────────────┼─────────────────────────────────────────────
   [S2 OCR-only normalisation: orientation · deskew · enhance  (exact affine maps kept)]
   [S3 Layout: regions · table cells · header/footer bands · reading order]
   [S4 Region detectors: barcode · face · signature/handwriting · stamp/seal · logo]
   [S5 OCR  A=PP-OCRv6  B=Tesseract 5  D=embedded text  C=PaddleOCR-VL (tier, targeted crops)]
   [S6 OCR fusion → canonical tokens + alternates]  →  [S7 Ink accounting → re-read or remove]
 PER DOCUMENT (single-threaded, canonical order) ───────────────────────────────────────────────────
   [S8 Text views: prose · line · cell/KV · header-footer · case-folded · OCR-normalised (+offset↔polygon maps)]
   [S9 Detectors: AU-ID rules · label→value · address grammar · NER×2(+1) · gazetteers · seeds · LLM sweep (tier)]
   [S10 Evidence fusion + contextual allowlists] → [S11 Entity resolution + propagation (to fixpoint)]
   [S12 Policy profile → actions] → [S13 Surrogate assignment (keyed HMAC-DRBG)]
 PER PAGE (parallel) ───────────────────────────────────────────────────────────────────────────────
   [S14 Raster editing: erase+typeset · region retypeset · remove · faint-ink suppression · notice]
 PER DOCUMENT ──────────────────────────────────────────────────────────────────────────────────────
   [S15 PDF build (pikepdf): image-only pages + invisible post-redaction text layer]
   [S16 Independent verification: structure · text · pixels · re-render+re-OCR+re-detect · surrogate safety]
            │ pass                                              │ fail
            ▼                                                   ▼
   redacted.pdf + manifest.json + run.json              quarantine record (reason codes only)
   (+ optional encrypted sensitive manifest)
```

## 2. Reasoning behind the architecture

- **Scanned documents come first, so OCR quality caps recall.** Detection can't find what OCR didn't read. Most of the investment therefore goes into OCR redundancy (two independent engines, with a VLM second reader as an extra tier), "ink accounting", and conservative handling of anything unreadable.
- **Any single detector misses 10–60% of PII on real clinical/legal text.** The plan therefore combines:
  - heterogeneous detectors with recall-biased fusion;
  - **label-anchored rules**, because most direct identifiers sit in forms, letterheads, "Re:" lines and patient labels;
  - **document-level propagation**, where one confident hit finds every other mention, including OCR-corrupted ones;
  - optional **job seeds**: known claimant details supplied by the calling system.
- **Encoders and rules are cheap and deterministic. Generative models are valuable only where encoders fail:** reading hard pixels (handwriting, stamps) and context-heavy recall sweeps. They also hallucinate and are expensive, so they're confined to grounded, add-only roles behind feature flags. They're enabled by default only once the external evaluation shows a recall gain.
- **Rasterise-and-rebuild removes whole classes of PDF leak.** It eliminates hidden or invisible text, metadata, annotations, incremental updates, glyph-position side channels and embedded original images. It also gives *one* code path for scanned, digital and mixed pages.
- **Fully automated release means the verifier must be independent of the renderer, and the system must fail closed.**
- **Determinism is designed in at every layer:**
  - the data model uses content-derived IDs and canonical ordering;
  - numerics use pinned runtimes, fixed threads and batches, and deterministic kernels;
  - generative tiers are greedy, batch-invariant and grammar-constrained.

  Guarantees are **per hardware profile**, because bit-exact results across different CPU instruction sets or GPU architectures are not achievable in general.

## 3–4. Alternatives considered and why they were rejected

| Area | Chosen | Alternative | Why rejected |
|---|---|---|---|
| Redaction mechanism | Rasterise + rebuild a new PDF | In-place content-stream redaction (PyMuPDF `apply_redactions`, pdf-redactor, qpdf stream surgery) | Many leak paths: invisible/hidden text, glyph-position shifts, font subsets, shared form XObjects, incremental updates. Needs two code paths. PyMuPDF is AGPL |
| | | Overlay boxes / annotation "redaction" | Content underneath stays recoverable |
| OCR | PP-OCRv6 + Tesseract 5 fusion; VLM second reader (tier) | Tesseract alone | Weaker on noise, faint text and handwriting; no redundancy |
| | | VLM-only OCR (PaddleOCR-VL, olmOCR, Qwen-VL) | Generative: can invent or drop text, gives imprecise geometry, costs more, and is harder to make deterministic. Used as a *second reader* only |
| | | Surya / Chandra | Weights are CC-BY-NC-SA with a revenue-gated waiver; code is GPL |
| | | MinerU | AGPL |
| | | Cloud OCR (Textract, Azure, Google) | Breaks the local-only requirement |
| | | ABBYY FineReader Engine (commercial, on-prem) | Strong on poor scans, but proprietary, costly and opaque. Can plug in later behind the `OcrEngine` interface |
| | | docTR (Apache-2.0) | Viable, but adds less engine diversity than Tesseract. Candidate third engine if the evaluator shows a gain |
| PII detection | Heterogeneous ensemble + rules + propagation + optional grounded LLM sweep | One NER model | Measured recall gaps on in-domain text |
| | | Local LLM alone | Low precision, span-boundary errors, hallucination, cost, determinism |
| | | Presidio as the core framework | Works on text only, with no awareness of geometry or OCR alternates; AU recognisers are basic. We reuse its ideas, not the framework |
| | | Commercial on-prem (John Snow Labs, Private AI) | Strong but proprietary, costly and opaque. Possible extra detector later |
| Layout | PP-DocLayout (RT-DETR, Apache-2.0) | LayoutLMv3 (CC BY-NC-SA); DocLayout-YOLO/Ultralytics (AGPL) | Licence |
| | | Docling Heron (Apache-2.0) | Viable, but has no seal or header/footer-image classes. Kept as an alternate |
| Runtime | ONNX Runtime for all encoder models; vLLM/SGLang on GPU and llama.cpp on CPU for generative tiers | Native PyTorch/Paddle throughout | Bigger footprint and more determinism controls to manage. ORT gives one place to enforce deterministic compute |
| PDF libraries | PDFium (pypdfium2), pikepdf/qpdf | MuPDF (AGPL), Poppler (GPL), Ghostscript (AGPL) | Licence |
| Surrogate mapping | Stateless derivation from a keyed HMAC | Stored real→fake mapping table | The table would itself be a PII store |
| | | Unkeyed hash, or Faker seeded with the value | Reversible by dictionary attack |

---

## 5. End-to-end data flow

| Stage | Scope | Input → output | Notes |
|---|---|---|---|
| S0 Intake | doc | Job request → validated job, input SHA-256, inspection report | Parsing runs in a sandboxed subprocess (qpdf/pikepdf). Quarantines: encrypted without password, XFA-only, over size or page limits. Records what will be **dropped**: attachments, JavaScript, hidden layers, annotation popups |
| S1 Rasterise | page | Page → canonical RGB raster + embedded char layer | PDFium; CropBox and /Rotate applied; visible annotations and form fields rendered; pixel-budget guard |
| S2 Normalise | page | Raster → OCR-ready copies + affine/projective maps | **The output always edits the canonical raster, never these copies** |
| S3 Layout | page | Regions, table cells, columns, reading order, header/footer bands | |
| S4 Region detectors | page | Barcodes, faces, signatures, handwriting lines, stamps, logos | |
| S5–S6 OCR + fusion | page | Canonical tokens (text, alternates, polygon, confidence vector, script type) | |
| S7 Ink accounting | page | Residual ink → re-read tokens or unreadable regions | |
| S8 Text views | doc | Length-preserving views + offset↔token maps | Joins text across lines and pages; de-hyphenates |
| S9–S10 Detection + fusion | doc | Mentions with an evidence record | |
| S11 Resolution | doc | Entity clusters, roles, propagated mentions | Iterates to a fixpoint |
| S12–S13 Policy + surrogates | doc | Action per mention/region; surrogate per entity component | |
| S14 Render | page | Edited rasters | Drawing ops in canonical order |
| S15 Build | doc | PDF bytes | |
| S16 Verify | doc | Pass/fail + metrics | One residual re-plan loop at most, then quarantine |
| S17 Emit + cleanup | doc | Artifacts; workspace deleted and its key destroyed | |

The per-job workspace is encrypted, and every stage writes **content-addressed artifacts**. A crashed job therefore resumes with byte-identical results.

---

## 6. OCR strategy

**Canonical raster**
- Rendered with PDFium at 300 dpi by default (configurable 200–400). Image-only pages can optionally render at their native resolution.
- Rendering includes visible annotations and AcroForm appearances, so the output reflects exactly what a viewer shows.
- Content that is never rendered (popups, attachments, hidden layers) cannot reach the output.

**Normalisation (OCR copies only)**
- **Orientation:** page orientation from a PP-LCNet document-orientation classifier, with Tesseract OSD as tie-breaker. Per-line orientation (0/180°) is also checked, and small rotated regions such as stamps or vertical table headers are OCR'd at four rotations.
- **Deskew** of up to ±15°.
- **Enhancement:** background normalisation and CLAHE; Sauvola binarisation for Tesseract; upscaling so the x-height is at least about 20 px.
- **Dewarping** (UVDoc) only for camera-captured pages.
- **Coordinate maps:** all transforms are composed and inverted in float64, and polygons are rounded *outward* to integer pixels.

**Engines**

| Engine | Role | Output |
|---|---|---|
| **A: PP-OCRv6 medium** (DB detection + CTC recognition), via ONNX Runtime with RapidOCR-style pre/post-processing (vendored and pinned) | Primary | Line polygons; per-character positions turned into word polygons; **per-character top-k alternatives** from CTC |
| **B: Tesseract 5.5.x LSTM** (`tessdata_best` eng) | Independent second engine | Word boxes + confidences. Runs page segmentation mode 3, plus a sparse-text pass (mode 11) in form regions |
| **D: Embedded text** (PDFium text API) | Exact text on born-digital pages | Char boxes. Treated as a hypothesis and checked against visible ink |
| **C: PaddleOCR-VL-1.6** (tier) | Second reader on targeted crops | Text only. Crops are low-confidence lines, A/B disagreements, handwriting, stamps/seals, tiny text and uncovered ink. On GPU it can optionally read whole pages |

**Fusion**
- Lines are matched across engines by geometric overlap.
- Within a line, characters are aligned by weighted edit distance using OCR confusion costs (rn↔m, cl↔d, 0↔O, 1↔l↔I, 5↔S, 8↔B…).
- Each token's text is chosen by confidence-weighted voting (ROVER-style). **All distinct hypotheses are kept as alternates.**
- Tokens found by only one engine are **kept**, because detection misses are a top source of false negatives.

**Ink accounting (safety net)**
- Start from the ink mask, then subtract token polygons and classified non-text graphics (ruling lines, checkboxes, figures handled by policy). What remains is residual ink.
- Text-like residual components are re-read: contrast stretch, ×2 upscale, four rotations, plus tier C.
- If still unreadable, the Broad profile applies:
  - **remove** the region in PII-prone zones (header/footer bands, form fields, near PII labels, margins);
  - **suppress** faint, non-text-like ink such as show-through from the reverse side.

**Handwriting**
- A line-level printed/handwritten classifier decides what is handwriting.
- Handwritten lines are read by engine A plus tier C.
- Handwritten tokens with detected PII, or with confidence below `τ_hw`, are **removed**. Handwriting is never re-typeset.
- Signatures and handwritten initials are always removed.

**Quality gate.** Each page gets a legibility score (mean confidence plus unreadable-ink fraction). An illegible page has all its content removed and a notice added (default), or the whole document is quarantined (configurable).

---

## 7. PII detection strategy

### 7.1 Taxonomy and Broad default actions

| Category | Includes (AU specifics) | Broad default |
|---|---|---|
| PERSON | Full or partial names, initials, nicknames, maiden names and aliases, for **every role**: claimant, relatives/associates, clinicians, lawyers, insurer/employer staff, witnesses | SURROGATE |
| SIGNATURE / HANDWRITTEN INITIALS | Image regions | REMOVE |
| DOB / DATE OF DEATH | Any format: "D.O.B.", dots, 2-digit years, written months | SURROGATE (DOB keeps its year; day and month are derived from the key) |
| STREET ADDRESS | Unit, number, street, lot, RMB, PO Box, Locked Bag, property names, c/- lines | SURROGATE |
| LOCALITY + POSTCODE | Suburbs and towns (outside an allowlist of capitals and major cities) and postcodes | SURROGATE (a real locality + postcode pair in the **same state**) |
| CONTACT | Phone/mobile/fax, email, personal URLs, social handles, IP addresses | SURROGATE (ACMA fictitious number ranges; `example.com`) |
| GOVERNMENT IDs | Medicare (+IRN), IHI, DVA file number, Centrelink CRN, TFN, NDIS, driver licence and card numbers, passport, visa/ImmiCard | SURROGATE (format-preserving, **check digit deliberately invalid**) |
| CLINICAL IDs | MRN/UR/URN, episode/admission numbers, pathology/imaging accession numbers, implant serials | SURROGATE |
| PROVIDER IDs | Medicare provider numbers, prescriber numbers, AHPRA registration, HPI-I/HPI-O (clinician names are redacted, so these would re-identify them) | SURROGATE |
| CLAIM / LEGAL / FINANCIAL | Claim, policy and insurer references; law-firm "Our/Your ref"; court/tribunal file numbers; BSB + account; card numbers; employee/payroll IDs; ABN/ACN of an employer or sole trader | SURROGATE |
| VEHICLE | Registration plates, VINs | SURROGATE |
| LINKED ORGANISATIONS | Employer/workplace, treating hospitals, clinics and practices, law firms, schools, clubs, small businesses | SURROGATE (fictional name; legal suffix such as "Pty Ltd" kept) |
| VISUAL | Faces and photos of people, barcodes/QR/DataMatrix, logos and letterhead images, stamps/seals | REMOVE |
| UNREADABLE | Illegible text or handwriting; unexplained ink in PII-prone zones | REMOVE (faint show-through: SUPPRESS) |

**Not redacted by default (Broad):**
- Clinical content: diagnoses, symptoms, findings, procedures, medications and doses, measurements and results, anatomy, clinical scales, impairment ratings (WPI).
- Personal context: occupation, gender, age, household composition, language or interpreter use.
- Dates other than DOB/DoD (injury, consultation and report dates).
- Places at state level and above: state, country, capital and major cities.
- Public references: legislation and guidelines (e.g. AMA Guides), government agencies and schemes (Medicare, Centrelink, NDIS, SIRA, icare, WorkSafe), insurers (configurable), names of courts and tribunals (but not file numbers), generic facility types.

**Other profiles** (same engine, different YAML):
- **Claimant-focused:** person and organisation actions are filtered by role.
- **Maximal:** adds document-level date shifting, localities generalised to state, ages ≥ 90 shown as "90+", configurable quasi-identifier categories, and line-retypeset positional hardening (§10).

### 7.2 Detectors

Every detector implements one `Detector` interface: pure, deterministic, versioned, and registered in config.

| ID | Detector | Evidence |
|---|---|---|
| R1 | **AU structured IDs** (regex + check-digit + context). Covers: Medicare (weights 1,3,7,9, mod 10; IRN), IHI/HPI-I/HPI-O (16 digits, prefixes 800360/1/2, Luhn), Medicare provider number (stem + location char + check char), DVA (state + war code + digits), CRN, TFN (mod 11), ABN (mod 89), ACN, NDIS, AHPRA, passport, state licence/card formats, BSB/account, cards (Luhn). Candidates are normalised for OCR confusions (O→0, l/I→1, S→5…) before validation | **Strong** if the checksum is valid *or* a label is present. **A failed checksum never suppresses a match when a label is present** (OCR damage) |
| R2 | **Contact / date / plate patterns.** AU phone formats (+61, 04xx, (0x), 13/1300/1800), email, URL, IP. Dates are parsed **day-first** by default, plus DOB/DoD cues. Plates and VINs need vehicle context | Strong with context, otherwise weak |
| R3 | **Label→value extractor**, layout-aware. A lexicon of AU form labels ("Name", "Surname", "Given names", "Patient", "Claimant", "Injured worker", "Re:", "DOB", "UR No", "MRN", "Claim No", "Our ref", "Medicare No", "NOK", "Employer", "Signed"…) marks the value region **regardless of NER**: to the right of the label, below it, or in the same table row or cell. Handwritten values cause the whole value region to be removed | Strong |
| R4 | **AU address grammar** plus ABS SAL / G-NAF gazetteer validation (CC BY 4.0) | Strong if validated |
| N1 | **OpenAI Privacy Filter**, run on ONNX Runtime with our own constrained Viterbi decoder; its 6 transition biases are set in config at a recall-biased operating point. Windowing is exact: 8 layers × 128-token band means a ±1024-token receptive field, so we stitch logits from windows with ≥1024-token overlap and run **one** Viterbi pass over the whole document. Later fine-tuned to our taxonomy | Strong ≥ τ_high, weak ≥ τ_low |
| N2 | **GLiNER-PII** (Knowledgator, Apache-2.0) on ONNX Runtime, with a label set matched to the taxonomy (person, organisation, hospital, employer, address…); fixed windows with overlap | Strong / weak |
| N3 | Reserved slot: an **in-house AU medico-legal de-identification model** (ModernBERT or DeBERTa-v3 student, from the model repo, Phase 7) | Strong / weak |
| G1 | **Gazetteers:** given names and surnames, localities and postcodes, healthcare facilities, honorifics and post-nominals, and (optionally) business names from the ABN bulk extract | Weak (corroborating) |
| S1 | **Seeds:** known identifiers supplied with the job (claimant name, DOB, address, claim number, family names) | Strong |
| P1 | **Propagation:** every surface variant of a resolved entity is searched across all views *and* OCR alternates (§8–9) | Strong |
| L1 | **LLM recall sweep** (tier). Each chunk is shown with existing detections marked, and the model is asked for remaining identifying spans "copied exactly" as grammar-constrained JSON. **Proposals are accepted only if they match the source text verbatim or with OCR-tolerant fuzzy matching (≥ 0.9)**. Add-only | Strong if grounded |
| — | **Context cues:** titles, kinship terms, signature-block and salutation patterns, "Dear …", "Re:", and role vocabulary | Modifies evidence and role |

**Text views.** Detectors run on several views, each with an exact offset→token map:
- **prose** (reading order, joined lines, de-hyphenated, crossing pages; headers/footers kept as a separate stream);
- **line**;
- **cell/key-value**;
- **title-cased** copy of ALL-CAPS lines (length-preserving, so offsets stay identical);
- **OCR-normalised** copy for IDs (also length-preserving);
- **per-engine views** (A/B/C/D).

The cheap detectors (R*, G1, P1, N1) run on all views. N2 runs on the fused prose view.

### 7.3 Evidence fusion and decisions (deterministic)

- Candidates are mapped to **canonical token sets**. Overlaps take the union of extents. Type is resolved by a policy priority table (ID > PERSON > ORG > LOCATION > DATE), with ties broken by detector ID.
- **Decision rule:**
  - any **strong** evidence → redact;
  - **two or more independent weak** signals → redact;
  - **one weak** signal → redact if the type is PERSON, ID, CONTACT or ADDRESS and no contextual allowlist applies;
  - otherwise keep, and record it in the manifest as a low-evidence candidate.
- **Neighbour rule:** tokens with confidence below τ_adj directly next to a PII span join that span. This protects partial reads such as "Smi?h".
- **Contextual allowlists** suppress *weak* evidence only, and **only inside exact patterns**. They never override seed, label→value or ID-rule evidence. They cover:
  - eponyms ("Tinel's sign", "Phalen's test", "Lachman", "McMurray", "Spurling", "Colles/Smith's fracture", "Parkinson's", "Crohn's"…);
  - drug brand names (from the PBS schedule);
  - scales and instruments;
  - legislation and agencies.

  The verifier uses the **same** allowlist rules (shared module), so policy and verification can't disagree.
- **Thresholds** are set per detector × type in versioned config. They start recall-first and are tuned from the external evaluator's results.

---

## 8. Entity resolution strategy

- **People.**
  - A name parser splits each mention into title, given names, middle, surname, suffix and initials. It handles inverted "SMITH, John", ALL-CAPS surnames, particles (van, de, O', Mc/Mac), hyphenated names, and "née", "formerly", "aka" and "preferred name" constructions.
  - **Deterministic agglomerative clustering:**
    - Mentions are processed in order of evidence strength, then length (descending), then first occurrence.
    - Clusters are seeded from full-name mentions.
    - Partial mentions attach only if compatible: surname equal or OCR-equivalent, given name or initial compatible, title gender compatible, role compatible.
    - Ambiguous partial mentions (e.g. "Mr Smith" when there are two Smiths) attach to a **surname group** rather than to an individual. They are still redacted; only the choice of surrogate depends on the grouping.
  - Each cluster's canonical spelling is the strongest variant: seed, then digital text, then highest OCR confidence, then frequency, then lexicographic order.
- **Addresses:** street types, unit notation and postcode are normalised. Locality-only mentions link to address clusters in the same locality.
- **Organisations:** legal suffixes (Pty Ltd, P/L, Limited) and punctuation are normalised. Acronyms and head nouns are linked ("Bunnings Warehouse" ↔ "Bunnings").
- **Dates:** parsed to ISO (day-first, windowed 2-digit years). The same date in different formats is the same entity.
- **IDs:** separators are stripped, OCR confusions are canonicalised in digit contexts, and equivalence is check-digit aware.
- **Propagation loop:** variants are generated (§9), all views and OCR alternates are searched, and hits become new mentions. This repeats until nothing new is found (bounded; usually 2 rounds).
- **Roles** (claimant/patient, relative/associate, clinician, legal, insurer, employer staff, other) come from deterministic cue scoring:
  - labels: "Re:", "Patient:", "Claimant";
  - kinship terms: "his wife", "son";
  - clinician markers: Dr/Prof, FRACS/MBBS, provider numbers, signature blocks;
  - legal markers: "Solicitor", "Lawyers".

  Under Broad every person is replaced anyway, so a role error can never cause a miss.

## 9. Alias handling

| Variant class | Handling |
|---|---|
| Full, given+surname, title+surname, initial(s)+surname, "SMITH J", "SMITH, John", ALL CAPS, possessives, hyphen parts | Generated from the parsed components of each cluster |
| Nicknames and transliterations | Pinned tables, e.g. William↔Bill/Will/Liam, Robert↔Bob/Rob, Elizabeth↔Liz/Beth, Mohammed↔Muhammad/Mohamed |
| Maiden / married / aka / preferred names | Linked when cue patterns are present |
| Initials only ("JS") | Only inside a person context, or when they match a cluster's initials alongside a person cue. Medical abbreviations (OT, PT, GP, CT, MR…) are on a blocklist |
| OCR-corrupted forms ("Srnith", "5mith") | Weighted edit distance using OCR confusion costs, with length-dependent thresholds; also searched in CTC alternates |
| Common-word names (May, Will, Grace, Hope, Mark, Rose, Bill) | Propagated only when capitalised mid-sentence, not in date context (May), or with a title or person cue |

**Consistent surrogates for aliases.** Mappings are kept per *component* (`given_map`, `surname_map`), keyed by canonical value, so every variant renders consistently:

- "John Smith" → "Peter Walsh"; "Mr Smith" → "Mr Walsh"
- "J. Smith" → "P. Walsh"; "SMITH J" → "WALSH P"
- "John" → "Peter"; "JS" → "PW"
- **Family members share a surrogate surname.**
- Casing, possessives and titles are preserved.

---

## 10. Redaction strategy

**Actions:**
- **SURROGATE:** erase, then typeset the surrogate.
- **RETYPESET_REGION:** erase a whole block, then typeset its lines with PII replaced.
- **REMOVE_REGION:** opaque neutral fill with a small optional label.
- **SUPPRESS_INK.**
- **KEEP.**
- Other profiles also support **BLACKOUT**, **LABEL**, **GENERALISE** and **DATE_SHIFT**.

**Geometry**
- A span maps to canonical tokens, then to the **union of polygons across all engines**.
- Coverage is always whole tokens.
- Padding is max(2 px, 8% of line height), skew-aware; multi-line spans are split per line.

**Escalation**
- **PII-dense blocks** are re-typeset as whole regions: patient-label stickers, address blocks, recipient blocks, signature blocks and fax headers (≥ 2 entities, or ≥ 40% PII tokens in a small block). No original pixels survive in them.
- **Crop-level hits from the VLM** (no word geometry) re-typeset the whole line.

**Surrogate generation (stateless and keyed)**
- `K_scope = HMAC(K_master[key_id], scope_id)`. The `scope_id` is the job's **matter/claim ID** (recommended, so surrogates stay consistent across a claim's documents) or, failing that, the document's SHA-256.
- Each component is drawn by HMAC-DRBG rejection sampling from **pinned, versioned pools**:
  - given names by gender (from title or name-lexicon cues), from state baby-name lists (CC BY);
  - surnames from public-domain and CC0 sources;
  - real AU localities and postcodes from the same state (ABS SAL);
  - fictional organisation and street names.
- The draw is filtered to the original's length bucket for visual fit; the box width already reveals that much.
- **Collision rules:** a surrogate is never equal or fuzzy-near any original value in the document, and is unique per entity. Collisions are redrawn deterministically.
- **Safe by construction:**
  - phone numbers come only from **ACMA's fictitious ranges** ((0x) 5550/7010 xxxx, the 0491 570 xxx list, 1800 160 401, 1300/1800 975 707–711);
  - emails and URLs use `example.com`;
  - IDs keep their format but **fail their checksum**.
- **DOB:** same year, with day and month derived from the key. **Maximal:** dates shift by a per-document offset, so intervals are preserved.

**Typesetting**
- **Erase** to an opaque fill of the local background: median of a ring of non-ink pixels, falling back to the page median.
- **Font:** Liberation Sans/Serif/Mono (OFL; metrically compatible with Arial/Times/Courier). The family comes from the embedded font name on digital pages, or from glyph statistics on scans.
- **Size** from x-height and cap-height; **colour** from the median ink colour; **baseline and skew** from line geometry.
- **Fit:** condense horizontally down to 0.75, then shrink. **Never draw outside the erased polygon.** Bitonal pages are drawn without anti-aliasing.

**Notice**
- A small margin banner reads: *"De-identified copy — personal details replaced with fictitious values."*
- It is placed only in blank margin. If there is none, the notice goes in metadata and the manifest only. Configurable.

**Positional side channel**
- An erased gap reveals roughly how long the original was.
- Option `positional_hardening: line_retypeset` (default in Maximal) re-typesets the whole line, so word positions no longer encode the original width.

## 11. PDF reconstruction strategy

- **Build:** pikepdf/qpdf writes a **new** PDF 1.7, with PDF/A-2b as an option.
- **Pages:** each page's MediaBox is the original visible page size in points. The page holds **one image XObject** and an optional **invisible text layer**: render mode 3, glyphless font, positioned per word. That layer contains only post-redaction text, meaning non-PII OCR text plus surrogates, so it matches what is visible.
- **Image encoding** depends on the page's colour class:
  - bitonal → CCITT G4 (lossless);
  - grey or colour → JPEG with pinned libjpeg-turbo and fixed quality and subsampling (lossless Flate is an option).
  - Output dpi defaults to the canonical dpi. Downsampling after editing is safe.
- **Metadata:** a fixed Info dictionary (Producer = mlredact version; Title = "De-identified document"). Dates appear only if the job supplies them; XMP only in PDF/A mode, with deterministic values.
- **Deterministic structure:** deterministic `/ID` (`deterministic_id`), fixed object order, single revision.
- **Never written:** outlines, annotations, forms, JavaScript, OpenAction, attachments, optional-content groups, structure tree, page labels, thumbnails.

## 12. Redaction verification (independent, fail-closed)

The verifier lives in `verify/`. It imports only the shared normalisers and allowlist rules, never the rendering code, so a rendering bug can't hide itself.

| Check | Method | On failure |
|---|---|---|
| V1 Structure | Walk the object graph with pikepdf *and* open with pypdfium2. Only the allowlisted object kinds may appear; single revision; nothing listed in §11 is present | Quarantine |
| V2 Text leakage | Extract text with pypdfium2, then scan every decoded content stream and string. Search for **every removed value and its variants**: case, space and separator-insensitive; digits-only for IDs; OCR-normalised | Quarantine |
| V3 Pixels | Checked on pre-encoding rasters: REMOVE regions are 100% fill colour; inside SURROGATE regions, original ink not explained by the surrogate's glyph mask is zero; outside planned regions and the notice area, pixels are unchanged. Plus a decode-consistency check of the PDF images | Quarantine |
| V4 Re-render + re-read | Render the **output PDF** with PDFium, OCR it with A and B, then run R1–R4 and P1 with removed values as seeds. Any fuzzy match of a removed value, or any new strong PII, triggers **one** re-plan and re-render; if it persists, fail | Re-plan, then quarantine |
| V5 Surrogate safety | Surrogates must not equal or be near any original; IDs must fail their checksums; phone numbers must be in ACMA ranges; emails must use reserved domains; date shifts must be consistent | Quarantine |
| V6 Integrity | Page count and sizes match; every planned operation was applied (count + hash); manifest and PDF agree; output re-opens | Quarantine |

**Limit, stated plainly:** verification proves that *planned* removals happened and can't be recovered. It cannot find PII that every detector missed. Recall is measured by the external evaluator.

---

## 13. Determinism strategy

| Source of nondeterminism | Control |
|---|---|
| Inputs and configuration | A job fingerprint is recorded in the manifest: input SHA-256, config hash, policy hash, model-registry snapshot hash, surrogate `key_id`, runtime profile, software version |
| Build | Container images pinned by digest; `uv.lock` with hashes; OS packages from snapshot repositories; models, fonts and resources verified by SHA-256; `SOURCE_DATE_EPOCH` and reproducible image builds; **no network at runtime** (`HF_HUB_OFFLINE=1` etc.) |
| ML numerics | ONNX Runtime: fixed intra/inter-op threads, `use_deterministic_compute`, fixed graph-optimisation level, no dynamic quantisation. CUDA: `cudnn_conv_algo_search=DEFAULT`; any PyTorch uses deterministic algorithms and `CUBLAS_WORKSPACE_CONFIG`. Tesseract `OMP_THREAD_LIMIT=1`; OpenCV and BLAS threads fixed |
| Batching | Fixed batch shapes that don't depend on concurrency (batch 1 or fixed padded batches); fixed NER windows |
| LLM / VLM tiers | GPU: vLLM `VLLM_BATCH_INVARIANT=1` or SGLang `--enable-deterministic-inference`. CPU: llama.cpp with fixed threads and batch sizes. Greedy decoding, grammar-constrained JSON, fixed max tokens, prefix caching off, versioned prompt templates; responses replayed within a job on retry |
| Ordering | Total-order sort keys everywhere (page, reading order, offset, type rank, detector ID). **IDs are content- and position-derived, never UUID or time.** Unordered sets are never iterated without sorting; `PYTHONHASHSEED=0` as a backstop |
| Concurrency | Per-page stages are pure functions merged by page index. Document stages are single-threaded. CI checks that changing the worker count or page order cannot change outputs |
| Randomness and time | The only randomness is the keyed HMAC-DRBG. No wall-clock values in `manifest.json` (operational data goes to `run.json`) |
| Geometry | float64 transforms; explicit outward rounding; integer pixel operations |
| Rendering and encoding | PDFium, Pillow/FreeType, libjpeg-turbo and zlib pinned, with fixed flags |
| Output PDF bytes | Deterministic object order and `/ID` |
| Retries and crashes | Content-addressed stage artifacts, so a resumed job produces identical bytes |

**Limits that can't be removed, and their mitigation.**
- Float results differ across CPU instruction sets (AVX2 vs AVX-512) and GPU architectures.
- **Runtime profiles** (e.g. `cpu-x86-64-v3`, `gpu-cuda-sm89`) are therefore pinned in production config and checked at startup.
- Golden hashes are kept per profile.
- Decisions favour the union of detectors, so tiny numeric drift rarely flips one.
- Production **determinism canaries** periodically re-run synthetic documents and compare hashes.
- Native Windows runs are for development convenience only and are not a determinism target.

## 14. Security / privacy architecture

**Threat model.**
1. Recipients trying to recover PII: de-redaction, metadata, hidden layers, width side channels, surrogate inversion.
2. Insiders with access to infrastructure, logs or caches.
3. Malicious PDFs: parser exploits, decompression bombs.
4. Supply chain: tampered packages or model weights.
5. Accidental leaks: logs, telemetry, crash dumps, temp files, swap.

**Controls**
- **Data lifecycle.**
  - Each job gets an encrypted workspace: tmpfs or dm-crypt with a per-job data key.
  - At job end the artifacts are deleted **and the key destroyed** (crypto-shredding).
  - Inputs and outputs are encrypted at rest, with configurable retention.
- **Isolation.**
  - Workers have **no network egress**.
  - PDF parsing and rendering run in a **sandboxed subprocess** (seccomp, rlimits, timeouts; optionally nsjail or gVisor).
  - Containers run non-root with a read-only root filesystem.
  - **No core dumps** (`RLIMIT_CORE=0`, `PR_SET_DUMPABLE=0`); swap off or encrypted.
  - Model servers listen only on loopback or a Unix socket, with telemetry disabled (vLLM usage stats, HF telemetry, PaddleX model-source check).
- **Logging and observability.**
  - A typed log facade accepts **only allowlisted fields**: IDs, counts, durations, stage names, reason codes, keyed hashes.
  - A `Sensitive[T]` wrapper redacts `__repr__`/`__str__` and refuses serialisation.
  - An exception sanitiser logs only the exception type, code location and reason code — never messages from third-party exceptions, which can embed content.
  - As a last resort, a scrubber masks anything in the current job's removal set.
  - Metrics use only low-cardinality labels; trace IDs are hashed.
  - **Filenames are never logged** — they often contain names.
- **Debug.** Production images can't write debug bundles; dev bundles require a synthetic-data flag.
- **Secrets.** `K_master` and the manifest encryption keys live in a secret store (OpenBao/Vault/KMS). Manifests carry only `key_id`.
- **Access.** OIDC or mTLS authentication. RBAC roles: submit, read-output, read-sensitive-manifest (evaluator only; rare), admin. An audit log records who did what, when, on which job hash — no content.
- **Supply chain.**
  - SBOM, vulnerability scanning and **licence gate** (no AGPL, GPL or non-commercial licences in the runtime image).
  - Signed images and model artifacts.
  - Weights only as safetensors, ONNX or GGUF (**no pickle**); `trust_remote_code=False`, with any required model code vendored after review.
- **Outputs.** The safe manifest contains no original text. The sensitive manifest is opt-in and encrypted to the evaluator's public key (X25519/age).

## 15. Model selection

| Role | Choice | Licence | Runtime | Alternates |
|---|---|---|---|---|
| Render / inspect / build | PDFium (pypdfium2); pikepdf/qpdf | BSD/Apache; MPL-2.0/Apache | in-process (sandboxed) | — |
| Orientation | PP-LCNet document and text-line orientation classifiers | Apache-2.0 | ONNX Runtime | Tesseract OSD |
| Layout + table cells | PP-DocLayout_plus-L (or newer PP-DocLayout at pin time); PP-StructureV3 table-cell detector | Apache-2.0 | ONNX Runtime | Docling Heron |
| OCR A | PP-OCRv6 medium detection + recognition | Apache-2.0 | ONNX Runtime | PP-OCRv5 server |
| OCR B | Tesseract 5.5.x LSTM (`tessdata_best`) | Apache-2.0 | tesserocr | docTR |
| OCR C (tier) | PaddleOCR-VL-1.6 (0.9B) | Apache-2.0 | vLLM (GPU); transformers or llama.cpp (CPU), validated in Phase 5 | olmOCR, Qwen-VL |
| Barcodes / faces | zxing-cpp; YuNet (OpenCV Zoo) | Apache-2.0; MIT | native / ONNX Runtime | — |
| Signature / handwriting | In-house RT-DETRv2/D-FINE detector + line classifier (model repo); heuristics until then | Apache training stack | ONNX Runtime | — |
| NER 1 | OpenAI Privacy Filter (fine-tuned in Phase 7) | Apache-2.0 | ONNX Runtime + own Viterbi | — |
| NER 2 | Knowledgator GLiNER-PII (large) | Apache-2.0 | ONNX Runtime | nvidia/gliner-PII (NVIDIA Open Model Licence); GLiNER2 |
| NER 3 | In-house ModernBERT/DeBERTa-v3 student distilled from a local LLM teacher on synthetic + annotated AU documents | MIT/Apache bases | ONNX Runtime | — |
| LLM tier | Bake-off between **Qwen3.5/3.6 (~14–32B)**, **Gemma 4 (12B / 26B-A4B)** and **gpt-oss-20b** | Apache-2.0 | vLLM/SGLang (GPU, BF16/FP8); llama.cpp Q8_0 with a smaller model in CPU targeted mode | — |
| Fonts | Liberation Sans/Serif/Mono | SIL OFL | Pillow/FreeType | Noto |

**Tier defaults**

| Profile | VLM second reader | LLM tier |
|---|---|---|
| GPU | On | On |
| CPU | Targeted crops only | Off, or targeted mode |

Either tier is promoted to default only if the external evaluator shows a recall gain.

## 16. Infrastructure requirements

- **CPU profile node:** x86-64-v3 (AVX2) or newer; 16–32 vCPU; 64 GB RAM; 200 GB encrypted NVMe scratch.
  - *Rough* core-path cost: 12–20 s per page per core, including verification. That's about 50–80 pages/min on 16 cores; to be measured in Phase 6.
- **GPU profile node:** NVIDIA sm80+ (required for batch-invariant serving).
  - L40S 48 GB or A100/H100 80 GB to host the VLM plus a 14–32B LLM.
  - L4 24 GB if the LLM is 8B or smaller.
  - CPU, RAM and scratch as for the CPU profile.
- **Services:**
  - PostgreSQL (job queue via `SKIP LOCKED`, metadata only);
  - encrypted storage: local volume or customer-provided S3-compatible storage with server-side encryption;
  - secret store;
  - on-prem OpenTelemetry collector, Prometheus and Grafana.
- **Development:** your laptop (Ryzen 5 5600H, 16 GB, no NVIDIA GPU) builds the **CPU profile inside a Linux dev container** (Docker Desktop on WSL2). GPU-tier work needs a GPU box.

## 17. Deployment strategy

- **Artifacts:**
  - `mlredact-cpu` and `mlredact-gpu` images, pinned by digest;
  - a `mlredact-models` bundle, hash-verified and signed;
  - a Helm chart and Docker Compose file;
  - an **air-gapped install bundle** (images, models, SBOM, signatures).
- **Topology:** stateless API → Postgres queue → CPU/GPU worker pools, each with sidecar model servers → encrypted storage. No egress anywhere.
- **Startup self-checks:** model hashes; detected runtime profile matches the pinned one; config schema; a **determinism smoke test** (tiny synthetic document → expected hash). Any failure means the worker refuses to start.
- **Configuration:** layered (defaults < profile < deployment < limited job overrides), validated by pydantic, with unknown keys rejected. Canonical JSON produces a hash recorded in provenance.
- **Rollout:**
  - a versioned "pipeline version" (app + image digest + models snapshot + config + policy);
  - blue/green deploys;
  - promotion gated on golden diffs and sign-off from the external evaluation;
  - previous versions retained for the reproducibility window.

## 18. Project structure

```
redaction_tool_from_scratch_2/
  pyproject.toml  uv.lock  README.md  Makefile  .devcontainer/
  src/mlredact/
    core/         # types (Page, Token, Span, Mention, Entity, Action), ids, ordering, errors, Sensitive[T]
    config/       # pydantic models, layered loader, canonical hashing
    runtime/      # determinism setup (threads, env, flags), runtime-profile detection, startup self-checks
    pipeline/     # stage interface, DAG runner (process pool, spawn), encrypted workspace, artifact cache
    ingest/       # sandboxed inspection, PDFium renderer, embedded-text extractor, image inputs
    imaging/      # orientation, deskew, enhancement, transform algebra
    layout/       # layout model, table cells, reading order, header/footer, region detectors
    ocr/          # OcrEngine interface; ppocr_onnx, tesseract, vlm_reader, embedded; fusion; ink accounting
    text/         # views, offset↔token maps, de-hyphenation, OCR-confusion tables, normalisers (shared w/ verify)
    detect/       # Detector interface; rules/ (au_ids, contact, dates, labels, address); ner/ (privacy_filter,
                  #   gliner, viterbi); gazetteer/; seeds/; llm/; fusion/; allowlists/
    resolve/      # name parser, clustering, aliases, nicknames, propagation, roles, addr/org/date/id normalisation
    policy/       # profiles → actions
    surrogate/    # HMAC-DRBG, pools, per-type generators, checksum-invalidation, date shift, collision checks
    render/       # erase+typeset, region retypeset, fills, ink suppression, notice
    pdfout/       # PDF builder, glyphless text layer, metadata, PDF/A
    verify/       # V1–V6 (independent of render/)
    manifest/     # schema models, canonical JSON writer, sensitive-manifest encryption
    security/     # log facade, scrubber, exception sanitiser, secrets, crypto
    observability/# metrics, tracing
    api/  cli/    # FastAPI service; Typer CLI (same artifacts)
  resources/      # versioned gazetteers, surrogate pools, allowlists, label lexicon, confusion matrix, fonts (+SHA256SUMS)
  models/registry.yaml   # model id/version, file SHA-256, licence, format, export-recipe & parity-report hashes
  configs/profiles/{broad,claimant_focused,maximal}.yaml  configs/runtime/{cpu-x86-64-v3,gpu-cuda-sm89}.yaml
  schemas/        # job-request.v1.json, manifest.v1.json, sensitive-manifest.v1.json
  docker/  deploy/helm/  deploy/compose/
  tests/{unit,property,integration,determinism,security,golden}/
  tools/synthdocs/  # synthetic AU medico-legal fixture generator (tests only); model export/parity helpers
  docs/{architecture.md,threat-model.md,adr/,runbooks/}
```

Fine-tuning, annotation tooling and training data live in a **separate repository** (`mlredact-models`). This repo only consumes signed, pinned artifacts.

## 19. Testing strategy

This is correctness testing, not an evaluation framework.

- **Unit tests:**
  - every recogniser, with positive cases, negative cases and OCR-corrupted variants;
  - the checksum validators;
  - date parsing (day-first, 2-digit years);
  - the name parser and clustering;
  - alias variants and propagation guards;
  - each surrogate generator (format kept, checksum invalid, ACMA ranges, no collisions);
  - transform algebra; render primitives; the PDF builder.
- **Property-based tests (Hypothesis):** shuffling the input order gives the same output; idempotence; transforms are invertible; surrogates never equal originals; span merging is associative and commutative.
- **Synthetic fixtures (`tools/synthdocs`):**
  - templated IME reports, clinical notes, hospital forms with patient labels, letters with letterheads, claim forms and court documents;
  - filled with fake AU data;
  - degraded with Augraphy (MIT): skew, noise, blur, JPEG, fax binarisation, show-through, stamps, handwriting fonts, multi-column layouts, tables;
  - the ground truth is known by construction. **Real documents never enter the repo or CI.**
- **Integration and golden tests:** the end-to-end manifest and PDF hashes for the fixture corpus are pinned per runtime profile. Any change needs an explicit golden update with a reviewed manifest diff. Planted values are asserted **unrecoverable**.
- **Determinism tests:** each fixture runs twice in CI, with different worker counts, page orders and cold/warm caches → byte-identical output. GPU-profile tests run on the GPU runner.
- **Security tests:**
  - canary PII strings must never appear in logs, metrics, exceptions, leftover temp files or `run.json`;
  - a malformed and malicious PDF corpus plus fuzzing (atheris/Hypothesis);
  - recovery attempts on outputs (multiple extractors, raw stream dumps, `strings`, OCR).
- **Verifier mutation tests:** deliberately skip a box, leak a text-layer word, or copy metadata → V1–V6 must fail.
- **Model smoke tests:** fixed inputs give expected output hashes per profile. Parity between ONNX and the reference model is tested in the model repo.
- **CI:** ruff, mypy `--strict`, pytest, licence gate, SBOM, container build reproducibility check.

## 20. Development phases (each phase ends in a runnable, verified state)

| Phase | Deliverables | Exit criteria |
|---|---|---|
| **P0 Foundations** | Repo + uv + CI + devcontainer; core types; config/profiles; runtime determinism module; security log facade and `Sensitive[T]`; model registry + hash verification; synthdocs v0 | CI green; determinism harness and leak tests run |
| **P1 Safe minimal pipeline** | Intake/sandbox, PDFium render, OCR A, R1/R2, REMOVE fill, PDF builder, V1–V4 and V6, manifest v0, CLI | Synthetic documents redacted and verified; byte-identical reruns; malicious-PDF corpus handled |
| **P2 OCR robustness + layout** | Normalisation; OCR B + embedded text; fusion; layout and tables; header/footer; R3 label→value; R4 addresses; barcode, face, stamp and logo detection; interim signature/handwriting handling; ink accounting; legibility gate | Degraded corpus passes; page-parallel determinism holds |
| **P3 Detection ensemble** | N1 + N2 on ONNX Runtime; text views; gazetteers; allowlists; fusion rules; threshold config; seeds; propagation | **Handoff to external evaluator: baseline 1** |
| **P4 Resolution + surrogates** | Name parser, clustering, aliases, OCR-variant matching, roles; surrogate generators; erase+typeset; region retypeset; text layer; notice; V5; Claimant-focused and Maximal profiles | Consistency and safety tests pass; **evaluator baseline 2** |
| **P5 Generative tiers** | VLM reader and LLM sweep with deterministic serving (GPU) and targeted CPU mode; grounding; replay | Determinism on both profiles; evaluator shows recall gain, otherwise tiers stay off |
| **P6 Productionisation** | API, queue, workers, storage and secret adapters; cpu/gpu images; Helm/Compose; observability; canaries; runbooks; load tests; threat-model review and output pen-test; SBOM and licence; air-gapped bundle; optional PDF/A | Production-readiness review |
| **P7 Models (parallel, separate repo)** | Annotation guideline (AU taxonomy); secure annotation (self-hosted Label Studio); synthetic + real in-domain data; Privacy Filter fine-tune; N3 student; signature/handwriting detectors; ONNX export, parity, signing | Each release is promoted only via evaluator results + golden diffs |

## 21. Major risks

| # | Risk | Impact | Mitigation |
|---|---|---|---|
| R1 | Off-the-shelf NER recall is far below target on AU medico-legal text | Missed PII | Ensemble + rules + label→value + propagation + seeds; **P7 fine-tuning is the main lever**; LLM sweep |
| R2 | OCR misses or misreads text (faint, handwriting, stamps, dot-matrix) | Missed PII | Two engines + VLM reader; CTC alternates; ink accounting; conservative removal |
| R3 | No in-domain labelled data | Can't fine-tune or calibrate | Synthetic data (MedDeID-style) + LLM-teacher distillation + targeted annotation |
| R4 | Realistic surrogates mistaken for real people or data | Misidentification, misuse | Notice banner; reserved phone/email ranges; checksum-invalid IDs; common-name pools |
| R5 | Over-redaction (eponyms, drugs, common words) harms usability | Lower utility | Contextual allowlists for weak evidence only |
| R6 | Determinism breaks on hardware or software drift | Irreproducible outputs | Runtime profiles, startup checks, canaries, digest pinning |
| R7 | Malicious PDFs | Compromise or denial of service | Sandbox, resource limits, timeouts, fuzzing |
| R8 | PII leaks through the system itself (logs, crash dumps, caches) | Breach | Typed logging, `Sensitive[T]`, no core dumps, encrypted workspaces with crypto-shredding, canary leak tests |
| R9 | Licence contamination | Legal exposure | Licence gate in CI; only permissive components |
| R10 | Generative tiers too slow on CPU | Throughput | Targeted mode or off on CPU; GPU profile |
| R11 | Positional side channel (erased-gap width) | Partial inference of names | `line_retypeset` hardening option |
| R12 | Quarantine rate too high under fail-closed | Operational backlog | Reason-code analytics; conservative removal instead of quarantine where safe |

## 22. Expected failure modes

| Failure mode | How it shows | Handling |
|---|---|---|
| Name OCR'd as "Srnith", "J0HN" | Text-only NER misses it | Fuzzy propagation over alternates; seeds; neighbour rule |
| Uncommon or non-Anglo names, initials, lowercase names | Missed by NER | Gazetteers, context cues, label→value, LLM sweep, fine-tuning |
| Patient-label stickers, fax headers, letterheads | Dense PII with partial reads | Region retypeset (escalation) |
| Value in a different table cell from its label | Label→value miss | Table-cell geometry + below/right search |
| Handwritten notes and annotations | Low confidence | VLM reader; remove if PII or unreadable |
| Eponyms that look like names ("Phalen's test") | False positives | Contextual allowlist (weak evidence only) |
| Surname shared by two people | Wrong cluster | Surname group; doesn't affect recall |
| Surrogate too long or skewed line | Visual artefact | Condense, shrink, never overflow; fall back to region retypeset |
| Coordinate-transform bug | Box misplaced | V4 re-render + re-read → re-plan or quarantine |
| New ID or claim-number format | Pattern miss | Label anchors + NER + LLM; patterns live in config |
| Non-Latin script, interpreter letters | Weak NER | PP-OCRv6 multilingual; non-Latin tokens in PII contexts removed |
| Very long records (500+ pages) | Memory and latency | Page streaming, exact NER windowing, limits |
| XFA forms / encrypted PDFs | Can't render | Quarantine with reason code |
| Show-through text from the reverse page | Faint mirrored ink | Faint-ink suppression |
| Verifier false alarm (e.g. allowlisted eponym) | Spurious quarantine | Allowlist rules shared between policy and verifier |

## 23. Mitigations (defence-in-depth summary)

| Layer | What it protects against |
|---|---|
| M1 Multi-engine OCR + alternates + ink accounting | Unread PII |
| M2 Heterogeneous detectors + label→value + seeds + propagation | Missed detections |
| M3 Recall-biased fusion + neighbour and escalation rules | Partial or boundary misses |
| M4 Rasterise-and-rebuild | Hidden-content and PDF-structure leaks |
| M5 Independent verification V1–V6, fail-closed | Execution errors and leaks |
| M6 Keyed, stateless, safe-by-construction surrogates | Re-identification via surrogates; accidentally creating real identifiers |
| M7 Determinism controls + runtime profiles + canaries | Irreproducibility |
| M8 Isolation, typed logging, crypto-shredding, no egress | Leakage from the system itself |
| M9 Pinning, signing, SBOM, licence gate | Supply-chain risk |
| M10 Evaluator-gated promotion of models, thresholds and tiers | Silent regressions |

## 24. Production operational considerations

- **SLOs and alerts:**
  - per-page latency (p95) and queue age;
  - success and quarantine rates by reason code;
  - unreadable-ink and OCR-confidence distributions, which reveal scanner or quality drift;
  - entity counts per type per document (a sudden drop means a regression);
  - verifier re-plan rate;
  - determinism canary status.
- **Quarantine:**
  - quarantined documents are never released;
  - reason codes go back to the caller;
  - options are a stricter-profile retry or manual handling outside the system.
- **Reproducibility:** re-running with the same pipeline version, key ID and runtime profile gives byte-identical output. Images and models are kept for the retention window.
- **Surrogate key management:**
  - rotating `K_master` changes future surrogates;
  - keep the `key_id` stable per matter if cross-document consistency must hold;
  - key IDs are recorded in manifests.
- **Change management:**
  - policy, threshold, model and tier changes all bump the pipeline version;
  - each needs golden diffs plus external evaluation;
  - policy profiles are signed off by your privacy officer (Privacy Act / APPs, NSW HRIP Act, Vic Health Records Act; OAIC de-identification guidance).
- **Incident response (suspected leak):**
  1. freeze releases for that pipeline version;
  2. use manifests to identify affected jobs;
  3. recall and reprocess;
  4. assess Notifiable Data Breaches obligations.
- **Capacity:** autoscale workers on queue depth; keep CPU and GPU pools separate; GPU sidecars sized to the chosen LLM.
- **Retention and backups:**
  - back up only metadata, configs and manifests (no PII);
  - workspaces are never backed up;
  - inputs and outputs follow your retention policy.

---

## Appendix A — Output contract (stable interface for the external evaluator)

- **Job request** (`schemas/job-request.v1.json`):
  - `input` (PDF/TIFF/PNG/JPEG);
  - `profile`;
  - optional `scope_id` (matter/claim);
  - optional `seed_entities` [{type, value, role}] (sensitive);
  - `options` {output_dpi, text_layer, notice, sensitive_manifest: off | encrypt-to `<pubkey_id>`}.
- **Artifacts:**
  - `redacted.pdf`;
  - `manifest.json` — deterministic and **contains no original text**;
  - `run.json` — operational: timings, host, job ID; not deterministic;
  - optional `sensitive-manifest.json.age`.
  - A failed job produces `quarantine.json` (reason codes) instead.
- **`manifest.json` (v1, semver):**
  - **Schema and job:** `schema_version`; `job` {input_sha256, page_count}.
  - **`provenance`:** app version, git commit, image digest, model registry entries (id, version, sha256), config/policy hashes, runtime profile, key_id.
  - **`pages`:** size in points, raster dpi, orientation and transforms.
  - **`entities`:** each with:
    - entity_id (position-derived, not value-derived), cluster_id, type, subtype, role, action, surrogate_value (safe: it is visible in the output);
    - **`mentions`**: mention_id, page, polygons in **PDF points and pixels**, view offsets, detector evidence [{detector_id, version, score, rule_id}], decision reason codes.
  - **`regions`:** non-text removals (signature, face, barcode, logo, stamp, illegible, suppressed ink), with polygons and reasons.
  - **`low_evidence_candidates`:** kept items, for the evaluator's error analysis.
  - **`verification`:** V1–V6 results and metrics.
- **Sensitive manifest** (opt-in, evaluator role, encrypted): original mention text, OCR tokens and alternates per page, cluster canonical strings.
- **Interfaces:**
  - CLI: `mlredact run --input in.pdf --profile broad --out out/ [--scope-id …] [--seeds seeds.json]`;
  - Python: `Redactor(config).run(...)`;
  - HTTP: `POST /v1/jobs`, `GET /v1/jobs/{id}`, `GET /v1/jobs/{id}/artifacts/{name}`. The job key is a hash of input + request, so resubmissions are idempotent.

## Appendix B — How the build is verified end-to-end

1. **Every phase:**
   - `make test` runs unit, property and integration tests;
   - `make determinism` runs each fixture twice with shuffled page order and different worker counts, and asserts byte-identical `redacted.pdf` and `manifest.json`;
   - `make leaktest` checks canary PII in logs, metrics, temp files and `run.json`;
   - `make verify-mutations`.
2. **From P1 onward:** `mlredact run` on the synthdocs corpus inside the dev container must pass V1–V6. Outputs are then attacked manually: open them in multiple viewers, run `qpdf --qdf` and grep, extract text with pypdfium2/pdfminer, re-OCR with Tesseract. No planted value may be recoverable.
3. **From P3 onward:** the CLI produces manifests for the external evaluation project, which reports recall and precision. Thresholds and tier defaults change only through config versions backed by those reports.
4. **P6:** run the Compose stack offline (`--network none` for workers), process a 500-page synthetic record end to end, and confirm startup self-checks, canaries, quarantine paths and metrics with no PII in any label.

## Sources

- [OpenAI Privacy Filter model card (HF)](https://huggingface.co/openai/privacy-filter)
- [Tonic.ai benchmark of Privacy Filter](https://tonic.ai/blog/benchmarking-openai-privacy-filter-pii-detection)
- [Grepture: open PII models](https://grepture.com/blog/best-open-source-models-pii-redaction)
- [nvidia/gliner-PII](https://huggingface.co/nvidia/gliner-PII)
- [knowledgator/gliner-pii-base-v1.0](https://huggingface.co/knowledgator/gliner-pii-base-v1.0)
- [SHIELD (arXiv 2605.03301)](https://arxiv.org/abs/2605.03301)
- [MedDeID (arXiv 2609.10049)](https://arxiv.org/abs/2609.10049)
- [PP-OCRv6 release](https://x.com/PaddlePaddle/status/2065299834756902995)
- [PaddleOCR-VL-1.6](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6)
- [PaddleX layout detection](https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/layout_detection.html)
- [Docling layout models (arXiv 2509.11720)](https://arxiv.org/abs/2509.11720)
- [vLLM batch invariance](https://docs.vllm.ai/en/latest/features/batch_invariance/)
- [SGLang deterministic inference](https://lmsys.org/blog/2025-09-22-sglang-deterministic/)
- [ONNX Runtime CUDA EP](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
- [Story Beyond the Eye (PETS 2023)](https://petsymposium.org/popets/2023/popets-2023-0069.php)
- [pypdfium2](https://pypi.org/project/pypdfium2/)
- [OAIC de-identification guidance](https://www.oaic.gov.au/privacy/privacy-guidance-for-organisations-and-government-agencies/handling-personal-information/de-identification-and-the-privacy-act)
- [ACMA fictitious numbers](https://www.acma.gov.au/phone-numbers-use-tv-shows-films-and-creative-works)
- [G-NAF](https://data.gov.au/data/dataset/geocoded-national-address-file-g-naf)
- [Medicare check digit](https://rdrr.io/cran/starling/man/check_medicare.html)
- [Medicare provider number](https://developer.digitalhealth.gov.au/namespaces/id/medicare-provider-number/index.html)
- [DVA number format](https://hl7.org.au/fhir/StructureDefinition-au-dvanumber.html)
- [Surya licence](https://pypi.org/project/surya-ocr/0.5.0)
- [LLM de-identification survey (arXiv 2509.14464)](https://arxiv.org/abs/2509.14464)
