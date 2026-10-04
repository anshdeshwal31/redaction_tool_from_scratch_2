# ADR 0004 — Non-text regions and ink accounting

- Status: accepted (heuristic classifier; a trained signature/handwriting detector replaces parts of it in P7)
- Date: 2026-10-04

## Context

A text pipeline only redacts what OCR reads.  On scanned medico-legal documents the PII that OCR
does not read is exactly the dangerous part: signatures, handwritten notes, received-stamps with
dates and names, QR/DataMatrix codes encoding claimant details, ID photos, and show-through of the
reverse page.  Plan §6 requires that every patch of ink is explained; plan §7.1 lists these visual
categories as REMOVE.

Probing on synthetic scans showed two failure modes of naive ink accounting:

* OCR "reads" a signature as a tall, low-confidence garbage word ("soooor", weakest character 0.42),
  and that token then *explains* the signature's ink;
* OCR "reads" mirrored show-through as confident garbage (a 585-px-wide "T"; "888" mirrors to
  "888" at 0.95), hiding it.

## Decision

* **Detectors**: zxing-cpp (all symbologies, `return_errors=True` so located-but-undecodable codes
  are removed too) and OpenCV Zoo YuNet (MIT, 233 KB, score threshold 0.6 for recall, run on a copy
  with long side ≤ 1600 px, boxes padded 35%).
* **Ink accounting** (`layout/regions.py`): ink = darkness against a *local* background ≥ 60;
  ruling lines (straight runs ≥ ~1 cm) are subtracted; only *plausible* tokens explain ink
  (weakest character ≥ 0.5, height ≤ 1.8 lines, width ≤ 1.5 line-heights per character).
* **Classification** of what remains, in order: coloured ink not confidently read → `stamp`;
  large mid-tone area → `photo`; inside a signature zone (below a sign-off, around "Signed" /
  "Signature") → `signature`; text-like → re-read (2× upscaled crop through the same OCR engine:
  accepted as tokens only if every line ≥ 0.85 confidence and the tokens explain ≥ 80% of the ink,
  else `handwriting`); header/footer band → `logo`; outside the text column → `handwriting`;
  large framed marks → `graphic`; anything else → `handwriting`.
* **Faint ink** (darkness 18–60 after resolution-scaled smoothing): suppressed to paper colour unless
  explained by tokens printed in real ink (or read at ≥ 0.95).  Tokens lying on faint-only ink
  inside a faint region are *ghosts* and are dropped before detection, so no surrogate is typeset
  into the blanked area.
* Fragments of one mark are merged (signature gap 2 lines, show-through 2, handwriting 1) so no
  stroke survives between two boxes.
* **`graphic` is removed by default in every profile**: a figure cannot be told apart from
  handwriting in a form cell reliably, and handwriting in form cells is the commonest handwriting.
* **V4** also re-runs barcode and face detection on the *output*; any symbol or face outside the
  planned removal boxes is a residual (re-plan, then quarantine).
* SUPPRESS_INK is a raster op with its own pixel check (every pixel is paper or strong ink).

## Consequences

* On the synthetic artefact letter every artefact is covered by exactly one region; clean, degraded
  and born-digital letters produce no regions (no false positives).
* Utility cost: figures, body charts and unreadable printed text are removed (grey boxes).  The
  manifest records the kind and rule of each region for the evaluator.
* Region analysis costs ~1 s per page at 300 dpi plus any re-reads.
* A trained signature/handwriting detector (plan P7) should replace the zone and text-likeness
  heuristics; the interfaces (`Region`, region actions, V4 probe) stay.
