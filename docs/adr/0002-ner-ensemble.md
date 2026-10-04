# ADR 0002 — NER ensemble: models, precision, windowing, decoding and contextual allowlist

- Status: accepted (thresholds and biases to be tuned from the external evaluator's reports)
- Date: 2026-10-04

## Context

Plan §7.2 adds two statistical detectors to the rules: N1 OpenAI Privacy Filter and N2
GLiNER-PII.  Both run on ONNX Runtime without PyTorch.  Measurements below are from the development
profile (`windows-x86_64-v3`, Ryzen 5 5600H).

**Model variants.**

| model | variant | size | notes |
|---|---|---|---|
| GLiNER-PII base (deberta-v3-small, span `markerV0`, max width 12) | fp32 | 665 MB | **default N2** |
| GLiNER-PII large (deberta-v3-large, token-level) | fp32 | 1.76 GB | registered, optional |
| Privacy Filter (8 layers, 128 experts / top-4, banded attention) | q4 (int4 weight-only `MatMulNBits` + `QMoE`, fp32 activations) | 917 MB | **default N1** |

Published `quint8` / `quantized` variants use *dynamic* activation quantisation: the scale is
computed over the whole input, so a token's scores would depend on the rest of its window.  Plan §13
excludes that, so only fp32 and weight-only quantised graphs are registered.

**Speed.** GLiNER base costs ~2.3 s per 300-word window on 1 thread, mostly in the span head
(12 widths × an MLP per candidate span), and 0.8 s on 4 threads.  The Privacy Filter costs ~1.3 s
per call at 4 threads plus roughly quadratic attention cost: 2.9 s at 512 tokens, 4.8 s at 1,024,
12 s at 3,072, 22 s at 4,096.  The first call after loading takes ~20 s.  Both sessions
together hold ~2.5 GB.

**Thread invariance.** GLiNER logits are bit-identical for 1, 2, 4 and 6 intra-op threads.
Privacy Filter logits are *not* bit-identical between 1 and 4 threads.

**Receptive field.** The Privacy Filter's attention is banded (|i−j| ≤ 128) in all 8 layers.  So
a token's logits can depend only on tokens within 1,024 positions.  Checked empirically: logits
from a window with ≥ 768 tokens of left context match whole-sequence logits to ~2e-5, which is
float-reduction noise.  With 512 tokens of context they differ by up to 2.3.

## Decision

* **N2 = GLiNER-PII base fp32**, word windows of 300 with 50 overlap, batch size 1.  Overlapping
  windows are unioned.
  * Runs on the line view and the title-cased view.  A window whose text is unchanged between views
    gives an identical result, so it is skipped; for ordinary prose this halves the cost.
  * The prompt is built in sorted label order, so the configuration hash fixes it.
  * There is no generic `date` label: flat decoding keeps one label per span, so `date` could only
    take spans away from `dob`.
  * Threshold 0.3 (model card).  Hits are strong at ≥ 0.8.  Per-type overrides set organisations
    to 0.5 and localities to 0.3: no other detector produces those types, and fusion needs strong
    evidence for them.
* **N1 = Privacy Filter q4** with our own decoder:
  * exact windowing — 3,072-token windows, context margin = layers × band = 1,024 (derived from the
    model config);
  * each window contributes only its core, and the stitched logits feed one constrained BIOES
    Viterbi pass over the whole document;
  * the six transition biases are configuration, defaulting to the model's calibrated zero point;
  * a posterior sweep adds weak candidates where P(not background) ≥ 0.25 but the path chose
    background;
  * `private_date` maps to `date`, which the Broad policy keeps; DOBs come from cue rules and
    GLiNER's `dob`.
* NER runs **in the parent process** while the page workers are idle.  It has its own pinned
  thread count (`detection.ner.threads`, 4).  That value is part of the configuration hash,
  because Privacy Filter output depends on it.
* **Contextual allowlist and refinement** (`detect/allowlist.py`) apply to statistical evidence
  only, never to rule, label, seed or propagation evidence.  This matters because propagation
  amplifies any accepted false positive across the document.  It handles:
  * eponym patterns ("Tinel's sign", "Tinel and Phalen tests");
  * medicine names (curated list in `resources/allowlists/drugs.txt`, interim until the PBS
    extract);
  * public bodies, capital cities and states, and non-name words;
  * person spans, which drop leading titles and trailing post-nominals (kept visible, plan §10);
  * implausible types: an identifier type with no digit is re-typed as an organisation if it reads
    like a proper name, otherwise dropped.
* V4 (re-OCR residual check) stays rules-only: it searches for the removed values themselves.

## Consequences

* Both detectors find PII that the rules miss: lone given names ("His wife Mary attended"), suburbs
  ("lives in Blacktown"), facility names ("Westmead Hospital") and titled partial names.
* CPU cost is dominated by the Privacy Filter at ~11.5 s per 1,024 tokens of long documents.
  Short documents (< 3,072 tokens) need one pass.  `context_tokens` can be lowered (768 is at the
  noise floor in the measurement above) to trade strict exactness for ~1.5× throughput.
* Loading both models takes ~35 s and ~2.5 GB in the service process.  It happens once per
  `Redactor`.
* Open items for the evaluator loop: the Viterbi operating point, the GLiNER thresholds, whether
  the large GLiNER or a fine-tuned N3 (plan P7) replaces base, and replacing the drug list with the
  PBS schedule.
