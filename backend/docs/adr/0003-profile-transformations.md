# ADR 0003 — Maximal date shifting, age top-coding and claimant-focused roles

- Status: accepted
- Date: 2026-10-04

## Context

Plan §7.1 defines two profiles beyond Broad:

* **Maximal** (dataset release): every date shifted by a per-document offset that preserves
  intervals, and ages of 90 and over generalised.
* **Claimant-focused**: professionals (treating clinicians, lawyers, case managers) stay visible,
  and everyone else is replaced.

In both profiles a mistake can either leak or merely cost utility, and the requirements put false
negatives first.

## Decision

**Date shifting (`DATE_SHIFT`)**

* One offset of ±30–365 days per surrogate scope.  It is drawn from the scope-keyed HMAC-DRBG, so
  it is consistent across a matter's documents when a scope id is supplied.  The offset is never
  recorded in the manifest.
* Every date keeps its layout and surrounding punctuation:
  * numeric dates keep their separators, zero-padding and 2- or 4-digit year;
  * ISO dates stay ISO;
  * textual dates keep ordinals and the case and abbreviation of the month;
  * month-year dates move by whole months.
* Bare years are kept: an offset under a year cannot move them, and two different bare years would
  make every offset collide.
* Dates that cannot be parsed become `[date]`.
* **Collisions.**  The offset is redrawn deterministically until no shifted date *contains* a
  removed value, using exactly the needles V2 searches for.  Otherwise a shifted "18 March 2023"
  next to an original "March 2023" would look like a leak and quarantine the document.  If no
  offset works, every date becomes `[date]` (fail safe).
* **V5 checks**, using a different date parser from the shifter's:
  * no shifted date equals an original value;
  * all shifted dates share one non-zero offset;
  * a parseable original never becomes unexplained text.

**Age top-coding (`GENERALISE` of `age`)**

* A rule detector (`rules.ages`) matches only explicit phrasing: "aged 92", "92-year-old",
  "92 years of age", "92yo".  Durations ("for 5 years") are not ages.
* Ages of 90 and over become "90+".  V5 checks that no age of 90 or over remains.
* A transformation that would not change the text (an age under 90, a bare year) is settled to
  `KEEP` at the policy stage, in every render style.  `GENERALISE` of any other type is never
  settled to `KEEP`; it falls back to blackout.

**Claimant-focused roles** (`resolve/roles.py`)

The rule is one-sided.  A person cluster stays visible only if all three hold:

* at least one mention has explicit professional evidence: a Dr/Prof title, an adjacent
  post-nominal, or a role word *adjacent* to the name.  A role word elsewhere on the line does not
  count ("Mr Smith was examined by his treating surgeon");
* no mention has any subject cue on its line ("Re:", patient/claimant/worker labels, kinship
  words, DOB, address);
* its surname is not shared with any non-professional cluster, because family members share
  surnames.

Unclustered mentions are always replaced.

**V4 and deliberate keeps**

* Re-detection on the output skips tokens inside regions the policy deliberately kept, even though
  their type is normally acted on (professionals, settled ages and bare years).
* The removed-value probe is never exempted.

## Consequences

* Maximal outputs keep clinical timelines (verified end to end: 41- and 454-day intervals survive)
  while no original date remains.
* Claimant-focused trades some utility for safety: a professional named without explicit evidence
  (for example "Ms Jane Citizen" in an address block) is still replaced.
* Role inference is rule-based.  The evaluator should report both missed professionals (utility)
  and wrongly kept subjects (a privacy failure, to be treated as a blocker).
