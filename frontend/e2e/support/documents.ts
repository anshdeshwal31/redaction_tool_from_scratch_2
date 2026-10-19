import { readFileSync } from "node:fs";
import path from "node:path";

import { REAL, REAL_DOCS_DIR } from "./env";

/** A file for `setInputFiles`. */
export interface Doc {
  name: string;
  mimeType: string;
  buffer: Buffer;
}

// The stub engine reads its scenario from the document's first line (backend/tests/api/stub_engine.py).
const stub = (scenario: string): Doc => ({
  name: `e2e-${scenario}.pdf`,
  mimeType: "application/pdf",
  buffer: Buffer.from(`%STUB ${scenario}\n`),
});

// Written by global-setup.ts with e2e/support/make_documents.py.
const generated = (name: string): Doc => ({
  name,
  mimeType: "application/pdf",
  buffer: readFileSync(path.join(REAL_DOCS_DIR, name)),
});

/** The document each scenario submits, for the engine this run uses. */
export const documents = {
  /** Every check passes and a PDF is released. */
  released: () => (REAL ? generated("letter.pdf") : stub("released")),
  /** Held back: the stub fails V4; the real engine rejects the encrypted letter at intake. */
  quarantined: () => (REAL ? generated("encrypted.pdf") : stub("quarantined")),
  /** The engine exits with E_CONFIG_INVALID.  The real engine does so for a surrogate job with no key. */
  failing: () => (REAL ? generated("letter.pdf") : stub("error")),
  /** Still running when the test acts on it: the stub never finishes; a real job takes minutes. */
  longRunning: () => (REAL ? generated("letter.pdf") : stub("hang")),
};

export const QUARANTINE_REASON = REAL ? "E_INPUT_ENCRYPTED" : "V4_RESIDUAL";
