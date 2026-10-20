import { readFileSync } from "node:fs";
import path from "node:path";

import { REAL, REAL_DOCS_DIR } from "./env";

/** A file for `setInputFiles`. */
export interface Doc {
  name: string;
  mimeType: string;
  buffer: Buffer;
}

// The stub engine reads its scenario from a line near the document's start (backend/tests/api/stub_engine.py).
const stub = (scenario: string): Doc => ({
  name: `e2e-${scenario}.pdf`,
  mimeType: "application/pdf",
  buffer: Buffer.from(`%STUB ${scenario}\n`),
});

/** A one-page PDF that pdf.js can draw, carrying the stub's scenario line right after its header. */
function stubPdf(scenario: string): Doc {
  const objects = [
    "<< /Type /Catalog /Pages 2 0 R >>",
    "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
    "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595.276 841.89] >>",
  ];
  let body = `%PDF-1.7\n%STUB ${scenario}\n`;
  const offsets: number[] = [];
  objects.forEach((obj, i) => {
    offsets.push(body.length);
    body += `${i + 1} 0 obj\n${obj}\nendobj\n`;
  });
  const xref = body.length;
  body += `xref\n0 ${objects.length + 1}\n0000000000 65535 f \n`;
  body += offsets.map((at) => `${String(at).padStart(10, "0")} 00000 n \n`).join("");
  body += `trailer\n<< /Size ${objects.length + 1} /Root 1 0 R >>\nstartxref\n${xref}\n%%EOF\n`;
  return { name: `e2e-${scenario}-original.pdf`, mimeType: "application/pdf", buffer: Buffer.from(body, "latin1") };
}

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
  /** Released, and a real PDF, so a kept original can be drawn beside the redacted pages. */
  comparable: () => (REAL ? generated("letter.pdf") : stubPdf("released")),
  /** Held back: the stub fails V4; the real engine rejects the encrypted letter at intake. */
  quarantined: () => (REAL ? generated("encrypted.pdf") : stub("quarantined")),
  /** The engine exits with E_CONFIG_INVALID.  The real engine does so for a surrogate job with no key. */
  failing: () => (REAL ? generated("letter.pdf") : stub("error")),
  /** Still running when the test acts on it: the stub never finishes; a real job takes minutes. */
  longRunning: () => (REAL ? generated("letter.pdf") : stub("hang")),
};

export const QUARANTINE_REASON = REAL ? "E_INPUT_ENCRYPTED" : "V4_RESIDUAL";
