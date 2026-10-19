import { execFileSync } from "node:child_process";
import path from "node:path";

import { backendPython, FRONTEND_DIR, REAL, REAL_DOCS_DIR } from "./support/env";

/** Real-engine runs submit synthetic documents; generate them with the backend's synthdocs tool. */
export default function globalSetup() {
  if (!REAL) return;
  execFileSync(backendPython(), [path.join(FRONTEND_DIR, "e2e/support/make_documents.py"), REAL_DOCS_DIR], {
    stdio: "inherit",
  });
}
