// `npm run test:e2e:real`: runs Playwright with E2E_ENGINE=real on any shell (npm runs scripts with
// cmd.exe on Windows, where `VAR=value command` does not work).  Extra arguments go to Playwright.
import { spawnSync } from "node:child_process";
import { createRequire } from "node:module";

const [engine, ...args] = process.argv.slice(2);
const cli = createRequire(import.meta.url).resolve("@playwright/test/cli");
const result = spawnSync(process.execPath, [cli, "test", ...args], {
  stdio: "inherit",
  env: { ...process.env, E2E_ENGINE: engine },
});
process.exit(result.status ?? 1);
