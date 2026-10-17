/**
 * Manifests for tests, generated from the backend's stub engine (backend/tests/api/stub_engine.py):
 *
 *   python backend/tests/api/stub_engine.py run <input starting "%STUB <scenario>"> --out <dir> --style <style>
 *
 * The contract test validates every fixture against the backend's schema, so a stale fixture fails
 * loudly instead of drifting.
 */
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import type { Manifest, RenderStyle } from "@/lib/manifest";

export const FRONTEND_ROOT = fileURLToPath(new URL("../..", import.meta.url));
export const SCHEMA_PATH = path.resolve(FRONTEND_ROOT, "..", "backend", "schemas", "manifest.v1.json");

export type Scenario = "released" | "quarantined" | "intake";
export const SCENARIOS: Scenario[] = ["released", "quarantined", "intake"];

/** A fresh copy each call, so tests may mutate it. */
export function fixture(scenario: Scenario, style: RenderStyle = "surrogate"): Manifest {
  const file = path.join(FRONTEND_ROOT, "test", "fixtures", `manifest.${scenario}.${style}.json`);
  return JSON.parse(readFileSync(file, "utf8")) as Manifest;
}

export function schema(): Record<string, any> {
  return JSON.parse(readFileSync(SCHEMA_PATH, "utf8")) as Record<string, any>;
}
