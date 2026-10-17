/**
 * The UI's types and its test fixtures must both follow the engine's output contract,
 * backend/schemas/manifest.v1.json.  If the schema changes, these fail before the UI silently drifts.
 */
import { Ajv2020 } from "ajv/dist/2020.js";
import { describe, expect, it } from "vitest";

import { ACTIONS, CHECKS, ENTITY_TYPES, REGION_KINDS, RENDER_STYLES } from "@/lib/manifest";

import { fixture, SCENARIOS, schema as loadSchema, type Scenario } from "../support/fixtures";

const schema = loadSchema();
const validate = new Ajv2020({ allErrors: true, strict: false }).compile(schema);

describe("UI enumerations match the manifest schema", () => {
  it.each([
    ["entity types", ENTITY_TYPES, schema.$defs.entity_type.enum],
    ["region kinds", REGION_KINDS, schema.$defs.region_kind.enum],
    ["actions", ACTIONS, schema.$defs.action.enum],
    ["checks", CHECKS, schema.properties.verification.properties.checks.items.properties.check.enum],
    ["render styles", RENDER_STYLES, schema.properties.job.properties.render_style.enum],
  ])("%s", (_name, ours, theirs) => {
    expect([...ours]).toEqual(theirs);
  });
});

describe("test fixtures are schema-valid manifests", () => {
  const cases: [Scenario, (typeof RENDER_STYLES)[number]][] = [];
  for (const scenario of SCENARIOS) for (const style of RENDER_STYLES) cases.push([scenario, style]);

  it.each(cases)("%s, %s style", (scenario, style) => {
    const ok = validate(fixture(scenario, style));
    expect(validate.errors ?? []).toEqual([]);
    expect(ok).toBe(true);
  });
});
