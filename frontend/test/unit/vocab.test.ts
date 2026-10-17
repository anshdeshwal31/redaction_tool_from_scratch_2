import { describe, expect, it } from "vitest";

import { ACTIONS, CHECKS, ENTITY_TYPES, REGION_KINDS } from "@/lib/manifest";
import {
  ACTION_LABELS,
  CATEGORIES,
  CHECK_INFO,
  ENTITY_LABELS,
  REGION_LABELS,
  categoryColor,
  categoryOf,
  humanize,
  reasonText,
  typeLabel,
} from "@/lib/vocab";

import { schema as loadSchema } from "../support/fixtures";

const schema = loadSchema();
const UNKNOWN = reasonText("NOT_A_CODE");

describe("every code the engine can emit has a human label", () => {
  it.each(ENTITY_TYPES)("entity type %s", (t) => {
    expect(ENTITY_LABELS[t]).toBeTruthy();
    expect(CATEGORIES.map((c) => c.id)).toContain(categoryOf(t));
  });

  it.each(REGION_KINDS)("region kind %s", (k) => expect(REGION_LABELS[k]).toBeTruthy());
  it.each(ACTIONS)("action %s", (a) => expect(ACTION_LABELS[a]).toBeTruthy());
  it.each(CHECKS)("check %s", (c) => expect(CHECK_INFO[c].title).toBeTruthy());

  it.each(schema.$defs.reason.enum as string[])("reason %s", (code) => {
    expect(reasonText(code)).not.toBe(UNKNOWN);
  });
});

describe("categories", () => {
  it("use each categorical slot once, in order", () => {
    expect(CATEGORIES.map((c) => c.slot)).toEqual([1, 2, 3, 4, 5, 6, 7, 8]);
  });

  it("colour by fixed slot, with a neutral colour for verifier additions", () => {
    expect(categoryColor("person")).toBe("var(--series-1)");
    expect(categoryColor("region")).toBe("var(--series-8)");
    expect(categoryColor("residual")).toBe("var(--residual)");
  });
});

describe("labels", () => {
  it("prefer entity, then region names, then a humanised code", () => {
    expect(typeLabel("medicare")).toBe("Medicare number");
    expect(typeLabel("faint_ink")).toBe("Faint ink");
    expect(typeLabel("something_new")).toBe("Something new");
  });

  it("humanise timing keys without the unit suffix", () => {
    expect(humanize("analyse_ms")).toBe("Analyse");
    expect(humanize("pages_failed")).toBe("Pages failed");
  });
});
