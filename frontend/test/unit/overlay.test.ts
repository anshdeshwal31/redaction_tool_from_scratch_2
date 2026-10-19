import { describe, expect, it } from "vitest";

import type { Manifest } from "@/lib/manifest";
import { buildOverlay } from "@/lib/overlay";

import { fixture } from "../support/fixtures";

function released(): Manifest {
  return fixture("released");
}

describe("buildOverlay", () => {
  it("makes one item per mention and region, in reading order", () => {
    const model = buildOverlay(released());
    expect(model.items.map((i) => [i.kind, i.type])).toEqual([
      ["region", "barcode"], // y 114
      ["mention", "street_address"], // y 296, same x as the postcode: ordered by id
      ["mention", "postcode"],
      ["mention", "person"], // y 500
      ["mention", "date"], // y 700
    ]);
  });

  it("shares one box between mentions on the same re-typeset line, at the erased rectangle", () => {
    const model = buildOverlay(released());
    expect(model.boxes).toHaveLength(4);
    const line = model.boxes.find((b) => b.itemIds.length === 2);
    expect(line?.rect).toEqual([227, 296, 1252, 367]);
    expect(line?.category).toBe("location");
    expect(model.itemsById.get("M-000000000000000000000003")?.boxKeys).toEqual([line?.key]);
  });

  it("marks kept mentions, and their boxes, as kept", () => {
    const model = buildOverlay(released());
    const date = model.items.find((i) => i.type === "date");
    expect(date?.kept).toBe(true);
    expect(model.boxes.find((b) => b.key === date?.boxKeys[0])?.kept).toBe(true);
  });

  it("draws a shared box as replaced, in the replaced mention's colour, even if a kept one came first", () => {
    const m = released();
    const date = m.mentions.find((x) => x.type === "date")!;
    const street = m.mentions.find((x) => x.type === "street_address")!;
    date.boxes = structuredClone(street.boxes);
    m.mentions = [date, ...m.mentions.filter((x) => x !== date)];
    const shared = buildOverlay(m).boxes.find((b) => b.itemIds.includes(date.mention_id));
    expect(shared?.kept).toBe(false);
    expect(shared?.category).toBe("location");
  });

  it("adds the verifier's residual boxes as their own category", () => {
    const residual = buildOverlay(fixture("quarantined")).items.filter((i) => i.kind === "residual");
    expect(residual).toHaveLength(1);
    expect(residual[0]).toMatchObject({ category: "residual", kept: false, action: "blackout" });
  });

  it("groups boxes by page with larger boxes first, so small ones stay on top", () => {
    const page = buildOverlay(released()).boxesByPage.get(0) ?? [];
    const areas = page.map(({ rect: [x0, y0, x1, y1] }) => (x1 - x0) * (y1 - y0));
    expect(areas).toEqual([...areas].sort((a, b) => b - a));
  });

  it("is empty for a job rejected at intake", () => {
    const model = buildOverlay(fixture("intake"));
    expect(model.items).toEqual([]);
    expect(model.boxes).toEqual([]);
  });
});
