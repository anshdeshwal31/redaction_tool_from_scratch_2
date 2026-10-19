/**
 * Turns a manifest into what the document viewer draws: one list item per mention, region or
 * verifier addition, and one overlay box per distinct rectangle (a re-typeset line holds several
 * mentions in a single erased box, so boxes are shared).
 */
import type { Action, Evidence, Manifest, Rect } from "./manifest";
import { type CategoryId, categoryOf } from "./vocab";

export type ItemKind = "mention" | "region" | "residual";

export interface OverlayItem {
  id: string;
  kind: ItemKind;
  /** Entity type or region kind. */
  type: string;
  category: CategoryId;
  action: Action | string;
  render: string[];
  replacement?: string;
  evidence: Evidence[];
  kept: boolean;
  boxKeys: string[];
  page: number;
  /** Top-left of the first box, for reading-order sorting. */
  y: number;
  x: number;
}

export interface OverlayBox {
  key: string;
  page: number;
  rect: Rect;
  itemIds: string[];
  category: CategoryId;
  kept: boolean;
}

export interface OverlayModel {
  items: OverlayItem[];
  itemsById: Map<string, OverlayItem>;
  boxes: OverlayBox[];
  boxesByPage: Map<number, OverlayBox[]>;
}

const boxKey = (page: number, r: Rect) => `${page}:${r.map((v) => Math.round(v)).join(",")}`;

export function buildOverlay(manifest: Manifest): OverlayModel {
  const items: OverlayItem[] = [];
  const boxes = new Map<string, OverlayBox>();

  const addBox = (item: OverlayItem, page: number, rect: Rect) => {
    const key = boxKey(page, rect);
    let box = boxes.get(key);
    if (!box) {
      box = { key, page, rect, itemIds: [], category: item.category, kept: true };
      boxes.set(key, box);
    }
    box.itemIds.push(item.id);
    if (!item.kept) {
      // A box that replaces anything is drawn as replaced, in the colour of what it replaces.
      if (box.kept) box.category = item.category;
      box.kept = false;
    }
    item.boxKeys.push(key);
  };

  for (const m of manifest.mentions) {
    const first = m.boxes[0];
    const item: OverlayItem = {
      id: m.mention_id,
      kind: "mention",
      type: m.type,
      category: categoryOf(m.type),
      action: m.action,
      render: m.render,
      replacement: m.surrogate ?? m.label,
      evidence: m.evidence,
      kept: m.action === "keep",
      boxKeys: [],
      page: first?.page ?? 0,
      y: first?.px[1] ?? 0,
      x: first?.px[0] ?? 0,
    };
    for (const b of m.boxes) addBox(item, b.page, b.erased_px ?? b.px);
    items.push(item);
  }

  for (const r of manifest.regions ?? []) {
    const item: OverlayItem = {
      id: r.region_id,
      kind: "region",
      type: r.kind,
      category: "region",
      action: r.action,
      render: [],
      evidence: [{ detector: r.detector, version: "", rule: r.rule, strength: "strong", score: r.score }],
      kept: r.action === "keep",
      boxKeys: [],
      page: r.box.page,
      y: r.box.px[1],
      x: r.box.px[0],
    };
    addBox(item, r.box.page, r.box.erased_px ?? r.box.px);
    items.push(item);
  }

  manifest.residual_boxes.forEach((r, i) => {
    const item: OverlayItem = {
      id: `${r.ref_id}#${i}`,
      kind: "residual",
      type: r.type,
      category: "residual",
      action: "blackout",
      render: [],
      evidence: [],
      kept: false,
      boxKeys: [],
      page: r.page,
      y: r.px[1],
      x: r.px[0],
    };
    addBox(item, r.page, r.erased_px ?? r.px);
    items.push(item);
  });

  items.sort((a, b) => a.page - b.page || a.y - b.y || a.x - b.x || a.id.localeCompare(b.id));
  const boxList = [...boxes.values()];
  const boxesByPage = new Map<number, OverlayBox[]>();
  for (const box of boxList) {
    const list = boxesByPage.get(box.page) ?? [];
    list.push(box);
    boxesByPage.set(box.page, list);
  }
  // Larger boxes first, so smaller ones stay on top and remain hoverable.
  for (const list of boxesByPage.values()) list.sort((a, b) => area(b.rect) - area(a.rect));
  return { items, itemsById: new Map(items.map((i) => [i.id, i])), boxes: boxList, boxesByPage };
}

function area(r: Rect): number {
  return Math.max(0, r[2] - r[0]) * Math.max(0, r[3] - r[1]);
}
