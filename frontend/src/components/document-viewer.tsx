"use client";

import { Eye, EyeOff, FileWarning, Minus, Plus, RotateCcw } from "lucide-react";
import type { PDFDocumentProxy, RenderTask } from "pdfjs-dist";
import { memo, useEffect, useMemo, useRef, useState } from "react";

import type { Manifest, PageInfo } from "@/lib/manifest";
import { buildOverlay, type OverlayBox, type OverlayItem, type OverlayModel } from "@/lib/overlay";
import { ACTION_LABELS, CATEGORIES, type CategoryId, categoryColor, categoryLabel, typeLabel } from "@/lib/vocab";

import { cx, Notice, Spinner, Swatch } from "./ui";

/* ------------------------------------------------------------------------------------ pdf.js */

type PdfState = { doc: PDFDocumentProxy | null; error: boolean };

/** Either a URL, or a function returning a fresh one (signed links expire). */
export type PdfSource = string | (() => Promise<string>);

function usePdf(source: PdfSource | null): PdfState {
  const [state, setState] = useState<PdfState>({ doc: null, error: false });
  useEffect(() => {
    if (!source) return;
    let cancelled = false;
    let destroy: (() => Promise<void>) | null = null;
    (async () => {
      try {
        const pdfjs = await import("pdfjs-dist");
        pdfjs.GlobalWorkerOptions.workerSrc = "/pdf.worker.min.mjs";
        const url = typeof source === "function" ? await source() : source;
        const res = await fetch(url, { cache: "no-store" });
        if (!res.ok) throw new Error("fetch");
        const data = new Uint8Array(await res.arrayBuffer());
        // Unmounted while downloading: the cleanup has already run, so never start pdf.js for it.
        if (cancelled) return;
        const task = pdfjs.getDocument({ data, enableXfa: false });
        destroy = () => task.destroy();
        const doc = await task.promise;
        if (!cancelled) setState({ doc, error: false });
      } catch {
        if (!cancelled) setState({ doc: null, error: true });
      }
    })();
    return () => {
      cancelled = true;
      void destroy?.();
    };
  }, [source]);
  return state;
}

function useWidth(ref: React.RefObject<HTMLElement | null>): number {
  const [width, setWidth] = useState(0);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => {
      if (entry) setWidth(Math.floor(entry.contentRect.width));
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [ref]);
  return width;
}

/* ------------------------------------------------------------------------------------ page */

const MAX_CANVAS_PX = 2600;

interface PageProps {
  info: PageInfo;
  pageCount: number;
  doc: PDFDocumentProxy | null;
  root: HTMLElement | null;
  cssWidth: number;
  boxes: OverlayBox[];
  items: Map<string, OverlayItem>;
  isVisibleItem: (item: OverlayItem) => boolean;
  selectedId: string | null;
  flashKey: string | null;
  hoverKey: string | null;
  onHover: (key: string | null) => void;
  onPick: (box: OverlayBox) => void;
  showOverlay: boolean;
}

const Page = memo(function Page(props: PageProps) {
  const { info, doc, root, cssWidth, boxes, items, isVisibleItem, showOverlay } = props;
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [near, setNear] = useState(false);
  const [rendered, setRendered] = useState(false);

  // Only pages near the viewport hold a canvas: long records would otherwise exhaust memory.
  useEffect(() => {
    const el = wrapRef.current;
    if (!el || !root) return;
    const io = new IntersectionObserver(([e]) => setNear(Boolean(e?.isIntersecting)), {
      root,
      rootMargin: "1500px 0px",
    });
    io.observe(el);
    return () => io.disconnect();
  }, [root]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!doc || !canvas || !near || cssWidth < 50) {
      if (canvas && !near) {
        canvas.width = 0;
        canvas.height = 0;
        setRendered(false);
      }
      return;
    }
    let task: RenderTask | null = null;
    let cancelled = false;
    const handle = setTimeout(async () => {
      try {
        const page = await doc.getPage(info.index + 1);
        if (cancelled) return;
        const base = page.getViewport({ scale: 1 });
        const target = Math.min(MAX_CANVAS_PX, cssWidth * Math.min(window.devicePixelRatio || 1, 2));
        const viewport = page.getViewport({ scale: target / base.width });
        const off = document.createElement("canvas");
        off.width = Math.floor(viewport.width);
        off.height = Math.floor(viewport.height);
        task = page.render({ canvas: off, viewport });
        await task.promise;
        if (cancelled) return;
        // Swap in the finished bitmap at once, so a re-render never blanks the page.
        canvas.width = off.width;
        canvas.height = off.height;
        canvas.getContext("2d")?.drawImage(off, 0, 0);
        setRendered(true);
      } catch {
        /* cancelled or failed: the page keeps its placeholder */
      }
    }, 60);
    return () => {
      cancelled = true;
      clearTimeout(handle);
      task?.cancel();
    };
  }, [doc, near, cssWidth, info.index]);

  const visibleBoxes = showOverlay
    ? boxes.filter((b) => b.itemIds.some((id) => {
        const item = items.get(id);
        return item ? isVisibleItem(item) : false;
      }))
    : [];
  const hovered = visibleBoxes.find((b) => b.key === props.hoverKey) ?? null;

  return (
    <div className="mx-auto" style={{ width: cssWidth || "100%" }}>
      <div className="mb-1.5 flex items-center justify-between text-xs text-muted">
        <span className="font-medium">
          Page {info.index + 1} of {props.pageCount}
        </span>
        <span>{info.kind.replace("_", " ")}</span>
      </div>
      <div
        ref={wrapRef}
        id={`page-${info.index}`}
        className="relative overflow-visible bg-paper shadow-[0_1px_3px_rgba(0,0,0,0.18)] ring-1 ring-line"
        style={{ aspectRatio: `${info.width_px} / ${info.height_px}` }}
      >
        {doc ? (
          <canvas ref={canvasRef} className="absolute inset-0 size-full" aria-label={`Redacted page ${info.index + 1}`} />
        ) : null}
        {doc && !rendered ? (
          <div className="absolute inset-0 flex items-center justify-center text-muted">
            <Spinner />
          </div>
        ) : null}
        {visibleBoxes.map((box) => (
          <BoxMark
            key={box.key}
            box={box}
            info={info}
            count={box.itemIds.filter((id) => {
              const item = items.get(id);
              return item ? isVisibleItem(item) : false;
            }).length}
            active={box.key === props.hoverKey || (props.selectedId !== null && box.itemIds.includes(props.selectedId))}
            flash={box.key === props.flashKey}
            onHover={props.onHover}
            onPick={props.onPick}
          />
        ))}
        {hovered ? <BoxTooltip box={hovered} info={info} items={items} isVisibleItem={isVisibleItem} /> : null}
      </div>
    </div>
  );
});

function pct(v: number, total: number) {
  return `${(v / total) * 100}%`;
}

function BoxMark({
  box,
  info,
  count,
  active,
  flash,
  onHover,
  onPick,
}: {
  box: OverlayBox;
  info: PageInfo;
  count: number;
  active: boolean;
  flash: boolean;
  onHover: (key: string | null) => void;
  onPick: (box: OverlayBox) => void;
}) {
  const [x0, y0, x1, y1] = box.rect;
  const color = categoryColor(box.category);
  return (
    <div
      data-box={box.key}
      onPointerEnter={() => onHover(box.key)}
      onPointerLeave={() => onHover(null)}
      onClick={() => onPick(box)}
      className={cx("absolute cursor-pointer transition-[background-color,box-shadow]", flash && "flash")}
      style={{
        left: pct(x0, info.width_px),
        top: pct(y0, info.height_px),
        width: pct(x1 - x0, info.width_px),
        height: pct(y1 - y0, info.height_px),
        border: `1.5px ${box.kept ? "dashed" : "solid"} ${color}`,
        background: box.kept
          ? active
            ? `color-mix(in srgb, ${color} 14%, transparent)`
            : "transparent"
          : `color-mix(in srgb, ${color} ${active ? 38 : 16}%, transparent)`,
        boxShadow: active ? `0 0 0 2px var(--surface), 0 0 0 4px ${color}` : undefined,
        zIndex: active ? 5 : 1,
      }}
    >
      {count > 1 ? (
        <span
          className="absolute -top-2 -right-2 flex h-4 min-w-4 items-center justify-center bg-ink px-1 text-[0.625rem] font-semibold text-page"
          aria-hidden
        >
          {count}
        </span>
      ) : null}
    </div>
  );
}

function BoxTooltip({
  box,
  info,
  items,
  isVisibleItem,
}: {
  box: OverlayBox;
  info: PageInfo;
  items: Map<string, OverlayItem>;
  isVisibleItem: (item: OverlayItem) => boolean;
}) {
  const [x0, y0, x1, y1] = box.rect;
  const below = y1 / info.height_px < 0.72;
  const alignRight = x0 / info.width_px > 0.55;
  const entries = box.itemIds.map((id) => items.get(id)).filter((i): i is OverlayItem => Boolean(i && isVisibleItem(i)));
  return (
    <div
      role="tooltip"
      className="pointer-events-none absolute z-20 w-72 max-w-[90%] border border-line bg-surface p-3 text-xs shadow-lg"
      style={{
        [below ? "top" : "bottom"]: below
          ? `calc(${pct(y1, info.height_px)} + 8px)`
          : `calc(${pct(info.height_px - y0, info.height_px)} + 8px)`,
        [alignRight ? "right" : "left"]: alignRight ? pct(info.width_px - x1, info.width_px) : pct(x0, info.width_px),
      }}
    >
      <ul className="space-y-2">
        {entries.slice(0, 6).map((item) => (
          <li key={item.id}>
            <div className="flex items-center gap-1.5">
              <Swatch color={categoryColor(item.category)} dashed={item.kept} />
              <span className="font-medium text-ink">{typeLabel(item.type)}</span>
              <span className="ml-auto text-muted">{ACTION_LABELS[item.action as keyof typeof ACTION_LABELS] ?? item.action}</span>
            </div>
            {item.replacement ? (
              <div className="mt-0.5 truncate pl-4 font-mono text-[0.6875rem] text-ink-2">→ {item.replacement}</div>
            ) : null}
          </li>
        ))}
      </ul>
      {entries.length > 6 ? <div className="mt-2 text-muted">+{entries.length - 6} more in the list</div> : null}
    </div>
  );
}

/* ------------------------------------------------------------------------------------ list */

function ItemRow({
  item,
  active,
  onSelect,
  onHover,
}: {
  item: OverlayItem;
  active: boolean;
  onSelect: (item: OverlayItem) => void;
  onHover: (item: OverlayItem | null) => void;
}) {
  const detectors = [...new Set(item.evidence.map((e) => e.detector))];
  const strong = item.evidence.some((e) => e.strength === "strong");
  const actionText = item.kept
    ? "Kept"
    : item.render.length && item.render[0] !== item.action
      ? `${ACTION_LABELS[item.action as keyof typeof ACTION_LABELS] ?? item.action} · ${item.render
          .map((r) => ACTION_LABELS[r as keyof typeof ACTION_LABELS] ?? r)
          .join(", ")}`
      : (ACTION_LABELS[item.action as keyof typeof ACTION_LABELS] ?? item.action);
  return (
    <li>
      <button
        type="button"
        data-item={item.id}
        onClick={() => onSelect(item)}
        onPointerEnter={() => onHover(item)}
        onPointerLeave={() => onHover(null)}
        onFocus={() => onHover(item)}
        onBlur={() => onHover(null)}
        className={cx(
          "w-full px-2.5 py-2 text-left transition-colors",
          active ? "bg-accent-soft" : "hover:bg-surface-2",
        )}
      >
        <div className="flex items-center gap-2">
          <Swatch color={categoryColor(item.category)} dashed={item.kept} />
          <span className="min-w-0 truncate text-sm font-medium text-ink">
            {item.kind === "residual" ? `Verifier: ${typeLabel(item.type)}` : typeLabel(item.type)}
          </span>
          <span className={cx("ml-auto shrink-0 text-xs", item.kept ? "text-muted" : "text-ink-2")}>{actionText}</span>
        </div>
        {item.replacement ? (
          <div className="mt-0.5 truncate pl-[1.125rem] font-mono text-xs text-ink-2" title={item.replacement}>
            → {item.replacement}
          </div>
        ) : null}
        {detectors.length ? (
          <div className="mt-0.5 truncate pl-[1.125rem] text-[0.6875rem] text-muted">
            {detectors.join(" · ")}
            {item.kind === "mention" ? (strong ? " · strong" : " · weak") : ""}
          </div>
        ) : null}
      </button>
    </li>
  );
}

/* ------------------------------------------------------------------------------------ viewer */

const ZOOMS = [1, 1.25, 1.5, 2];

export function DocumentViewer({ manifest, pdfUrl }: { manifest: Manifest; pdfUrl: PdfSource | null }) {
  const model: OverlayModel = useMemo(() => buildOverlay(manifest), [manifest]);
  const pdf = usePdf(pdfUrl);
  const scrollRef = useRef<HTMLDivElement>(null);
  const listRef = useRef<HTMLDivElement>(null);
  const [scrollEl, setScrollEl] = useState<HTMLDivElement | null>(null);
  const width = useWidth(scrollRef);

  const present = useMemo(() => {
    const counts = new Map<CategoryId, { replaced: number; kept: number }>();
    for (const item of model.items) {
      const c = counts.get(item.category) ?? { replaced: 0, kept: 0 };
      if (item.kept) c.kept++;
      else c.replaced++;
      counts.set(item.category, c);
    }
    return counts;
  }, [model]);
  const allCategories: CategoryId[] = [...CATEGORIES.map((c) => c.id), "residual"];
  const shown = allCategories.filter((c) => present.has(c));
  const keptTotal = model.items.filter((i) => i.kept).length;

  const [hidden, setHidden] = useState<Set<CategoryId>>(new Set());
  const [showKept, setShowKept] = useState(false);
  const [showOverlay, setShowOverlay] = useState(true);
  const [zoom, setZoom] = useState(1);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [hoverKey, setHoverKey] = useState<string | null>(null);
  const [hoverItem, setHoverItem] = useState<string | null>(null);
  const [flash, setFlash] = useState<{ key: string; n: number } | null>(null);

  const isVisibleItem = useMemo(
    () => (item: OverlayItem) => !hidden.has(item.category) && (showKept || !item.kept),
    [hidden, showKept],
  );
  const visibleItems = model.items.filter(isVisibleItem);
  const byPage = new Map<number, OverlayItem[]>();
  for (const item of visibleItems) byPage.set(item.page, [...(byPage.get(item.page) ?? []), item]);

  // The box under the pointer highlights its list rows; a hovered row highlights its boxes.
  const hoverBoxItems = useMemo(
    () => new Set(hoverKey ? model.boxes.find((b) => b.key === hoverKey)?.itemIds ?? [] : []),
    [hoverKey, model],
  );
  const effectiveHoverKey = hoverKey ?? (hoverItem ? model.itemsById.get(hoverItem)?.boxKeys[0] ?? null : null);

  const select = (item: OverlayItem) => {
    setSelectedId(item.id);
    const key = item.boxKeys[0];
    if (!key) return;
    setShowOverlay(true);
    requestAnimationFrame(() => {
      const el = scrollRef.current?.querySelector(`[data-box="${CSS.escape(key)}"]`);
      el?.scrollIntoView({ behavior: "smooth", block: "center", inline: "center" });
      setFlash((f) => ({ key, n: (f?.n ?? 0) + 1 }));
    });
  };

  const pick = (box: OverlayBox) => {
    const id = box.itemIds.find((i) => {
      const item = model.itemsById.get(i);
      return item ? isVisibleItem(item) : false;
    });
    if (!id) return;
    setSelectedId(id);
    listRef.current
      ?.querySelector(`[data-item="${CSS.escape(id)}"]`)
      ?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  };

  // `width` is the scroller's content box (padding and scrollbar already excluded).
  const pageWidth = Math.max(0, Math.floor(width * zoom));

  return (
    <div className="flex flex-col gap-3">
      {/* Filters: one row above the content they scope. */}
      <div className="flex flex-wrap items-center gap-2">
        {shown.map((c) => {
          const on = !hidden.has(c);
          const n = present.get(c);
          return (
            <button
              key={c}
              type="button"
              aria-pressed={on}
              onClick={() =>
                setHidden((h) => {
                  const next = new Set(h);
                  if (next.has(c)) next.delete(c);
                  else next.add(c);
                  return next;
                })
              }
              className={cx(
                "inline-flex h-7 items-center gap-1.5 border px-2.5 text-xs font-medium transition-colors",
                on ? "border-line-strong bg-surface text-ink" : "border-line bg-transparent text-muted line-through",
              )}
            >
              <Swatch color={categoryColor(c)} className={on ? "" : "opacity-40"} />
              {categoryLabel(c)}
              <span className="tabular text-muted">{(n?.replaced ?? 0) + (showKept ? (n?.kept ?? 0) : 0)}</span>
            </button>
          );
        })}
        {keptTotal ? (
          <label className="ml-1 inline-flex h-7 cursor-pointer items-center gap-1.5 text-xs text-ink-2">
            <input
              type="checkbox"
              checked={showKept}
              onChange={(e) => setShowKept(e.target.checked)}
              className="check size-3.5"
            />
            Show kept ({keptTotal})
          </label>
        ) : null}
        <div className="ml-auto flex items-center gap-1">
          <button
            type="button"
            onClick={() => setShowOverlay((v) => !v)}
            aria-pressed={showOverlay}
            className="inline-flex h-7 items-center gap-1.5 px-2 text-xs font-medium text-ink-2 hover:bg-surface-2 hover:text-ink"
          >
            {showOverlay ? <Eye className="size-3.5" aria-hidden /> : <EyeOff className="size-3.5" aria-hidden />}
            Overlay
          </button>
          <div className="flex items-center border border-line" role="group" aria-label="Zoom">
            <button
              type="button"
              className="inline-flex size-7 items-center justify-center text-ink-2 hover:bg-surface-2 disabled:opacity-40"
              onClick={() => setZoom((z) => ZOOMS[Math.max(0, ZOOMS.indexOf(z) - 1)] ?? 1)}
              disabled={zoom === ZOOMS[0]}
              aria-label="Zoom out"
            >
              <Minus className="size-3.5" aria-hidden />
            </button>
            <span className="tabular w-11 text-center text-xs text-ink-2">{Math.round(zoom * 100)}%</span>
            <button
              type="button"
              className="inline-flex size-7 items-center justify-center text-ink-2 hover:bg-surface-2 disabled:opacity-40"
              onClick={() => setZoom((z) => ZOOMS[Math.min(ZOOMS.length - 1, ZOOMS.indexOf(z) + 1)] ?? z)}
              disabled={zoom === ZOOMS[ZOOMS.length - 1]}
              aria-label="Zoom in"
            >
              <Plus className="size-3.5" aria-hidden />
            </button>
            {zoom !== 1 ? (
              <button
                type="button"
                className="inline-flex size-7 items-center justify-center border-l border-line text-ink-2 hover:bg-surface-2"
                onClick={() => setZoom(1)}
                aria-label="Fit width"
              >
                <RotateCcw className="size-3.5" aria-hidden />
              </button>
            ) : null}
          </div>
        </div>
      </div>

      {!pdfUrl ? (
        <Notice tone="warn" icon={<FileWarning className="size-4" />} title="No PDF was released for this job">
          The pages below are outlines. Boxes show where the engine planned its changes, from the manifest.
        </Notice>
      ) : pdf.error ? (
        <Notice tone="crit" icon={<FileWarning className="size-4" />} title="The redacted PDF could not be displayed">
          Download it to view it in another reader. The overlay is still shown on page outlines.
        </Notice>
      ) : null}

      <div className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_22rem]">
        <div
          ref={(el) => {
            scrollRef.current = el;
            setScrollEl(el);
          }}
          className="h-[78vh] min-h-[28rem] overflow-auto border border-line bg-surface-3/60 p-4"
        >
          <div className="flex flex-col gap-6" style={{ width: zoom > 1 ? pageWidth : undefined }}>
            {manifest.pages.length === 0 ? (
              <div className="py-24 text-center text-sm text-muted">The job ended before any page was analysed.</div>
            ) : (
              manifest.pages.map((info) => (
                <Page
                  key={info.index}
                  info={info}
                  pageCount={manifest.pages.length}
                  doc={pdf.error ? null : pdf.doc}
                  root={scrollEl}
                  cssWidth={pageWidth}
                  boxes={model.boxesByPage.get(info.index) ?? []}
                  items={model.itemsById}
                  isVisibleItem={isVisibleItem}
                  selectedId={selectedId}
                  flashKey={flash?.key ?? null}
                  hoverKey={effectiveHoverKey}
                  onHover={setHoverKey}
                  onPick={pick}
                  showOverlay={showOverlay}
                />
              ))
            )}
          </div>
        </div>

        <div ref={listRef} className="h-[78vh] min-h-[28rem] overflow-auto border border-line bg-surface">
          <div className="sticky top-0 z-10 border-b border-line bg-surface/95 px-3 py-2.5 backdrop-blur">
            <div className="text-sm font-semibold text-ink">Detections</div>
            <div className="text-xs text-muted">
              {visibleItems.length} shown of {model.items.length} · select one to find it on the page
            </div>
          </div>
          {visibleItems.length === 0 ? (
            <div className="px-4 py-12 text-center text-sm text-muted">
              {model.items.length === 0 ? "Nothing was detected in this document." : "Every category is filtered out."}
            </div>
          ) : (
            <div className="px-1.5 py-1.5">
              {[...byPage.entries()].map(([page, items]) => (
                <section key={page} className="mb-1">
                  <h3 className="px-2.5 pt-2 pb-1 text-[0.6875rem] font-semibold tracking-wide text-muted uppercase">
                    Page {page + 1}
                  </h3>
                  <ul>
                    {items.map((item) => (
                      <ItemRow
                        key={item.id}
                        item={item}
                        active={item.id === selectedId || hoverBoxItems.has(item.id)}
                        onSelect={select}
                        onHover={(i) => setHoverItem(i?.id ?? null)}
                      />
                    ))}
                  </ul>
                </section>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
