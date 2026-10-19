"use client";

import { CheckCircle2, XCircle } from "lucide-react";

import { formatDuration } from "@/lib/format";
import type { Manifest, RunRecord } from "@/lib/manifest";
import { CHECK_INFO, type CategoryId, categoryColor, categoryOf, typeLabel } from "@/lib/vocab";

import { Card, CardHeader, cx, Swatch } from "./ui";

interface Row {
  key: string;
  label: string;
  category: CategoryId;
  replaced: number;
  kept: number;
}

function rows(manifest: Manifest): { mentions: Row[]; regions: Row[] } {
  const m = new Map<string, Row>();
  for (const x of manifest.mentions) {
    const r = m.get(x.type) ?? { key: x.type, label: typeLabel(x.type), category: categoryOf(x.type), replaced: 0, kept: 0 };
    if (x.action === "keep") r.kept++;
    else r.replaced++;
    m.set(x.type, r);
  }
  const g = new Map<string, Row>();
  for (const x of manifest.regions ?? []) {
    const r = g.get(x.kind) ?? { key: x.kind, label: typeLabel(x.kind), category: "region" as const, replaced: 0, kept: 0 };
    if (x.action === "keep") r.kept++;
    else r.replaced++;
    g.set(x.kind, r);
  }
  const order = (a: Row, b: Row) => b.replaced + b.kept - (a.replaced + a.kept) || a.label.localeCompare(b.label);
  return { mentions: [...m.values()].sort(order), regions: [...g.values()].sort(order) };
}

/** Horizontal bars: solid = replaced, light wash of the same hue = kept, 2px surface gap between. */
function BarList({ data, max }: { data: Row[]; max: number }) {
  return (
    <ul className="space-y-1.5">
      {data.map((r) => {
        const total = r.replaced + r.kept;
        const color = categoryColor(r.category);
        return (
          <li
            key={r.key}
            className="grid grid-cols-[9.5rem_minmax(0,1fr)_5.5rem] items-center gap-3 text-sm"
            title={`${r.label}: ${r.replaced} replaced${r.kept ? `, ${r.kept} kept` : ""}`}
          >
            <span className="flex min-w-0 items-center gap-2 text-ink-2">
              <Swatch color={color} />
              <span className="truncate">{r.label}</span>
            </span>
            <span className="flex h-3.5 items-center gap-0.5" style={{ width: `${(total / max) * 100}%` }}>
              {r.replaced ? (
                <span
                  className="h-full"
                  style={{ flex: r.replaced, background: color }}
                />
              ) : null}
              {r.kept ? (
                <span
                  className="h-full"
                  style={{ flex: r.kept, background: `color-mix(in srgb, ${color} 22%, transparent)`, boxShadow: `inset 0 0 0 1px color-mix(in srgb, ${color} 55%, transparent)` }}
                />
              ) : null}
            </span>
            <span className="tabular text-right whitespace-nowrap text-ink">
              {r.replaced || null}
              {r.kept ? <span className="text-muted">{r.replaced ? ` + ${r.kept} kept` : `${r.kept} kept`}</span> : null}
            </span>
          </li>
        );
      })}
    </ul>
  );
}

const STAGES: [RegExp, string][] = [
  [/^convert_ms$/, "Convert image to PDF"],
  [/^inspect_ms$/, "Inspect input"],
  [/^analyse_ms$/, "Analyse pages (render, OCR, regions)"],
  [/^detect_ms$/, "Detect, decide and plan"],
  [/^render_build_ms_(\d+)$/, "Edit rasters and rebuild PDF"],
  [/^verify_ms_(\d+)$/, "Verify (V1–V6, re-OCR)"],
];

function stageRows(run: RunRecord): { key: string; label: string; ms: number }[] {
  const out: { key: string; label: string; ms: number }[] = [];
  for (const [key, ms] of Object.entries(run.timings_ms)) {
    if (key === "total_ms") continue;
    const stage = STAGES.find(([re]) => re.test(key));
    const pass = key.match(/_(\d+)$/)?.[1];
    const label = stage ? stage[1] + (pass && pass !== "0" ? ` · re-plan ${pass}` : "") : key;
    out.push({ key, label, ms });
  }
  const rank = (k: string) => STAGES.findIndex(([re]) => re.test(k));
  return out.sort((a, b) => {
    const pa = Number(a.key.match(/_(\d+)$/)?.[1] ?? -1);
    const pb = Number(b.key.match(/_(\d+)$/)?.[1] ?? -1);
    return pa - pb || rank(a.key) - rank(b.key);
  });
}

export function OverviewPanel({
  manifest,
  run,
  onOpenTab,
}: {
  manifest: Manifest;
  run: RunRecord | null;
  onOpenTab: (tab: "verification" | "document") => void;
}) {
  const { mentions, regions } = rows(manifest);
  const max = Math.max(1, ...[...mentions, ...regions].map((r) => r.replaced + r.kept));
  const lowByType = new Map<string, number>();
  for (const l of manifest.low_evidence) lowByType.set(l.type, (lowByType.get(l.type) ?? 0) + 1);
  const stages = run ? stageRows(run) : [];
  const stageMax = Math.max(1, ...stages.map((s) => s.ms));
  const keptAny = mentions.some((r) => r.kept);

  return (
    <div data-stagger className="grid gap-2 lg:grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)]">
      <Card>
        <CardHeader
          title="Detections by type"
          description="Mentions the policy acted on, and visual regions removed."
          actions={
            keptAny ? (
              <span className="flex items-center gap-3 text-xs text-muted">
                <span className="flex items-center gap-1.5">
                  <span className="inline-block h-2.5 w-3.5 bg-ink-2" aria-hidden /> Replaced
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="inline-block h-2.5 w-3.5 border border-ink-2/60 bg-ink-2/20" aria-hidden /> Kept
                </span>
              </span>
            ) : null
          }
        />
        <div className="space-y-5 px-6 pb-6">
          {mentions.length ? (
            <BarList data={mentions} max={max} />
          ) : (
            <p className="text-sm text-muted">No text mentions were detected.</p>
          )}
          {regions.length ? (
            <div>
              <h3 className="mb-2 text-xs font-semibold tracking-wide text-muted uppercase">Visual regions</h3>
              <BarList data={regions} max={max} />
            </div>
          ) : null}
          {manifest.residual_boxes.length ? (
            <p className="text-sm text-ink-2">
              The V4 re-read added <span className="font-medium text-ink">{manifest.residual_boxes.length}</span> blackout
              box{manifest.residual_boxes.length === 1 ? "" : "es"} for content still readable after the first render.
            </p>
          ) : null}
          {mentions.length || regions.length ? (
            <button
              type="button"
              onClick={() => onOpenTab("document")}
              className="text-sm font-medium text-ink underline decoration-line-strong underline-offset-4 hover:decoration-current"
            >
              Inspect them on the pages →
            </button>
          ) : null}
        </div>
      </Card>

      <div className="space-y-2">
        <Card>
          <CardHeader
            title="Verification"
            description={manifest.verification.checks.length ? undefined : "Verification did not run."}
          />
          <ul className="px-3 pb-3">
            {manifest.verification.checks.map((c) => {
              const info = CHECK_INFO[c.check];
              return (
                <li key={c.check}>
                  <button
                    type="button"
                    onClick={() => onOpenTab("verification")}
                    className="flex w-full items-center gap-3 px-2 py-1.5 text-left text-sm hover:bg-surface-2"
                  >
                    {c.passed ? (
                      <CheckCircle2 className="size-4 shrink-0 text-good" aria-hidden />
                    ) : (
                      <XCircle className="size-4 shrink-0 text-crit" aria-hidden />
                    )}
                    <span className="w-6 font-mono text-xs text-muted">{info.code}</span>
                    <span className="text-ink">{info.title}</span>
                    <span className={cx("ml-auto text-xs font-medium", c.passed ? "text-good-text" : "text-crit-text")}>
                      {c.passed ? "Passed" : "Failed"}
                    </span>
                  </button>
                </li>
              );
            })}
          </ul>
        </Card>

        {stages.length ? (
          <Card>
            <CardHeader
              title="Run time"
              description={`${formatDuration(run?.timings_ms.total_ms)} in total with ${run?.workers ?? "?"} worker${run?.workers === 1 ? "" : "s"}.`}
            />
            <ul className="space-y-2.5 px-6 pb-6">
              {stages.map((s) => (
                <li key={s.key} className="text-sm">
                  <div className="mb-1 flex justify-between gap-3">
                    <span className="truncate text-ink-2">{s.label}</span>
                    <span className="tabular shrink-0 text-ink">{formatDuration(s.ms)}</span>
                  </div>
                  <div className="h-1.5 bg-surface-3">
                    <div className="h-full bg-(--series-1)" style={{ width: `${(s.ms / stageMax) * 100}%` }} />
                  </div>
                </li>
              ))}
            </ul>
          </Card>
        ) : null}

        <Card>
          <CardHeader
            title="Low-evidence candidates"
            description="Seen by a detector but below the acceptance bar, so left visible. Listed for error analysis."
          />
          <div className="px-6 pb-6">
            {lowByType.size ? (
              <ul className="flex flex-wrap gap-1.5">
                {[...lowByType.entries()]
                  .sort((a, b) => b[1] - a[1])
                  .map(([type, n]) => (
                    <li key={type} className="inline-flex items-center gap-1.5 border border-line px-2 py-0.5 text-xs text-ink-2">
                      <Swatch color={categoryColor(categoryOf(type as never))} />
                      {typeLabel(type)}
                      <span className="tabular font-medium text-ink">{n}</span>
                    </li>
                  ))}
              </ul>
            ) : (
              <p className="text-sm text-muted">None.</p>
            )}
          </div>
        </Card>
      </div>
    </div>
  );
}
