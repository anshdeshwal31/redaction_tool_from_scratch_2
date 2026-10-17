import type { Manifest } from "@/lib/manifest";
import { percent } from "@/lib/format";

import { Card, CardHeader, cx, KeyValue } from "./ui";

const KIND: Record<string, string> = {
  image_only: "Scanned image",
  born_digital: "Born digital",
  mixed: "Mixed",
  empty: "Empty",
};

const mm = (pt: number) => Math.round((pt * 25.4) / 72);

/** A meter: the fill carries the value; the track is a lighter step of the same hue. */
function Meter({ value, tone }: { value: number | undefined; tone: "accent" | "warn" }) {
  if (value === undefined) return <span className="text-muted">–</span>;
  const fill = tone === "warn" ? "var(--series-2)" : "var(--series-1)";
  return (
    <div className="flex items-center justify-end gap-2">
      <div
        className="h-1.5 w-16 overflow-hidden"
        style={{ background: `color-mix(in srgb, ${fill} 18%, transparent)` }}
        aria-hidden
      >
        <div className="h-full" style={{ width: `${Math.min(1, value) * 100}%`, background: fill }} />
      </div>
      <span className="tabular w-12 text-right">{percent(value)}</span>
    </div>
  );
}

function YesNo({ value, risky = false }: { value: boolean; risky?: boolean }) {
  return <span className={cx(value && risky ? "font-medium text-warn-text" : "text-ink")}>{value ? "Yes" : "No"}</span>;
}

export function PagesPanel({ manifest }: { manifest: Manifest }) {
  const perPage = new Map<number, { replaced: number; regions: number }>();
  const bump = (page: number, key: "replaced" | "regions") => {
    const p = perPage.get(page) ?? { replaced: 0, regions: 0 };
    p[key]++;
    perPage.set(page, p);
  };
  for (const m of manifest.mentions) {
    if (m.action === "keep") continue;
    for (const page of new Set(m.boxes.map((b) => b.page))) bump(page, "replaced");
  }
  for (const r of manifest.regions ?? []) bump(r.box.page, "regions");
  const input = manifest.input;

  return (
    <div className="space-y-2">
      <Card className="overflow-hidden">
        <CardHeader
          title="Pages"
          description="Unexplained ink is the share of a page's ink that no readable token accounts for; it becomes regions."
        />
        <div className="overflow-x-auto">
          <table className="w-full min-w-184 text-sm">
            <thead>
              <tr className="border-y border-line text-left text-[0.6875rem] tracking-wider text-muted uppercase">
                <th scope="col" className="px-5 py-3 font-medium">Page</th>
                <th scope="col" className="px-5 py-3 font-medium">Kind</th>
                <th scope="col" className="px-5 py-3 font-medium">Size</th>
                <th scope="col" className="px-5 py-3 text-right font-medium">Lines</th>
                <th scope="col" className="px-5 py-3 text-right font-medium">Tokens</th>
                <th scope="col" className="px-5 py-3 text-right font-medium">OCR confidence</th>
                <th scope="col" className="px-5 py-3 text-right font-medium">Unexplained ink</th>
                <th scope="col" className="px-5 py-3 text-right font-medium">Replaced</th>
                <th scope="col" className="px-5 py-3 text-right font-medium">Regions</th>
                <th scope="col" className="px-5 py-3 font-medium">Notice</th>
              </tr>
            </thead>
            <tbody className="tabular">
              {manifest.pages.map((p) => (
                <tr key={p.index} className="border-b border-line last:border-b-0">
                  <td className="px-5 py-3 font-medium text-ink">{p.index + 1}</td>
                  <td className="px-5 py-3 whitespace-nowrap text-ink-2">{KIND[p.kind] ?? p.kind}</td>
                  <td className="px-5 py-3 whitespace-nowrap text-ink-2" title={`${p.width_px} × ${p.height_px} px at ${p.dpi} dpi`}>
                    {mm(p.width_pt)} × {mm(p.height_pt)} mm
                  </td>
                  <td className="px-5 py-3 text-right">{p.lines}</td>
                  <td className="px-5 py-3 text-right">{p.tokens}</td>
                  <td className="px-5 py-3">
                    <Meter value={p.mean_ocr_confidence} tone="accent" />
                  </td>
                  <td className="px-5 py-3">
                    <Meter value={p.unexplained_ink} tone="warn" />
                  </td>
                  <td className="px-5 py-3 text-right">{perPage.get(p.index)?.replaced ?? 0}</td>
                  <td className="px-5 py-3 text-right">{perPage.get(p.index)?.regions ?? 0}</td>
                  <td className="px-5 py-3 text-ink-2">{p.notice ? "Drawn" : "–"}</td>
                </tr>
              ))}
              {manifest.pages.length === 0 ? (
                <tr>
                  <td colSpan={10} className="px-4 py-10 text-center text-muted">
                    No page was analysed.
                  </td>
                </tr>
              ) : null}
            </tbody>
          </table>
        </div>
      </Card>

      <Card>
        <CardHeader title="Input inspection" description="Structure of the submitted file, checked before rendering." />
        <div className="px-6 pb-6">
          {input ? (
            <KeyValue
              className="md:grid-cols-[minmax(8rem,auto)_1fr_minmax(8rem,auto)_1fr] lg:grid-cols-[minmax(8rem,auto)_1fr_minmax(8rem,auto)_1fr_minmax(8rem,auto)_1fr]"
              items={[
                ["Source format", (manifest.job.source_format ?? "pdf").toUpperCase()],
                ["Pages", input.page_count],
                ["Encrypted", <YesNo key="e" value={input.encrypted} risky />],
                ["AcroForm", <YesNo key="a" value={input.has_acroform} />],
                ["XFA", <YesNo key="x" value={input.has_xfa} risky />],
                ["JavaScript", <YesNo key="j" value={input.has_javascript} risky />],
                ["Optional content", <YesNo key="o" value={input.has_optional_content} />],
                ["Attachments dropped", input.attachments_dropped],
                ["Annotations flattened or dropped", input.annotations_flattened_or_dropped],
                ["UserUnit pages", input.user_unit_pages],
              ]}
            />
          ) : (
            <p className="text-sm text-muted">The input was rejected before inspection completed.</p>
          )}
        </div>
      </Card>
    </div>
  );
}
