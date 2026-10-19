import { CheckCircle2, ShieldQuestion, XCircle } from "lucide-react";

import type { Manifest } from "@/lib/manifest";
import { BAD_WHEN_NONZERO, CHECK_INFO, humanize, reasonText } from "@/lib/vocab";

import { Card, cx, EmptyState, Mono } from "./ui";

export function VerificationPanel({ manifest }: { manifest: Manifest }) {
  const checks = manifest.verification.checks;
  if (checks.length === 0) {
    return (
      <Card>
        <EmptyState icon={<ShieldQuestion className="size-8" />} title="Verification did not run">
          The job stopped before a PDF was built
          {manifest.reasons.length ? (
            <>
              : <Mono>{manifest.reasons.join(", ")}</Mono>. {reasonText(manifest.reasons[0] ?? "")}
            </>
          ) : (
            "."
          )}
        </EmptyState>
      </Card>
    );
  }

  return (
    <div className="space-y-2">
      <p className="max-w-3xl py-4 text-[0.9375rem] leading-relaxed text-ink-2">
        Release is fail-closed: the PDF is released only when all six checks pass. Verification proves that planned
        removals happened; it cannot find PII that every detector missed.
      </p>
      <div data-stagger className="grid gap-2 md:grid-cols-2 xl:grid-cols-3">
        {checks.map((c) => {
          const info = CHECK_INFO[c.check];
          return (
            <Card key={c.check} className={cx(!c.passed && "border-crit/50")}>
              <header className="flex items-start gap-3 px-6 pt-6">
                {c.passed ? (
                  <CheckCircle2 className="mt-0.5 size-5 shrink-0 text-good" aria-hidden />
                ) : (
                  <XCircle className="mt-0.5 size-5 shrink-0 text-crit" aria-hidden />
                )}
                <div className="min-w-0 flex-1">
                  <div className="flex items-baseline justify-between gap-2">
                    <h3 className="font-serif text-xl leading-tight text-ink">
                      <span className="mr-2 font-sans text-xs text-muted tabular">{info.code}</span>
                      {info.title}
                    </h3>
                    <span className={cx("text-xs font-medium", c.passed ? "text-good-text" : "text-crit-text")}>
                      {c.passed ? "Passed" : "Failed"}
                    </span>
                  </div>
                  <p className="mt-2 text-sm leading-relaxed text-ink-2">{info.summary}</p>
                  <p className="mt-1 text-xs text-muted">Independence: {info.independent}</p>
                </div>
              </header>
              <dl className="mx-6 mt-4 mb-6 grid grid-cols-[1fr_auto] gap-x-4 gap-y-1 border-t border-line pt-3 text-sm">
                {Object.entries(c.metrics).map(([k, v]) => {
                  const bad = BAD_WHEN_NONZERO.has(k) && typeof v === "number" && v > 0;
                  return (
                    <div key={k} className="contents">
                      <dt className={cx(bad ? "font-medium text-crit-text" : "text-muted")}>{humanize(k)}</dt>
                      <dd className={cx("tabular text-right", bad ? "font-semibold text-crit-text" : "text-ink")}>
                        {typeof v === "boolean" ? (v ? "yes" : "no") : v}
                      </dd>
                    </div>
                  );
                })}
              </dl>
            </Card>
          );
        })}
      </div>
    </div>
  );
}
