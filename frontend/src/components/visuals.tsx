/**
 * Decorative media drawn entirely in CSS: nothing is fetched, so the interface keeps its
 * no-external-requests posture.  These panels are always dark, in both themes, like photographs.
 */
import { Fragment, type CSSProperties, type ReactNode } from "react";

import { cx } from "./ui";

const PETALS = [
  { r: -12, l: 46, d: 0 },
  { r: 26, l: 33, d: -2 },
  { r: 58, l: 44, d: -4 },
  { r: 99, l: 37, d: -6 },
  { r: 134, l: 47, d: -1 },
  { r: 173, l: 32, d: -3 },
  { r: 208, l: 45, d: -5 },
  { r: 252, l: 36, d: -7 },
  { r: 291, l: 48, d: -2.5 },
  { r: 330, l: 31, d: -4.5 },
];

function Petals({ className }: { className: string }) {
  return (
    <div className={className}>
      {PETALS.map((p) => (
        <span
          key={p.r}
          className="petal"
          style={{ "--r": `${p.r}deg`, "--l": `${p.l}%`, "--d": `${p.d}s` } as CSSProperties}
        />
      ))}
    </div>
  );
}

/** Rainbow loops of light around a white core, with a wider bloom behind them. */
export function Prism({ className }: { className?: string }) {
  return (
    <div className={cx("prism grain", className)} aria-hidden="true">
      <Petals className="prism-spin prism-bloom" />
      <Petals className="prism-spin" />
      <div className="prism-core" />
    </div>
  );
}

export type AuroraTone = "rose" | "ember" | "moss" | "amber" | "sand" | "dusk" | "slate";

const TONES: Record<AuroraTone, [string, string, string, string, string]> = {
  rose: ["#2f1a33", "#71385f", "#1f0f1b", "rgba(255,150,170,.78)", "rgba(184,112,196,.55)"],
  ember: ["#4a140c", "#b3331d", "#220906", "rgba(255,118,78,.72)", "rgba(110,16,8,.65)"],
  moss: ["#262d14", "#66753a", "#0f1209", "rgba(214,224,170,.72)", "rgba(84,104,38,.6)"],
  amber: ["#2b3049", "#b0632a", "#231107", "rgba(255,178,98,.78)", "rgba(86,94,146,.5)"],
  sand: ["#5d4730", "#b8946a", "#2e2115", "rgba(255,236,204,.72)", "rgba(132,94,58,.55)"],
  dusk: ["#1b222c", "#36445a", "#121519", "rgba(214,152,122,.62)", "rgba(112,134,164,.62)"],
  slate: ["#18202a", "#344456", "#0d1116", "rgba(156,182,212,.55)", "rgba(58,78,104,.62)"],
};

/** Silky diagonal streaks of colour under film grain. */
export function Aurora({ tone, className }: { tone: AuroraTone; className?: string }) {
  const [a1, a2, a3, a4, a5] = TONES[tone];
  return (
    <div
      className={cx("aurora grain", className)}
      style={{ "--a1": a1, "--a2": a2, "--a3": a3, "--a4": a4, "--a5": a5 } as CSSProperties}
      aria-hidden="true"
    >
      <span />
      <span />
      <span />
    </div>
  );
}

/** A frosted chip for use over dark media. */
export function GlassChip({ children, className, style }: { children: ReactNode; className?: string; style?: CSSProperties }) {
  return (
    <span
      style={style}
      className={cx(
        "inline-flex h-9 items-center gap-2 border border-white/12 bg-black/35 px-3.5 text-sm text-white backdrop-blur-md",
        className,
      )}
    >
      {children}
    </span>
  );
}

/**
 * Steps joined by a thin rule with a node at its midpoint.  Each chip is keyed by its label, so a
 * changed step pops back in while the others stay put; the nodes pulse along the line.
 */
export function Flow({ steps, className }: { steps: string[]; className?: string }) {
  return (
    <div className={cx("flex flex-wrap items-center justify-center gap-y-3", className)}>
      {steps.map((s, i) => (
        <Fragment key={i}>
          {i > 0 ? (
            <span className="relative mx-1 hidden h-px w-7 bg-white/45 sm:block" aria-hidden="true">
              <span
                className="flow-node absolute top-1/2 left-1/2 size-1.5 -translate-1/2 bg-white"
                style={{ animationDelay: `${i * 400}ms` }}
              />
            </span>
          ) : null}
          <GlassChip key={s} className="fade-up mx-1 sm:mx-0" style={{ animationDelay: `${i * 70}ms` }}>
            <span className="size-1.5 shrink-0 bg-white" aria-hidden="true" />
            {s}
          </GlassChip>
        </Fragment>
      ))}
    </div>
  );
}
