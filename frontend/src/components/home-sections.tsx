"use client";

import { ArrowLeft, ArrowRight, Check } from "lucide-react";
import { useRef, useState } from "react";

import { Mark } from "./brand";
import { ArrowLink, cx, SectionHeading } from "./ui";
import { Aurora, type AuroraTone, Flow, GlassChip } from "./visuals";

/* ------------------------------------------------------------------------------ how it works */

const STAGES = [
  {
    label: "Stage 1",
    title: "Inspect",
    statement: "Encrypted, XFA-only or over-limit inputs are rejected before a single page is drawn.",
    checks: ["Encryption and XFA", "Size and page limits", "Structure inventory"],
  },
  {
    label: "Stage 2",
    title: "Analyse pages",
    statement: "Each page is rasterised and OCR'd, and ink that no reading explains becomes a region to remove.",
    checks: ["Rasterise", "OCR every page", "Unexplained ink"],
  },
  {
    label: "Stage 3",
    title: "Detect",
    statement: "Rules, NER and your known identifiers are fused, then propagated across the whole document.",
    checks: ["Rule detectors", "NER models", "Known identifiers", "Propagation"],
  },
  {
    label: "Stage 4",
    title: "Replace",
    statement: "The policy picks an action for every mention, and the page raster itself is edited.",
    checks: ["Policy profile", "Surrogate, label or blackout", "Raster edit"],
  },
  {
    label: "Stage 5",
    title: "Rebuild & verify",
    statement: "A brand-new PDF is built, then six checks run against it, including a full re-OCR.",
    checks: ["Fresh PDF", "Checks V1–V6", "Release or quarantine"],
  },
];

export function HowItWorks() {
  const [active, setActive] = useState(0);
  const refs = useRef<(HTMLButtonElement | null)[]>([]);
  const stage = STAGES[active]!;

  return (
    <section id="how-it-works" className="pt-32">
      <SectionHeading
        eyebrow="How it works"
        title={
          <>
            From upload to release
            <br className="hidden sm:block" /> in five checked stages.
          </>
        }
      />
      <div role="tablist" aria-label="Pipeline stages" data-stagger className="mt-14 flex gap-2 overflow-x-auto">
        {STAGES.map((s, i) => (
          <button
            key={s.title}
            ref={(el) => {
              refs.current[i] = el;
            }}
            type="button"
            role="tab"
            id={`stage-tab-${i}`}
            aria-selected={active === i}
            aria-controls="stage-panel"
            tabIndex={active === i ? 0 : -1}
            onClick={() => setActive(i)}
            onKeyDown={(e) => {
              if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
              const next = (i + (e.key === "ArrowRight" ? 1 : STAGES.length - 1)) % STAGES.length;
              setActive(next);
              refs.current[next]?.focus();
            }}
            className={cx(
              "flex min-w-48 flex-1 flex-col items-start px-6 pt-5 text-left transition-colors",
              active === i ? "bg-surface-2" : "bg-surface hover:bg-surface-2",
            )}
          >
            <span className="text-xs text-muted">{s.label}</span>
            <span className="mt-2 font-serif text-[1.375rem] leading-tight whitespace-nowrap text-ink">
              {s.title}
            </span>
            <span className="relative mt-6 mb-5 h-px w-full bg-line-strong" aria-hidden>
              <span
                className={cx(
                  "absolute -top-px left-0 h-0.5 bg-ink transition-[width] duration-500",
                  active === i ? "w-full" : "w-0",
                )}
              />
            </span>
          </button>
        ))}
      </div>

      <div
        id="stage-panel"
        role="tabpanel"
        aria-labelledby={`stage-tab-${active}`}
        data-reveal
        className="mt-2 grid gap-12 bg-surface px-7 py-12 sm:px-14 sm:py-16 lg:grid-cols-[minmax(0,1.1fr)_minmax(0,1fr)] lg:gap-20"
      >
        <div key={active} className="fade-up">
          <div className="text-sm text-muted tabular">0{active + 1}</div>
          <p className="mt-6 font-serif text-[clamp(1.625rem,2.6vw,2.125rem)] leading-tight text-pretty text-ink">
            {stage.statement}
          </p>
          <ul data-stagger className="mt-9 flex flex-wrap gap-2">
            {stage.checks.map((c) => (
              <li key={c} className="inline-flex h-10 items-center gap-2 bg-surface-2 px-4 text-sm text-ink">
                <Check className="size-4" aria-hidden />
                {c}
              </li>
            ))}
          </ul>
        </div>
        <div className="flex flex-col justify-center">
          <div className="flex justify-between text-xs text-muted">
            <span>Upload</span>
            <span>Release</span>
          </div>
          <div className="mt-3 flex gap-1 border border-line bg-page p-1.5" aria-hidden>
            {STAGES.map((s, i) => (
              <span
                key={s.title}
                className={cx(
                  "h-9 flex-1 transition-colors duration-500",
                  i === active ? "bg-ink" : i < active ? "bg-surface-3" : "bg-surface-2",
                )}
              />
            ))}
          </div>
          <ArrowLink href="/redact" className="mt-8 self-start">
            Start with a document
          </ArrowLink>
        </div>
      </div>
    </section>
  );
}

/* -------------------------------------------------------------------------------- guarantees */

const GUARANTEES: {
  title: string;
  text: string;
  tone: AuroraTone;
  flow: string[];
  badge: string;
}[] = [
  {
    title: "Rebuild, never edit",
    text: "Every page is rebuilt from its edited raster. Nothing from the input's PDF structure reaches the output.",
    tone: "ember",
    flow: ["Input PDF", "Page rasters", "Edit", "New PDF"],
    badge: "No input objects carried over",
  },
  {
    title: "Fail closed",
    text: "Any failed check quarantines the job, with content-free reason codes. There is no partial release.",
    tone: "moss",
    flow: ["Rebuilt PDF", "V1–V6", "Release"],
    badge: "Six of six checks, or quarantine",
  },
  {
    title: "Content-free manifest",
    text: "The manifest records types, boxes and evidence for every change, never the document's text.",
    tone: "sand",
    flow: ["Mention", "Type", "Box", "Evidence"],
    badge: "No document text stored",
  },
  {
    title: "Your own server",
    text: "Documents go straight from the browser to your redaction server and are processed there. Inputs are deleted when a job ends, unless you keep the original for a few days to compare.",
    tone: "slate",
    flow: ["Browser", "Redaction server", "Engine", "Disk"],
    badge: "No external requests",
  },
];

const ASSURANCES = ["Same inputs give byte-identical output", "Pinned models, hash-verified", "Reason codes, never content"];

export function Guarantees() {
  const [open, setOpen] = useState(1);
  const g = GUARANTEES[open]!;

  return (
    <section id="guarantees" className="pt-32">
      <SectionHeading
        eyebrow="Guarantees"
        title={
          <>
            Built to fail closed.
            <br className="hidden sm:block" /> Never to fail open.
          </>
        }
      />
      <div className="mt-14 grid gap-2 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.22fr)]">
        <ul data-stagger className="space-y-2">
          {GUARANTEES.map((item, i) => {
            const on = open === i;
            return (
              <li key={item.title}>
                <button
                  type="button"
                  aria-expanded={on}
                  aria-controls={`guarantee-${i}`}
                  onClick={() => setOpen(i)}
                  className={cx(
                    "group w-full px-7 py-6 text-left transition-colors duration-300",
                    on ? "bg-ink text-page" : "bg-surface text-ink hover:bg-surface-2",
                  )}
                >
                  <span className="flex items-center gap-5">
                    <span className={cx("text-xs tabular", on ? "text-page/55" : "text-muted")}>0{i + 1}</span>
                    <span className="font-serif text-[1.625rem] leading-tight">{item.title}</span>
                    <ArrowRight
                      className={cx(
                        "ml-auto size-4 shrink-0 transition-transform duration-300 group-hover:translate-x-1",
                        on ? "" : "text-muted",
                      )}
                      aria-hidden
                    />
                  </span>
                  <span
                    id={`guarantee-${i}`}
                    className={cx(
                      "grid transition-[grid-template-rows] duration-300",
                      on ? "grid-rows-[1fr]" : "grid-rows-[0fr]",
                    )}
                  >
                    <span className="overflow-hidden">
                      <span className="block pt-3 pl-9 text-[0.9375rem] leading-relaxed text-page/70">{item.text}</span>
                    </span>
                  </span>
                </button>
              </li>
            );
          })}
        </ul>

        <div data-reveal className="flex flex-col gap-2">
          <div className="relative min-h-88 flex-1 overflow-hidden lg:min-h-96">
            <Aurora key={g.tone} tone={g.tone} className="fade-up" />
            <div key={open} className="fade-up absolute inset-0 flex items-center justify-center px-6">
              <Flow steps={g.flow} />
            </div>
            <GlassChip className="absolute bottom-5 left-5">{g.badge}</GlassChip>
          </div>
          <ul className="grid gap-2 sm:grid-cols-3">
            {ASSURANCES.map((a) => (
              <li key={a} className="flex gap-3 bg-surface px-5 py-4 text-[0.9375rem] leading-snug text-ink-2">
                <Mark className="mt-0.5 size-3.5 text-muted" />
                {a}
              </li>
            ))}
          </ul>
        </div>
      </div>
    </section>
  );
}

/* -------------------------------------------------------------------------------- principles */

const QUOTES = [
  {
    quote: "Verification proves that planned removals happened. It cannot find what every detector missed.",
    name: "On verification",
    role: "Checks V1–V6",
  },
  {
    quote: "The same input, config, models, key and runtime profile give a byte-identical PDF and manifest.",
    name: "On reproducibility",
    role: "The provenance record",
  },
  {
    quote: "Reason codes are content-free by design. A quarantine says why, never what the document said.",
    name: "On quarantine",
    role: "Fail-closed release",
  },
];

export function Principles() {
  const [i, setI] = useState(0);
  const q = QUOTES[i]!;
  const go = (d: number) => setI((n) => (n + d + QUOTES.length) % QUOTES.length);

  return (
    <section aria-label="Principles" data-reveal className="flex flex-col items-center pt-36 text-center">
      <Mark className="size-6 text-muted" />
      <figure key={i} className="fade-up mt-8 flex flex-col items-center" aria-live="polite">
        <blockquote className="display max-w-5xl text-[clamp(1.875rem,3.6vw,3rem)] leading-[1.18] text-balance text-ink">
          &ldquo;{q.quote}&rdquo;
        </blockquote>
        <figcaption className="mt-10 flex items-center gap-3 text-left">
          <span className="grid size-11 place-items-center bg-surface-3 text-ink">
            <Mark className="size-4" />
          </span>
          <span>
            <span className="block text-[0.9375rem] font-medium text-ink">{q.name}</span>
            <span className="block text-sm text-muted">{q.role}</span>
          </span>
        </figcaption>
      </figure>
      <div className="mt-10 flex items-center gap-4">
        <button
          type="button"
          onClick={() => go(-1)}
          className="grid size-12 place-items-center bg-surface-2 text-ink transition-colors hover:bg-surface-3"
          aria-label="Previous principle"
        >
          <ArrowLeft className="size-4" aria-hidden />
        </button>
        <div className="flex items-center gap-1.5">
          {QUOTES.map((x, n) => (
            <button
              key={x.name}
              type="button"
              onClick={() => setI(n)}
              aria-label={`Principle ${n + 1} of ${QUOTES.length}`}
              aria-current={n === i || undefined}
              className={cx("h-1.5 transition-all", n === i ? "w-7 bg-ink" : "w-1.5 bg-muted/60 hover:bg-muted")}
            />
          ))}
        </div>
        <button
          type="button"
          onClick={() => go(1)}
          className="grid size-12 place-items-center bg-surface-2 text-ink transition-colors hover:bg-surface-3"
          aria-label="Next principle"
        >
          <ArrowRight className="size-4" aria-hidden />
        </button>
      </div>
    </section>
  );
}
