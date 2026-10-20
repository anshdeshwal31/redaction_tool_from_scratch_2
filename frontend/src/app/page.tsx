import { ArrowRight } from "lucide-react";
import Link from "next/link";

import { Mark } from "@/components/brand";
import { Guarantees, HowItWorks, Principles } from "@/components/home-sections";
import { IdentifierMarquee } from "@/components/marquee";
import { CountUp } from "@/components/motion";
import { RecentJobs } from "@/components/recent-jobs";
import { ButtonLink, Stat } from "@/components/ui";
import { Aurora, Prism } from "@/components/visuals";
import { CHECKS, ENTITY_TYPES, REGION_KINDS } from "@/lib/manifest";

const FACTS = [
  { value: ENTITY_TYPES.length, label: "Identifier types detected and replaced, from names to Medicare numbers" },
  { value: REGION_KINDS.length, label: "Kinds of visual region removed, from signatures to faces" },
  { value: CHECKS.length, label: "Verification checks a document must pass before release" },
  {
    value: 0,
    label: "Copies of the original kept once a job ends, unless you keep one for a side-by-side comparison",
  },
];

function Hero() {
  return (
    <section className="flex flex-col items-center pt-20 text-center sm:pt-28">
      <h1 className="display text-[clamp(3.25rem,7.4vw,6.25rem)] leading-[0.98] text-balance text-ink">
        <span className="rise block">Redact the record.</span>
        <span className="rise block" style={{ animationDelay: "140ms" }}>
          Verify every page.
        </span>
      </h1>
      <p
        className="rise mt-8 max-w-136 text-lg leading-relaxed text-pretty text-ink-2 sm:text-[1.1875rem]"
        style={{ animationDelay: "320ms" }}
      >
        Find and replace personal information in medico-legal documents, rebuild a brand-new PDF, and release it only
        when every check passes — on your own redaction server.
      </p>
      <div className="rise mt-10 flex flex-wrap justify-center gap-2.5" style={{ animationDelay: "460ms" }}>
        <ButtonLink href="/redact" variant="primary" size="lg">
          Start a redaction
        </ButtonLink>
        <ButtonLink href="#how-it-works" size="lg">
          See how it works
        </ButtonLink>
      </div>
    </section>
  );
}

function Showcase() {
  return (
    <section
      aria-label="mlredact"
      className="rise relative mt-20 h-[clamp(24rem,46vw,38rem)] overflow-hidden"
      style={{ animationDelay: "600ms" }}
    >
      <Prism />
      <div className="pointer-events-none absolute inset-0 flex items-center justify-center" aria-hidden>
        <span className="display text-[clamp(4.5rem,14vw,12rem)] text-transparent select-none [-webkit-text-stroke:1px_rgba(255,255,255,0.55)]">
          mlredact
        </span>
      </div>
      <div className="absolute top-5 left-5 flex items-center gap-3.5 border border-white/10 bg-black/45 px-4 py-3 text-white backdrop-blur-md">
        <Mark className="size-4.5" />
        <div>
          <div className="text-sm font-semibold">Fail-closed release</div>
          <div className="text-xs text-white/70">Six checks. No partial output.</div>
        </div>
      </div>
      <div className="absolute right-5 bottom-5 hidden items-center gap-4 border border-white/10 bg-black/45 px-4 py-3 text-white backdrop-blur-md sm:flex">
        <div className="flex gap-1" aria-hidden>
          {CHECKS.map((c, i) => (
            <span
              key={c}
              className="tick grid size-7 place-items-center border border-white/15 bg-white/10 font-mono text-[0.625rem]"
              style={{ animationDelay: `${1200 + i * 180}ms` }}
            >
              V{i + 1}
            </span>
          ))}
        </div>
        <div>
          <div className="text-sm font-semibold">All six pass</div>
          <div className="text-xs text-white/70">or nothing is released</div>
        </div>
        <Mark className="size-4" />
      </div>
    </section>
  );
}

function CallToAction() {
  return (
    <section data-reveal className="relative mt-36 grid min-h-120 place-items-center overflow-hidden px-6 py-20">
      <Aurora tone="dusk" />
      <div data-stagger className="relative flex flex-col items-center text-center text-white">
        <h2 className="display text-[clamp(2.5rem,5vw,4.25rem)] text-balance">
          Check the engine.
          <br /> Then trust the output.
        </h2>
        <p className="mt-6 max-w-lg text-[1.0625rem] leading-relaxed text-white/80">
          Run the self-check to verify the configuration, pinned models and engine binaries, then redact your first
          document.
        </p>
        <div className="mt-9 flex flex-wrap justify-center gap-2.5">
          <Link
            href="/redact"
            className="group inline-flex h-13 items-center gap-2 bg-white px-7 text-[0.9375rem] font-medium text-[#0e1214] transition-[background-color,scale] duration-150 hover:bg-white/90 motion-safe:active:scale-[0.97]"
          >
            Start a redaction
            <ArrowRight className="size-4 transition-transform group-hover:translate-x-0.5" aria-hidden />
          </Link>
          <Link
            href="/system"
            className="inline-flex h-13 items-center border border-white/15 bg-black/35 px-7 text-[0.9375rem] font-medium text-white backdrop-blur-md transition-[background-color,scale] duration-150 hover:bg-black/55 motion-safe:active:scale-[0.97]"
          >
            Run the self-check
          </Link>
        </div>
        <p className="mt-5 text-xs text-white/65">It hashes every pinned model, so it can take a minute.</p>
      </div>
    </section>
  );
}

export default function HomePage() {
  return (
    <>
      <Hero />
      <Showcase />
      <IdentifierMarquee caption="Identifiers it finds and replaces in Australian medico-legal records" />
      <section aria-label="At a glance" data-stagger className="grid gap-2 sm:grid-cols-2">
        {FACTS.map((f) => (
          <Stat key={f.label} size="lg" value={<CountUp value={f.value} />} label={f.label} />
        ))}
      </section>

      <HowItWorks />
      <Guarantees />
      <RecentJobs />
      <Principles />
      <CallToAction />
    </>
  );
}
