"use client";

import { ArrowRight } from "lucide-react";
import Link from "next/link";

import { usePolling, useNow } from "@/lib/api";
import { formatRelative, shortId } from "@/lib/format";
import { ACTIVE_STATES, type JobRecord, type JobState } from "@/lib/jobs";
import { PROFILE_INFO, STYLE_INFO } from "@/lib/vocab";

import { Mark } from "./brand";
import { ButtonLink, SectionHeading } from "./ui";
import { Aurora, type AuroraTone } from "./visuals";

/** Each outcome wears its own light, so a glance tells released from quarantined. */
export const STATE_TONE: Record<JobState, AuroraTone> = {
  awaiting_upload: "rose",
  released: "moss",
  quarantined: "amber",
  failed: "ember",
  running: "rose",
  queued: "rose",
  cancelled: "slate",
  interrupted: "slate",
};

const STATE_WORD: Record<JobState, string> = {
  awaiting_upload: "Uploading",
  released: "Released",
  quarantined: "Quarantined",
  failed: "Failed",
  running: "Running",
  queued: "Queued",
  cancelled: "Cancelled",
  interrupted: "Interrupted",
};

const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? "" : "s"}`;

function story(job: JobRecord): string {
  const s = job.summary;
  switch (job.state) {
    case "released":
      return s
        ? `${plural(s.pages, "page")} rebuilt, ${plural(s.replaced, "mention")} replaced and ${plural(s.regions, "region")} removed. ${s.checksPassed} of ${s.checksTotal} checks passed.`
        : "Released after every check passed.";
    case "quarantined":
      return s?.reasons.length
        ? `Held back by ${s.reasons.join(", ")}. No PDF was released; the reasons are content-free.`
        : "Held back by verification. No PDF was released.";
    case "failed":
      return `The engine could not complete this job${job.errorCode ? ` (${job.errorCode})` : ""}.`;
    case "running":
      return "Inspecting, analysing, detecting, rendering and verifying now.";
    case "queued":
      return "Waiting for the job ahead of it to finish.";
    default:
      return "Stopped before the engine finished. Its input has been deleted.";
  }
}

function JobCard({ job, now }: { job: JobRecord; now: number }) {
  const o = job.options;
  return (
    <Link href={`/jobs/${job.id}`} className="group flex flex-col gap-2">
      <div className="relative grid h-64 place-items-center overflow-hidden sm:h-72">
        <Aurora tone={STATE_TONE[job.state]} className="transition-transform duration-700 group-hover:scale-[1.03]" />
        <span className="relative flex items-center gap-3 text-white">
          <Mark className="size-6" />
          <span className="font-mono text-[1.75rem] font-semibold tracking-[-0.02em]">{shortId(job.id)}</span>
        </span>
        <span className="absolute top-4 right-4 bg-black/55 px-3 py-1.5 text-xs font-medium text-white backdrop-blur-md">
          {STATE_WORD[job.state]}
        </span>
      </div>
      <div className="flex flex-1 flex-col bg-surface px-7 pt-6 pb-6 transition-colors group-hover:bg-surface-2">
        <p className="max-w-md text-[1.0625rem] leading-relaxed text-ink">{story(job)}</p>
        <p className="mt-2 text-sm text-muted">
          {PROFILE_INFO[o.profile]?.title ?? o.profile} · {STYLE_INFO[o.style].title} · {formatRelative(job.createdAt, now)}
        </p>
        <div className="mt-auto flex items-center justify-between pt-8">
          <span className="inline-flex items-center gap-1.5 text-[0.9375rem] text-ink">
            Open job
            <ArrowRight className="size-3.5 transition-transform group-hover:translate-x-0.5" aria-hidden />
          </span>
          <Mark className="size-4 text-muted" />
        </div>
      </div>
    </Link>
  );
}

export function RecentJobs() {
  // Idle polling too: jobs started from another tab or by a colleague must still appear.
  const { data } = usePolling<{ jobs: JobRecord[] }>(
    "/api/jobs",
    3000,
    (d) => d.jobs.some((j) => ACTIVE_STATES.has(j.state)),
    15000,
  );
  const now = useNow(false);
  const jobs = data?.jobs.slice(0, 2) ?? [];

  return (
    <section id="recent" className="pt-32">
      <SectionHeading
        eyebrow="Recent jobs"
        title={
          <>
            What ran last,
            <br /> and how it ended.
          </>
        }
      />
      <div className="mt-14">
        {!data ? (
          <div className="grid gap-2 md:grid-cols-2" aria-hidden>
            <div className="h-104 bg-surface" />
            <div className="h-104 bg-surface" />
          </div>
        ) : jobs.length === 0 ? (
          <div data-reveal className="relative grid h-72 place-items-center overflow-hidden">
            <Aurora tone="slate" />
            <div className="relative px-6 text-center text-white">
              <p className="font-serif text-[1.75rem]">Nothing on the record yet.</p>
              <p className="mt-2 text-sm text-white/70">Your first redaction appears here with its verification result.</p>
            </div>
          </div>
        ) : (
          <div data-stagger className="grid gap-2 md:grid-cols-2">
            {jobs.map((job) => (
              <JobCard key={job.id} job={job} now={now} />
            ))}
          </div>
        )}
      </div>
      {jobs.length ? (
        <div className="mt-10 flex justify-center">
          <ButtonLink href="/jobs" size="lg">
            All jobs
            <ArrowRight className="size-4" aria-hidden />
          </ButtonLink>
        </div>
      ) : null}
    </section>
  );
}
