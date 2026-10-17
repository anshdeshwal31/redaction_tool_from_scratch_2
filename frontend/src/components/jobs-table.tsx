"use client";

import { ArrowRight, Inbox } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";

import { usePolling, useNow } from "@/lib/api";
import { formatBytes, formatClock, formatDate, formatDuration, formatRelative, shortHash, shortId } from "@/lib/format";
import { ACTIVE_STATES, type JobRecord } from "@/lib/jobs";
import { PROFILE_INFO, STYLE_INFO } from "@/lib/vocab";

import { CountUp } from "./motion";
import { StatusBadge } from "./status-badge";
import { ButtonLink, Card, EmptyState, Mono, Notice, Spinner, Stat } from "./ui";

type Row = JobRecord & { queuePosition?: number };

function Checks({ job }: { job: Row }) {
  const s = job.summary;
  if (!s || s.checksTotal === 0) return <span className="text-muted">–</span>;
  const ok = s.checksPassed === s.checksTotal;
  return (
    <span className={ok ? "text-ink" : "font-medium text-warn-text"}>
      {s.checksPassed}/{s.checksTotal}
    </span>
  );
}

function Duration({ job, now }: { job: Row; now: number }) {
  if (job.state === "running" && job.startedAt) {
    return <span className="text-ink">{formatClock(now - new Date(job.startedAt).getTime())}</span>;
  }
  if (job.state === "queued") {
    return <span className="text-muted">{job.queuePosition ? `#${job.queuePosition + 1} in queue` : "next"}</span>;
  }
  const ms =
    job.summary?.durationMs ??
    (job.startedAt && job.finishedAt ? new Date(job.finishedAt).getTime() - new Date(job.startedAt).getTime() : null);
  return <>{formatDuration(ms)}</>;
}

/** Totals across every job on the redaction server, in the same serif figures as the home page. */
function Totals({ jobs }: { jobs: Row[] }) {
  const count = (state: Row["state"]) => jobs.filter((j) => j.state === state).length;
  const pages = jobs.reduce((n, j) => n + (j.summary?.pages ?? 0), 0);
  return (
    <div data-stagger className="mb-2 grid grid-cols-2 gap-2 lg:grid-cols-4">
      <Stat label="Jobs on the server" value={<CountUp value={jobs.length} />} />
      <Stat label="Released" value={<CountUp value={count("released")} />} hint="every check passed" />
      <Stat label="Quarantined" value={<CountUp value={count("quarantined")} />} hint="held back by verification" />
      <Stat label="Pages processed" value={<CountUp value={pages} />} />
    </div>
  );
}

const TH = "px-5 py-3.5 text-[0.6875rem] font-medium tracking-wider uppercase";

export function JobsTable() {
  const router = useRouter();
  const { data, error } = usePolling<{ jobs: Row[] }>("/api/jobs", 2000, (d) =>
    d.jobs.some((j) => ACTIVE_STATES.has(j.state)),
  );
  const jobs = data?.jobs ?? [];
  const now = useNow(jobs.some((j) => j.state === "running"));

  if (!data) {
    return (
      <Card>
        {error ? (
          <Notice tone="crit" title="Could not load jobs" className="m-4">
            {error.code}
          </Notice>
        ) : (
          <div className="flex items-center justify-center gap-2 py-20 text-sm text-muted">
            <Spinner /> Loading jobs…
          </div>
        )}
      </Card>
    );
  }

  if (jobs.length === 0) {
    return (
      <Card>
        <EmptyState icon={<Inbox className="size-8" strokeWidth={1.5} />} title="No jobs yet">
          Jobs you start appear here with their verification result and artefacts.
          <div className="mt-6">
            <ButtonLink href="/redact" variant="primary">
              New redaction
            </ButtonLink>
          </div>
        </EmptyState>
      </Card>
    );
  }

  return (
    <>
      <Totals jobs={jobs} />
      <Card data-reveal className="overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full min-w-4xl text-sm">
            <thead>
              <tr className="border-b border-line text-left text-muted">
                <th scope="col" className={TH}>Status</th>
                <th scope="col" className={TH}>Job</th>
                <th scope="col" className={TH}>Submitted</th>
                <th scope="col" className={TH}>Profile · style</th>
                <th scope="col" className={`${TH} text-right`}>Pages</th>
                <th scope="col" className={`${TH} text-right`}>Replaced</th>
                <th scope="col" className={`${TH} text-right`}>Regions</th>
                <th scope="col" className={`${TH} text-right`}>Checks</th>
                <th scope="col" className={`${TH} text-right`}>Run time</th>
                <th scope="col" className="w-10 px-3 py-3.5"><span className="sr-only">Open</span></th>
              </tr>
            </thead>
            <tbody data-stagger className="tabular">
              {jobs.map((job) => (
                <tr
                  key={job.id}
                  onClick={() => router.push(`/jobs/${job.id}`)}
                  className="group cursor-pointer border-b border-line transition-colors last:border-b-0 hover:bg-surface-2"
                >
                  <td className="px-5 py-4">
                    <StatusBadge state={job.state} />
                  </td>
                  <td className="px-5 py-4">
                    <Link
                      href={`/jobs/${job.id}`}
                      className="font-mono text-[0.8125rem] font-medium text-ink underline-offset-4 hover:underline"
                      onClick={(e) => e.stopPropagation()}
                    >
                      {shortId(job.id)}
                    </Link>
                    <div className="mt-0.5 text-xs text-muted" title={job.input ? `SHA-256 ${job.input.sha256}` : undefined}>
                      {job.input ? (
                        <>
                          <Mono className="text-xs">{shortHash(job.input.sha256, 8)}</Mono> · {formatBytes(job.input.bytes)}
                        </>
                      ) : (
                        "awaiting upload"
                      )}
                    </div>
                  </td>
                  <td className="px-5 py-4 whitespace-nowrap text-ink-2" title={formatDate(job.createdAt)}>
                    {formatRelative(job.createdAt, now)}
                  </td>
                  <td className="px-5 py-4 whitespace-nowrap">
                    <span className="font-serif text-base text-ink">
                      {PROFILE_INFO[job.options.profile]?.title ?? job.options.profile}
                    </span>
                    <span className="text-muted"> · {STYLE_INFO[job.options.style].title}</span>
                  </td>
                  <td className="px-5 py-4 text-right">{job.summary?.pages ?? "–"}</td>
                  <td className="px-5 py-4 text-right">{job.summary?.replaced ?? "–"}</td>
                  <td className="px-5 py-4 text-right">{job.summary?.regions ?? "–"}</td>
                  <td className="px-5 py-4 text-right">
                    <Checks job={job} />
                  </td>
                  <td className="px-5 py-4 text-right whitespace-nowrap text-ink-2">
                    <Duration job={job} now={now} />
                  </td>
                  <td className="px-3 py-4 text-muted">
                    <ArrowRight className="size-4 transition-transform group-hover:translate-x-0.5 group-hover:text-ink" aria-hidden />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </>
  );
}
