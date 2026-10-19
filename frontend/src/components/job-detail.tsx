"use client";

import {
  AlertOctagon,
  ArrowLeft,
  Ban,
  Download,
  FileJson,
  FileLock2,
  FileText,
  ShieldAlert,
  Square,
  Trash2,
} from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";

import { ApiError, downloadUrl, freshFileUrl, send, useNow, usePolling } from "@/lib/api";
import { formatBytes, formatClock, formatDate, formatDuration, shortHash, shortId } from "@/lib/format";
import { ACTIVE_STATES, type JobView, type OutputFile } from "@/lib/jobs";
import type { Manifest } from "@/lib/manifest";
import { PROFILE_INFO, reasonText, STYLE_INFO } from "@/lib/vocab";

import { CopyButton } from "./copy-button";
import { DocumentViewer } from "./document-viewer";
import { OverviewPanel } from "./overview-panel";
import { PagesPanel } from "./pages-panel";
import { ProvenancePanel } from "./provenance-panel";
import { STATE_TONE } from "./recent-jobs";
import { StatusBadge } from "./status-badge";
import { Button, ButtonAnchor, Card, cx, EmptyState, Mono, Notice, Spinner, Stat } from "./ui";
import { VerificationPanel } from "./verification-panel";
import { Aurora } from "./visuals";

const TABS = [
  { id: "overview", label: "Overview" },
  { id: "document", label: "Document" },
  { id: "verification", label: "Verification" },
  { id: "pages", label: "Pages" },
  { id: "provenance", label: "Provenance" },
] as const;
type Tab = (typeof TABS)[number]["id"];

const ARTEFACTS: { name: OutputFile; label: string; icon: React.ReactNode }[] = [
  { name: "manifest.json", label: "Manifest", icon: <FileJson className="size-4" aria-hidden /> },
  { name: "run.json", label: "Run record", icon: <FileJson className="size-4" aria-hidden /> },
  { name: "quarantine.json", label: "Quarantine record", icon: <ShieldAlert className="size-4" aria-hidden /> },
  { name: "sensitive-manifest.sealed.json", label: "Sealed manifest", icon: <FileLock2 className="size-4" aria-hidden /> },
];

function Tabs({ tab, onChange }: { tab: Tab; onChange: (t: Tab) => void }) {
  const refs = useRef<(HTMLButtonElement | null)[]>([]);
  return (
    <div role="tablist" aria-label="Job details" data-stagger className="flex gap-2 overflow-x-auto">
      {TABS.map((t, i) => (
        <button
          key={t.id}
          ref={(el) => {
            refs.current[i] = el;
          }}
          role="tab"
          id={`tab-${t.id}`}
          aria-selected={tab === t.id}
          aria-controls={`panel-${t.id}`}
          tabIndex={tab === t.id ? 0 : -1}
          onClick={() => onChange(t.id)}
          onKeyDown={(e) => {
            if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
            const next = (i + (e.key === "ArrowRight" ? 1 : TABS.length - 1)) % TABS.length;
            onChange(TABS[next]!.id);
            refs.current[next]?.focus();
          }}
          className={cx(
            "flex min-w-36 flex-1 flex-col items-start px-5 pt-4 text-left transition-colors",
            tab === t.id ? "bg-surface-2" : "bg-surface hover:bg-surface-2",
          )}
        >
          <span className="text-xs text-muted tabular">0{i + 1}</span>
          <span className="mt-1 font-serif text-xl leading-tight whitespace-nowrap text-ink">
            {t.label}
          </span>
          <span className="relative mt-4 mb-4 h-px w-full bg-line-strong" aria-hidden>
            <span
              className={cx(
                "absolute -top-px left-0 h-0.5 bg-ink transition-[width] duration-500",
                tab === t.id ? "w-full" : "w-0",
              )}
            />
          </span>
        </button>
      ))}
    </div>
  );
}

/**
 * The manifest comes straight from the backend through a fresh signed link: a long record's
 * manifest can exceed what a serverless host relays (Vercel: 4.5 MB).
 */
function useManifest(id: string, available: boolean): { manifest: Manifest | null; loading: boolean } {
  const [state, setState] = useState<{ manifest: Manifest | null; loading: boolean }>({ manifest: null, loading: false });
  useEffect(() => {
    if (!available) return;
    let cancelled = false;
    setState((s) => ({ ...s, loading: true }));
    (async () => {
      try {
        const res = await fetch(await freshFileUrl(id, "manifest.json"), { cache: "no-store" });
        if (!res.ok) throw new Error("manifest");
        const manifest = (await res.json()) as Manifest;
        if (!cancelled) setState({ manifest, loading: false });
      } catch {
        if (!cancelled) setState({ manifest: null, loading: false });
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [id, available]);
  return state;
}

function Progress({ view, onCancel, cancelling }: { view: JobView; onCancel: () => void; cancelling: boolean }) {
  const { job } = view;
  const running = job.state === "running";
  const now = useNow(running);
  const elapsed = running && job.startedAt ? now - new Date(job.startedAt).getTime() : 0;
  return (
    <section className="relative overflow-hidden">
      <Aurora tone="rose" />
      <div className="absolute inset-x-0 top-0 h-0.5 overflow-hidden bg-white/15" aria-hidden>
        <div className={cx("h-full w-2/5 bg-white", running ? "indeterminate" : "opacity-0")} />
      </div>
      <div className="relative flex flex-wrap items-center gap-x-12 gap-y-6 px-8 py-12 text-white sm:px-12">
        <div>
          <div className="text-xs text-white/70">
            {running ? "Elapsed" : job.state === "awaiting_upload" ? "Waiting for the upload" : "Position in queue"}
          </div>
          <div className="display tabular mt-2 text-[clamp(3.5rem,7vw,5.5rem)] leading-none" aria-live="polite">
            {running
              ? formatClock(elapsed)
              : job.state === "awaiting_upload"
                ? "…"
                : view.queuePosition == null
                  ? "Next"
                  : `#${view.queuePosition + 1}`}
          </div>
        </div>
        <div className="min-w-64 flex-1 text-[0.9375rem] leading-relaxed text-white/85">
          {running ? (
            <>
              <p>
                The engine is inspecting, analysing, detecting, rendering and verifying this document. This page updates
                when it finishes.
              </p>
              <p className="mt-2 text-sm text-white/65">
                Each job starts its own engine, so expect model loading first (about 35 s with NER), then roughly
                20–40 s per page on CPU.
              </p>
            </>
          ) : (
            <p>Jobs run one at a time. This one starts when the jobs ahead of it finish.</p>
          )}
        </div>
        <button
          type="button"
          onClick={onCancel}
          disabled={cancelling}
          className="inline-flex h-12 items-center gap-2 border border-white/15 bg-black/35 px-5 text-sm font-medium text-white backdrop-blur-md transition-colors hover:bg-black/55 disabled:opacity-50"
        >
          {cancelling ? <Spinner /> : <Square className="size-3.5" aria-hidden />}
          {running ? "Stop job" : job.state === "awaiting_upload" ? "Discard" : "Cancel"}
        </button>
      </div>
    </section>
  );
}

export function JobDetail({ id }: { id: string }) {
  const router = useRouter();
  const { data, error, refresh } = usePolling<JobView>(`/api/jobs/${id}`, 1500, (d) => ACTIVE_STATES.has(d.job.state));
  const [tab, setTab] = useState<Tab | null>(null);
  const [busy, setBusy] = useState<"cancel" | "delete" | null>(null);
  const [actionError, setActionError] = useState<{ action: "cancel" | "delete"; code: string } | null>(null);
  // The viewer stays mounted once opened, so switching tabs never downloads the PDF again.
  const [documentOpened, setDocumentOpened] = useState(false);
  const files = data?.files ?? [];
  const { manifest, loading: manifestLoading } = useManifest(id, files.includes("manifest.json"));
  // A fresh signed link each time the viewer opens, so a page left open never shows an expired one.
  const pdfSource = useCallback(() => freshFileUrl(id, "redacted.pdf"), [id]);

  useEffect(() => {
    const fromHash = window.location.hash.slice(1);
    if (TABS.some((t) => t.id === fromHash)) setTab(fromHash as Tab);
  }, []);

  const changeTab = (t: Tab) => {
    setTab(t);
    window.history.replaceState(null, "", `#${t}`);
  };

  if (!data) {
    if (error?.status === 404) {
      return (
        <div className="pt-16">
          <Card>
            <EmptyState title="Job not found">
              It may have been deleted.{" "}
              <Link href="/jobs" className="text-ink underline underline-offset-4">
                Back to jobs
              </Link>
            </EmptyState>
          </Card>
        </div>
      );
    }
    return (
      <div className="flex items-center justify-center gap-2 py-32 text-sm text-muted">
        {error ? `Could not load the job (${error.code}). Retrying…` : <><Spinner /> Loading job…</>}
      </div>
    );
  }

  const { job } = data;
  const run = data.run ?? null;
  const active = ACTIVE_STATES.has(job.state);
  const released = job.state === "released";
  const activeTab: Tab = tab ?? (released ? "document" : "overview");
  if (activeTab === "document" && !documentOpened) setDocumentOpened(true);
  const has = (name: OutputFile) => files.includes(name);
  const o = job.options;

  async function cancel() {
    setBusy("cancel");
    setActionError(null);
    try {
      await send(`/api/jobs/${id}/cancel`, "POST");
    } catch (err) {
      setActionError({ action: "cancel", code: err instanceof ApiError ? err.code : "E_UI_NETWORK" });
    } finally {
      setBusy(null);
      refresh();
    }
  }

  async function remove() {
    if (!window.confirm("Delete this job and all of its artefacts? This cannot be undone.")) return;
    setBusy("delete");
    setActionError(null);
    try {
      await send(`/api/jobs/${id}`, "DELETE");
      router.push("/jobs");
    } catch (err) {
      setActionError({ action: "delete", code: err instanceof ApiError ? err.code : "E_UI_NETWORK" });
      setBusy(null);
      refresh();
    }
  }

  const meta: React.ReactNode[] = [
    <span key="d">Submitted {formatDate(job.createdAt)}</span>,
    <span key="p">
      {PROFILE_INFO[o.profile]?.title ?? o.profile} · {STYLE_INFO[o.style].title}
    </span>,
  ];
  if (job.input) {
    meta.push(
      <span key="s">{formatBytes(job.input.bytes)}</span>,
      <span key="h" className="inline-flex items-center gap-1">
        SHA-256 <Mono title={job.input.sha256}>{shortHash(job.input.sha256)}</Mono>
        <CopyButton value={job.input.sha256} label="Copy input SHA-256" />
      </span>,
    );
  }
  if (o.scope === "caller") meta.push(<span key="sc">Caller scope</span>);
  if (o.seedCount) meta.push(<span key="se">{o.seedCount} known identifier{o.seedCount === 1 ? "" : "s"}</span>);
  if (o.devKey) meta.push(<span key="dk" className="font-medium text-warn-text">Development key</span>);

  const reasons = manifest?.reasons ?? job.summary?.reasons ?? [];
  const summary = job.summary;
  // Rejected at intake (encrypted, unsupported…): nothing was analysed, so only provenance is meaningful.
  const analysed = Boolean(manifest && manifest.pages.length > 0);
  const missingKey = job.errorCode === "E_CONFIG_INVALID" && o.style === "surrogate" && !o.devKey;

  return (
    <div className="space-y-8 pt-10 sm:pt-14">
      <header data-stagger>
        <Link
          href="/jobs"
          className="inline-flex items-center gap-1.5 text-sm text-muted transition-colors hover:text-ink"
        >
          <ArrowLeft className="size-4" aria-hidden /> All jobs
        </Link>
        <div className="mt-8 flex flex-wrap items-end justify-between gap-6">
          <div className="min-w-0">
            <StatusBadge state={job.state} size="lg" />
            <h1 className="display mt-5 text-[clamp(2.75rem,5.4vw,4.5rem)] text-ink">
              Job <span className="font-mono text-[0.6em] tracking-[-0.03em]">{shortId(job.id)}</span>
            </h1>
            <div className="mt-4 flex flex-wrap items-center gap-x-3 gap-y-1 text-sm text-muted [&>*:not(:last-child)]:after:ml-3 [&>*:not(:last-child)]:after:text-line-strong [&>*:not(:last-child)]:after:content-['·']">
              {meta}
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            {has("redacted.pdf") ? (
              <ButtonAnchor href={downloadUrl(id, "redacted.pdf")} variant="primary" size="lg">
                <Download className="size-4" aria-hidden />
                Redacted PDF
              </ButtonAnchor>
            ) : null}
            {ARTEFACTS.filter((a) => has(a.name)).map((a) => (
              <ButtonAnchor key={a.name} href={downloadUrl(id, a.name)} size="lg">
                {a.icon}
                {a.label}
              </ButtonAnchor>
            ))}
            {!active ? (
              <Button variant="danger" size="lg" onClick={remove} disabled={busy !== null} aria-label="Delete job">
                {busy === "delete" ? <Spinner /> : <Trash2 className="size-4" aria-hidden />}
                <span className="hidden sm:inline">Delete</span>
              </Button>
            ) : null}
          </div>
        </div>
      </header>

      {active ? (
        // A job still waiting for its upload has nothing to stop: it can only be discarded.
        <Progress
          view={data}
          onCancel={job.state === "awaiting_upload" ? remove : cancel}
          cancelling={busy !== null}
        />
      ) : null}

      {actionError ? (
        <Notice
          tone="warn"
          icon={<AlertOctagon className="size-4" />}
          title={actionError.action === "delete" ? "The job could not be deleted" : "The job could not be stopped"}
        >
          <Mono className="font-medium text-ink">{actionError.code}</Mono> — {reasonText(actionError.code)}
        </Notice>
      ) : null}

      {job.state === "quarantined" ? (
        <Notice tone="warn" icon={<ShieldAlert className="size-4" />} title="Quarantined: no PDF was released">
          {reasons.length ? (
            <ul className="mt-1 space-y-1">
              {reasons.map((r) => (
                <li key={r}>
                  <Mono className="font-medium text-ink">{r}</Mono> — {reasonText(r)}
                </li>
              ))}
            </ul>
          ) : (
            "Verification failed."
          )}
          <p className="mt-2 text-xs text-muted">
            Reason codes are content-free by design.
            {analysed ? " The manifest below still shows what was detected and planned." : " The input was rejected before any page was analysed."}
          </p>
        </Notice>
      ) : null}

      {job.state === "failed" ? (
        <Notice tone="crit" icon={<AlertOctagon className="size-4" />} title="The engine could not complete this job">
          {job.errorCode ? (
            <>
              <Mono className="font-medium text-ink">{job.errorCode}</Mono> — {reasonText(job.errorCode)}
            </>
          ) : (
            "No reason code was reported."
          )}
          {missingKey ? (
            <p className="mt-2">
              This surrogate-style job most likely had no surrogate key. Set <Mono>MLREDACT_SURROGATE_KEY</Mono> on the
              redaction server, name <Mono>surrogate.key_file</Mono> in a deployment config, or choose the label or blackout
              style.
            </p>
          ) : null}
          <p className="mt-2">
            Check the environment on the{" "}
            <Link href="/system" className="font-medium text-ink underline underline-offset-4">
              System page
            </Link>
            , then submit the document again. The input was deleted when the job ended.
          </p>
        </Notice>
      ) : null}

      {job.state === "cancelled" || job.state === "interrupted" ? (
        <Notice
          tone="info"
          icon={<Ban className="size-4" />}
          title={job.state === "cancelled" ? "This job was cancelled" : "This job was interrupted"}
        >
          {job.state === "interrupted" ? reasonText("E_API_INTERRUPTED") : "It was stopped before the engine finished."}{" "}
          Its input has been deleted; submit the document again to rerun it.
        </Notice>
      ) : null}

      {manifest && !analysed ? <ProvenancePanel manifest={manifest} run={run} /> : null}

      {manifest && analysed ? (
        <>
          <div data-stagger className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-6">
            <Stat label="Pages" value={manifest.pages.length || manifest.job.page_count} />
            <Stat label="Replaced" value={summary?.replaced ?? "–"} hint="mentions acted on" />
            <Stat label="Kept" value={summary?.kept ?? "–"} hint="policy kept visible" />
            <Stat label="Regions removed" value={summary?.regions ?? "–"} hint="signatures, stamps, codes…" />
            <Stat
              label="Checks passed"
              value={summary && summary.checksTotal ? `${summary.checksPassed}/${summary.checksTotal}` : "–"}
            />
            <Stat label="Run time" value={formatDuration(run?.timings_ms.total_ms)} hint={run ? `${run.workers} workers` : undefined} />
          </div>

          <div>
            <Tabs tab={activeTab} onChange={changeTab} />
            {documentOpened ? (
              <div
                role="tabpanel"
                id="panel-document"
                aria-labelledby="tab-document"
                hidden={activeTab !== "document"}
                className="fade-up pt-2"
              >
                <DocumentViewer manifest={manifest} pdfUrl={has("redacted.pdf") ? pdfSource : null} />
              </div>
            ) : null}
            {activeTab !== "document" ? (
              <div
                key={activeTab}
                role="tabpanel"
                id={`panel-${activeTab}`}
                aria-labelledby={`tab-${activeTab}`}
                className="fade-up pt-2"
              >
                {activeTab === "overview" ? <OverviewPanel manifest={manifest} run={run} onOpenTab={changeTab} /> : null}
                {activeTab === "verification" ? <VerificationPanel manifest={manifest} /> : null}
                {activeTab === "pages" ? <PagesPanel manifest={manifest} /> : null}
                {activeTab === "provenance" ? <ProvenancePanel manifest={manifest} run={run} /> : null}
              </div>
            ) : null}
          </div>
        </>
      ) : manifestLoading ? (
        <div className="flex items-center justify-center gap-2 py-16 text-sm text-muted">
          <Spinner /> Loading the manifest…
        </div>
      ) : !manifest && !active && job.state !== "failed" && job.state !== "cancelled" && job.state !== "interrupted" ? (
        <Card>
          <EmptyState icon={<FileText className="size-8" strokeWidth={1.5} />} title="No manifest was written for this job" />
        </Card>
      ) : null}

      {!active && !manifest && !manifestLoading ? (
        <section className="relative grid min-h-56 place-items-center overflow-hidden px-6 py-12">
          <Aurora tone={STATE_TONE[job.state]} />
          <div className="relative flex flex-col items-center text-center text-white">
            <p className="font-serif text-[1.75rem]">Run it again from the start.</p>
            <p className="mt-2 text-sm text-white/70">The input was deleted when this job ended.</p>
            <Link
              href="/redact"
              className="mt-6 inline-flex h-13 items-center bg-white px-7 text-[0.9375rem] font-medium text-[#0e1214] transition-colors hover:bg-white/90"
            >
              New redaction
            </Link>
          </div>
        </section>
      ) : null}
    </div>
  );
}
