"use client";

import { AlertTriangle, ArrowRight, ChevronRight, KeyRound, Plus, Trash2, Upload } from "lucide-react";
import { useRouter } from "next/navigation";
import { useState } from "react";

import { ApiError, send, uploadFile, usePolling } from "@/lib/api";
import { formatBytes } from "@/lib/format";
import { useHydrated } from "@/lib/hydrated";
import type { CreatedJob, SystemInfo } from "@/lib/jobs";
import { ENTITY_TYPES, type EntityType, RENDER_STYLES, type RenderStyle } from "@/lib/manifest";
import { ENTITY_LABELS, PROFILE_INFO, STYLE_INFO } from "@/lib/vocab";

import { Mark } from "./brand";
import { FileDrop } from "./file-drop";
import { Button, cx, Notice, Spinner } from "./ui";
import { Aurora, Flow, GlassChip } from "./visuals";

// The engine's default profile (the client's policy), listed first.
const DEFAULT_PROFILE = "client";
const FALLBACK_PROFILES = ["client", "broad", "claimant_focused", "maximal"];

const SUBMIT_ERRORS: Record<string, string> = {
  E_API_UPLOAD_TOO_LARGE: "The document is larger than the server's upload limit.",
  E_API_EMPTY_UPLOAD: "That file is empty.",
  E_API_UPLOAD_SIZE_MISMATCH: "The upload did not match the file's size. The file may have changed; try again.",
  E_API_BAD_PROFILE: "The engine does not know that profile.",
  E_API_BAD_REQUEST:
    "The server refused these options. Check the known identifiers (type and value, up to 500 characters), the scope ID and the evaluator key.",
  E_API_DEV_KEY_DISABLED: "The development key is disabled on this server.",
  E_API_LINK_EXPIRED: "The upload link expired before the upload started. Try again.",
  E_UI_BACKEND_UNREACHABLE: "The redaction server could not be reached.",
  E_UI_UPLOAD_NETWORK:
    "The upload to the redaction server failed. Check the connection, and that the server allows this site's address.",
  E_UI_NETWORK: "This app's server could not be reached.",
};

const FACTS = ["The filename stays in your browser", "The input is deleted when the job ends", "Jobs run one at a time"];

type SeedRow = { type: EntityType; value: string };

function Step({ n, title, hint, children }: { n: number; title: string; hint?: string; children: React.ReactNode }) {
  return (
    <fieldset className="bg-surface px-6 py-7 sm:px-8">
      <legend className="sr-only">{title}</legend>
      <div className="flex items-baseline gap-5">
        <span className="text-xs text-muted tabular">0{n}</span>
        <div className="min-w-0">
          <div className="font-serif text-[1.625rem] leading-tight text-ink">{title}</div>
          {hint ? <p className="mt-1.5 text-sm text-muted">{hint}</p> : null}
        </div>
      </div>
      <div className="mt-6 sm:pl-9">{children}</div>
    </fieldset>
  );
}

/** A radio drawn as a row: selected rows invert and open their summary, like an accordion. */
function ChoiceRow({
  name,
  value,
  checked,
  onSelect,
  title,
  summary,
  code,
}: {
  name: string;
  value: string;
  checked: boolean;
  onSelect: () => void;
  title: string;
  summary?: string;
  code?: string;
}) {
  return (
    <label
      className={cx(
        "choice group flex cursor-pointer items-start gap-4 px-5 py-4 transition-colors duration-300",
        checked ? "bg-ink text-page" : "bg-surface-2 text-ink hover:bg-surface-3",
      )}
    >
      <input type="radio" name={name} value={value} checked={checked} onChange={onSelect} className="sr-only" />
      <span className="min-w-0 flex-1">
        <span className="flex items-baseline justify-between gap-3">
          <span className="font-serif text-xl leading-tight">{title}</span>
          {code ? <span className={cx("font-mono text-[0.6875rem]", checked ? "text-page/55" : "text-muted")}>{code}</span> : null}
        </span>
        {summary ? (
          <span
            className={cx("mt-1.5 text-sm leading-relaxed", checked ? "block text-page/75" : "line-clamp-1 text-muted")}
          >
            {summary}
          </span>
        ) : null}
      </span>
      <ArrowRight
        className={cx(
          "mt-1.5 size-4 shrink-0 transition-transform duration-300 group-hover:translate-x-1",
          checked ? "translate-x-1" : "text-muted",
        )}
        aria-hidden
      />
    </label>
  );
}

const inputClass =
  "h-11 w-full border border-line-strong bg-surface-2 px-3.5 text-sm text-ink placeholder:text-muted focus:border-ink focus:outline-none";

export function NewJobForm() {
  const router = useRouter();
  const { data: system } = usePolling<SystemInfo>("/api/system", 5000, () => true);
  const [file, setFile] = useState<File | null>(null);
  const [fileError, setFileError] = useState<string | null>(null);
  const [profile, setProfile] = useState(DEFAULT_PROFILE);
  const [style, setStyle] = useState<RenderStyle>("surrogate");
  const [scopeId, setScopeId] = useState("");
  const [workers, setWorkers] = useState<string>("");
  const [devKey, setDevKey] = useState(false);
  const [seeds, setSeeds] = useState<SeedRow[]>([]);
  const [evaluatorKey, setEvaluatorKey] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [progress, setProgress] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const hydrated = useHydrated();

  const profiles = [...(system?.profiles ?? FALLBACK_PROFILES)].sort(
    (a, b) => Number(b === DEFAULT_PROFILE) - Number(a === DEFAULT_PROFILE),
  );
  const maxBytes = (system?.defaults.maxUploadMb ?? 200) * 1024 * 1024;
  const keyConfigured = system?.surrogateKey.fromEnv ?? false;
  const keyHex = evaluatorKey.trim();
  const keyValid = keyHex === "" || /^[0-9a-fA-F]{64}$/.test(keyHex);
  const seedsValid = seeds.every((s) => s.value.trim().length > 0 && s.value.length <= 500);
  const canSubmit = Boolean(file) && keyValid && seedsValid && !submitting;

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!file || !canSubmit) return;
    setSubmitting(true);
    setError(null);
    let created: CreatedJob | null = null;
    try {
      // 1. Create the job (small JSON through this app's server, which holds the API token).
      created = await send<CreatedJob>("/api/jobs", "POST", {
        profile,
        style,
        ...(workers ? { workers: Number(workers) } : {}),
        ...(scopeId.trim() ? { scopeId: scopeId.trim() } : {}),
        ...(devKey && style === "surrogate" ? { devKey: true } : {}),
        ...(seeds.length ? { seeds: seeds.map((s) => ({ type: s.type, value: s.value.trim() })) } : {}),
        ...(keyHex ? { evaluatorKey: keyHex } : {}),
        inputBytes: file.size,
      });
      // 2. Send the document straight to the redaction server: any size, and only the bytes,
      //    never the filename.
      setProgress(0);
      await uploadFile(created.uploadUrl, file, setProgress);
      router.push(`/jobs/${created.id}`);
    } catch (err) {
      // A job whose upload failed would only wait for its link to expire: discard it now. Best
      // effort, as the server sweeps it away later anyway.
      if (created) send(`/api/jobs/${created.id}`, "DELETE").catch(() => {});
      const code = err instanceof ApiError ? err.code : "E_UI_NETWORK";
      setError(SUBMIT_ERRORS[code] ?? `The server refused the job (${code}).`);
      setSubmitting(false);
      setProgress(null);
    }
  }

  // The summary never shows the filename: only its kind and size.
  const docLabel = file ? `${/\.pdf$/i.test(file.name) ? "PDF" : "Scan"} · ${formatBytes(file.size)}` : "Document";

  return (
    // data-hydrated: the end-to-end tests wait for it before choosing a file.
    <form
      onSubmit={submit}
      noValidate
      className="grid items-start gap-2 lg:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)]"
      data-hydrated={hydrated ? "" : undefined}
    >
      <div data-stagger className="space-y-2">
        <Step n={1} title="Document" hint="A medico-legal PDF, or a scanned page as TIFF, PNG or JPEG.">
          <FileDrop
            file={file}
            maxBytes={maxBytes}
            disabled={submitting}
            onChange={(f, err) => {
              setFile(f);
              setFileError(err ?? null);
            }}
          />
          {fileError ? <p className="mt-2 text-sm text-crit-text">{fileError}</p> : null}
          <p className="mt-3 text-xs text-muted">
            The filename stays in your browser. The server stores the document under a fixed name and deletes it as soon
            as the job ends.
          </p>
        </Step>

        <Step n={2} title="Policy profile" hint="Decides which entity types are replaced and which stay visible.">
          <div className="space-y-1.5">
            {profiles.map((p) => (
              <ChoiceRow
                key={p}
                name="profile"
                value={p}
                checked={profile === p}
                onSelect={() => setProfile(p)}
                title={PROFILE_INFO[p]?.title ?? p}
                summary={PROFILE_INFO[p]?.summary}
                code={p}
              />
            ))}
          </div>
        </Step>

        <Step n={3} title="Render style" hint="How replaced text is drawn on the rebuilt page.">
          <div className="space-y-1.5">
            {RENDER_STYLES.map((s) => (
              <ChoiceRow
                key={s}
                name="style"
                value={s}
                checked={style === s}
                onSelect={() => setStyle(s)}
                title={STYLE_INFO[s].title}
                summary={STYLE_INFO[s].summary}
              />
            ))}
          </div>
          {style === "surrogate" && system && !keyConfigured && !devKey ? (
            <Notice tone="info" icon={<KeyRound className="size-4" />} title="Surrogates need a key" className="mt-4">
              MLREDACT_SURROGATE_KEY is not set on the redaction server. The job will only run if a deployment config sets{" "}
              <code className="font-mono text-xs">surrogate.key_file</code>
              {system.surrogateKey.devKeyAllowed ? ", or if you turn on the development key under Advanced." : "."}
            </Notice>
          ) : null}
        </Step>

        <Step
          n={4}
          title="Surrogate scope"
          hint="Optional. Documents sharing a scope, such as a matter or claim ID, get the same surrogate for the same value."
        >
          <input
            className={cx(inputClass, "max-w-md")}
            value={scopeId}
            onChange={(e) => setScopeId(e.target.value)}
            placeholder="e.g. MATTER-2026-0142"
            maxLength={200}
            autoComplete="off"
            spellCheck={false}
            aria-label="Scope ID"
          />
          <p className="mt-3 text-xs text-muted">
            Passed to the engine only. Neither the UI nor the manifest stores it; the manifest records just that a caller
            scope was used.
          </p>
        </Step>

        <details className="group bg-surface">
          <summary className="flex cursor-pointer list-none items-center gap-5 px-6 py-6 select-none hover:bg-surface-2 sm:px-8 [&::-webkit-details-marker]:hidden">
            <span className="text-xs text-muted tabular">05</span>
            <span className="font-serif text-[1.625rem] leading-tight text-ink">Advanced</span>
            <span className="hidden text-sm text-muted sm:inline">workers, known identifiers, evaluator key</span>
            <ChevronRight className="ml-auto size-4 text-muted transition-transform group-open:rotate-90" aria-hidden />
          </summary>
          <div className="space-y-8 px-6 pt-2 pb-8 sm:px-8 sm:pl-17">
            <div className="grid gap-4 sm:grid-cols-[10rem_1fr] sm:items-start">
              <label htmlFor="workers" className="pt-3 text-sm font-medium text-ink">
                Worker processes
              </label>
              <div>
                <input
                  id="workers"
                  type="number"
                  min={1}
                  max={64}
                  className={cx(inputClass, "w-28")}
                  value={workers}
                  placeholder={String(system?.defaults.workers ?? 2)}
                  onChange={(e) => setWorkers(e.target.value)}
                />
                <p className="mt-2 text-xs text-muted">
                  Page OCR runs in parallel workers, each holding about 0.4 GB. Output is identical for any worker count.
                </p>
              </div>
            </div>

            {system?.surrogateKey.devKeyAllowed ? (
              <div className="grid gap-4 sm:grid-cols-[10rem_1fr] sm:items-start">
                <span className="text-sm font-medium text-ink">Development key</span>
                <label className="flex items-start gap-3 text-sm text-ink-2">
                  <input
                    type="checkbox"
                    className="check mt-0.5"
                    checked={devKey}
                    onChange={(e) => setDevKey(e.target.checked)}
                    disabled={style !== "surrogate"}
                  />
                  <span>
                    Use the public development surrogate key.{" "}
                    <span className="font-medium text-warn-text">Local experiments with synthetic documents only.</span>{" "}
                    Anyone can reverse surrogates made with it, and the engine refuses it when a production runtime
                    profile is pinned.
                  </span>
                </label>
              </div>
            ) : null}

            <div className="grid gap-4 sm:grid-cols-[10rem_1fr] sm:items-start">
              <span className="pt-3 text-sm font-medium text-ink">Known identifiers</span>
              <div className="space-y-2">
                <p className="text-xs text-muted">
                  Values the caller already knows, such as the claimant&apos;s name or Medicare number. Each is searched for
                  across the whole document as strong evidence. They are deleted with the input when the job ends.
                </p>
                {seeds.map((seed, i) => (
                  <div key={i} className="flex gap-2">
                    <select
                      className={cx(inputClass, "w-44 shrink-0 px-2")}
                      value={seed.type}
                      aria-label={`Identifier ${i + 1} type`}
                      onChange={(e) =>
                        setSeeds((rows) => rows.map((r, j) => (j === i ? { ...r, type: e.target.value as EntityType } : r)))
                      }
                    >
                      {ENTITY_TYPES.map((t) => (
                        <option key={t} value={t}>
                          {ENTITY_LABELS[t]}
                        </option>
                      ))}
                    </select>
                    <input
                      className={inputClass}
                      value={seed.value}
                      maxLength={500}
                      autoComplete="off"
                      spellCheck={false}
                      aria-label={`Identifier ${i + 1} value`}
                      placeholder="Value"
                      onChange={(e) =>
                        setSeeds((rows) => rows.map((r, j) => (j === i ? { ...r, value: e.target.value } : r)))
                      }
                    />
                    <Button
                      variant="ghost"
                      className="h-11 w-11 shrink-0 px-0"
                      aria-label={`Remove identifier ${i + 1}`}
                      onClick={() => setSeeds((rows) => rows.filter((_, j) => j !== i))}
                    >
                      <Trash2 className="size-4" aria-hidden />
                    </Button>
                  </div>
                ))}
                <Button variant="secondary" onClick={() => setSeeds((rows) => [...rows, { type: "person", value: "" }])}>
                  <Plus className="size-4" aria-hidden />
                  Add identifier
                </Button>
              </div>
            </div>

            <div className="grid gap-4 sm:grid-cols-[10rem_1fr] sm:items-start">
              <label htmlFor="evaluator" className="pt-3 text-sm font-medium text-ink">
                Evaluator public key
              </label>
              <div>
                <div className="flex gap-2">
                  <input
                    id="evaluator"
                    className={cx(inputClass, "font-mono text-xs", !keyValid && "border-crit focus:border-crit")}
                    value={evaluatorKey}
                    onChange={(e) => setEvaluatorKey(e.target.value)}
                    placeholder="64 hex characters from evaluator.pub"
                    autoComplete="off"
                    spellCheck={false}
                    aria-invalid={!keyValid}
                  />
                  <label className="choice inline-flex h-11 shrink-0 cursor-pointer items-center gap-2 bg-surface-2 px-4 text-sm font-medium text-ink hover:bg-surface-3">
                    <Upload className="size-4" aria-hidden />
                    Load file
                    <input
                      type="file"
                      accept=".pub,.txt"
                      className="sr-only"
                      onChange={async (e) => {
                        const f = e.target.files?.[0];
                        if (f && f.size < 4096) setEvaluatorKey((await f.text()).trim());
                        e.target.value = "";
                      }}
                    />
                  </label>
                </div>
                <p className={cx("mt-2 text-xs", keyValid ? "text-muted" : "text-crit-text")}>
                  {keyValid
                    ? "Optional. Adds a sensitive manifest with the original text, sealed to this key (X25519). Only the evaluator's private key can open it."
                    : "Must be exactly 64 hexadecimal characters."}
                </p>
              </div>
            </div>
          </div>
        </details>
      </div>

      <aside data-stagger className="flex flex-col gap-2 lg:sticky lg:top-24">
        <div className="relative grid min-h-80 place-items-center overflow-hidden px-6 py-16">
          <Aurora tone="rose" />
          <Flow
            className="relative"
            steps={[docLabel, PROFILE_INFO[profile]?.title ?? profile, STYLE_INFO[style].title, "V1–V6"]}
          />
          <GlassChip className="absolute bottom-5 left-5">Released only if every check passes</GlassChip>
        </div>
        <ul className="grid gap-2 sm:grid-cols-3">
          {FACTS.map((f) => (
            <li key={f} className="flex gap-3 bg-surface px-5 py-4 text-sm leading-snug text-ink-2">
              <Mark className="mt-0.5 size-3.5 text-muted" />
              {f}
            </li>
          ))}
        </ul>
        <div className="bg-surface px-6 py-6">
          <div className="min-h-10 text-sm" aria-live="polite">
            {error ? (
              <span className="inline-flex items-start gap-1.5 text-crit-text" role="alert">
                <AlertTriangle className="mt-0.5 size-4 shrink-0" aria-hidden />
                {error}
              </span>
            ) : system?.queue.running ? (
              <span className="text-muted">
                A job is running{system.queue.queued ? ` and ${system.queue.queued} queued` : ""}. Yours will wait its
                turn.
              </span>
            ) : file ? (
              <span className="text-ink-2">
                {system ? "Ready. Nothing else is running, so this job starts at once." : "Ready to start."}
              </span>
            ) : (
              <span className="text-muted">Choose a document to begin. Jobs run one at a time on the redaction server.</span>
            )}
          </div>
          <Button type="submit" variant="primary" size="lg" disabled={!canSubmit} className="mt-4 w-full">
            {submitting ? <Spinner /> : null}
            {!submitting
              ? "Start redaction"
              : progress === null
                ? "Creating job…"
                : progress < 1
                  ? `Uploading ${Math.round(progress * 100)}%`
                  : "Starting…"}
          </Button>
          {progress !== null ? (
            <div className="mt-3 h-1 bg-surface-3" aria-hidden>
              <div className="h-full bg-ink transition-[width] duration-200" style={{ width: `${Math.round(progress * 100)}%` }} />
            </div>
          ) : null}
        </div>
      </aside>
    </form>
  );
}
