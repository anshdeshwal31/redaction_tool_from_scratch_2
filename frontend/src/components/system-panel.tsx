"use client";

import { CheckCircle2, KeyRound, Play, TerminalSquare, XCircle } from "lucide-react";
import { useState } from "react";

import { ApiError, send, usePolling } from "@/lib/api";
import { formatDuration } from "@/lib/format";
import type { SelfcheckReport, SystemInfo } from "@/lib/jobs";
import { PROFILE_INFO, reasonText } from "@/lib/vocab";

import { Hash } from "./copy-button";
import { Button, Card, CardHeader, cx, KeyValue, Mono, Notice, Pill, Spinner } from "./ui";

export function SystemPanel() {
  const { data: info, error } = usePolling<SystemInfo>("/api/system", 3000, () => true);
  const [profile, setProfile] = useState("client");
  const [check, setCheck] = useState<SelfcheckReport | null>(null);
  const [running, setRunning] = useState(false);

  async function runSelfcheck() {
    setRunning(true);
    try {
      setCheck(await send<SelfcheckReport>("/api/system/selfcheck", "POST", { profile }));
    } catch (err) {
      setCheck({ ok: false, errorCode: err instanceof ApiError ? err.code : "E_UI_NETWORK", durationMs: 0 });
    } finally {
      setRunning(false);
    }
  }

  if (!info) {
    return error ? (
      <Notice tone="crit" title="Could not reach the redaction server">
        {error.code}
      </Notice>
    ) : (
      <div className="flex items-center justify-center gap-2 py-24 text-sm text-muted">
        <Spinner /> Loading…
      </div>
    );
  }

  const r = check?.report;
  return (
    <div data-stagger className="grid gap-2 xl:grid-cols-2">
      <Card className="xl:col-span-2">
        <CardHeader
          title="Self-check"
          description="Runs mlredact selfcheck: configuration, runtime profile, pinned resources and model hashes, and engine binaries. Hashing every model can take a minute."
          actions={
            <>
              <select
                value={profile}
                onChange={(e) => setProfile(e.target.value)}
                className="h-10 border border-line-strong bg-surface-2 px-3 text-sm text-ink"
                aria-label="Profile to check"
              >
                {info.profiles.map((p) => (
                  <option key={p} value={p}>
                    {PROFILE_INFO[p]?.title ?? p}
                  </option>
                ))}
              </select>
              <Button variant="primary" onClick={runSelfcheck} disabled={running}>
                {running ? <Spinner /> : <Play className="size-4" aria-hidden />}
                {running ? "Checking…" : "Run self-check"}
              </Button>
            </>
          }
        />
        <div className="px-6 pb-6">
          {!check ? (
            <p className="text-sm text-muted">Not run yet in this session.</p>
          ) : check.ok && r ? (
            <div className="space-y-2">
              <Notice tone="good" icon={<CheckCircle2 className="size-4" />} title="The engine is ready">
                Checked in {formatDuration(check.durationMs)}: {r.resources_verified} resources and{" "}
                {r.models_verified.length} models verified.
              </Notice>
              <div className="grid gap-6 lg:grid-cols-2">
                <KeyValue
                  items={[
                    ["mlredact", <Mono key="v">{r.mlredact_version}</Mono>],
                    ["Profile", <Mono key="p">{r.profile}</Mono>],
                    ["Runtime profile", <Mono key="r">{r.runtime_profile}</Mono>],
                    ["Config hash", <Hash key="c" value={r.config_hash} />],
                    [
                      "Models",
                      <span key="m" className="flex flex-wrap gap-1">
                        {r.models_verified.map((m) => (
                          <Pill key={m} className="font-mono">
                            {m}
                          </Pill>
                        ))}
                      </span>,
                    ],
                  ]}
                />
                <dl className="grid grid-cols-2 content-start gap-x-6 gap-y-1 text-sm">
                  {Object.entries(r.libraries).map(([name, version]) => (
                    <div key={name} className="flex min-w-0 justify-between gap-2">
                      <dt className="truncate text-muted">{name}</dt>
                      <dd className="font-mono text-xs leading-5 text-ink">{version}</dd>
                    </div>
                  ))}
                </dl>
              </div>
              {r.runtime_profile.startsWith("windows") ? (
                <p className="text-xs text-muted">
                  Windows is a development platform, not a determinism target. Production runs pin a Linux runtime
                  profile.
                </p>
              ) : null}
            </div>
          ) : (
            <Notice tone="crit" icon={<XCircle className="size-4" />} title="The self-check failed">
              <Mono className="font-medium text-ink">{check.errorCode}</Mono> — {reasonText(check.errorCode ?? "")}
              {check.errorCode === "E_MODEL_MISSING" ? (
                <p className="mt-2">
                  Fetch the models with <Mono>uv run mlredact models fetch --profile {profile}</Mono>, or, on a laptop
                  without the NER models, point <Mono>MLREDACT_UI_CONFIG</Mono> at a deployment file that disables
                  them (lower recall: never for real documents).
                </p>
              ) : null}
            </Notice>
          )}
        </div>
      </Card>

      <Card>
        <CardHeader title="Engine" description="How the redaction server runs the engine. Set with MLREDACT_API_* variables on the server." />
        <div className="px-6 pb-6">
          <KeyValue
            items={[
              [
                "Command",
                <span key="c" className="inline-flex items-center gap-1.5">
                  <TerminalSquare className="size-4 text-muted" aria-hidden />
                  <Mono>{info.engine.command}</Mono>
                </span>,
              ],
              ["Working directory", <Mono key="w" className="break-all">{info.engine.cwd}</Mono>],
              [
                "Deployment configs",
                info.engine.configFiles.length ? (
                  <ul key="f" className="space-y-0.5">
                    {info.engine.configFiles.map((f) => (
                      <li key={f}>
                        <Mono className="break-all">{f}</Mono>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <span key="f" className="text-muted">None (built-in profile defaults)</span>
                ),
              ],
              ["Profiles", info.profiles.join(", ")],
              ["Jobs directory", <Mono key="j" className="break-all">{info.jobsDir}</Mono>],
            ]}
          />
        </div>
      </Card>

      <Card>
        <CardHeader title="Limits and queue" />
        <div className="space-y-4 px-6 pb-6">
          <KeyValue
            items={[
              ["Default workers", info.defaults.workers],
              ["Upload limit", `${info.defaults.maxUploadMb} MB`],
              ["Job time limit", `${info.defaults.jobTimeoutMin} min`],
              [
                "Queue",
                info.queue.running ? (
                  <span key="q">
                    1 running{info.queue.queued ? `, ${info.queue.queued} waiting` : ""}
                  </span>
                ) : (
                  <span key="q" className="text-muted">Idle</span>
                ),
              ],
            ]}
          />
          <div className="bg-surface-2 px-5 py-4">
            <div className="flex items-center gap-2 text-sm font-medium text-ink">
              <KeyRound className="size-4 text-muted" aria-hidden />
              Surrogate key
            </div>
            <ul className="mt-2 space-y-1 text-sm">
              <li className="flex items-start gap-2">
                <span className={cx("mt-1.5 size-2 shrink-0", info.surrogateKey.fromEnv ? "bg-good" : "bg-line-strong")} />
                {info.surrogateKey.fromEnv
                  ? "MLREDACT_SURROGATE_KEY is set for the engine."
                  : "MLREDACT_SURROGATE_KEY is not set (a deployment config may still name surrogate.key_file)."}
              </li>
              <li className="flex items-start gap-2">
                <span className={cx("mt-1.5 size-2 shrink-0", info.surrogateKey.devKeyAllowed ? "bg-warn" : "bg-line-strong")} />
                {info.surrogateKey.devKeyAllowed
                  ? "The public development key is offered in the form (local experiments only)."
                  : "The public development key is not offered."}
              </li>
            </ul>
            <p className="mt-2 text-xs text-muted">The key never passes through this interface; only its presence is shown.</p>
          </div>
        </div>
      </Card>
    </div>
  );
}
