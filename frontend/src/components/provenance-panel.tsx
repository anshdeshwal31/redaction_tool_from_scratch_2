import type { Manifest, RunRecord } from "@/lib/manifest";

import { Hash } from "./copy-button";
import { Card, CardHeader, KeyValue, Mono, Pill } from "./ui";

export function ProvenancePanel({ manifest, run }: { manifest: Manifest; run: RunRecord | null }) {
  const p = manifest.provenance;
  return (
    <div data-stagger className="grid gap-2 xl:grid-cols-2">
      <Card>
        <CardHeader
          title="Reproducibility"
          description="The same input, config, models, key and runtime profile give a byte-identical PDF and manifest."
        />
        <div className="px-6 pb-6">
          <KeyValue
            items={[
              ["mlredact", <Mono key="v">{p.mlredact_version}</Mono>],
              ["Manifest schema", <Mono key="s">{manifest.schema_version}</Mono>],
              ["Runtime profile", <Mono key="r">{p.runtime_profile.id}</Mono>],
              ["Config hash", <Hash key="c" value={p.config_hash} />],
              ["Input SHA-256", <Hash key="i" value={manifest.job.input_sha256} />],
              ["Profile · style", `${manifest.job.profile} · ${manifest.job.render_style}`],
              [
                "Surrogate key",
                p.surrogate ? (
                  <span key="k">
                    <Mono>{p.surrogate.key_id}</Mono>
                    <span className="text-muted"> · {p.surrogate.scope === "caller" ? "caller scope" : "document scope"}</span>
                  </span>
                ) : (
                  <span key="k" className="text-muted">Not used</span>
                ),
              ],
              ["Propagation rounds", manifest.detection.propagation_rounds],
              ...(run
                ? ([
                    ["Engine job id", <Mono key="j">{run.job_id}</Mono>],
                    ["Workers", run.workers],
                  ] as [string, React.ReactNode][])
                : []),
            ]}
          />
        </div>
      </Card>

      <Card>
        <CardHeader title="Detectors" description="Every detector that ran, with its version." />
        <div className="flex flex-wrap gap-1.5 px-6 pb-6">
          {p.detectors.length ? (
            p.detectors.map((d) => (
              <Pill key={d.id}>
                <span className="font-mono">{d.id}</span>
                <span className="text-muted">v{d.version}</span>
              </Pill>
            ))
          ) : (
            <span className="text-sm text-muted">Detection did not run.</span>
          )}
        </div>
        <CardHeader title="Native libraries" className="pt-1" />
        <div className="px-6 pb-6">
          <dl className="grid grid-cols-2 gap-x-6 gap-y-1 text-sm sm:grid-cols-3">
            {Object.entries(p.runtime_profile.libraries).map(([name, version]) => (
              <div key={name} className="flex min-w-0 justify-between gap-2">
                <dt className="truncate text-muted">{name}</dt>
                <dd className="font-mono text-xs leading-5 text-ink">{version}</dd>
              </div>
            ))}
          </dl>
        </div>
      </Card>

      <Card className="overflow-hidden xl:col-span-2">
        <CardHeader title="Models" description="Pinned artefacts, each verified against its SHA-256 before use." />
        <div className="overflow-x-auto">
          <table className="w-full min-w-[44rem] text-sm">
            <thead>
              <tr className="border-y border-line text-left text-[0.6875rem] tracking-wider text-muted uppercase">
                <th scope="col" className="px-6 py-3 font-medium">Model</th>
                <th scope="col" className="px-6 py-3 font-medium">Version</th>
                <th scope="col" className="px-6 py-3 font-medium">Licence</th>
                <th scope="col" className="px-6 py-3 font-medium">Files</th>
              </tr>
            </thead>
            <tbody>
              {p.models.map((m) => (
                <tr key={m.id} className="border-b border-line align-top last:border-b-0">
                  <td className="px-6 py-3 font-mono text-[0.8125rem] text-ink">{m.id}</td>
                  <td className="max-w-xs px-6 py-3 text-ink-2">{m.version}</td>
                  <td className="px-6 py-3 whitespace-nowrap text-ink-2">{m.license}</td>
                  <td className="px-6 py-3">
                    <ul className="space-y-0.5">
                      {m.files.map((f) => (
                        <li key={f.name} className="flex min-w-0 items-center gap-2">
                          <span className="truncate text-xs text-ink-2" title={f.name}>
                            {f.name}
                          </span>
                          <Hash value={f.sha256} n={10} />
                        </li>
                      ))}
                    </ul>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
