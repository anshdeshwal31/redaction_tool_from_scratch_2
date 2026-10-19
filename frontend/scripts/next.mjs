// Runs the Next.js CLI with telemetry off: this UI must make no outbound requests.
process.env.NEXT_TELEMETRY_DISABLED = "1";
await import("next/dist/bin/next");
