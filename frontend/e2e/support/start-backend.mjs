// Starts the backend API for an end-to-end run (Playwright's webServer runs this).  It empties the
// run's jobs directory first, so every run starts with no jobs.  Settings come from the environment
// the Playwright config passes: E2E_PYTHON, E2E_BACKEND_DIR, E2E_BACKEND_PORT, plus MLREDACT_API_*.
import { spawn } from "node:child_process";
import { rmSync } from "node:fs";

const python = process.env.E2E_PYTHON;
const cwd = process.env.E2E_BACKEND_DIR;
const port = process.env.E2E_BACKEND_PORT;
const jobsDir = process.env.MLREDACT_API_JOBS_DIR;
if (!python || !cwd || !port || !jobsDir) {
  console.error("start-backend: E2E_PYTHON, E2E_BACKEND_DIR, E2E_BACKEND_PORT and MLREDACT_API_JOBS_DIR are required");
  process.exit(1);
}

rmSync(jobsDir, { recursive: true, force: true });

const child = spawn(python, ["-m", "mlredact.cli.main", "serve", "--port", port], { cwd, stdio: "inherit" });
for (const signal of ["SIGINT", "SIGTERM"]) process.on(signal, () => child.kill(signal));
child.on("exit", (code, signal) => process.exit(code ?? (signal ? 1 : 0)));
