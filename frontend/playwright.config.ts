import path from "node:path";

import { defineConfig, devices } from "@playwright/test";

import {
  API_TOKEN,
  BACKEND_DIR,
  BACKEND_PORT,
  BACKEND_URL,
  backendPython,
  ENGINE,
  EXTERNAL_BACKEND,
  FRONTEND_PORT,
  FRONTEND_URL,
  JOBS_DIR,
  NEXTAUTH_SECRET,
  STORAGE_STATE,
  STUB_JOB_LIMIT_S,
  TIMEOUTS,
  usersEnv,
} from "./e2e/support/env";

/*
 * End-to-end tests: the real frontend and backend in a browser.  `npm run test:e2e` uses the stub
 * engine; `npm run test:e2e:real` the real one.  See e2e/README.md.
 */
const python = EXTERNAL_BACKEND ? "" : backendPython();

const backendServer = {
  command: "node e2e/support/start-backend.mjs",
  url: `${BACKEND_URL}/v1/health`,
  timeout: 60_000,
  reuseExistingServer: false,
  stdout: "pipe" as const,
  env: {
    E2E_PYTHON: python,
    E2E_BACKEND_DIR: BACKEND_DIR,
    E2E_BACKEND_PORT: String(BACKEND_PORT),
    MLREDACT_API_TOKEN: API_TOKEN,
    MLREDACT_API_JOBS_DIR: JOBS_DIR,
    MLREDACT_API_CORS_ORIGINS: FRONTEND_URL,
    ...(ENGINE === "stub"
      ? {
          MLREDACT_API_ENGINE: JSON.stringify([python, path.join(BACKEND_DIR, "tests/api/stub_engine.py")]),
          MLREDACT_API_JOB_TIMEOUT_S: String(STUB_JOB_LIMIT_S),
        }
      : {}),
  },
};

const frontendServer = {
  command: `node scripts/copy-pdf-worker.mjs && node scripts/next.mjs dev --hostname 127.0.0.1 --port ${FRONTEND_PORT}`,
  url: `${FRONTEND_URL}/login`,
  timeout: 180_000,
  reuseExistingServer: false,
  env: {
    // Its own build directory, so it starts beside `npm run dev` (see next.config.ts).
    MLREDACT_UI_DIST_DIR: ".next-e2e",
    // These win over .env.local: Next.js never overrides variables already set.
    MLREDACT_API_URL: BACKEND_URL,
    MLREDACT_API_PUBLIC_URL: BACKEND_URL,
    MLREDACT_API_TOKEN: API_TOKEN,
    NEXTAUTH_SECRET,
    NEXTAUTH_URL: FRONTEND_URL,
    MLREDACT_UI_USERS: usersEnv(),
    GOOGLE_CLIENT_ID: "",
    GOOGLE_CLIENT_SECRET: "",
  },
};

export default defineConfig({
  testDir: "./e2e",
  outputDir: "./e2e/.generated/test-results",
  // The backend runs one job at a time and every test shares its job list: one worker.
  workers: 1,
  fullyParallel: false,
  forbidOnly: Boolean(process.env.CI),
  retries: 0,
  timeout: TIMEOUTS.test,
  expect: { timeout: 15_000 },
  reporter: [["list"], ["html", { open: "never", outputFolder: "e2e/.generated/report" }]],
  globalSetup: "./e2e/global-setup.ts",
  use: {
    baseURL: FRONTEND_URL,
    // `next dev` compiles each route on its first visit.
    navigationTimeout: 60_000,
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  projects: [
    { name: "setup", testMatch: /auth\.setup\.ts/ },
    {
      name: `chromium-${ENGINE}`,
      use: { ...devices["Desktop Chrome"], storageState: STORAGE_STATE },
      dependencies: ["setup"],
    },
  ],
  webServer: EXTERNAL_BACKEND ? [frontendServer] : [backendServer, frontendServer],
});
