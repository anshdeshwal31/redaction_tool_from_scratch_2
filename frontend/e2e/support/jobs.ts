import { type APIRequestContext, expect, type Page } from "@playwright/test";

import { ACTIVE_STATES, type CreatedJob, type JobRecord, type JobState } from "@/lib/jobs";
import type { RenderStyle } from "@/lib/manifest";
import { MUTATION_HEADERS } from "@/lib/ui-header";

import type { Doc } from "./documents";
import { TIMEOUTS } from "./env";
import { formHydrated } from "./hydration";

/** Jobs this run submitted: the only ones cancelSubmittedJobs touches, as the backend may be shared. */
const submitted = new Set<string>();

/** Fills the New redaction form, submits it and returns the new job's ID from the job page's URL. */
export async function submitJob(page: Page, doc: Doc, options: { style?: RenderStyle } = {}): Promise<string> {
  await page.goto("/redact");
  await formHydrated(page);
  // The document input, not the one Next.js's dev overlay adds.
  await page.locator('input[type="file"][accept*=".pdf"]').setInputFiles(doc);
  await expect(page.getByText(doc.name)).toBeVisible();
  if (options.style) {
    // The radios are visually hidden; their rows are the labels.
    await page.locator("label.choice", { has: page.locator(`input[name="style"][value="${options.style}"]`) }).click();
  }
  // The job exists, and starts once uploaded, before the page moves to it: record it from the
  // create answer, so it is cancelled even if the navigation never comes.
  const create = page.waitForResponse(
    (res) => res.request().method() === "POST" && new URL(res.url()).pathname === "/api/jobs",
  );
  await page.getByRole("button", { name: "Start redaction" }).click();
  const res = await create;
  expect(res.ok(), `creating the job answered ${res.status()}`).toBe(true);
  const { id } = (await res.json()) as CreatedJob;
  submitted.add(id);
  await page.waitForURL((url) => url.pathname === `/jobs/${id}`);
  return id;
}

/**
 * Waits for the job to finish, then checks the job page's status badge reads `label` (Released,
 * Quarantined, Failed, Cancelled).  A job that ends any other way fails at once, naming its state, and
 * so does a 4xx answer from the API.  A 5xx answer (a busy backend) is polled through.
 */
export async function expectJobState(page: Page, id: string, label: string, timeout = TIMEOUTS.job): Promise<void> {
  let state = "";
  let refused = "";
  let lastError = "";
  try {
    await expect
      .poll(
        async () => {
          const res = await page.request.get(`/api/jobs/${id}`);
          const answer = () => `GET /api/jobs/${id} answered ${res.status()}`;
          if (res.status() >= 500) {
            lastError = `${answer()}: ${(await res.text()).slice(0, 200)}`;
            return true;
          }
          // Stop at a refusal: polling on would only hide it until the timeout.
          if (!res.ok()) {
            refused = `${answer()}: ${(await res.text()).slice(0, 200)}`;
            return false;
          }
          lastError = "";
          const { job } = (await res.json()) as { job: JobRecord };
          state = job.state;
          return ACTIVE_STATES.has(job.state);
        },
        { timeout, intervals: [250, 500, 1000, 2000] },
      )
      .toBe(false);
  } catch (err) {
    if (lastError) throw new Error(`job ${id} did not finish; the last poll ${lastError}`, { cause: err });
    throw err;
  }
  if (refused) throw new Error(refused);
  await expect(page.locator("header:has(h1)").getByText(label, { exact: true }), `job ended as "${state}"`).toBeVisible();
}

// The states the backend can cancel; a job awaiting its upload expires instead.
const CANCELLABLE: ReadonlySet<JobState> = new Set(["queued", "running"]);

/**
 * Cancels this run's jobs that are still queued or running.  The backend runs one job at a time, so a
 * job a failed test leaves behind would hold up every later test.
 */
export async function cancelSubmittedJobs(request: APIRequestContext): Promise<void> {
  if (!submitted.size) return;
  const res = await request.get("/api/jobs");
  await expect(res, "listing jobs to cancel").toBeOK();
  const { jobs } = (await res.json()) as { jobs: JobRecord[] };
  for (const job of jobs.filter((j) => submitted.has(j.id) && CANCELLABLE.has(j.state))) {
    const cancel = await request.post(`/api/jobs/${job.id}/cancel`, { headers: MUTATION_HEADERS });
    // 409: the job ended since the list was read.
    expect(cancel.ok() || cancel.status() === 409, `cancelling job ${job.id}: ${cancel.status()}`).toBe(true);
  }
}
