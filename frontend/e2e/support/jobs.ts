import { expect, type Page } from "@playwright/test";

import type { Doc } from "./documents";
import { TIMEOUTS } from "./env";

type Style = "surrogate" | "label" | "blackout";

/** Fills the New redaction form, submits it and returns the new job's ID from the job page's URL. */
export async function submitJob(page: Page, doc: Doc, options: { style?: Style; devKey?: boolean } = {}): Promise<string> {
  await page.goto("/redact");
  // The document input, not the one Next.js's dev overlay adds.
  await page.locator('input[type="file"][accept*=".pdf"]').setInputFiles(doc);
  await expect(page.getByText(doc.name)).toBeVisible();
  if (options.style) {
    // The radios are visually hidden; their rows are the labels.
    await page.locator("label.choice", { has: page.locator(`input[name="style"][value="${options.style}"]`) }).click();
  }
  if (options.devKey) {
    await page.getByText("Advanced", { exact: true }).click();
    await page.getByRole("checkbox", { name: /public development surrogate key/ }).check();
  }
  await page.getByRole("button", { name: "Start redaction" }).click();
  await page.waitForURL(/\/jobs\/[^/?#]+/);
  const id = new URL(page.url()).pathname.split("/").pop();
  if (!id) throw new Error(`no job ID in ${page.url()}`);
  return id;
}

const ACTIVE = new Set(["awaiting_upload", "queued", "running"]);

/**
 * Waits for the job to finish, then checks the job page's status badge reads `label` (Released,
 * Quarantined, Failed, Cancelled).  A job that ends any other way fails at once, naming its state.
 */
export async function expectJobState(page: Page, id: string, label: string, timeout = TIMEOUTS.job): Promise<void> {
  let state = "";
  await expect
    .poll(
      async () => {
        const res = await page.request.get(`/api/jobs/${id}`);
        state = ((await res.json()) as { job: { state: string } }).job.state;
        return ACTIVE.has(state);
      },
      { timeout, intervals: [250, 500, 1000, 2000] },
    )
    .toBe(false);
  await expect(page.locator("header:has(h1)").getByText(label, { exact: true }), `job ended as "${state}"`).toBeVisible();
}
