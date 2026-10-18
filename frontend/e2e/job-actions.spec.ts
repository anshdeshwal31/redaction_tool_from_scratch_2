import { expect, type Page, test } from "@playwright/test";

import { documents } from "./support/documents";
import { expectJobState, submitJob } from "./support/jobs";

/** Submits a job that is still going when the test acts, then stops it from the job page. */
async function submitAndStop(page: Page): Promise<string> {
  const id = await submitJob(page, documents.longRunning(), { style: "label" });
  // "Cancel" while queued, "Stop job" once running.
  await page.getByRole("button", { name: /^(Cancel|Stop job)$/ }).click();
  await expectJobState(page, id, "Cancelled", 30_000);
  return id;
}

test("a running job can be stopped", async ({ page }) => {
  await submitAndStop(page);
  await expect(page.getByText("This job was cancelled")).toBeVisible();
  await expect(page.getByRole("button", { name: /^(Cancel|Stop job)$/ })).toHaveCount(0);
});

test("a finished job can be deleted", async ({ page }) => {
  const id = await submitAndStop(page);

  page.once("dialog", (dialog) => void dialog.accept());
  await page.getByRole("button", { name: "Delete job" }).click();
  await page.waitForURL((url) => url.pathname === "/jobs");

  await expect(page.getByText(/Jobs on the server|No jobs yet/).first()).toBeVisible();
  await expect(page.locator(`a[href="/jobs/${id}"]`)).toHaveCount(0);
  expect((await page.request.get(`/api/jobs/${id}`)).status()).toBe(404);
});
