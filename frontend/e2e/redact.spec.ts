import { expect, test } from "@playwright/test";

import { documents, QUARANTINE_REASON } from "./support/documents";
import { REAL } from "./support/env";
import { expectJobState, submitJob } from "./support/jobs";

test("a released job shows the redacted PDF, passed checks and a working download", async ({ page }) => {
  // Label style needs no surrogate key, so this runs against any backend.
  const id = await submitJob(page, documents.released(), { style: "label" });
  await expectJobState(page, id, "Released");

  // A released job opens on the Document tab: pdf.js draws the page under the manifest's boxes.
  await expect(page.getByRole("tab", { name: /Document/ })).toHaveAttribute("aria-selected", "true");
  await expect(page.getByLabel("Redacted page 1")).toBeVisible();
  await expect(page.getByText("The redacted PDF could not be displayed")).toHaveCount(0);

  await page.getByRole("tab", { name: /Verification/ }).click();
  const panel = page.getByRole("tabpanel");
  await expect(panel.getByText("Passed", { exact: true })).toHaveCount(6);
  await expect(panel.getByText("Failed", { exact: true })).toHaveCount(0);

  // The download link hands out a signed link to the backend; the file comes straight from there.
  const href = await page.getByRole("link", { name: "Redacted PDF" }).getAttribute("href");
  expect(href).toBeTruthy();
  const res = await page.request.get(href!);
  expect(res.ok()).toBe(true);
  expect((await res.body()).subarray(0, 5).toString("latin1")).toBe("%PDF-");
});

test("a quarantined job lists its reasons and releases no PDF", async ({ page }) => {
  const id = await submitJob(page, documents.quarantined(), { style: "label" });
  await expectJobState(page, id, "Quarantined");

  await expect(page.getByText("Quarantined: no PDF was released")).toBeVisible();
  await expect(page.getByText(QUARANTINE_REASON, { exact: true }).first()).toBeVisible();
  await expect(page.getByRole("link", { name: "Redacted PDF" })).toHaveCount(0);
  await expect(page.getByRole("link", { name: "Quarantine record" })).toBeVisible();
});

test("an engine failure shows its reason code", async ({ page }) => {
  if (REAL) {
    // The real engine fails a surrogate job that has no key; a backend with a key would release it.
    const system = (await (await page.request.get("/api/system")).json()) as { surrogateKey: { fromEnv: boolean } };
    test.skip(system.surrogateKey.fromEnv, "the backend has MLREDACT_SURROGATE_KEY, so a surrogate job won't fail");
  }
  const id = await submitJob(page, documents.failing(), { style: "surrogate" });
  await expectJobState(page, id, "Failed");

  await expect(page.getByText("The engine could not complete this job")).toBeVisible();
  await expect(page.getByText("E_CONFIG_INVALID", { exact: true }).first()).toBeVisible();
  await expect(page.getByRole("link", { name: "Redacted PDF" })).toHaveCount(0);
});
