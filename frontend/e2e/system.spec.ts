import { expect, test } from "@playwright/test";

import { TIMEOUTS } from "./support/env";

test("the system page shows the server's settings and runs the self-check", async ({ page }) => {
  await page.goto("/system");

  const profiles = page.getByLabel("Profile to check").locator("option");
  await expect(profiles).toHaveCount(4);
  await expect(page.getByText("Upload limit")).toBeVisible();

  await page.getByRole("button", { name: "Run self-check" }).click();
  // Wait for either outcome, so a failed check fails the test at once with its reason code.
  const ready = page.getByText("The engine is ready");
  const failed = page.getByText("The self-check failed");
  await expect(ready.or(failed)).toBeVisible({ timeout: TIMEOUTS.selfcheck });
  const reason = (await failed.isVisible()) ? await failed.locator("xpath=..").innerText() : "";
  expect(reason, "the self-check failed").toBe("");
  await expect(ready).toBeVisible();
});
