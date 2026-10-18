import { expect, test } from "@playwright/test";

import { documents } from "./support/documents";
import { expectJobState, submitJob } from "./support/jobs";

test("a finished job is listed with its result and opens from the list", async ({ page }) => {
  const id = await submitJob(page, documents.released(), { style: "label" });
  await expectJobState(page, id, "Released");

  await page.getByRole("navigation", { name: "Main" }).getByRole("link", { name: "Jobs" }).click();
  await expect(page).toHaveURL(/\/jobs$/);

  const row = page.getByRole("row").filter({ has: page.locator(`a[href="/jobs/${id}"]`) });
  await expect(row.getByText("Released", { exact: true })).toBeVisible();
  await expect(row.getByText("6/6", { exact: true })).toBeVisible();
  await expect(row.getByText("Label", { exact: false })).toBeVisible();

  await row.locator(`a[href="/jobs/${id}"]`).click();
  await expect(page).toHaveURL(new RegExp(`/jobs/${id}`));
  await expectJobState(page, id, "Released");
});
