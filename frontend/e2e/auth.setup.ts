import { expect, test as setup } from "@playwright/test";

import { STORAGE_STATE, USER } from "./support/env";

// Signs in once; every other test starts from this saved session.
setup("sign in", async ({ page }) => {
  await page.goto("/login");
  await page.getByLabel("Email").fill(USER.email);
  await page.getByLabel("Password").fill(USER.password);
  await page.getByRole("button", { name: "Sign in" }).click();
  await page.waitForURL((url) => url.pathname === "/");
  await expect(page.getByRole("button", { name: /Sign out/ })).toBeVisible();
  await page.context().storageState({ path: STORAGE_STATE });
});
