import { expect, type Page, test } from "@playwright/test";

import { USER } from "./support/env";

test.use({ storageState: { cookies: [], origins: [] } });

async function signIn(page: Page, password: string) {
  await page.getByLabel("Email").fill(USER.email);
  await page.getByLabel("Password").fill(password);
  await page.getByRole("button", { name: "Sign in" }).click();
}

test("a signed-out visitor is sent to the login page", async ({ page }) => {
  await page.goto("/jobs");
  await expect(page).toHaveURL(/\/login\?callbackUrl=%2Fjobs$/);
  await expect(page.getByRole("heading", { name: "Welcome back." })).toBeVisible();
});

test("the API refuses a request without a session", async ({ request }) => {
  const res = await request.get("/api/jobs");
  expect(res.status()).toBe(401);
  expect(await res.json()).toEqual({ error: "E_UI_UNAUTHENTICATED" });
});

test("a wrong password is refused without saying which part was wrong", async ({ page }) => {
  await page.goto("/login");
  await signIn(page, "not-the-password");
  // Scoped to the form: Next.js's route announcer is an alert too.
  await expect(page.locator("form").getByRole("alert")).toHaveText("That email and password don't match an account.");
  await expect(page).toHaveURL(/\/login/);
});

test("signing in returns to the page that was asked for", async ({ page }) => {
  await page.goto("/system");
  await expect(page).toHaveURL(/\/login\?callbackUrl=%2Fsystem$/);
  await signIn(page, USER.password);
  await expect(page).toHaveURL(/\/system$/);
  await expect(page.getByRole("button", { name: "Run self-check" })).toBeVisible();
});

test("signing out ends the session", async ({ page }) => {
  await page.goto("/login");
  await signIn(page, USER.password);
  await page.waitForURL((url) => url.pathname === "/");

  await page.getByRole("button", { name: /Sign out/ }).click();
  await expect(page).toHaveURL(/\/login/);

  await page.goto("/jobs");
  await expect(page).toHaveURL(/\/login\?callbackUrl=%2Fjobs$/);
});
