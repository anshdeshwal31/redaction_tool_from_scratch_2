import { expect, type Locator, type Page } from "@playwright/test";

/**
 * Waits until React has hydrated `target`, a control that then carries `data-hydrated`.  `next dev`
 * hydrates a route after its load event; a file or text set before then never reaches the
 * component's state, a click does nothing, and a form submits natively.
 */
export async function hydrated(target: Locator): Promise<void> {
  await expect(target, "React never hydrated it").toHaveAttribute("data-hydrated", "", { timeout: 60_000 });
}

/** Waits until React has hydrated the page's form. */
export async function formHydrated(page: Page): Promise<void> {
  await hydrated(page.locator("form"));
}
