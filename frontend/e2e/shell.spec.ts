import { expect, test } from "@playwright/test";

test("dashboard routes explain the inspection system", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Industrial Defect Detection" })).toBeVisible();

  const nav = page.getByRole("navigation", { name: "Primary" });
  await nav.getByRole("link", { name: "Datasets" }).click();
  await expect(page.getByRole("heading", { name: "Datasets" })).toBeVisible();
  await expect(page.getByRole("link", { name: "CC BY-NC-SA 4.0" })).toBeVisible();

  await nav.getByRole("link", { name: "Inference Lab" }).click();
  await expect(page.getByRole("heading", { name: "Inference Lab" })).toBeVisible();

  await nav.getByRole("link", { name: "Experiments" }).click();
  await expect(page.getByRole("heading", { name: "Experiments" })).toBeVisible();

  await nav.getByRole("link", { name: "Comparison" }).click();
  await expect(page.getByRole("heading", { name: "Model Comparison" })).toBeVisible();
});
