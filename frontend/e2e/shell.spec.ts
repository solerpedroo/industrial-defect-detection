import { expect, test } from "@playwright/test";

test("dashboard routes explain the inspection system", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Industrial Defect Detection" })).toBeVisible();

  const nav = page.getByRole("navigation", { name: "Primary" });
  const datasetsResponse = page.waitForResponse((response) =>
    new URL(response.url()).pathname === "/api/datasets" && response.request().method() === "GET",
  );
  await nav.getByRole("link", { name: "Datasets" }).click();
  const response = await datasetsResponse;
  expect(response.status()).toBe(200);
  const catalog = await response.json();
  expect(catalog.categories).toHaveLength(3);
  for (const category of catalog.categories) {
    expect(Number.isInteger(category.validation_normal)).toBe(true);
    expect(category.validation_normal).toBeGreaterThanOrEqual(0);
    expect(["not_created", "created", "invalid"]).toContain(category.validation_split);
  }
  await expect(page.getByRole("heading", { name: "Datasets" })).toBeVisible();
  await expect(page.getByRole("link", { name: "CC BY-NC-SA 4.0" })).toBeVisible();
  await expect(page.getByText("Validation normal", { exact: true })).toHaveCount(3);

  await nav.getByRole("link", { name: "Inference Lab" }).click();
  await expect(page.getByRole("heading", { name: "Inference Lab" })).toBeVisible();

  await nav.getByRole("link", { name: "Experiments" }).click();
  await expect(page.getByRole("heading", { name: "Experiments" })).toBeVisible();

  await nav.getByRole("link", { name: "Comparison" }).click();
  await expect(page.getByRole("heading", { name: "Model Comparison" })).toBeVisible();
});
