import { expect, test, type Page } from "@playwright/test";

test.skip(process.env.PLAYWRIGHT_REAL_INFERENCE === "0", "Requires synthetic real-model API fixture");

async function upload(page: Page) {
  const encoded = await page.evaluate(() => {
    const canvas = document.createElement("canvas");
    canvas.width = 21;
    canvas.height = 35;
    const context = canvas.getContext("2d")!;
    context.fillStyle = "rgb(42,43,44)";
    context.fillRect(0, 0, 21, 35);
    return canvas.toDataURL("image/png").split(",")[1];
  });
  await page.getByLabel("Image", { exact: true }).setInputFiles({
    name: "synthetic.png", mimeType: "image/png", buffer: Buffer.from(encoded, "base64"),
  });
}

test("real API results, opacity, download, selection cleanup and failure cleanup", async ({ page }, testInfo) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/inference");
  await expect(page.getByRole("button", { name: "Run inference" })).toBeEnabled();
  await upload(page);
  const responsePromise = page.waitForResponse((response) =>
    response.url().endsWith("/api/inference") && response.request().method() === "POST");
  await page.getByRole("button", { name: "Run inference" }).click();
  const response = await responsePromise;
  expect(response.status()).toBe(200);
  const result = await response.json();
  expect(result.data_kind).toBe("synthetic");
  expect(result.map_width).toBe(32);
  expect(result.label).toBe(result.score > result.threshold ? "anomalous" : "normal");
  await expect(page.getByRole("heading", { name: "Inspection result" })).toBeVisible();
  await expect(page.getByText(/Synthetic demonstration/)).toBeVisible();
  const heatmap = page.getByRole("img", { name: "Anomaly heatmap", exact: true });
  await expect(heatmap).toBeVisible();
  expect(await heatmap.evaluate((image: HTMLImageElement) => image.naturalWidth)).toBe(32);
  await page.getByRole("slider", { name: "Overlay opacity" }).fill("75");
  await expect(page.getByRole("img", { name: "Adjustable anomaly overlay" })).toHaveCSS("opacity", "0.75");
  await page.screenshot({ path: testInfo.outputPath("inference-result.png"), fullPage: true });
  const download = page.waitForEvent("download");
  await page.getByRole("link", { name: "Download mask PNG" }).click();
  const downloaded = await download;
  expect(downloaded.suggestedFilename()).toBe("bottle-autoencoder-mask.png");
  const stream = await downloaded.createReadStream();
  const chunks: Buffer[] = [];
  for await (const chunk of stream!) chunks.push(Buffer.from(chunk));
  expect(Buffer.concat(chunks).subarray(0, 8)).toEqual(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]));
  await page.getByRole("combobox", { name: "Model", exact: true }).selectOption("patchcore");
  await expect(page.getByRole("heading", { name: "Inspection result" })).toHaveCount(0);
  await page.getByRole("button", { name: "Run inference" }).click();
  await expect(page.getByRole("heading", { name: "Inspection result" })).toBeVisible();
  await expect(page.getByRole("img", { name: "Reconstruction", exact: true })).toHaveCount(0);
  await page.route("**/api/inference", (route) => route.fulfill({
    status: 503, contentType: "application/json", body: JSON.stringify({ detail: "Controlled inference failure." }),
  }));
  await page.getByRole("button", { name: "Run inference" }).click();
  await expect(page.getByRole("alert").filter({ hasText: "Controlled inference failure." })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Inspection result" })).toHaveCount(0);
  await expect(page.getByRole("link", { name: "Download mask PNG" })).toHaveCount(0);
  await page.getByRole("combobox", { name: "Category", exact: true }).selectOption("leather");
  await expect(page.getByRole("button", { name: "Run inference" })).toBeDisabled();
  await expect(page.getByText("Controlled inference failure.", { exact: true })).toHaveCount(0);
  expect(errors).toEqual([]);
});

test("obsolete successful response cannot repopulate a changed selection", async ({ page }) => {
  await page.goto("/inference");
  await expect(page.getByRole("button", { name: "Run inference" })).toBeEnabled();
  await upload(page);
  let release!: () => void;
  const hold = new Promise<void>((resolve) => { release = resolve; });
  let received!: () => void;
  const started = new Promise<void>((resolve) => { received = resolve; });
  await page.route("**/api/inference", async (route) => {
    const response = await route.fetch();
    received();
    await hold;
    await route.fulfill({ response }).catch(() => {}); // client may already have aborted.
  });
  await page.getByRole("button", { name: "Run inference" }).click();
  await started;
  await page.getByRole("combobox", { name: "Category", exact: true }).selectOption("leather");
  release();
  await expect(page.getByRole("button", { name: "Run inference" })).toBeDisabled();
  await expect(page.getByRole("heading", { name: "Inspection result" })).toHaveCount(0);
});
