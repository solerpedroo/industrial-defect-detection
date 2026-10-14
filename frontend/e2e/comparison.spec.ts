import fs from "node:fs/promises";
import path from "node:path";
import { expect, test, type Page } from "@playwright/test";

test.skip(process.env.PLAYWRIGHT_REAL_INFERENCE === "0", "Requires synthetic real-model API/report fixture");
const reportPath = path.join(__dirname, "../.next/e2e-results.json");

async function upload(page: Page) {
  const encoded = await page.evaluate(() => {
    const canvas = document.createElement("canvas"); canvas.width = 21; canvas.height = 35;
    const context = canvas.getContext("2d")!; context.fillStyle = "rgb(42,43,44)";
    context.fillRect(0, 0, 21, 35); return canvas.toDataURL("image/png").split(",")[1];
  });
  await page.getByLabel("Image", { exact: true }).setInputFiles({
    name: "synthetic.png", mimeType: "image/png", buffer: Buffer.from(encoded, "base64"),
  });
}

test("same upload produces two real maps; shared opacity, distinct selection and error cleanup", async ({ page }, testInfo) => {
  const errors: string[] = []; page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/comparison");
  const submit = page.getByRole("button", { name: "Compare models" });
  await expect(submit).toBeEnabled(); await upload(page);
  const responsePromise = page.waitForResponse((response) => response.url().endsWith("/api/inference/compare"));
  await submit.click();
  const response = await responsePromise; expect(response.status()).toBe(200);
  const payload = await response.json(); expect(payload.results).toHaveLength(2);
  expect(payload.results[0].input_png).toBe(payload.results[1].input_png);
  const live = page.getByRole("region", { name: "Live comparison" });
  await expect(live).toBeVisible();
  for (const result of payload.results) {
    const card = live.getByRole("article", { name: `Result ${result.model_id}` });
    await expect(card).toContainText(result.score.toPrecision(5));
    const image = card.getByRole("img", { name: `${result.model_id} heatmap` });
    await expect(image).toBeVisible();
    expect(await image.evaluate((image: HTMLImageElement) => image.naturalWidth)).toBe(32);
  }
  await page.getByRole("slider", { name: "Shared overlay opacity" }).fill("75");
  for (const model of ["autoencoder", "patchcore"]) {
    await expect(live.getByRole("img", { name: `${model} overlay` })).toHaveCSS("opacity", "0.75");
  }
  await page.screenshot({ path: testInfo.outputPath("comparison.png"), fullPage: true });
  await page.getByRole("combobox", { name: "Model B", exact: true }).selectOption("autoencoder");
  await expect(submit).toBeDisabled(); await expect(live).toHaveCount(0);
  await page.getByRole("combobox", { name: "Model B", exact: true }).selectOption("patchcore");
  await submit.click(); await expect(live).toBeVisible();
  await page.route("**/api/inference/compare", (route) => route.fulfill({
    status: 503, contentType: "application/json", body: JSON.stringify({ detail: "Controlled comparison failure." }),
  }));
  await submit.click();
  await expect(page.getByRole("alert").filter({ hasText: "Controlled comparison failure." })).toBeVisible();
  await expect(live).toHaveCount(0);
  await page.getByRole("combobox", { name: "Category", exact: true }).selectOption("leather");
  await expect(submit).toBeDisabled(); expect(errors).toEqual([]);
});

test("obsolete comparison cannot repopulate changed controls", async ({ page }) => {
  await page.goto("/comparison");
  const submit = page.getByRole("button", { name: "Compare models" });
  await expect(submit).toBeEnabled(); await upload(page);
  let release!: () => void, received!: () => void;
  const hold = new Promise<void>((resolve) => { release = resolve; });
  const started = new Promise<void>((resolve) => { received = resolve; });
  await page.route("**/api/inference/compare", async (route) => {
    const response = await route.fetch(); received(); await hold;
    await route.fulfill({ response }).catch(() => {});
  });
  await submit.click(); await started;
  await page.getByRole("combobox", { name: "Category", exact: true }).selectOption("leather");
  release(); await expect(submit).toBeDisabled();
  await expect(page.getByRole("region", { name: "Live comparison" })).toHaveCount(0);
});

test("actual evaluator report imports locally, exports values and clears after invalid replacement", async ({ page }, testInfo) => {
  const original = JSON.parse(await fs.readFile(reportPath, "utf8"));
  let posts = 0; page.on("request", (request) => { if (request.method() === "POST") posts += 1; });
  await page.goto("/comparison");
  const input = page.getByLabel("Experiment report", { exact: true });
  await input.setInputFiles(reportPath);
  const saved = page.getByRole("region", { name: "Saved metrics" });
  await expect(saved).toBeVisible(); await expect(saved).toContainText("Synthetic experiment");
  await page.getByRole("combobox", { name: "Saved category", exact: true }).selectOption("bottle");
  const metric = saved.getByRole("row").filter({ has: page.getByRole("rowheader", { name: "Image AUROC (ratio)", exact: true }) });
  await expect(metric.getByRole("cell").nth(0)).toHaveText(String(original.results[0].metrics.image_auroc));
  await expect(metric.getByRole("cell").nth(1)).toHaveText(String(original.results[1].metrics.image_auroc));
  await page.screenshot({ path: testInfo.outputPath("saved-metrics.png"), fullPage: true });
  const download = page.waitForEvent("download");
  await page.getByRole("button", { name: "Export visible table CSV" }).click();
  const file = await download; expect(file.suggestedFilename()).toBe("comparison-bottle.csv");
  const stream = await file.createReadStream(); const chunks: Buffer[] = [];
  for await (const chunk of stream!) chunks.push(Buffer.from(chunk));
  const csv = Buffer.concat(chunks).toString("utf8");
  expect(csv).toContain('"bottle","official_test","synthetic","partial"');
  expect(csv).toContain(original.config_hash);
  expect(csv).toContain(`"image_auroc","ratio","${original.results[0].metrics.image_auroc}"`);
  await input.setInputFiles({ name: "bad.json", mimeType: "application/json", buffer: Buffer.from('{"complete":false}') });
  await expect(saved).toHaveCount(0);
  await expect(page.getByRole("alert").filter({ hasText: "Invalid report" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Export visible table CSV" })).toHaveCount(0);
  expect(posts).toBe(0);
});

test("a slow report read cannot restore metrics after a newer invalid selection", async ({ page }) => {
  await page.addInitScript(() => {
    const read = File.prototype.text;
    File.prototype.text = async function () {
      const text = await read.call(this);
      if (this.name === "slow.json") {
        await new Promise<void>((resolve) => {
          Object.defineProperty(window, "releaseSlowReport", { value: resolve, configurable: true });
        });
      }
      return text;
    };
  });
  await page.goto("/comparison");
  const input = page.getByLabel("Experiment report", { exact: true });
  await input.setInputFiles({ name: "slow.json", mimeType: "application/json", buffer: await fs.readFile(reportPath) });
  await page.waitForFunction(() => typeof Reflect.get(window, "releaseSlowReport") === "function");
  await input.setInputFiles({ name: "invalid.json", mimeType: "application/json", buffer: Buffer.from("null") });
  await expect(page.getByRole("alert").filter({ hasText: "Invalid report" })).toBeVisible();
  await page.evaluate(async () => { Reflect.get(window, "releaseSlowReport")(); await new Promise(requestAnimationFrame); });
  await expect(page.getByRole("region", { name: "Saved metrics" })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Export visible table CSV" })).toHaveCount(0);
});
