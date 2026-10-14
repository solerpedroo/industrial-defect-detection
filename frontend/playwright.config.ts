import path from "node:path";

import { defineConfig, devices } from "@playwright/test";

const repoRoot = path.join(__dirname, "..");
const pythonBin = process.env.PLAYWRIGHT_PYTHON ?? path.join(repoRoot, ".venv/bin/python");
const apiUrl = process.env.PLAYWRIGHT_API_URL ?? "http://127.0.0.1:8000";
const webUrl = process.env.PLAYWRIGHT_WEB_URL ?? "http://localhost:3000";

export default defineConfig({
  testDir: "./e2e",
  timeout: 90_000,
  expect: { timeout: 20_000 },
  fullyParallel: false,
  workers: 1,
  reporter: [["list"]],
  use: {
    ...devices["Desktop Chrome"],
    baseURL: webUrl,
  },
  webServer: [
    {
      command: `"${pythonBin}" -m uvicorn app.main:app --app-dir "${repoRoot}/backend" --host 127.0.0.1 --port 8000`,
      url: `${apiUrl}/api/health`,
      reuseExistingServer: !process.env.CI,
      cwd: __dirname,
      timeout: 120_000,
    },
    {
      command: "npm run dev -- --port 3000",
      url: webUrl,
      reuseExistingServer: !process.env.CI,
      cwd: __dirname,
      timeout: 120_000,
      env: {
        ...process.env,
        NEXT_PUBLIC_API_URL: apiUrl,
      },
    },
  ],
});
