import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./tests/e2e",
  timeout: 180_000,
  expect: { timeout: 10_000 },
  fullyParallel: true,
  reporter: [["list"], ["html", { open: "never" }]],
  use: {
    baseURL: "http://127.0.0.1:5273",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  webServer: [
    {
      command: ".\\.venv\\Scripts\\python.exe scripts\\run_playwright_backend.py",
      cwd: "..",
      url: "http://127.0.0.1:8117/health",
      reuseExistingServer: false,
      timeout: 300_000,
    },
    {
      command: "npm run dev -- --host 127.0.0.1 --port 5273 --strictPort --force",
      env: {
        VITE_API_BASE: "http://127.0.0.1:8117",
      },
      url: "http://127.0.0.1:5273/login",
      reuseExistingServer: false,
      timeout: 180_000,
    },
  ],
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
});
