/// <reference types="vitest" />
import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  esbuild: {
    jsx: "automatic",
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./tests/setup.ts"],
    include: ["tests/**/*.test.{ts,tsx}"],
    exclude: ["tests/e2e/**", "node_modules/**"],
    css: false,
    // jsdom is slow on Windows and v8 coverage instrumentation roughly doubles
    // it, which pushed a few component tests past the 5s default and made the
    // suite flaky under `test:coverage` while passing under `test`. The tests
    // themselves are deterministic; only the wall-clock budget was wrong.
    testTimeout: 20_000,
    hookTimeout: 20_000,
    coverage: {
      provider: "v8",
      reportsDirectory: "coverage",
      reporter: ["text-summary", "json-summary", "lcov"],
      include: ["src/**/*.{ts,tsx}"],
      // Excluded from the denominator on purpose:
      //   - generated-openapi.d.ts and types/*     : declarations, no runtime code
      //   - main.tsx                               : DOM bootstrap, covered by e2e
      //   - assets                                 : non-code
      exclude: [
        "src/types/**",
        "src/main.tsx",
        "src/vite-env.d.ts",
        "src/assets/**",
      ],
      /**
       * Regression gate, not a target.
       *
       * Measured on 2026-08-21: 39.10% statements, 65.42% branches, 34.24%
       * functions, 39.10% lines. Each threshold sits a few points below its
       * measured value so ordinary churn and v8's slight run-to-run variance
       * cannot redden CI, while a real drop — someone deleting a suite or
       * landing a large untested surface — still fails the build.
       *
       * Raise these when coverage genuinely improves. Do not lower one to make
       * a failing build pass; that converts the gate into decoration.
       */
      thresholds: {
        statements: 35,
        branches: 62,
        functions: 31,
        lines: 35,
      },
    },
  },
});
