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
    },
  },
});
