import { axe } from "vitest-axe";
import type { AxeResults, Result } from "axe-core";

/**
 * Accessibility assertions for component tests.
 *
 * Deliberately not a blanket `expect(results).toHaveNoViolations()` on every
 * render. This app styles almost entirely through CSS custom properties, which
 * jsdom does not resolve, so axe's colour-contrast rule cannot produce a
 * meaningful verdict here and would only generate noise. Contrast is a real
 * concern — it just cannot be checked in this environment, and a check that
 * always passes for the wrong reason is worse than an explicit gap.
 *
 * The rules kept are the ones jsdom *can* decide from the DOM alone: names,
 * roles, relationships, and structure.
 */

/** Rules that need real layout or computed styles, so jsdom cannot judge them. */
const ENVIRONMENT_BLIND_RULES = {
  "color-contrast": { enabled: false },
  // Landmark/region rules assume a whole page; these tests mount fragments.
  region: { enabled: false },
} as const;

export interface A11yOptions {
  /** Additional rule ids to disable, with a reason recorded at the call site. */
  disableRules?: string[];
}

/** Run axe over a container with the environment-blind rules switched off. */
export async function checkA11y(
  container: Element,
  options: A11yOptions = {},
): Promise<AxeResults> {
  const disabled = Object.fromEntries(
    (options.disableRules ?? []).map((id) => [id, { enabled: false }]),
  );
  return (await axe(container, {
    rules: { ...ENVIRONMENT_BLIND_RULES, ...disabled },
  })) as AxeResults;
}

/** Readable failure text: rule id, impact, and the offending markup. */
export function formatViolations(violations: Result[]): string {
  return violations
    .map((violation) => {
      const nodes = violation.nodes.map((node) => `      ${node.html}`).join("\n");
      return `  [${violation.id}] (${violation.impact ?? "unknown"}) ${violation.help}\n${nodes}`;
    })
    .join("\n");
}

/**
 * Assert a container has no detectable accessibility violations.
 *
 * Uses a plain assertion rather than vitest-axe's custom matcher so the
 * failure message lists every violation with its markup, instead of only the
 * first.
 */
export async function expectNoA11yViolations(
  container: Element,
  options: A11yOptions = {},
): Promise<void> {
  const results = await checkA11y(container, options);
  if (results.violations.length > 0) {
    throw new Error(
      `Expected no accessibility violations, found ${results.violations.length}:\n${formatViolations(results.violations)}`,
    );
  }
}
