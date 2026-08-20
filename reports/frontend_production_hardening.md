# Frontend Production Hardening Sprint

**Scope:** `frontend-react/` (plus one additive change to `.github/workflows/ci.yml`)
**Date:** 2026-08-21
**Baseline commit:** `f65109dab4af20919447e5c17b5799533d106d0c` (branch `main`)

---

## 1. Baseline state

The working tree was **clean** at the start of this sprint — `git status --porcelain`
returned no output. The in-progress backend/repository-hardening sprint's work was
already committed at `f65109d`, so there were no uncommitted modifications to protect
from overwrite. No commits, stashes, resets, or pushes were performed during this
sprint; all changes are left in the working tree for review.

Verified baseline of the frontend as it actually was (all commands executed):

| Check | Result |
| ----- | ------ |
| `npm run lint` | pass (0 errors, 0 warnings) |
| `npm run typecheck` | pass |
| `npm test` | **pass — 60 tests across 10 files** |
| `npm run build` | pass |
| `npx vitest run --coverage` | 9.19% statements / 53.09% branches / 19.51% functions |

Architecture as found: React 19 + TypeScript 6 (strict) + Vite 8, react-router-dom 7
with `React.lazy` route splitting, Tailwind 4 over CSS custom properties, no global
state library, a single typed `api/client.ts`, `AuthContext` over role-keyed
`sessionStorage`, a `useApi` hook with request-id race fencing and in-flight GET
de-duplication, and `ErrorBoundary` already wrapping every route and admin tab.

---

## 2. Verified external findings

| # | External finding | Verdict | Evidence |
| - | ---------------- | ------- | -------- |
| 1 | `SafetyCenterSection.tsx` ~1,023 LOC, needs decomposition | **CONFIRMED** | Measured at exactly 1,023 lines; 11 components in one file. |
| 4 | No frontend-specific architecture documentation | **CONFIRMED** | Only a 37-line README existed, and it was partly inaccurate. |
| 5 | No structured frontend error tracking/logging strategy | **CONFIRMED** | `ErrorBoundary` did a bare `console.error`; no classification, no redaction, no provider seam. |
| 7 | Distinguish generated files from god components | **CONFIRMED and honoured** | `generated-openapi.d.ts` is 9,987 lines and generated. Not touched. |
| 8 | Must stay compatible with in-progress backend work | **CONFIRMED** | Zero backend files modified. See §19. |

---

## 3. Findings that were inaccurate or stale

| # | External finding | Reality |
| - | ---------------- | ------- |
| 2 | "Frontend-specific automated tests were not clearly detected" | **Stale.** 10 Vitest files with 60 passing tests plus a Playwright e2e suite already existed and passed. The detector likely missed them because they live in `tests/unit/` rather than beside sources. |
| 3 | "CI surfaces lint and TypeScript checking but not a clear frontend test step" | **Partly stale.** `ci.yml` already ran `npm run test` (line 162) and `npm run build` — but only in `quality-gates`, which `needs:` three other jobs including the full offline backend suite. So the finding was wrong on substance, right on effect: a frontend-only regression was not reported for tens of minutes. Addressed in §13. |
| 6 | "Frontend input-validation patterns were not clearly detected" | **Stale.** `CBCForm.tsx` and its siblings already had per-field presence, numeric, plausibility-range, and max-length validation via `useToolForm`, with `noValidate` forms and inline `Field` errors. |
| — | Test Coverage rated 35 | Coverage was measurable at **9.19% statements**, worse than the rating implied — but the *test suite* existed and was meaningful. The score conflated "no tests" with "no coverage tooling"; there was no coverage provider installed at all. |

**Conclusion:** the external assessment was directionally right about the god component,
documentation, and observability, and wrong about tests, CI, and validation. Roughly
half the stated weaknesses did not exist.

---

## 4. Architecture changes

1. **`SafetyCenterSection` split into a hook + 12 modules** under
   `src/pages/admin/sections/safety/`. The section is now composition only.
2. **Centralised telemetry** (`src/lib/telemetry.ts`) as the single error-reporting
   entry point, with redaction and a pluggable sink.
3. **`ApiError`** added to the API client, carrying HTTP status and an `isExpected`
   getter so 4xx (product state) is distinguishable from 5xx (defect).
4. **`useArtifactRunner`** hook replacing 14 hand-written run/refetch blocks in
   `MleSection` that leaked unhandled promise rejections.
5. **URL trust boundary** (`src/lib/safeUrl.ts` + `ExternalLink`) for backend-supplied
   hrefs.
6. **App-level error boundary** and global handlers installed in `main.tsx`.

Deliberately *not* done: no state-management library, no schema-validation library, no
monitoring SDK. Each would add weight without addressing a verified defect. Runtime
validation was added only at the two real trust boundaries (untyped eval artifacts,
external URLs).

---

## 5. Components / modules extracted

`SafetyCenterSection.tsx`: **1,023 → 242 lines.**

| New module | LOC | Responsibility |
| ---------- | --- | -------------- |
| `safety/useSafetyCenter.ts` | 172 | All data + actions; race fencing; separates fatal from non-fatal errors |
| `safety/safetyFormat.ts` | 108 | Pure formatters, status-tone mapping, guarded artifact readers |
| `safety/BenchmarkLadderBlock.tsx` | 106 | Six benchmark families |
| `safety/SafetyRedTeamBlock.tsx` | 101 | Red-team refusal results |
| `safety/DriftBlock.tsx` | 135 | Drift, shift panels, subgroup drift |
| `safety/RagEvalBlock.tsx` | 71 | RAG evaluation metrics |
| `safety/AdversarialGeneralizationBlock.tsx` | 69 | Held-out adversarial eval (untyped artifact) |
| `safety/MultilingualRefusalBlock.tsx` | 68 | Taglish refusal benchmark table |
| `safety/FailureCaseGallery.tsx` | 58 | Known-failure catalogue |
| `safety/CategoryGrid.tsx` | 53 | Per-category pass rates |
| `safety/LlmJudgeBlock.tsx` | 49 | Optional LLM judge, incl. `unavailable` state |
| `safety/ClinicianFeedbackBlock.tsx` | 40 | Human-oversight counts |
| `safety/CalibrationBlock.tsx` | 36 | Calibration metrics |

Also new: `src/lib/telemetry.ts`, `src/lib/safeUrl.ts`,
`src/hooks/useArtifactRunner.ts`, `src/components/ui/ExternalLink.tsx`.

Boundaries were derived from responsibility (one artifact → one block), not from a
line budget. Every block is a pure function of its props; only the hook touches the
network.

### Other hand-written files above ~500 LOC (inspected)

| File | LOC | Assessment |
| ---- | --- | ---------- |
| `api/client.ts` | 831 | **Not a god object.** A flat registry of one typed wrapper per endpoint sharing a single `request<T>()`. Splitting it would add import churn without reducing complexity. Left as is. |
| `MleSection.tsx` | 717 → 654 | **Genuine god component** — one function, 24 `useApi` calls, 14 boolean flags. Extracted the duplicated runner logic (fixing a real bug class, §10); the panel split remains outstanding. |
| `RagSection.tsx` | 625 | Not decomposed this sprint. Outstanding. |
| `ChatPanel.tsx` | 601 | Long but already internally decomposed into 6 named components + a hook. Low priority. |
| `PatientDashboard.tsx` | 590 | Route container; moderate. Outstanding. |
| `types/api.*.ts` | 502–721 | Type declarations, no runtime code. Correctly excluded. |
| `types/generated-openapi.d.ts` | 9,987 | **Generated — not touched by policy.** |

---

## 6. Tests added

9 new files, all deterministic and offline (`src/api/client` mocked with `vi.mock`).

| File | Tests | Focus |
| ---- | ----- | ----- |
| `telemetry.test.ts` | 16 | Token/email/PHI redaction, path templating, sink isolation, global handlers |
| `SafetyBlocks.test.tsx` | 20 | Empty / error / missing-summary / malformed states across 8 blocks |
| `safetyFormat.test.ts` | 15 | Status-tone mapping, absent-value formatting, guarded readers |
| `useSafetyCenter.test.tsx` | 9 | Load, fatal error, optional-eval degradation, action errors, stale-response fencing |
| `safeUrl.test.ts` | 9 | `javascript:`/`data:`/control-char/scheme rejection |
| `useArtifactRunner.test.tsx` | 8 | Unhandled-rejection capture, no-reload-on-failure, unmount safety |
| `SafetyCenterSection.test.tsx` | 6 | Loading/error/empty integration, action-error banner, a11y name uniqueness |
| `ErrorBoundary.test.tsx` | 5 | Fallback UI, telemetry classification, retry recovery, sink-failure isolation |
| `ExternalLink.test.tsx` | 4 | Scheme allow-listing, `rel` hardening, text degradation |

Tests target user-visible behaviour (`getByRole`, `getByText`) rather than internals.
Four are explicit regression tests for defects found this sprint, each naming the
defect in a comment.

---

## 7. Frontend test counts

| | Files | Tests |
| - | ----- | ----- |
| Before | 10 | 60 |
| After | 19 | **152** |

All 152 pass. No pre-existing test was modified, weakened, skipped, or deleted.

---

## 8. Coverage before / after

Measured with `@vitest/coverage-v8` (added this sprint; no provider existed before).
`src/types/**` and `src/main.tsx` are excluded from the denominator — declarations and
DOM bootstrap.

| Metric | Before | After | Δ |
| ------ | ------ | ----- | - |
| Statements | 9.19% (1278/13899) | **19.82%** (2798/14117) | +10.63 pp |
| Branches | 53.09% (240/452) | **62.35%** (578/927) | +9.26 pp |
| Functions | 19.51% (56/287) | **35.19%** (126/358) | +15.68 pp |
| Lines | 9.19% | **19.82%** | +10.63 pp |

Branch count nearly doubled (452 → 927) because the new code adds explicit handling for
states that previously had no branch at all. Coverage was raised by testing behaviour
that matters, not by touching trivial lines; ~20% is an honest number for a codebase
this size and is called out as remaining work rather than presented as sufficient.

---

## 9. Accessibility improvements

- **Distinct accessible names for ambiguous controls.** "Fast", "Live agent", and "Run"
  each appeared on multiple panels with no way for a screen-reader user to tell which
  artifact they regenerate. All six now carry explicit `aria-label`s. Enforced by a
  test asserting every button on the section has a unique accessible name.
- **`aria-busy` on `Button`** while `loading` — the spinner swap was a purely visual cue.
- **Semantic tables** in `MultilingualRefusalBlock`: `<caption class="sr-only">`,
  `scope="col"`, and `scope="row"` on the case-id cell.
- **`<dl>/<dt>/<dd>`** in `CategoryGrid` so category labels and rates are announced as pairs.
- **`<section aria-label>` + real `<h4>` headings** in `BenchmarkLadderBlock`,
  `DriftBlock`, and `SafetyRedTeamBlock`, replacing anonymous `<div>` + styled `<p>`.
- **`<ul>/<li>`** for the failure-case gallery instead of a `<div>` soup.
- **`role="alert"`** on both new action-failure banners; **`role="status"`** on the
  LLM-judge unavailable notice.
- **`aria-hidden="true"`** added to decorative icons that lacked it.

Not done: no automated a11y audit (axe) was run. Improvements above were made by
inspection and are verified by role-based queries in tests.

---

## 10. Error-handling and observability improvements

**`src/lib/telemetry.ts`** — single reporting entry point:

- **Classification.** `expected` (4xx, disabled feature, ungenerated artifact →
  `console.warn`) vs `unexpected` (render crash, 5xx, TypeError → `console.error`).
  Previously everything was one undifferentiated `console.error`.
- **Redaction, applied to messages, stacks, and structured detail:** JWT/bearer
  patterns → `[redacted-token]`; emails → `[redacted-email]`; strings over 200 chars
  truncated (clinical prose is a disclosure risk); values under sensitive keys
  (`*token*`, `*authorization*`, `*patient_id*`, `*note*`, `*mrn*`, `*dob*`, …) →
  `[redacted]`; recursion depth-bounded at 4.
- **Request paths templated** — `/patients/P001/labs` → `/patients/:id/labs`, query
  strings dropped entirely. Routes stay groupable; subjects do not leak.
- **Provider seam.** `registerTelemetrySink()` accepts a hosted provider at bootstrap.
  No SDK is bundled — that is a data-egress decision for a deployment, not a default.
- **Fails safely.** Every sink dispatch is individually try/caught. A throwing or
  missing sink degrades observability and nothing else; tested explicitly, including
  that a broken sink does not starve healthy ones or mask the `ErrorBoundary` fallback.

**Boundaries:** app-level `ErrorBoundary` added in `main.tsx` (route- and tab-level
already existed); `installGlobalErrorHandlers()` catches unhandled rejections and
top-level `window.onerror`; `main.tsx` no longer uses a non-null assertion on
`#root` and reports a clear diagnostic instead.

### Defects found and fixed

1. **Failed artifact re-runs were invisible (Safety Center).** `regenerate()` wrote to
   the same `error` state as the fatal load error, which is only rendered in the
   `status === "error"` branch. The message was set but never displayed — the operator
   saw the spinner stop and nothing else. Fixed with a separate `actionError` and a
   dismissible `role="alert"` banner that keeps loaded artifacts visible.
   *Regression test: `useSafetyCenter.test.tsx`, `SafetyCenterSection.test.tsx`.*

2. **Unhandled promise rejections across 14 admin actions (`MleSection`).** Each runner
   used `try/finally` with no `catch`, so a rejected job reset the spinner and then
   escaped as an unhandled rejection, with the stale artifact still on screen implying
   success. `useArtifactRunner` captures the failure, reports it once, **skips the
   refetch on failure**, and surfaces a banner.
   *Regression test: `useArtifactRunner.test.tsx`.*

3. **Flash of "No safety center data" before loading.** The initial fetch is deferred to
   a macrotask, so the first render ran with `status === "idle"` and fell through to the
   empty pane — reading as "nothing to report" before the request had been made. On a
   safety surface that is a meaningful misstatement. Fixed by starting at `loading` and
   grouping `idle` with `loading`.
   *Regression test: `SafetyCenterSection.test.tsx`.*

4. **`NaN%` from untyped artifacts.** `readNumber` accepted `NaN` as a valid number.
   Now requires `Number.isFinite`, so it renders `—`. *Test: `safetyFormat.test.ts`.*

5. **Stale-response overwrite in the Safety Center.** No fencing existed; a slow initial
   load resolving after a fast re-run would overwrite fresh artifacts with stale ones.
   Added a load-id fence plus mount tracking. *Test: `useSafetyCenter.test.tsx`.*

---

## 11. Security improvements

| Finding | Severity | Status |
| ------- | -------- | ------ |
| `MleEvidencePanels.tsx:106` rendered `href={source.url}` directly from ingested knowledge-base metadata. React escapes text but does not block `javascript:` or `data:text/html` in an href. | Medium (stored XSS via ingested source metadata) | **Fixed** — `ExternalLink` + `safeExternalUrl`: scheme allow-list (`http`/`https`/`mailto`), control-character rejection (defeats `java\nscript:`), degrades to inert text. |
| Same anchor used `rel="noreferrer"` without `noopener`. | Low (reverse tabnabbing) | **Fixed** — `rel="noopener noreferrer"` always. |
| Error logs could carry bearer tokens, emails, patient identifiers, or clinical free text. | Medium (disclosure via logs/monitoring) | **Fixed** — redaction in `lib/telemetry.ts`, applied before any sink. |
| Request paths embedding patient identifiers reaching logs. | Low–Medium | **Fixed** — `redactUrlPath` templates ids and drops query strings. |
| Non-null assertion on `document.getElementById('root')`. | Low (opaque failure) | **Fixed** — explicit guard and diagnostic. |

**Audited and found already sound (no change needed):**

- No `dangerouslySetInnerHTML` or `innerHTML` anywhere in `src/`. `MarkdownMessage`
  constructs React elements via a left-to-right scanner — verified by inspection.
- Tokens are in `sessionStorage` (role-keyed, cleared on logout), not `localStorage`,
  so they do not outlive the tab. `localStorage` is used only for a UI collapse
  preference in `SectionCard.tsx`.
- No `window.location` assignment, no `eval`, no open-redirect surface. Routing is via
  react-router `navigate` with hard-coded paths.
- No file-upload interface exists in the frontend.
- Form input is validated for feedback only; the backend re-validates.

**Explicitly not done:** no attempt to replicate backend authorization. `RouteGuard`
remains a UX control. Its comment and the architecture doc now state plainly that the
backend is authoritative, so no reader mistakes it for a security boundary.

---

## 12. Documentation added

- **`frontend-react/docs/ARCHITECTURE.md`** (new, 12 sections): stack, directory layout,
  data-flow diagram, `useApi`/`useArtifactRunner`/`AuthContext` semantics, API client
  conventions, the loading/error/empty-state rule, error-handling and redaction
  contract, security boundaries, accessibility conventions, testing conventions,
  **generated-code policy**, local dev and production build, and a candid
  known-weaknesses section.
- **`frontend-react/README.md`** rewritten. It previously stated the app "uses
  role-keyed **localStorage** tokens" — **factually wrong**; the code uses
  `sessionStorage`. Corrected, and the omitted `typecheck`/`test`/`test:coverage`
  commands added.

Both avoid restating obvious code; they document invariants and rationale that are not
recoverable by reading a component.

---

## 13. CI changes

One additive change to `.github/workflows/ci.yml`:

- **Frontend unit tests added to the fast `static-quality` job**, running
  `npm run test:coverage`. Previously frontend tests ran only in `quality-gates`, which
  `needs: [static-quality, full-offline-tests, dependency-audit]` — so a frontend-only
  regression was not reported until after the full offline backend suite.
- **Coverage uploaded** as a `frontend-coverage` artifact (`if: always()`, 7-day retention).

The existing `Frontend unit tests` step in `quality-gates` was **left in place**. Removing
it would weaken an existing gate; the redundancy is a deliberate, cheap trade for fast
feedback.

No existing job, gate, threshold, or step was removed, weakened, or reordered. No
coverage threshold was introduced — gating on ~20% would be theatre, and gating higher
would fail the build today.

---

## 14. Exact verification commands executed

From a **clean dependency install** in `frontend-react/`:

```
npm ci
npm run lint
npm run typecheck
npm run test:coverage
npm run build
npm audit --audit-level=high
npm run typegen:file
```

Repository-level, from the project root:

```
.venv\Scripts\python.exe -m pytest tests/test_constants_sync.py -q
git diff --stat -- frontend-react/src/types/generated-openapi.d.ts
```

---

## 15. Exact pass/fail results

| Command | Result |
| ------- | ------ |
| `npm ci` | **pass** — 363 packages, 0 vulnerabilities |
| `npm run lint` | **pass** — exit 0, no errors, no warnings |
| `npm run typecheck` | **pass** — exit 0 (both `tsconfig.app.json` and `tsconfig.node.json`) |
| `npm run test:coverage` | **pass** — 19 files, **152/152 tests**, exit 0 |
| `npm run build` | **pass** — exit 0, 2,403 modules |
| `npm audit --audit-level=high` | **pass** — 0 vulnerabilities |
| `npm run typegen:file` | **pass** — regenerated types are **byte-identical** to the committed file (no diff) |
| `pytest tests/test_constants_sync.py` | **pass** — 7/7 |

**Not executed, with reasons:**

- `npm run test:e2e` (Playwright) — requires a running FastAPI backend and a
  `playwright install chromium` download. Not run; the e2e suite was not modified.
- The full `pytest tests` suite, `scripts/run_release_gate.py`, and the safety-eval
  scripts — backend-owned, unmodified by this sprint, and currently mid-change by the
  other agent. Running them would report that sprint's state, not this one's.
- No automated accessibility audit (axe) was run.

Bundle sizes are essentially unchanged: `AdminDashboard` 187.78 → 190.11 kB
(+2.33 kB raw, +1.25 kB gzip) for the telemetry, safe-URL, and error-handling code.

---

## 16. Files changed

**Modified (14):**

```
.github/workflows/ci.yml                                  +18
frontend-react/.gitignore                                  +1
frontend-react/README.md                                  ~43
frontend-react/eslint.config.js                            +3/-1
frontend-react/package.json                                +2
frontend-react/package-lock.json                        +664   (coverage provider)
frontend-react/vitest.config.ts                           +16
frontend-react/src/api/client.ts                          +42/-8
frontend-react/src/main.tsx                               +27/-6
frontend-react/src/components/ui/Button.tsx                +4
frontend-react/src/components/ui/ErrorBoundary.tsx        +13/-5
frontend-react/src/pages/admin/sections/MleEvidencePanels.tsx  +5/-3
frontend-react/src/pages/admin/sections/MleSection.tsx   +226/-160
frontend-react/src/pages/admin/sections/SafetyCenterSection.tsx  -1179/+242
```

**Added (23):**

```
frontend-react/docs/ARCHITECTURE.md
frontend-react/src/lib/telemetry.ts
frontend-react/src/lib/safeUrl.ts
frontend-react/src/hooks/useArtifactRunner.ts
frontend-react/src/components/ui/ExternalLink.tsx
frontend-react/src/pages/admin/sections/safety/   (13 modules)
frontend-react/tests/unit/   (9 new suites)
```

**Untouched by policy:** `src/types/generated-openapi.d.ts` (generated),
`Data/openapi.json`, all `backend/`, `scripts/`, `tests/`, `evals/`, and
`Data/evals/` paths.

---

## 17. Remaining frontend weaknesses

Ordered by value.

1. **`MleSection.tsx` (654 LOC) is still a god component.** One function orchestrating
   24 `useApi` calls. The duplicated runner logic is gone; the panel split is not done.
   The evidence-card components already exist in `sections/cards/` — the natural next
   step is one sub-section per artifact family owning its own data.
2. **Coverage is ~20% of statements.** Patient and clinician surfaces have far less
   component coverage than the admin safety surface. `PatientDashboard`, `LabsPanel`,
   `TimelinePanel`, `ReviewQueue`, and `ReviewPanel` have no direct tests.
3. **No coverage threshold enforced in CI.** Should be ratcheted once a baseline is
   agreed, rather than set aspirationally now.
4. **`RagSection.tsx` (625) and `PatientDashboard.tsx` (590) not decomposed.**
5. **No automated accessibility gate.** The improvements here were made by inspection.
   `vitest-axe` on key surfaces would catch regressions.
6. **`LabTrendsChart` is a 345 kB chunk** (101 kB gzip) — Recharts dominates the bundle.
   Worth evaluating a lighter charting approach for the patient portal.
7. **Telemetry has no remote sink wired.** The seam exists and is tested; no provider is
   configured, so production errors currently only reach the browser console.
8. **jsdom suite is slow on Windows** (~35–64 s wall for 152 tests, dominated by
   environment setup). Not a correctness issue.
9. **`MleSection` shows only the first runner error.** Adequate for operator-triggered
   one-at-a-time actions; would need per-panel placement if that changes.

---

## 18. Backend changes intentionally avoided

**Zero backend files were modified.** `git status` confirms every change is under
`frontend-react/` except the additive `ci.yml` step.

Documented rather than implemented:

1. **`Data/openapi.json` is stale relative to the live backend.** Regenerating it
   produces **26 additional endpoint paths** (164 live vs 138 committed; **zero paths
   removed**). This is drift produced by the in-progress backend sprint, not by this
   work. The CI step *"Export OpenAPI schema and verify frontend API types"* will fail
   until that sprint regenerates the schema and the frontend types together.
   **Deliberately not fixed here:** regenerating frontend types against a backend that
   is mid-change would bake in a moving contract and interfere with the other agent's
   work. It is their step to complete.
   *(I did briefly run the export during verification, which dirtied `Data/openapi.json`;
   it was immediately restored with `git checkout --` and confirmed clean.)*
2. **`AdminDashboard.tsx:119-123`** renders `<RagSection />` without `analytics` while
   `status === "loading"`, then again with `analytics` on success. This looks
   unintentional but may be deliberate prefetch behaviour. Not changed — it is outside
   the safety surface and the behaviour is currently harmless.
3. No backend validation, authorization, or response shape was altered or duplicated.

---

## 19. Conflicts with the existing unfinished backend sprint

**None encountered.** The working tree was clean at the start, so there was nothing to
overwrite. No file owned by the backend sprint was edited, reverted, reformatted, or
reset. No commits, stashes, or pushes were made.

The one point of contact is the OpenAPI schema drift described in §18. It is
**additive** (26 new paths, none removed), so no existing frontend contract is broken:
the frontend typechecks, builds, and regenerates byte-identically against the committed
schema. When the backend sprint regenerates `Data/openapi.json`, the frontend types
should be regenerated in the same change via `npm run typegen:file`.

`.github/workflows/ci.yml` is shared. The change is purely additive (two new steps in
`static-quality`) and touches no backend job, so a merge conflict is unlikely and would
be trivial.

---

## 20. Production-readiness assessment

**Assessment: improved but not production-ready. Suitable for a governed demo or
internal pilot; not for clinical deployment.**

What is genuinely production-grade now:

- Clean lint, strict typecheck, and green build from a clean install, all gated in CI.
- 152 deterministic offline tests, with the highest-risk surface — the safety and
  evaluation center — covered for absent, disabled, malformed, and failing states.
- A coherent, redacting error-reporting strategy with a provider seam and safe failure.
- Verified XSS and log-disclosure fixes at the two real frontend trust boundaries.
- The medical safety posture is preserved and now *enforced by tests*: unknown statuses
  never render green, unmeasured metrics render `—` rather than `0%`, a disabled
  evaluator says so instead of showing a clean board, and a pre-fetch frame no longer
  reads as "nothing to report".

What holds it back:

- ~20% statement coverage, concentrated in the admin surface. The patient and clinician
  portals — the surfaces a real user touches — are largely untested at the component level.
- One god component remains, and three files sit in the 500–650 LOC range.
- No accessibility gate, no error-budget or remote monitoring, no coverage threshold.
- The OpenAPI contract check is currently red for reasons outside this sprint.

**Internal frontend maturity estimate: 68–72 / C+**, up from a verified ~55. The
increase reflects the decomposition, five fixed defects, two fixed security issues,
+92 tests, +10.6 pp statement coverage, and real documentation — not the volume of
changed files. It is held below 75 by coverage that is still thin outside the admin
surface and by the god component that remains.

**Recommended next task:** decompose `MleSection.tsx` into per-artifact sub-sections
that each own their own `useApi` call (the `sections/cards/` components already provide
the seams), and add component tests for `PatientDashboard` and `ReviewQueue` — the two
highest-traffic untested surfaces.
