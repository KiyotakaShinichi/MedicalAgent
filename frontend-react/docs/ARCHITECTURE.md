# NLCare Frontend Architecture

Scope: everything under `frontend-react/`. Backend contracts are described only
where the frontend depends on them. For backend architecture see the
repository-root docs.

---

## 1. Stack

| Concern        | Choice                                | Notes |
| -------------- | ------------------------------------- | ----- |
| Framework      | React 19                              | Function components only, except `ErrorBoundary` (boundaries must be classes). |
| Language       | TypeScript 6, `strict`                | `npm run typecheck` covers both the app and the Node-side config projects. |
| Build          | Vite 8 (rolldown)                     | `npm run build` runs `tsc -b` first, so a type error fails the build. |
| Routing        | react-router-dom 7 (`BrowserRouter`)  | Route-level code splitting via `React.lazy`. |
| Styling        | Tailwind 4 + CSS custom properties    | Colours come from `var(--token)` in `index.css`, not Tailwind palette classes. |
| Charts         | Recharts 3                            | Lazily loaded — it is the largest chunk in the bundle. |
| Icons          | lucide-react                          | Always pass `aria-hidden="true"` for decorative icons. |
| Tests          | Vitest 3 + Testing Library + jsdom    | `npm test`. Playwright covers e2e smoke separately. |

No global state library. State is either local, in a route-level container, or
in the single `AuthContext`. Adding Redux/Zustand for the current surface area
would not pay for itself.

---

## 2. Directory layout

```
src/
  api/client.ts        Single typed API surface. Every network call lives here.
  components/
    layout/            AppShell, RouteGuard
    ui/                Presentational primitives (Button, Card, MetricCard, …)
    charts/            Recharts wrappers
  context/             AuthContext + its core (split for react-refresh)
  hooks/               useApi, useAuth, useToolForm, useArtifactRunner
  lib/                 Framework-free helpers: telemetry, safeUrl, constants
  pages/
    patient/           Patient portal + tool forms
    clinician/         Review queue and approvals
    admin/sections/    Admin dashboard, one file per tab
    admin/sections/safety/   Safety & Evaluation Center blocks
    workspace/         AI-assurance workspace
  types/               Hand-written API types + generated OpenAPI declarations
tests/
  unit/                Vitest suites
  e2e/                 Playwright smoke
```

Rule of thumb: a file under `pages/` owns data fetching and composition; a file
under `components/ui/` is a pure function of its props and never calls the API.

---

## 3. Data flow

```
component
   │  calls
   ▼
hooks/useApi(fn, deps)        ← owns status/error/staleness, fences races
   │  calls
   ▼
api/client.ts                 ← auth header, in-flight GET dedupe, ApiError
   │  fetch
   ▼
FastAPI backend
```

### `useApi`

`useApi(fn, deps)` returns `{ data, status, error, refetch, lastFetchedAt }`.

Two properties worth knowing:

- **`fn` is held in a ref.** Re-creating the callback inline every render does
  *not* re-fetch; only a change in `deps` does. This is why call sites can
  write `useApi(() => getThing(id), [id])` without a `useCallback`.
- **Responses are fenced by request id.** A slow response that resolves after a
  newer one has already landed is discarded, so a fast refetch cannot be
  overwritten by a stale in-flight request.

### `useArtifactRunner`

For admin "regenerate this artifact, then reload it" actions. Owns the spinner
flag, catches failures (a bare `try/finally` leaks an unhandled rejection), and
skips the reload on failure so a failed run cannot look like a successful one.

### `AuthContext`

Holds `{ token, role, patientId }`, hydrated from `sessionStorage` on mount.
Split into `AuthContext.tsx` (provider) and `authContextCore.ts` (context
object + storage keys) so react-refresh can hot-reload the provider.

---

## 4. API client conventions

`src/api/client.ts` is a flat list of named, typed wrappers — one exported
function per endpoint. It is long by design; it is a registry, not a god
object. Everything routes through one private `request<T>()` that:

- resolves the base URL from `VITE_API_BASE`, falling back to
  `http://127.0.0.1:8017`;
- attaches the bearer token from `sessionStorage`;
- sends `X-NLCare-Data-Class: synthetic`;
- **de-duplicates concurrent identical GETs**, keyed by token + path, so two
  panels mounting at once share a single request;
- throws `ApiError` (carrying `status` and an `isExpected` getter for 4xx);
- reports the failure to telemetry once, at the boundary.

Do not call `fetch` from a component. If an endpoint is missing, add a wrapper.

### Nullable data

Backend fields are frequently optional even when the schema marks them
required — artifacts may not have been generated yet. Read defensively
(`?.`, `??`) and render an explicit empty state rather than a zeroed metric.

---

## 5. Loading / error / empty states

Every data-driven panel renders one of four states, via
`LoadingPane` / `ErrorPane` / `EmptyPane` from `components/ui/Spinner.tsx`.

The distinction that matters in this product:

> **"Not measured" must never render as "measured and fine."**

Concretely: an absent metric formats as `—`, not `0%`; an unknown status maps
to the muted badge tone, never green; a disabled evaluator states that it is
disabled instead of showing an empty pass board. `statusBadge()` in
`pages/admin/sections/safety/safetyFormat.ts` encodes this, and the rule is
covered by tests in `tests/unit/safetyFormat.test.ts`.

An "idle" pre-fetch frame counts as loading, not empty.

---

## 6. Error handling and observability

`src/lib/telemetry.ts` is the single reporting entry point.

- `reportError(error, { surface, kind, detail })` — `kind` is `"expected"`
  (4xx, disabled feature, missing artifact → `console.warn`) or `"unexpected"`
  (render crash, 5xx, TypeError → `console.error`).
- **Redaction is not optional.** Messages, stacks, and `detail` all pass
  through it: JWT/bearer patterns and emails are replaced, strings over 200
  chars are truncated (clinical prose is a disclosure risk), values under
  sensitive keys (`*token*`, `*patient_id*`, `*note*`, …) become `[redacted]`,
  and request paths are templated to `/patients/:id/labs` with query strings
  dropped.
- `registerTelemetrySink(sink)` attaches a hosted provider at bootstrap. No
  SDK is bundled — that is a data-egress decision for a deployment, not a
  default. Sinks are individually try/caught: **telemetry failing must never
  become an application failure.**

Boundaries in place:

- `ErrorBoundary` wraps the whole app in `main.tsx`, each route in `App.tsx`,
  and each admin tab in `AdminDashboard.tsx`, so one bad panel cannot blank the
  dashboard.
- `installGlobalErrorHandlers()` in `main.tsx` catches unhandled rejections and
  top-level `window.onerror`.

---

## 7. Security boundaries

The backend is authoritative for authorization. Frontend checks are UX only —
`RouteGuard` prevents a confusing blank screen, it is not a security control,
and no frontend change can grant access the API would refuse.

What the frontend *is* responsible for:

| Boundary | Control |
| -------- | ------- |
| Assistant/markdown output | `MarkdownMessage` builds React elements. No `dangerouslySetInnerHTML` anywhere in `src/`. |
| Backend-supplied links | `ExternalLink` + `lib/safeUrl.ts`: scheme allow-list (`http`/`https`/`mailto`), control-character rejection, always `rel="noopener noreferrer"`. |
| Token storage | `sessionStorage`, role-keyed, cleared on logout. Chosen over `localStorage` so a token does not outlive the tab. |
| Logs | Redacted by `lib/telemetry.ts` before any sink sees them. |
| Form input | Client-side validation is for feedback only; the backend re-validates. |

---

## 8. Accessibility conventions

- Interactive controls get an explicit accessible name when the visible label
  is ambiguous in isolation ("Fast", "Live agent", "Run" appear on several
  panels — each carries a distinct `aria-label`).
- `Button` sets `aria-busy` while `loading`, since the spinner swap is purely
  visual.
- Tables use `<caption class="sr-only">`, `scope="col"`, and `scope="row"`.
- Label/value pairs use `<dl>/<dt>/<dd>`; grouped panels use `<section
  aria-label>` with a real heading.
- Errors that appear after an action render with `role="alert"`.
- Decorative icons are `aria-hidden="true"`.

---

## 9. Testing

```powershell
npm test                # unit + component (Vitest, jsdom)
npm run test:coverage   # same, with a v8 coverage report in coverage/
npm run test:watch
npm run test:e2e        # Playwright smoke; needs a running backend
```

Conventions:

- Test **user-visible behaviour and contracts**, not implementation. Prefer
  `getByRole` / `getByText` over test ids.
- Tests must be deterministic and offline. Mock `src/api/client` with
  `vi.mock`; never hit a live service.
- Every defect fixed gets a regression test that names the defect in a comment.
- **Coverage is gated, not merely measured.** `npm run test:coverage` enforces
  the thresholds declared in `coverage.thresholds` in `vitest.config.ts`
  (statements 35, branches 62, functions 31, lines 35) and exits non-zero when
  any is missed. CI runs that exact command in the `static-quality` job, so a
  coverage regression fails the build. `src/types/**` and `main.tsx` are
  excluded from the denominator (declarations and DOM bootstrap).

The safety blocks are the highest-value target: their tests assert that absent,
disabled, and malformed artifacts never render as passing results.

---

## 10. Generated code policy

`src/types/generated-openapi.d.ts` (~10k lines) is **generated and must never
be hand-edited or split**. Regenerate it instead:

```powershell
npm run typegen         # from a running backend on :8017
npm run typegen:file    # from ../Data/openapi.json
```

CI regenerates it from the exported schema and fails on a diff, so a backend
contract change that the frontend has not picked up breaks the build. If a
line-count or complexity tool flags this file, the correct response is to
exclude it, not to refactor it.

Hand-written types in `src/types/api*.ts` are the ones components import; they
are curated views over the generated surface.

---

## 11. Local development

```powershell
# backend, from repo root
uvicorn backend.api.main:app --host 127.0.0.1 --port 8017 --reload

# frontend
cd frontend-react
npm ci
npm run dev            # http://127.0.0.1:5173
```

Point at a non-default backend with `VITE_API_BASE` in `.env.local`.

Production build: `npm run build` → `dist/`, served by nginx via the provided
`Dockerfile` / `nginx.conf`.

---

## 12. Known weaknesses

Tracked honestly rather than hidden; see
`reports/frontend_production_hardening.md` for detail.

- `MleSection.tsx` (~650 lines) still fetches 20+ artifacts in one component.
  The duplicated run/refetch logic is extracted, but the panel split is not
  done.
- `RagSection.tsx`, `ChatPanel.tsx`, and `PatientDashboard.tsx` are all
  500–650 lines. `ChatPanel` is internally decomposed; the other two are not.
- Coverage is ~39% of statements (65% branches). Patient and clinician surfaces
  have far less component coverage than the admin safety surface. The gate in
  `vitest.config.ts` is a regression floor set below measured, not a target —
  raise it as coverage genuinely improves.
- `LabTrendsChart` (Recharts) is 345 kB — the largest chunk by a wide margin.
