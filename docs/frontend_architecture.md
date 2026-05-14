# Frontend Architecture

MedicalAgent has two frontend implementations:

| Frontend | Technology | Purpose |
|----------|-----------|---------|
| `frontend/` | Vanilla HTML + Chart.js | Legacy, left intact |
| `frontend-react/` | React 19 + TypeScript + Vite + Tailwind v4 | Current, production-ready |

---

## React Frontend Structure

```
frontend-react/src/
├── api/
│   └── client.ts           Typed API wrapper. Reads Bearer token from localStorage.
│                           All calls go to http://127.0.0.1:8017.
├── context/
│   ├── AuthContext.tsx     AuthProvider — wraps app, manages session lifecycle
│   └── authContextCore.ts  createContext + loadSession helper (split for tree-shaking)
├── hooks/
│   ├── useAuth.ts          Reads AuthContext; throws outside provider
│   └── useApi.ts           Generic data-fetching hook: status | data | error | refetch
├── types/
│   └── api.ts              TypeScript types mirroring backend Pydantic models
├── components/
│   ├── ui/
│   │   ├── Badge.tsx           Status badge with 7 colour variants
│   │   ├── badgeUtils.ts       statusVariant() helper (status string → Variant)
│   │   ├── Button.tsx          4 variants × 3 sizes, loading spinner
│   │   ├── Card.tsx            Surface card + CardHeader + SectionTitle
│   │   ├── ChatPanel.tsx       Full chat UI: Enter-send, Shift+Enter newline,
│   │   │                       tool-call chips, citation display, loading state
│   │   ├── MetricCard.tsx      KPI card with label / value / unit / status colour
│   │   ├── MetricInterpretation.tsx
│   │   │                       Metric spec + ideal/warning/bad bands + glossary grid
│   │   └── Spinner.tsx         Spinner, LoadingPane, ErrorPane, EmptyPane
│   ├── layout/
│   │   ├── AppShell.tsx        Collapsible sidebar + topbar + PoC disclaimer badge
│   │   └── RouteGuard.tsx      Role-based redirect: wrong role → own dashboard
│   └── charts/
│       ├── LabTrendsChart.tsx      Recharts multi-line CBC over time
│       └── ConfusionMatrixPanel.tsx 2×2 grid + sensitivity/specificity/precision/FNR
└── pages/
    ├── Login.tsx               Split-panel login: hero copy + credential form + demo pills
    ├── patient/
    │   ├── PatientDashboard.tsx    AppShell wrapper, tab: Overview | Support Chat
    │   ├── PatientBanner.tsx       Name, diagnosis, monitoring score, status badge
    │   ├── AiSummaryPanel.tsx      3-column: Key signals | Review with care team | About
    │   ├── LabsPanel.tsx           3 metric cards + LabTrendsChart
    │   ├── ModelSignalPanel.tsx    Hybrid score, calibrated P̂, SHAP bars, signal badges
    │   ├── TimelinePanel.tsx       Chronological dot-timeline with severity colour
    │   └── SymptomsTable.tsx       Severity bar chart inline + notes
    ├── clinician/
    │   ├── ClinicianDashboard.tsx  Left sidebar queue + right panel (patient detail)
    │   ├── ReviewQueue.tsx         Priority-sorted list with urgent flag count
    │   └── ReviewPanel.tsx         Approve/edit/reject/mark-unsafe + star scores + audit
    └── admin/
        ├── AdminDashboard.tsx      Tab-strip shell loading analytics once, sharing via prop
        └── sections/
            ├── OverviewSection.tsx     MLE gate statuses + quick metric grid
            ├── RagSection.tsx          RAG metrics + KB source table
            ├── GuardrailsSection.tsx   Input/output block counts + policy summary
            ├── MleSection.tsx          Training/holdout/ext-val + metric bands + cost-sensitive eval
            ├── RegressionSection.tsx   Run suite button + per-case pass/fail + category breakdown
            ├── AgentTraceSection.tsx   Pipeline architecture + per-call trace viewer
            └── FeedbackSection.tsx     Star rating summary + feedback log table
```

---

## Routing and Auth

- `BrowserRouter` with four top-level routes: `/login`, `/patient/*`, `/clinician/*`, `/admin/*`
- `RouteGuard` wraps each authenticated route — redirects unauthenticated users to `/login` and wrong-role users to their own dashboard
- Tokens stored in localStorage, keyed by role (`patientPortalAccessToken`, `clinicianAccessToken`, `adminAccessToken`)
- `AuthContext` clears all keys on logout; new login clears old keys before writing the new one

---

## State management

No global store. State is component-local or lifted as needed:

| Layer | Pattern |
|-------|---------|
| Auth | React Context (`AuthContext`) |
| Remote data | `useApi(fn, deps)` — runs once, exposes `refetch` |
| Chat history | Local state in `ChatPanel`, initialised from API |
| Admin section | Parent `AdminDashboard` fetches analytics once, passes via props |

---

## Design system

- **Dark theme** — CSS custom properties (`--bg`, `--surface`, `--text`, `--rose`, etc.)
- **Typography** — Inter (UI) + JetBrains Mono (code/values)
- **Tailwind v4** via `@tailwindcss/vite` — utility classes for layout, spacing, flex/grid
- **Inline styles** for design-token colours (avoids Tailwind colour purging issues)
- **Responsive** — desktop-first at 1366px+; mobile breakpoints collapse sidebar and stack grids
- **Code splitting** — Vite splits Admin/Patient/Clinician dashboards into separate chunks automatically

---

## Build

```bash
cd frontend-react
npm install
npm run build   # tsc -b && vite build
npm run preview # serve dist/ locally
```

Production build output: `dist/` (~650 kB JS gzipped ~194 kB).

---

## Safety design decisions

- No role-switching UI inside authenticated surfaces (clinicians cannot downgrade to patient view)
- PoC disclaimer badge in every topbar: "PoC — Not for clinical use"
- Chat disclaimer: "Not a substitute for clinical advice. Always consult your care team."
- AI summary always includes a "About this summary" panel explaining LLM limitations
- Model signal panel shows amber warning: "Exploratory engineering signal only — not a clinical prediction."

---

## Shared safe-language primitives

The following components are the **canonical** surface for any AI/model output. Inlining color codes or building one-off chips is a regression — use these instead.

| Primitive | File | Purpose |
|-----------|------|---------|
| `RiskBadge` | [components/ui/RiskBadge.tsx](../frontend-react/src/components/ui/RiskBadge.tsx) | Risk-level chip (`info` / `watch` / `urgent_review`) — never says "diagnosis" |
| `ConfidenceBadge` | [components/ui/ConfidenceBadge.tsx](../frontend-react/src/components/ui/ConfidenceBadge.tsx) | Confidence chip (`low` / `moderate` / `high`) — never says "certainty" |
| `ReviewStatusBadge` | [components/ui/ReviewStatusBadge.tsx](../frontend-react/src/components/ui/ReviewStatusBadge.tsx) | Clinician decision chip mirroring [REVIEW_DECISIONS](../frontend-react/src/lib/constants.ts) |
| `AIGeneratedLabel` | [components/ui/AIGeneratedLabel.tsx](../frontend-react/src/components/ui/AIGeneratedLabel.tsx) | "AI-generated • confidence • clinician review required • source/model · timestamp" row |
| `EmptyState`, `LoadingState`, `ErrorState` | [components/ui/states.ts](../frontend-react/src/components/ui/states.ts) | Re-exports of `EmptyPane` / `LoadingPane` / `ErrorPane` |

### Shared constants

[`lib/constants.ts`](../frontend-react/src/lib/constants.ts) is the single TypeScript source of truth for:

- `REVIEW_DECISIONS` / `REVIEW_DECISION_LABELS`
- `REVIEW_TARGETS`
- `REVIEW_REASON_CATEGORIES` / `REVIEW_REASON_LABELS`
- `RISK_LEVELS` / `RISK_LEVEL_LABELS`
- `CONFIDENCE_LEVELS`
- `ARTIFACT_STATUSES`
- `SAFETY_DISCLAIMER_SHORT` / `SAFETY_DISCLAIMER_LONG`
- `AI_GENERATED_LABEL` / `CLINICIAN_REVIEW_REQUIRED_LABEL`

These mirror [backend/services/review_constants.py](../backend/services/review_constants.py); update both when changing any string.

### Safe-language rule

- ✅ "For clinician review", "Possible concern", "Needs review", "Insufficient evidence", "Urgent escalation recommended", "Not a diagnosis"
- ❌ "Cancer detected", "Metastasis confirmed", "Diagnosis", "Treatment failed", "Guaranteed response"

If a label feels diagnostic when you read it back, replace it with one of the safe phrases above.

### Where new work goes

| If you are adding...    | Put it in... |
|-------------------------|--------------|
| A new API call          | `api/client.ts` + a type in `types/api.ts` |
| A new admin section     | `pages/admin/sections/XxxSection.tsx` |
| A new shared chip       | `components/ui/XxxBadge.tsx` |
| A new role-scoped page  | `pages/<role>/XxxPage.tsx` and wire it in `App.tsx` |
| A new safe label        | `lib/constants.ts` (so the BE mirror gets updated too) |

### Loading / error / empty discipline

Every async surface must render all four cases:

```tsx
if (status === "loading") return <LoadingState label="Loading patients..." />;
if (status === "error")   return <ErrorState message={error ?? "Failed to load"} />;
if (!data)                return <EmptyState label="No data" />;
return <ActualUI data={data} />;
```

Pages that consume optional artifacts (Safety & Eval Center cards) must render a "not yet generated" state instead of crashing — the backend returns `{ status: "not_generated", message, path }` for any missing artifact.
