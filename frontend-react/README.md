# NLCare React Frontend

React + TypeScript patient, clinician, and Admin/MLE interface for the NLCare
breast cancer monitoring POC.

**Architecture, conventions, and known weaknesses: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).**

## Run locally

Start the FastAPI backend from the repository root:

```powershell
uvicorn backend.api.main:app --host 127.0.0.1 --port 8017 --reload
```

Start the React dev server:

```powershell
cd frontend-react
npm ci
npm run dev
```

Open:

- Login: http://127.0.0.1:5173/login
- Patient portal: http://127.0.0.1:5173/patient
- Clinician portal: http://127.0.0.1:5173/clinician
- Admin/MLE dashboard: http://127.0.0.1:5173/admin

Set `VITE_API_BASE` in `.env.local` to target a backend other than
`http://127.0.0.1:8017`.

## Quality checks

```powershell
npm run lint            # eslint
npm run typecheck       # tsc --noEmit, app + node projects
npm test                # vitest unit and component suites
npm run test:coverage   # same, plus a coverage report in coverage/
npm run build           # tsc -b && vite build
npm run test:e2e        # Playwright smoke; requires a running backend
```

`lint`, `typecheck`, `test`, and `build` are all run in CI
(`.github/workflows/ci.yml`). Run the repository-level gate from the project
root with `python scripts/run_quality_gate.py --skip-slow-agent --include-e2e`.

## Notes

- Session tokens are role-keyed and held in `sessionStorage`, so they do not
  outlive the browser tab. Route guards are a UX control only — the backend
  remains authoritative for authorization, and patients are scoped to their own
  records by the `/me/*` API routes.
- `src/types/generated-openapi.d.ts` is generated from the backend OpenAPI
  schema. Do not edit it by hand; run `npm run typegen:file` and commit the
  result. CI fails if it drifts from the backend schema.
- The UI must never present an unavailable, abstained, or blocked backend state
  as a confident clinical result. See the "Loading / error / empty states"
  section of the architecture doc before adding a new panel.
