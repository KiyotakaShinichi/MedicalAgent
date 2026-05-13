# MedicalAgent React Frontend

React + TypeScript clinician, patient, and Admin/MLE interface for the MedicalAgent breast cancer monitoring POC.

## Run Locally

Start the FastAPI backend from the repository root:

```powershell
uvicorn backend.api.main:app --host 127.0.0.1 --port 8017 --reload
```

Start the React dev server:

```powershell
cd frontend-react
npm run dev
```

Open:

- Login: http://127.0.0.1:5173/login
- Patient portal: http://127.0.0.1:5173/patient
- Clinician portal: http://127.0.0.1:5173/clinician
- Admin/MLE dashboard: http://127.0.0.1:5173/admin

## Quality Checks

```powershell
npm run lint
npm run build
npm run test:e2e
```

The frontend uses role-keyed localStorage tokens, typed API wrappers, route guards, and the `/me/*` patient-scoped API routes so patients only see their own records.

Run the repository-level quality gate from the project root with `python scripts/run_quality_gate.py --skip-slow-agent --include-e2e`.
