# Role-Based Flows

MedicalAgent serves three portals from a single FastAPI backend and a single React frontend. Each role sees a different page set, has different API permissions, and surfaces different governance evidence.

## Patient portal

**Token:** `patientPortalAccessToken` (localStorage).
**Demo credentials:** any valid demo patient ID (e.g. `P001`) / `patient-demo`.

### Pages

| Page | Purpose |
|---|---|
| `pages/patient/PatientDashboard.tsx` | Banner + tabs for overview / support chat |
| `pages/patient/PatientBanner.tsx` | Name, diagnosis, monitoring score, status |
| `pages/patient/AiSummaryPanel.tsx` | "Key signals", "Review with care team", "About this summary" |
| `pages/patient/LabsPanel.tsx` | Latest WBC / hemoglobin / platelets + trend chart |
| `pages/patient/TimelinePanel.tsx` | Chronological events — labs, symptoms, imaging, AI flags |
| `pages/patient/SymptomsTable.tsx` | Symptom severity table |
| `pages/patient/ModelSignalPanel.tsx` | Hybrid score, calibrated P̂, SHAP bars (with engineering-only disclaimer) |

### Flow

1. Login → `RouteGuard` routes to `/patient`.
2. `useApi(getMyReport)` fetches `GET /me/patient-report`.
3. AI-flag timeline events render through the canonical `AIGeneratedLabel` row:
   `AI-generated` chip · confidence · "Clinician review required" · source/model · timestamp · uncertainty reason.
4. Support chat hits `POST /me/chat`, which runs through the safety guardrails before any retrieval or generation.
5. Symptom and lab uploads hit `POST /me/uploads`, scoped to the authenticated patient.

### Safety constraints

- Patient-facing copy never uses diagnostic vocabulary (`SAFETY_DISCLAIMER_SHORT` / `_LONG` in `lib/constants.ts`).
- Risk badges use `info` / `watch` / `urgent_review` — never "cancer detected" / "diagnosis".
- The chat panel disclaimer is always rendered: "Not a substitute for clinical advice."

## Clinician portal

**Token:** `clinicianAccessToken`.
**Demo credentials:** `clinician` / `clinician-demo`.

### Pages

| Page | Purpose |
|---|---|
| `pages/clinician/ClinicianDashboard.tsx` | Side-by-side queue + patient detail |
| `pages/clinician/ReviewQueue.tsx` | Priority-sorted list of patients with urgent flags / missing data |
| `pages/clinician/ReviewPanel.tsx` | The 7-decision review form: approve / edit / needs follow-up / missing evidence / wrong escalation / reject / unsafe |

### Flow

1. Login → routed to `/clinician`.
2. `getReviewQueue()` → `GET /clinician/review-queue?limit=25`.
   Items are sorted by `priority_score = urgent_count * 5 + review_flag_count * 2 + (no recent review ? 2 : 0) + missing_data_count`.
3. Clicking a queue item loads `getPatientReport(patientId)`.
4. `ReviewPanel` posts to `POST /patients/{id}/summary-review` with one of the canonical decisions from `REVIEW_DECISIONS`.
   Reason categories from `REVIEW_REASON_CATEGORIES` are required for `rejected` / `unsafe` / `missing_evidence` / `wrong_escalation` reviews so the audit log captures *why*.
5. Reviews never change the patient record — they are pure audit data.

### Decision vocabulary

| Decision | Use when |
|---|---|
| `approved` | AI summary is acceptable as-is for review purposes. |
| `edited` | Acceptable with the clinician's edit (included in payload). |
| `needs_followup` | Patient needs another touchpoint; AI output otherwise fine. |
| `missing_evidence` | Clinician cannot accept the summary without more data (labs, imaging). |
| `wrong_escalation` | Severity / urgency level is mis-set; route to recompute. |
| `rejected` | AI output is not useful for review purposes. |
| `unsafe` | AI output presented something that crossed the safety boundary. |

## Admin / MLE portal

**Token:** `adminAccessToken`.
**Demo credentials:** `admin` / `admin-demo`.

### Pages (tabs on `AdminDashboard`)

| Tab | Content |
|---|---|
| Overview | MLE gate statuses + quick metric grid |
| Safety & Eval Center | Safety red-team, RAG eval, drift, calibration, clinician feedback, failure case gallery |
| RAG / Cost | Retrieval metrics, KB source registry, cache stats |
| Guardrails | Input/output block counts + summary |
| MLE Gates | Training/holdout/external + cost-sensitive policy |
| Imaging MLE | Public imaging manifest + ultrasound baseline + CT lesion workflow + sim-to-public gap |
| Regression | Agent regression suite + per-case pass/fail |
| Agent Trace | RAG pipeline trace viewer |
| Feedback | Star ratings + feedback log |

### Flow

1. Login → routed to `/admin`.
2. `getAdminAnalytics()` runs once, results passed via prop to each section.
3. The Safety & Eval Center tab fetches `GET /admin/safety-center` independently and offers per-suite "Re-run" buttons that hit the corresponding `POST /admin/<suite>` endpoint.
4. Missing artifacts render as "not yet generated" empty states; freshness is conveyed via the artifact `generated_at` timestamp.

## Cross-role boundaries (auditable)

| Boundary | Enforced by |
|---|---|
| Patient role cannot read other patients' records | `get_patient_access_context` dependency — scopes every `/me/*` call to the bound `patient_id` |
| Clinician role can read all patients but cannot impersonate one in chat | `get_clinician_or_admin_context` — chat history is queue-scoped, not session-scoped |
| Admin role sees governance artifacts but does not see chat content unless explicitly opened via a patient context | `get_admin_access_context` |
| Cross-patient prompts in chat | Deterministic pattern-block in input guardrail; routed to `security_boundary` intent |

Every cross-role API call is recorded in `app_event_logs` with `actor_role`, `patient_id`, `route`, `status`, and `request_id`.

## Where new role-scoped work goes

| Adding... | Put it in... |
|---|---|
| A new patient-only endpoint | `backend/api/routers/patient.py` with `Depends(get_patient_access_context)` |
| A new clinician-only endpoint | new router file + `Depends(get_clinician_or_admin_context)` |
| A new admin-only endpoint | `backend/api/routers/admin_eval.py` or sibling, with `Depends(get_admin_access_context)` |
| A new portal page | `frontend-react/src/pages/<role>/` and `RouteGuard` in `App.tsx` |
| A new shared chip / label | `frontend-react/src/components/ui/` |
| A new constant string used by both sides | `backend/services/review_constants.py` **and** `frontend-react/src/lib/constants.ts` (the wire-contract test will fail otherwise) |
