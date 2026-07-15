import type {
  LoginResponse,
  DemoPatient,
  PatientReport,
  ChatResponse,
  ChatStreamHandlers,
  PatientSummary,
  ReviewQueueItem,
  SummaryReview,
  AdminAnalytics,
  AgentRegressionResult,
  MleReadinessSummary,
  AgentFeedbackItem,
  RagSourceEntry,
  AgentTraceLogsResponse,
  NoiseEvalResult,
  TemporalEvalResult,
  PredictionErrorTable,
  RagAblationResult,
  PublicDataManifest,
  PublicBiomarkerDatasetManifest,
  PublicBiomarkerMappingReadiness,
  CbioportalBiomarkerSchemaMapping,
  ClinicalSafetyReviewChecklist,
  SystemHealth,
  PublicImagingManifest,
  UltrasoundBaselineResult,
  CtLesionWorkflowReport,
  SimToPublicImagingReport,
  GeneticCounselingReadiness,
  HighRiskConversationAlertsResponse,
  HighRiskConversationAlert,
} from "../types/api";

/**
 * Backend base URL.  Resolved from (in order):
 *   1. Vite `VITE_API_BASE` env var (set in .env.local for non-default hosts)
 *   2. `http://127.0.0.1:8017` fallback for the local dev profile
 *
 * Exported so the ErrorPane + tool trace drawer can show the actual host
 * the frontend is trying to talk to.
 */
export const API_BASE: string =
  (import.meta as unknown as { env?: { VITE_API_BASE?: string } }).env?.VITE_API_BASE
    ?? "http://127.0.0.1:8017";

const BASE = API_BASE;
const inFlightGetRequests = new Map<string, Promise<unknown>>();

function getToken(): string | null {
  return (
    localStorage.getItem("patientPortalAccessToken") ||
    localStorage.getItem("clinicianAccessToken") ||
    localStorage.getItem("adminAccessToken")
  );
}

async function request<T>(
  method: string,
  path: string,
  body?: unknown
): Promise<T> {
  const token = getToken();
  const cacheKey = method === "GET" && body === undefined ? `${token ?? "anon"}:${path}` : null;
  if (cacheKey && inFlightGetRequests.has(cacheKey)) {
    return inFlightGetRequests.get(cacheKey) as Promise<T>;
  }

  const promise = fetch(`${BASE}${path}`, {
    method,
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    ...(body !== undefined ? { body: JSON.stringify(body) } : {}),
  })
    .then(async (res) => {
      if (!res.ok) {
        const text = await res.text().catch(() => res.statusText);
        throw new Error(`${res.status}: ${text}`);
      }
      return res.json() as Promise<T>;
    })
    .finally(() => {
      if (cacheKey) inFlightGetRequests.delete(cacheKey);
    });

  if (cacheKey) inFlightGetRequests.set(cacheKey, promise);
  return promise;
}

const get = <T>(path: string) => request<T>("GET", path);
const post = <T>(path: string, body?: unknown) => request<T>("POST", path, body);
const del = <T>(path: string) => request<T>("DELETE", path);

// Auth
export const login = (username: string, password: string) =>
  post<LoginResponse>("/auth/demo-credential-login", { username, password });

export const getDemoPatients = () =>
  get<{ patients: DemoPatient[] }>("/auth/demo-patients");

export const whoami = () =>
  get<{ role: string; patient_id: string | null }>("/auth/whoami");

// Patient
export const getMyReport = () => get<PatientReport>("/me/patient-report");

export const getMyReportCore = () => get<PatientReport>("/me/patient-report/core");

export const getMyReportEnrichment = () =>
  get<Partial<PatientReport>>("/me/patient-report/enrichment");

export const getMyChatHistory = () =>
  get<{ patient_id: string; messages: import("../types/api").ChatMessage[] }>("/me/chat");

export const sendMyChat = (message: string) =>
  post<ChatResponse>("/me/chat", { message });

export const undoMyConfirmedRecordWrite = (auditId: number) =>
  del<{ message: string; action: import("../types/api").SavedAction }>(`/me/record-write-actions/${auditId}`);

export async function sendMyChatStream(
  message: string,
  handlers: ChatStreamHandlers = {},
): Promise<ChatResponse> {
  return streamChat("/me/chat/stream", message, handlers);
}

async function streamChat(
  path: string,
  message: string,
  handlers: ChatStreamHandlers,
): Promise<ChatResponse> {
  const token = getToken();
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify({ message }),
  });
  if (!res.ok || !res.body) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`${res.status}: ${text}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finalAnswer: ChatResponse | null = null;

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split("\n\n");
    buffer = events.pop() ?? "";
    for (const eventBlock of events) {
      const event = parseSseEvent(eventBlock);
      if (!event) continue;
      if (event.name === "pipeline_stage") {
        handlers.onStage?.(String(event.data?.label ?? ""));
      } else if (event.name === "answer_delta") {
        handlers.onDelta?.(String(event.data?.text ?? ""));
      } else if (event.name === "answer") {
        finalAnswer = {
          reply: String(event.data?.reply ?? ""),
          saved_actions: Array.isArray(event.data?.saved_actions) ? event.data.saved_actions : [],
          citations: normalizeCitationLabels(event.data?.citations),
          assistant_message_id:
            typeof event.data?.assistant_message_id === "string" || typeof event.data?.assistant_message_id === "number"
              ? event.data.assistant_message_id
              : undefined,
        };
      } else if (event.name === "error") {
        throw new Error(String(event.data?.error ?? "Streaming chat failed"));
      }
    }
  }

  if (!finalAnswer) {
    throw new Error("Streaming chat ended without an answer.");
  }
  return finalAnswer;
}

function parseSseEvent(block: string): { name: string; data: Record<string, unknown> } | null {
  const eventLine = block.split("\n").find((line) => line.startsWith("event:"));
  const dataLine = block.split("\n").find((line) => line.startsWith("data:"));
  if (!eventLine || !dataLine) return null;
  try {
    return {
      name: eventLine.replace("event:", "").trim(),
      data: JSON.parse(dataLine.replace("data:", "").trim()),
    };
  } catch {
    return null;
  }
}

function normalizeCitationLabels(raw: unknown): string[] {
  if (!Array.isArray(raw)) return [];
  return raw
    .map((item) => {
      if (typeof item === "string") return item;
      if (item && typeof item === "object") {
        const source = item as { title?: unknown; source_name?: unknown; id?: unknown };
        return String(source.title ?? source.source_name ?? source.id ?? "").trim();
      }
      return "";
    })
    .filter(Boolean);
}

export const submitFeedback = (payload: {
  chat_message_id?: string;
  rating: number;
  thumbs_up: boolean;
  feedback_text: string;
}) => post<{ message: string }>("/me/agent-feedback", payload);

export const uploadFile = (payload: {
  upload_type: string;
  file_name: string;
  content_type: string;
  content_base64: string;
  notes: string;
  scan_date?: string;
}) => post<{ message: string; upload: unknown }>("/me/uploads", payload);

/**
 * Patient-scoped symptom save (manual-entry form).  Mirrors POST /me/symptoms
 * on the backend.  Date is yyyy-mm-dd.  ``urgent_flag`` is the explicit
 * patient-set checkbox; the backend folds it into the notes column with an
 * `[urgent flag]` tag so the existing review queue picks it up — it does
 * NOT auto-trigger any safety routing on its own.
 */
export interface AddMySymptomPayload {
  date: string;
  symptom: string;
  severity: number;
  notes?: string;
  duration?: string;
  urgent_flag?: boolean;
}

export interface AddMySymptomResponse {
  message: string;
  symptom_id: number;
  validation_warnings: { field: string; level: string; message: string }[];
  urgent_flag: boolean;
  ctcae_review_hint?: {
    schema_version: string;
    patient_severity: number;
    patient_severity_bucket: string;
    ctcae_hint: string;
    urgent_review: boolean;
    red_flag_terms: string[];
    review_focus: string[];
    claim_boundary: string;
  };
  safety_note: string;
}

export const addMySymptom = (payload: AddMySymptomPayload) =>
  post<AddMySymptomResponse>("/me/symptoms", payload);

// ─── Patient-scoped tool saves ───────────────────────────────────────────────

export interface AddMyLabPayload {
  date: string;
  wbc: number;
  hemoglobin: number;
  platelets: number;
  anc?: number;
  lab_source?: string;
  notes?: string;
}

export interface AddMyLabResponse {
  message: string;
  lab_id: number;
  validation_warnings: { field: string; level: string; message: string }[];
  reference_context?: {
    schema_version: string;
    demographics_used: Record<string, unknown>;
    context_type: string;
    labs: Record<string, {
      value: number;
      unit: string;
      reference_range: { low: number; high: number };
      status: string;
      range_source: string;
    }>;
    limitations: string[];
    claim_boundary: string;
  };
  safety_note: string;
}

export const addMyLab = (payload: AddMyLabPayload) =>
  post<AddMyLabResponse>("/me/labs", payload);

export interface AddMyImagingReportPayload {
  date: string;
  modality: string;          // MRI / CT / Ultrasound / Mammogram / Other
  report_type?: string;
  body_site?: string;
  findings?: string;
  impression?: string;
  notes?: string;
}

export interface AddMyImagingReportResponse {
  message: string;
  imaging_report_id: number;
  modality: string;
  validation_warnings: { field: string; level: string; message: string }[];
  safety_note: string;
}

export const addMyImagingReport = (payload: AddMyImagingReportPayload) =>
  post<AddMyImagingReportResponse>("/me/imaging-reports", payload);

export interface AddMyMedicationPayload {
  medication: string;
  dose?: string;
  frequency?: string;
  date: string;
  side_effects?: string;
  notes?: string;
}

export interface AddMyMedicationResponse {
  message: string;
  medication_id: number;
  safety_note: string;
  interaction_check?: {
    checker_version: string;
    status: string;
    flags: Array<{
      rule_id: string;
      severity: string;
      message: string;
      clinician_action: string;
      matched_trigger_terms?: string[];
      matched_context_terms?: string[];
    }>;
    claim_boundary: string;
  };
}

export const addMyMedication = (payload: AddMyMedicationPayload) =>
  post<AddMyMedicationResponse>("/me/medications", payload);

export interface AddMyTreatmentPayload {
  date: string;
  drug: string;
  cycle?: number;
  notes?: string;
}

export interface AddMyTreatmentResponse {
  message: string;
  treatment_id: number;
  validation_warnings: { field: string; level: string; message: string }[];
  safety_note: string;
}

export const addMyTreatment = (payload: AddMyTreatmentPayload) =>
  post<AddMyTreatmentResponse>("/me/treatments", payload);

export const getMyGeneticCounselingReadiness = () =>
  get<GeneticCounselingReadiness>("/me/genetic-counseling-readiness");

export const addMyFamilyHistory = (payload: {
  relationship: string;
  family_side: string;
  cancer_type: string;
  age_at_diagnosis?: number | null;
  relative_status?: string | null;
  multiple_relatives_affected?: string | null;
  male_breast_cancer?: string | null;
  known_familial_mutation?: string | null;
  bilateral_breast_cancer?: string | null;
  multiple_primary_cancers?: string | null;
  ancestry_ethnicity?: string | null;
  prior_breast_biopsy_atypia?: string | null;
  relation_degree?: string | null;
  notes?: string | null;
}) => post<{ message: string; record: unknown; boundary_note: string }>("/me/family-history", payload);

export const addMyGeneticTestRecord = (payload: {
  test_type: string;
  sample_type: string;
  gene?: string | null;
  variant_text?: string | null;
  classification?: string | null;
  report_date?: string | null;
  lab_provider?: string | null;
  upload_reference?: string | null;
  reviewed_by_genetic_counselor?: string | null;
  notes?: string | null;
}) => post<{ message: string; record: unknown; boundary_note: string }>("/me/genetic-test-records", payload);

export const addMyBiomarkerRecord = (payload: {
  source: string;
  er_status?: string | null;
  pr_status?: string | null;
  her2_status?: string | null;
  ki67_percent?: number | null;
  grade?: string | null;
  stage?: string | null;
  report_date?: string | null;
  report_text?: string | null;
  upload_reference?: string | null;
}) => post<{ message: string; record: unknown; boundary_note: string }>("/me/biomarker-records", payload);

export const addMyTumorMarkerRecord = (payload: {
  marker: string;
  value: number;
  unit?: string | null;
  reference_range?: string | null;
  date_collected?: string | null;
  trend_direction?: string | null;
  notes?: string | null;
}) => post<{ message: string; record: unknown; boundary_note: string }>("/me/tumor-marker-records", payload);

// Clinician
export const getPatients = () => get<PatientSummary[]>("/patients");

export const getPatientReport = (patientId: string) =>
  get<PatientReport>(`/patient-report/${patientId}`);

export const getPatientGeneticCounselingReadiness = (patientId: string) =>
  get<GeneticCounselingReadiness>(`/patients/${patientId}/genetic-counseling-readiness`);

export const getReviewQueue = () =>
  get<{ queue: ReviewQueueItem[] }>("/clinician/review-queue?limit=25");

export const getHighRiskConversationAlerts = () =>
  get<HighRiskConversationAlertsResponse>("/clinician/high-risk-conversation-alerts?limit=25");

export const acknowledgeHighRiskConversationAlert = (alertId: number, note?: string) =>
  post<{ message: string; alert: HighRiskConversationAlert; safety_note: string }>(
    `/clinician/high-risk-conversation-alerts/${alertId}/acknowledge`,
    { note: note ?? "Acknowledged from the clinician review dashboard." },
  );

export const getSummaryReviews = (patientId: string) =>
  get<{ summary_reviews: SummaryReview[] }>(
    `/summary-reviews?patient_id=${patientId}&limit=10`
  );

export const submitSummaryReview = (
  patientId: string,
  payload: {
    decision: string;
    clinician_notes: string;
    edited_patient_summary?: string;
    explanation_quality_score?: number;
    model_usefulness_score?: number;
    review_target?: string;
    reason_category?: string;
    model_version?: string;
    rag_version?: string;
  }
) => post<{ message: string; review: SummaryReview }>(`/patients/${patientId}/summary-review`, payload);

export const submitGeneticCounselingReview = (
  patientId: string,
  payload: { decision: string; notes?: string | null }
) => post<{ message: string; review: unknown }>(`/patients/${patientId}/genetic-counseling-review`, payload);

export const addLab = (
  patientId: string,
  payload: { date: string; wbc: number; hemoglobin: number; platelets: number }
) => post<{ message: string }>(`/patients/${patientId}/labs`, payload);

export const addSymptom = (
  patientId: string,
  payload: { date: string; symptom: string; severity: number; notes: string }
) => post<{ message: string }>(`/patients/${patientId}/symptoms`, payload);

export const sendClinicianChat = (patientId: string, message: string) =>
  post<ChatResponse>(`/patients/${patientId}/chat`, { message });

// Admin
export const getAdminAnalytics = () => get<AdminAnalytics>("/admin/analytics");

export const runAgentRegression = () =>
  post<{ message: string; result: AgentRegressionResult }>("/admin/agent-regression");

export const runMleReadiness = () =>
  post<{ message: string; result: MleReadinessSummary }>("/admin/mle-readiness");

export const getTrainingReport = () =>
  get<{ message: string; result: unknown }>("/admin/training-evaluation-report");

export const runTrainingReport = () =>
  post<{ message: string; result: unknown }>("/admin/training-evaluation-report");

export const getLockedHoldout = () =>
  get<{ message: string; result: unknown }>("/admin/locked-holdout-evaluation");

export const getExternalValidation = () =>
  get<{ message: string; result: unknown }>("/admin/external-validation");

export const getModelComparison = () =>
  get<{ message: string; result: unknown }>("/admin/model-comparison");

export const getAgentFeedback = () =>
  get<{ summary: import("../types/api").AgentFeedbackSummary; feedback: AgentFeedbackItem[] }>(
    "/agent-feedback?limit=50"
  );

export const getRagSourceRegistry = () =>
  get<{ sources: RagSourceEntry[]; metrics: unknown }>("/admin/rag-source-registry");

export const getAgentTraceLogs = (limit = 50) =>
  get<AgentTraceLogsResponse>(`/admin/agent-trace-logs?limit=${limit}`);

export const getNoiseEval = () =>
  get<NoiseEvalResult>("/admin/noise-eval");

export const runNoiseEval = () =>
  post<{ message: string; result: NoiseEvalResult }>("/admin/noise-eval");

export const getTemporalEval = () =>
  get<TemporalEvalResult>("/admin/temporal-eval");

export const runTemporalEval = () =>
  post<{ message: string; result: TemporalEvalResult }>("/admin/temporal-eval");

export const getPredictionErrorTable = (limit = 100) =>
  get<PredictionErrorTable>(`/admin/prediction-error-table?limit=${limit}`);

export const getRagAblation = () =>
  get<RagAblationResult>("/admin/rag-ablation");

export const runRagAblation = () =>
  post<{ message: string; result: RagAblationResult }>("/admin/rag-ablation");

export const getPublicDataManifest = () =>
  get<PublicDataManifest>("/admin/public-data-manifest");

export const runPublicDataManifest = () =>
  post<{ message: string; result: PublicDataManifest }>("/admin/public-data-manifest");

export const getPublicBiomarkerDatasetManifest = () =>
  get<PublicBiomarkerDatasetManifest>("/admin/public-biomarker-dataset-manifest");

export const runPublicBiomarkerDatasetManifest = () =>
  post<{ message: string; result: PublicBiomarkerDatasetManifest }>("/admin/public-biomarker-dataset-manifest");

export const getPublicBiomarkerMappingReadiness = () =>
  get<PublicBiomarkerMappingReadiness>("/admin/public-biomarker-mapping-readiness");

export const runPublicBiomarkerMappingReadiness = () =>
  post<{ message: string; result: PublicBiomarkerMappingReadiness }>("/admin/public-biomarker-mapping-readiness");

export const getCbioportalBiomarkerSchemaMapping = () =>
  get<CbioportalBiomarkerSchemaMapping>("/admin/cbioportal-biomarker-schema-mapping");

export const runCbioportalBiomarkerSchemaMapping = (liveFetch = true) =>
  post<{ message: string; result: CbioportalBiomarkerSchemaMapping }>(
    `/admin/cbioportal-biomarker-schema-mapping?live_fetch=${liveFetch ? "true" : "false"}`
  );

export const getClinicalSafetyReviewChecklist = () =>
  get<ClinicalSafetyReviewChecklist>("/admin/clinical-safety-review-checklist");

export const runClinicalSafetyReviewChecklist = () =>
  post<{ message: string; result: ClinicalSafetyReviewChecklist }>("/admin/clinical-safety-review-checklist");

export const getSystemHealth = () =>
  get<SystemHealth>("/admin/system-health");

export const runSystemHealth = () =>
  post<{ message: string; result: SystemHealth }>("/admin/system-health");

// FAST_MODE runtime toggle — emergency degradation switch when the
// Groq cloud provider is rate-limited / down.  enabled=true forces
// fast mode ON, false forces it OFF, null clears the runtime override
// and falls back to the ONCOTRACK_FAST_MODE env var.
export interface FastModeStatus {
  enabled: boolean;
  env_var_value: string | null;
  env_var_active: boolean;
  runtime_override: boolean | null;
  source: "env_var" | "runtime_override";
}

export const getAdminFastMode = () =>
  get<FastModeStatus>("/admin/fast-mode");

export const setAdminFastMode = (enabled: boolean | null) =>
  post<FastModeStatus>("/admin/fast-mode", { enabled });

// Compound-intent live probe — paste a message, see how the router
// would classify it (deterministic + LLM-merged envelope + raw LLM
// verdict).  Stateless: does NOT touch the chat DB.
export interface IntentProbeResponse {
  status: "ok" | "empty";
  message?: string;
  deterministic: import("../types/api").CompoundIntentTrace;
  merged: import("../types/api").CompoundIntentTrace;
  llm: {
    available: boolean;
    language?: string;
    llm_confidence?: number;
    provider?: string;
    model?: string;
    reason?: string;
  };
}

export const probeAdminIntent = (message: string, useLlm: boolean = true) =>
  post<IntentProbeResponse>("/admin/intent-classifier-probe", {
    message,
    use_llm: useLlm,
  });

export const getPublicImagingManifest = () =>
  get<PublicImagingManifest>("/admin/public-imaging-manifest");

export const runPublicImagingManifest = () =>
  post<{ message: string; result: PublicImagingManifest }>("/admin/public-imaging-manifest");

export const getUltrasoundBaseline = () =>
  get<UltrasoundBaselineResult>("/admin/ultrasound-baseline");

export const runUltrasoundBaseline = () =>
  post<{ message: string; result: UltrasoundBaselineResult }>("/admin/ultrasound-baseline");

export const getUltrasoundTransferBaseline = () =>
  get<import("../types/api").UltrasoundTransferBaselineResult>("/admin/ultrasound-transfer-baseline");

export const runUltrasoundTransferBaseline = (pretrained = false) =>
  post<{ message: string; result: import("../types/api").UltrasoundTransferBaselineResult }>(
    `/admin/ultrasound-transfer-baseline?pretrained=${pretrained ? "true" : "false"}`
  );

export const getUltrasoundSegmentationBaseline = () =>
  get<import("../types/api").UltrasoundSegmentationBaselineResult>("/admin/ultrasound-segmentation-baseline");

export const runUltrasoundSegmentationBaseline = () =>
  post<{ message: string; result: import("../types/api").UltrasoundSegmentationBaselineResult }>(
    "/admin/ultrasound-segmentation-baseline"
  );

export const getCtLesionWorkflow = () =>
  get<CtLesionWorkflowReport>("/admin/ct-lesion-workflow");

export const runCtLesionWorkflow = () =>
  post<{ message: string; result: CtLesionWorkflowReport }>("/admin/ct-lesion-workflow");

export const getSimToPublicImaging = () =>
  get<SimToPublicImagingReport>("/admin/sim-to-public-imaging");

export const runSimToPublicImaging = () =>
  post<{ message: string; result: SimToPublicImagingReport }>("/admin/sim-to-public-imaging");

export const getCurrentVsRealismCandidate = () =>
  get<import("../types/api").CurrentVsRealismCandidateReport>("/admin/current-vs-realism-candidate");

export const runCurrentVsRealismCandidate = () =>
  post<{ message: string; result: import("../types/api").CurrentVsRealismCandidateReport }>(
    "/admin/current-vs-realism-candidate"
  );

export const getBiomarkerFeatureBenchmark = () =>
  get<unknown>("/admin/biomarker-feature-benchmark");

export const runBiomarkerFeatureBenchmark = () =>
  post<{ message: string; result: unknown }>("/admin/biomarker-feature-benchmark");

export const getFullFeatureGroupAblation = () =>
  get<import("../types/api").FullFeatureGroupAblationReport>("/admin/full-feature-group-ablation");

export const runFullFeatureGroupAblation = () =>
  post<{ message: string; result: import("../types/api").FullFeatureGroupAblationReport }>(
    "/admin/full-feature-group-ablation",
  );

export const getLeakageAudit = () =>
  get<import("../types/api").LeakageAuditReport>("/admin/leakage-audit");

export const runLeakageAudit = () =>
  post<{ message: string; result: import("../types/api").LeakageAuditReport }>(
    "/admin/leakage-audit",
  );

export const getKbSourceGovernance = () =>
  get<import("../types/api").KbSourceGovernanceReport>("/admin/kb-source-governance");

export const runKbSourceGovernance = () =>
  post<{ message: string; result: import("../types/api").KbSourceGovernanceReport }>(
    "/admin/kb-source-governance",
  );

export const getSyntheticGeneratorCard = () =>
  get<import("../types/api").SyntheticGeneratorCard>("/admin/synthetic-generator-card");

export const runSyntheticGeneratorCard = () =>
  post<{ message: string; result: import("../types/api").SyntheticGeneratorCard }>(
    "/admin/synthetic-generator-card",
  );

export const getFailureModeRegistry = () =>
  get<import("../types/api").FailureModeRegistry>("/admin/failure-mode-registry");

export const runFailureModeRegistry = () =>
  post<{ message: string; result: import("../types/api").FailureModeRegistry }>(
    "/admin/failure-mode-registry",
  );

export const getModalityRobustnessComparison = () =>
  get<import("../types/api").ModalityRobustnessComparisonReport>(
    "/admin/modality-robustness-comparison",
  );

export const runModalityRobustnessComparison = () =>
  post<{ message: string; result: import("../types/api").ModalityRobustnessComparisonReport }>(
    "/admin/modality-robustness-comparison",
  );

export const getResponseConformalCalibration = () =>
  get<import("../types/api").ResponseConformalCalibrationReport>(
    "/admin/response-conformal-calibration",
  );

export const runResponseConformalCalibration = () =>
  post<{ message: string; result: import("../types/api").ResponseConformalCalibrationReport }>(
    "/admin/response-conformal-calibration",
  );

export const getRobustnessStress = () =>
  get<import("../types/api").RobustnessStressReport>("/admin/robustness-stress");

export const runRobustnessStress = () =>
  post<{ message: string; result: import("../types/api").RobustnessStressReport }>(
    "/admin/robustness-stress",
  );

export const getClinicianPatientPredictionTraces = (patient_id: string, params?: {
  limit?: number;
  abstained_only?: boolean;
}) => {
  const q = new URLSearchParams();
  if (params?.limit != null) q.set("limit", String(params.limit));
  if (params?.abstained_only) q.set("abstained_only", "true");
  const suffix = q.toString() ? `?${q.toString()}` : "";
  return get<import("../types/api").ClinicianPredictionTracesResponse>(
    `/clinician/patients/${encodeURIComponent(patient_id)}/prediction-traces${suffix}`,
  );
};

export const getPredictionTraces = (params?: {
  limit?: number;
  patient_id?: string;
  decision?: string;
  abstained_only?: boolean;
}) => {
  const q = new URLSearchParams();
  if (params?.limit != null) q.set("limit", String(params.limit));
  if (params?.patient_id) q.set("patient_id", params.patient_id);
  if (params?.decision) q.set("decision", params.decision);
  if (params?.abstained_only) q.set("abstained_only", "true");
  const suffix = q.toString() ? `?${q.toString()}` : "";
  return get<import("../types/api").PredictionTraceResponse>(`/admin/prediction-traces${suffix}`);
};

export const getEvidenceAbstentionEval = () =>
  get<import("../types/api").EvidenceAbstentionEvalReport>("/admin/evidence-abstention-eval");

export const runEvidenceAbstentionEval = () =>
  post<{ message: string; result: import("../types/api").EvidenceAbstentionEvalReport }>(
    "/admin/evidence-abstention-eval",
  );

export const getMultilingualRefusalEval = () =>
  get<import("../types/api").MultilingualRefusalEval>("/admin/multilingual-refusal-eval");

export const runMultilingualRefusalEval = () =>
  post<{ message: string; result: import("../types/api").MultilingualRefusalEval }>(
    "/admin/multilingual-refusal-eval"
  );

export const getLlmJudgeEval = () =>
  get<import("../types/api").LlmJudgeEval>("/admin/llm-judge-eval");

export const runLlmJudgeEval = (maxCases = 30) =>
  post<{ message: string; result: import("../types/api").LlmJudgeEval }>(
    `/admin/llm-judge-eval?max_cases=${maxCases}`
  );

export const getBenchmarkRegistry = () =>
  get<unknown>("/admin/benchmark-registry");

export const runBenchmarkRegistry = () =>
  post<{ message: string; result: unknown }>("/admin/benchmark-registry");

export const getNormalizedBenchmarkArtifact = (artifactId: string) =>
  get<import("../types/api").AdminBenchmarkResponse>(
    `/admin/benchmark-artifacts/${encodeURIComponent(artifactId)}`,
  );

export const getLiveRagEval = () =>
  get<unknown>("/admin/live-rag-eval");

export const runLiveRagEval = () =>
  post<{ message: string; result: unknown }>("/admin/live-rag-eval");

export const getClaimLevelCitationEval = () =>
  get<unknown>("/admin/claim-level-citation-eval");

export const getRagTraceReplay = (limit = 12) =>
  get<unknown>(`/admin/rag-trace-replay?limit=${limit}`);

// Safety & evaluation center
export const getSafetyCenter = () =>
  get<import("../types/api").SafetyCenter>("/admin/safety-center");

export const getSafetyRedTeam = () =>
  get<import("../types/api").SafetyRedTeamArtifact>("/admin/safety-red-team");

export const runSafetyRedTeam = (liveAgent = false) =>
  post<{ message: string; result: import("../types/api").SafetyRedTeamArtifact }>(
    `/admin/safety-red-team?live_agent=${liveAgent ? "true" : "false"}`
  );

export const getRagEvalArtifact = () =>
  get<import("../types/api").RagEvalArtifact>("/admin/rag-eval");

export const runRagEvalArtifact = (liveAgent = false) =>
  post<{ message: string; result: import("../types/api").RagEvalArtifact }>(
    `/admin/rag-eval?live_agent=${liveAgent ? "true" : "false"}`
  );

export const getDriftReport = () =>
  get<import("../types/api").DriftReport>("/admin/drift-report");

export const runDriftReport = () =>
  post<{ message: string; result: import("../types/api").DriftReport }>(
    "/admin/drift-report"
  );
