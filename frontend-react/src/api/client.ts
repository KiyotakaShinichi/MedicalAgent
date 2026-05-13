import type {
  LoginResponse,
  DemoPatient,
  PatientReport,
  ChatResponse,
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
} from "../types/api";

const BASE = "http://127.0.0.1:8017";

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
  const res = await fetch(`${BASE}${path}`, {
    method,
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    ...(body !== undefined ? { body: JSON.stringify(body) } : {}),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`${res.status}: ${text}`);
  }
  return res.json() as Promise<T>;
}

const get = <T>(path: string) => request<T>("GET", path);
const post = <T>(path: string, body?: unknown) => request<T>("POST", path, body);

// Auth
export const login = (username: string, password: string) =>
  post<LoginResponse>("/auth/demo-credential-login", { username, password });

export const getDemoPatients = () =>
  get<{ patients: DemoPatient[] }>("/auth/demo-patients");

export const whoami = () =>
  get<{ role: string; patient_id: string | null }>("/auth/whoami");

// Patient
export const getMyReport = () => get<PatientReport>("/me/patient-report");

export const getMyChatHistory = () =>
  get<{ patient_id: string; messages: import("../types/api").ChatMessage[] }>("/me/chat");

export const sendMyChat = (message: string) =>
  post<ChatResponse>("/me/chat", { message });

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

// Clinician
export const getPatients = () => get<PatientSummary[]>("/patients");

export const getPatientReport = (patientId: string) =>
  get<PatientReport>(`/patient-report/${patientId}`);

export const getReviewQueue = () =>
  get<{ queue: ReviewQueueItem[] }>("/clinician/review-queue?limit=25");

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
  }
) => post<{ message: string; review: SummaryReview }>(`/patients/${patientId}/summary-review`, payload);

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
