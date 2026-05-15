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
  PublicImagingManifest,
  UltrasoundBaselineResult,
  CtLesionWorkflowReport,
  SimToPublicImagingReport,
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
    review_target?: string;
    reason_category?: string;
    model_version?: string;
    rag_version?: string;
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

export const getPublicDataManifest = () =>
  get<PublicDataManifest>("/admin/public-data-manifest");

export const runPublicDataManifest = () =>
  post<{ message: string; result: PublicDataManifest }>("/admin/public-data-manifest");

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
