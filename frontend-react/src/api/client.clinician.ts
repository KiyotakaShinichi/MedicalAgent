import type * as Api from "../types/api";
import { get, post } from "./client.transport";

export const getPatients = () => get<Api.PatientSummary[]>("/patients");

export const getPatientReport = (patientId: string) =>
  get<Api.PatientReport>(`/patient-report/${patientId}`);

export const getPatientGeneticCounselingReadiness = (patientId: string) =>
  get<Api.GeneticCounselingReadiness>(`/patients/${patientId}/genetic-counseling-readiness`);

export const getReviewQueue = () =>
  get<{ queue: Api.ReviewQueueItem[] }>("/clinician/review-queue?limit=25");

export const getHighRiskConversationAlerts = () =>
  get<Api.HighRiskConversationAlertsResponse>("/clinician/high-risk-conversation-alerts?limit=25");

export const acknowledgeHighRiskConversationAlert = (alertId: number, note?: string) =>
  post<{ message: string; alert: Api.HighRiskConversationAlert; safety_note: string }>(
    `/clinician/high-risk-conversation-alerts/${alertId}/acknowledge`,
    { note: note ?? "Acknowledged from the clinician review dashboard." },
  );

export const getSummaryReviews = (patientId: string) =>
  get<{ summary_reviews: Api.SummaryReview[] }>(
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
) => post<{ message: string; review: Api.SummaryReview }>(`/patients/${patientId}/summary-review`, payload);

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
  post<Api.ChatResponse>(`/patients/${patientId}/chat`, { message });

// Admin
