// ─── Auth ─────────────────────────────────────────────────────────────────────
export type Role = "patient" | "clinician" | "admin";

export interface LoginResponse {
  role: Role;
  access_token: string;
  patient_id: string | null;
}

export interface DemoPatient {
  id: string;
  label: string;
  hint: string;
}

// ─── Patient report ───────────────────────────────────────────────────────────
export interface LabValues {
  wbc: number | null;
  hemoglobin: number | null;
  platelets: number | null;
}

export interface LabHistoryPoint {
  date: string;
  wbc: number | null;
  hemoglobin: number | null;
  platelets: number | null;
}

export interface Signal {
  status: string;
  message: string;
  response_probability?: number;
  pcr_probability?: number;
  response_signal_score?: number | null;
  urgent_count?: number | null;
  watch_count?: number | null;
  has_synthetic_labs?: boolean | null;
  max_severity?: number | null;
  symptom_count?: number | null;
}

export interface MonitoringScoreBreakdown {
  base_signal: number;
  urgent_review_flags: number;
  urgent_flag_deduction: number;
  watch_flags: number;
  watch_flag_deduction: number;
  peak_recorded_symptom_severity: number | null;
  symptom_deduction: number;
  synthetic_lab_provenance_deduction: number;
  total_deduction: number;
  final_score: number;
  formula: string;
  claim_boundary: string;
}

export interface MultimodalAssessment {
  treatment_monitoring_score: number | null;
  overall_status: string;
  overall_message: string;
  signals: {
    mri_response?: Signal;
    clinical_monitoring?: Signal;
    symptoms?: Signal;
  };
  score_breakdown?: MonitoringScoreBreakdown | null;
  patient_next_steps?: string[];
}

export interface DataAvailabilityItem {
  name: string;
  status: "available" | "insufficient_data" | "missing" | "model_unavailable" | string;
  detail: string;
  next_step: string;
}

export interface DataAvailability {
  status: string;
  items: DataAvailabilityItem[];
  clinician_style_summary: string;
  patient_friendly_summary: string;
  fallback_policy: string;
}

export interface HybridMleSignal {
  hybrid_score: number | null;
  classification_probability: number | null;
  response_score_percent: number | null;
  agreement: string | null;
}

export interface SyntheticModelPrediction {
  hybrid_mle_signal: HybridMleSignal;
  actual_label: string | null;
}

export interface FeatureContribution {
  feature: string;
  contribution: number;
  shap_value: number;
}

export interface SyntheticModelExplanation {
  positive_contributions: FeatureContribution[];
  negative_contributions: FeatureContribution[];
}

export interface PatientXaiEnvelope {
  schema_version: string;
  status: "available_synthetic_signal" | "abstained" | "unavailable" | string;
  output: {
    label: string;
    hybrid_score: number | null;
    classification_probability: number | null;
    decision: string | null;
    meaning: string;
    calculation: string;
  };
  evidence: {
    inputs_used: string[];
    inputs_missing: string[];
    sufficiency: string | null;
    abstained: boolean;
    abstain_reason: string | null;
  };
  uncertainty: {
    confidence: string | null;
    confidence_modifier: number | null;
    uncertainty_is_clinical_probability: false;
    explanation: string;
  };
  top_model_factors: Array<{
    feature: string | null;
    relative_contribution: number | null;
    direction: string;
    meaning: string | null;
    clinical_causality: false;
    rank_interpretation_allowed?: boolean;
  }>;
  explanation_reliability?: {
    display_mode?: string | null;
    ranked_feature_order_allowed?: boolean;
    numeric_shap_values_visible?: boolean;
    warning?: string | null;
  };
  provenance: {
    synthetic_only: true;
    model_version: string | null;
    explanation_method: string | null;
    causal_interpretation_allowed: false;
  };
  safe_next_steps: string[];
  clinical_validation: false;
  claim_boundary: string;
}

export interface AiSummary {
  patient_explanation: string | string[];
  clinical_summary: string | string[];
  review_reasons: string[];
}

export interface UncertaintyBlock {
  confidence_level?: "low" | "moderate" | "high" | string | null;
  uncertainty_reason?: string | null;
  missing_data_indicators?: string[] | null;
  clinician_review_required?: boolean | null;
}

export interface TimelineMedia {
  label?: string | null;
  modality?: string | null;
  upload_id?: number | null;
  artifact_url?: string | null;
  content_type?: string | null;
  previewable?: boolean | null;
  notes?: string | null;
}

export interface TimelineEventDetail {
  kind?: string | null;
  title?: string | null;
  fields?: Record<string, string | number | boolean | null | undefined> | null;
  findings?: string | null;
  impression?: string | null;
  message?: string | null;
  evidence?: Record<string, unknown> | null;
  media?: TimelineMedia[] | null;
  notes?: string | null;
}

export interface TimelineEvent {
  date: string;
  type: string;
  severity: string;
  title: string;
  summary: string;
  detail?: TimelineEventDetail | null;
  /** Optional uncertainty block produced by the risk_engine / agent layer. */
  uncertainty?: UncertaintyBlock | null;
  /** True when this event was produced by an AI/model layer (vs. raw record). */
  ai_generated?: boolean | null;
  /** Free-text source/evidence reference, e.g. "risk_engine", "rag_agent". */
  evidence_source?: string | null;
  /** Model identifier for the AI output, when available. */
  model_version?: string | null;
}

export interface TreatmentEffect {
  cycle: number;
  drug: string;
  min_wbc_post_cycle: number | null;
  min_hemoglobin_post_cycle: number | null;
  min_platelets_post_cycle: number | null;
}

export interface Symptom {
  date: string;
  symptom: string;
  severity: number;
  notes: string;
}

export interface MedicationLog {
  date: string;
  medication: string;
  dose: string;
  frequency: string;
}

export interface TreatmentOutcome {
  response_category: string | null;
  cancer_status: string | null;
  maintenance_plan: string | null;
}

export interface ClinicalIntervention {
  date: string;
  intervention_type: string;
  reason: string;
  medication_or_product: string;
}

export interface PatientUpload {
  upload_type: string;
  original_filename: string;
  notes: string;
}

export interface FamilyCancerHistoryRecord {
  id: number;
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
  review_status?: string | null;
  source?: string | null;
  created_at?: string | null;
}

export interface GeneticTestRecord {
  id: number;
  test_type: string;
  sample_type: string;
  gene?: string | null;
  variant_text?: string | null;
  classification?: string | null;
  report_date?: string | null;
  lab_provider?: string | null;
  upload_reference?: string | null;
  reviewed_by_genetic_counselor?: string | null;
  clinician_review_status?: string | null;
  notes?: string | null;
  created_at?: string | null;
}

export interface BiomarkerRecord {
  id: number;
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
  clinician_review_needed?: boolean | null;
  review_status?: string | null;
  created_at?: string | null;
}

export interface TumorMarkerRecord {
  id: number;
  marker: string;
  value: number;
  unit?: string | null;
  reference_range?: string | null;
  date_collected?: string | null;
  trend_direction?: string | null;
  notes?: string | null;
  review_status?: string | null;
  created_at?: string | null;
}

export interface GeneticCounselingReviewNote {
  id: number;
  reviewer_role: string;
  decision: string;
  notes?: string | null;
  created_at?: string | null;
}

export interface GeneticCounselingReadiness {
  schema_version: string;
  patient_id: string;
  boundary_note: string;
  family_history: FamilyCancerHistoryRecord[];
  genetic_test_records: GeneticTestRecord[];
  biomarker_records: BiomarkerRecord[];
  tumor_marker_records: TumorMarkerRecord[];
  review_notes: GeneticCounselingReviewNote[];
  flags: string[];
  missing_data: string[];
  readiness_status: string;
  questions_to_ask: string[];
}

export interface EvidenceAwarePrediction {
  decision: "favorable_pattern" | "concerning_pattern" | "uncertain" | "insufficient_evidence" | string;
  probability: number | null;
  raw_probability: number | null;
  calibrated: boolean;
  confidence: "low" | "moderate" | "high" | string;
  evidence: {
    modalities_present: string[];
    modalities_missing: string[];
    sufficiency: "sufficient" | "partial" | "insufficient" | string;
    abstain: boolean;
    reason: string | null;
    confidence_modifier: number;
  };
  model_version: string;
  question: string;
  claim_boundary: string;
}

export interface EvidenceAwareRegression {
  decision: "strong_response_signal" | "moderate_response_signal" | "weak_response_signal" | "insufficient_evidence" | string;
  response_score: number | null;
  raw_response_score: number | null;
  uncertainty_band: [number, number] | null;
  confidence: "low" | "moderate" | "high" | string;
  evidence: EvidenceAwarePrediction["evidence"];
  model_version: string;
  question: string;
  claim_boundary: string;
}

export interface HybridPrediction {
  classification: EvidenceAwarePrediction;
  response_score: EvidenceAwareRegression;
  toxicity: EvidenceAwarePrediction;
  claim_boundary: string;
}

export interface PatientReport {
  patient_id: string;
  patient_name: string;
  diagnosis: string;
  latest_labs: LabValues;
  lab_history: LabHistoryPoint[];
  monitoring_score: number | null;
  overall_status: string;
  multimodal_assessment: MultimodalAssessment | null;
  synthetic_model_prediction: SyntheticModelPrediction | null;
  synthetic_model_explanation: SyntheticModelExplanation | null;
  xai_explanation_envelope?: PatientXaiEnvelope | null;
  ai_summary: AiSummary | null;
  timeline: TimelineEvent[];
  treatment_effects: TreatmentEffect[];
  symptoms: Symptom[];
  medication_logs: MedicationLog[];
  chat_history: ChatMessage[];
  uploads: PatientUpload[];
  treatment_outcome: TreatmentOutcome | null;
  clinical_interventions: ClinicalIntervention[];
  breast_cancer_profile?: BreastCancerProfile;
  genetic_counseling_readiness?: GeneticCounselingReadiness | null;
  evidence_aware_prediction?: EvidenceAwarePrediction | null;
  hybrid_prediction?: HybridPrediction | null;
  data_availability?: DataAvailability | null;
  report_enrichment?: {
    status: "deferred" | "complete" | string;
    profile: string;
    generated_ms: number;
    message: string;
    requested_at?: string | null;
    started_at?: string | null;
    completed_at?: string | null;
    retry_after_ms?: number | null;
    error_code?: string | null;
    clinical_validation: false;
    healthcare_production_ready?: false;
    claim_boundary?: string;
  };
}

// ─── Chat ─────────────────────────────────────────────────────────────────────
export interface SavedAction {
  type: string;
  data?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface ChatMessage {
  role: "user" | "assistant";
  message: string;
  saved_actions_json?: string;
  citations?: string[];
}

export interface ChatResponse {
  reply: string;
  saved_actions: SavedAction[];
  citations?: string[];
  assistant_message_id?: string | number;
}

export interface ChatStreamHandlers {
  onStage?: (label: string) => void;
  onDelta?: (text: string) => void;
}

// ─── Clinician ────────────────────────────────────────────────────────────────
export interface BreastCancerProfile {
  er_status: string | null;
  pr_status: string | null;
  her2_status: string | null;
  molecular_subtype: string | null;
  cancer_stage: string | null;
  treatment_intent: string | null;
}

export interface PatientSummary {
  id: string;
  name: string;
  diagnosis: string;
  breast_cancer_profile: BreastCancerProfile | null;
}

export interface ReviewQueueItem {
  patient_id: string;
  patient_name: string;
  overall_status: string;
  priority_score: number;
  urgent_flags: string[];
  latest_decision: string | null;
}

export interface HighRiskConversationAlert {
  id: number;
  patient_id: string;
  source_chat_message_id: number;
  assistant_chat_message_id: number | null;
  category: string;
  severity: "urgent_review" | "critical_review" | string;
  trigger_summary: string;
  status: "queued" | "notified" | "acknowledged" | string;
  notification_channel: string;
  notification_status: "disabled" | "accepted_by_workflow" | "accepted_by_channel" | "delivered_to_channel" | "retry_scheduled" | "dead_lettered" | string;
  notification_event_id: string | null;
  notification_attempt_count: number;
  notification_max_attempts: number;
  last_notification_attempt_at: string | null;
  next_notification_retry_at: string | null;
  delivery_receipt_status: string;
  delivery_receipt_id: string | null;
  delivery_receipt_at: string | null;
  dead_lettered_at: string | null;
  dead_letter_reason: string | null;
  acknowledged_by_role: string | null;
  acknowledgement_note: string | null;
  created_at: string | null;
  notified_at: string | null;
  acknowledged_at: string | null;
  delivery_claim_boundary: string;
}

export interface HighRiskConversationAlertsResponse {
  alerts: HighRiskConversationAlert[];
  count: number;
  clinical_validation: false;
  safety_note: string;
}

export interface SummaryReview {
  id: number;
  patient_id: string;
  decision: string;
  clinician_notes: string;
  edited_patient_summary: string | null;
  explanation_quality_score: number | null;
  model_usefulness_score: number | null;
  created_at: string;
}

// ─── Admin / MLE ──────────────────────────────────────────────────────────────
