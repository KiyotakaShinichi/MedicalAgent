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

export interface AiSummary {
  patient_explanation: string | string[];
  clinical_summary: string | string[];
  review_reasons: string[];
}

export interface TimelineEvent {
  date: string;
  type: string;
  severity: string;
  title: string;
  summary: string;
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
}

// ─── Chat ─────────────────────────────────────────────────────────────────────
export interface SavedAction {
  type: string;
  data: Record<string, unknown>;
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
export interface RagEvaluationSummary {
  evaluations: number;
  grounding_score: number | null;
  hallucination_score: number | null;
  cache_hit_rate: number | null;
  precision_at_3: number | null;
  estimated_cost_usd: number | null;
  input_tokens: number | null;
  output_tokens: number | null;
  p95_latency_ms: number | null;
}

export interface GuardrailSummary {
  input_blocks: number;
  output_blocks: number;
  attack_block_rate: number | null;
  pass_rate: number | null;
}

export interface MleReadinessSummary {
  status: string;
  release_recommendation: string;
  hard_gate_status: string;
  hard_gate_failures: number;
  poc_demo_readiness: string;
  category_statuses: Record<string, string>;
}

export interface AgentFeedbackSummary {
  count: number;
  average_rating: number | null;
  thumbs_up_rate: number | null;
}

export interface AgentFeedbackItem {
  patient_id: string;
  rating: number | null;
  thumbs_up: boolean | null;
  feedback_text: string | null;
  created_at: string;
}

export interface AdminAnalytics {
  rag_evaluation: RagEvaluationSummary;
  guardrails: GuardrailSummary;
  mle_readiness: MleReadinessSummary;
  agent_feedback: AgentFeedbackSummary;
  api_cost?: { estimated_cost_usd: number | null };
}

export interface RagSourceEntry {
  id: string;
  source_name: string;
  trust_level: string;
  chunk_count: number;
  topics: string[];
}

export interface AgentRegressionResult {
  case_count: number;
  status: string;
  pass_rate: number;
  attack_block_rate: number;
  expected_source_hit_rate: number;
  cases?: AgentRegressionCase[];
}

export interface AgentRegressionCase {
  id: string;
  category: string;
  query: string;
  status: "passed" | "failed";
  checks: { name: string; passed: boolean; expected: unknown; observed: unknown }[];
}

export interface FeatureDriftEntry {
  feature: string;
  std_mean_shift: number;
  status: string;
}

export interface CalibrationBin {
  range: string;
  count: number;
  mean_predicted: number;
  observed_rate: number;
  gap: number;
}

export interface CalibrationReport {
  ece: number | null;
  brier_score: number | null;
  bins: CalibrationBin[];
}

export interface ConfusionMatrix {
  tp: number;
  fp: number;
  fn: number;
  tn: number;
  sensitivity: number | null;
  specificity: number | null;
  precision: number | null;
  fnr: number | null;
}

// ─── Agent Trace Observatory ──────────────────────────────────────────────────
export interface AgentTraceLog {
  id: number;
  patient_id: string | null;
  query_preview: string;
  intent: string | null;
  safety_level: string | null;
  cache_status: string | null;
  terminal_step: string | null;
  input_guardrail: string | null;
  output_guardrail: string | null;
  grounding_score: number | null;
  hallucination_score: number | null;
  hallucination_risk: string | null;
  latency_ms: number | null;
  estimated_total_tokens: number | null;
  retrieved_source_ids: string[];
  cited_source_ids: string[];
  created_at: string;
}

export interface AgentTraceLogsResponse {
  count: number;
  traces: AgentTraceLog[];
  note: string;
}

// ─── Noise / Temporal Robustness ─────────────────────────────────────────────
export interface NoiseResult {
  mode: string;
  auroc: number | null;
  brier_score: number | null;
  sensitivity: number | null;
  auroc_delta: number | null;
  sensitivity_delta: number | null;
  brier_delta: number | null;
  status: string;
}

export interface NoiseEvalResult {
  status: string;
  clean_baseline: {
    auroc: number | null;
    brier_score: number | null;
    sensitivity: number | null;
    pr_auc: number | null;
  };
  noise_results: NoiseResult[];
  summary: {
    worst_mode: string | null;
    max_auroc_drop: number | null;
    max_sensitivity_drop: number | null;
  };
  claim_boundary: string;
}

export interface SplitMetrics {
  auroc: number | null;
  brier_score: number | null;
  sensitivity: number | null;
  n_train: number;
  n_eval: number;
}

export interface TemporalEvalResult {
  status: string;
  temporal_split: SplitMetrics;
  cycle_split: SplitMetrics;
  random_split_baseline: SplitMetrics;
  generalization_gap: {
    temporal_auroc_gap: number | null;
    cycle_auroc_gap: number | null;
  };
  interpretation: string;
  claim_boundary: string;
}

// ─── Per-Prediction ML Error Table ───────────────────────────────────────────
export interface PredictionErrorRow {
  patient_id: string;
  actual_label: number;
  predicted_probability: number;
  predicted_class: number;
  threshold_used: number;
  absolute_error: number;
  confusion_type: "TP" | "FP" | "TN" | "FN";
  top_features: { feature: string; value: number }[];
  note: string;
}

export interface PredictionErrorTable {
  schema_version: string;
  model: string;
  threshold: number;
  total_predictions: number;
  confusion_summary: { TP: number; FP: number; TN: number; FN: number };
  mae: number | null;
  sensitivity: number | null;
  specificity: number | null;
  rows: PredictionErrorRow[];
  claim_boundary: string;
  shap_available: boolean;
}

// ─── RAG Ablation ─────────────────────────────────────────────────────────────
export interface AblationStrategyMetrics {
  case_count: number;
  pass_rate: number | null;
  expected_source_hit_rate: number | string | null;
  average_grounding_score: number | null;
  average_latency_ms: number | null;
  backend?: string | null;
}

export interface RagAblationResult {
  schema_version: string;
  generated_at: string;
  purpose: string;
  active_index?: {
    retrieval_backend?: string | null;
    dense_component?: string | null;
    sparse_component?: string | null;
    fusion?: string | null;
    dense_available?: boolean;
    bm25_available?: boolean;
    status?: string;
  };
  strategies: Record<string, AblationStrategyMetrics> & {
    bm25_only: AblationStrategyMetrics;
    hybrid: AblationStrategyMetrics;
    hybrid_reranked: AblationStrategyMetrics;
  };
  comparison: {
    notes: string[];
    winner: string;
    caveat: string;
  };
  limitations: string[];
  claim_boundary: string;
}
