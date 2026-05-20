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

export interface UncertaintyBlock {
  confidence_level?: "low" | "moderate" | "high" | string | null;
  uncertainty_reason?: string | null;
  missing_data_indicators?: string[] | null;
  clinician_review_required?: boolean | null;
}

export interface TimelineMedia {
  label?: string | null;
  modality?: string | null;
  local_path?: string | null;
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

export interface AdminBenchmarkResponse {
  schema_version?: string;
  generated_at?: string;
  status: string;
  headline_metric: string | null;
  metrics: Record<string, unknown>;
  rows: unknown[];
  artifact_path: string | null;
  last_run_at: string | null;
  claim_boundary: string;
  can_rerun: boolean;
  errors: string[];
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
export interface CompoundIntentSegment {
  intent: string;
  kind: string;
  span: string;
  tool_targets: string[];
}

export interface CompoundIntentTrace {
  segments: CompoundIntentSegment[];
  primary_intent: string;
  is_compound: boolean;
  has_casual_opener: boolean;
  has_tool_request: boolean;
  has_education_request: boolean;
  has_capability_request: boolean;
  tool_request_targets: string[];
  suggested_acknowledgment: string | null;
  llm?: {
    available: boolean;
    language?: string;
    llm_confidence?: number;
    provider?: string;
    model?: string;
    reason?: string;
  };
}

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
  compound_intent: CompoundIntentTrace | null;
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

export interface PublicDataSource {
  id: string;
  name: string;
  provider: string;
  url: string;
  access: string;
  modalities: string[];
  use_in_project: string[];
  covers: Record<string, boolean>;
  limitations: string[];
}

export interface PublicDataNeed {
  need: string;
  status: string;
  sources: string[];
  project_action: string;
}

export interface PublicDataManifest {
  schema_version: string;
  generated_at: string;
  status: string;
  central_data_reality: string;
  recommended_strategy: string;
  sources: PublicDataSource[];
  feature_feasibility: PublicDataNeed[];
  claim_boundary: string;
  manifest_hash: string;
}

export interface PublicBiomarkerDatasetEntry {
  id: string;
  name: string;
  provider: string;
  url: string;
  access: string;
  predictor_fields: string[];
  target_fields: string[];
  fit_for_oncotrack: string;
  use_in_project: string[];
  limitations: string[];
  license_or_terms: string;
  source_evidence: string;
}

export interface PublicBiomarkerDatasetManifest {
  schema_version: string;
  generated_at: string;
  status: string;
  dataset_count: number;
  datasets: PublicBiomarkerDatasetEntry[];
  recommended_order: string[];
  next_step: string;
  claim_boundary: string;
  manifest_hash: string;
}

export interface PublicBiomarkerMappingReadiness {
  schema_version: string;
  generated_at: string;
  status: string;
  source_manifest_hash?: string | null;
  datasets: Record<string, {
    status: string;
    mapped_now?: boolean;
    rows?: number;
    patients?: number;
    target?: string | null;
    direct_predictors?: string[];
    derived_predictors?: string[];
    imaging_predictors?: string[];
    predictors_to_map?: string[];
    target_to_map?: string;
    role?: string;
    next_action?: string;
  }>;
  three_stage_ablation_plan: Record<string, string>;
  tumor_marker_boundary: string;
  recommended_next_order: string[];
  claim_boundary: string;
  mapping_hash: string;
}

export interface CbioportalBiomarkerSchemaMapping {
  schema_version: string;
  generated_at: string;
  status: string;
  mapped_dataset_count: number;
  datasets: Record<string, {
    status: string;
    study_id: string;
    label: string;
    role: string;
    reason?: string;
    clinical_attribute_count?: number;
    mapped_groups?: Record<string, Array<{
      id: string;
      display_name: string;
      description: string;
      datatype: string;
    }>>;
    core_biomarker_group_hits?: number;
    next_action?: string;
  }>;
  recommended_use: string[];
  claim_boundary: string;
  mapping_hash: string;
}

export interface FullFeatureGroupAblationReport {
  schema_version?: string;
  status: "strong" | "acceptable" | "needs_attention" | "missing" | string;
  generated_at?: string;
  feature_groups?: Record<string, {
    purpose?: string;
    modalities?: string[];
    classification?: {
      patient_level_auroc?: number | null;
      auroc?: number | null;
      auprc?: number | null;
      brier?: number | null;
      ece?: number | null;
      sensitivity?: number | null;
      specificity?: number | null;
      false_negative_count?: number | null;
      subgroup_ece_max?: number | null;
    };
    regression?: {
      mae?: number | null;
      rmse?: number | null;
      r2?: number | null;
    };
  }>;
  deltas?: {
    full_vs_clinical_auroc_delta?: number | null;
    full_vs_clinical_brier_delta?: number | null;
    full_vs_clinical_ece_delta?: number | null;
    full_vs_clinical_regression_mae_delta?: number | null;
  };
  leakage_audit?: { status?: string; violations?: unknown[] };
  recommendation?: {
    status?: string;
    promote_feature_set?: boolean;
    recommended_use?: string;
    reason?: string;
  };
  claim_boundary?: string;
  message?: string;
}

export interface ClinicalSafetyReviewChecklist {
  schema_version: string;
  generated_at: string;
  status: string;
  purpose: string;
  review_frequency: string;
  sections: Array<{
    id: string;
    title: string;
    items: string[];
  }>;
  sign_off_fields: Record<string, string>;
  known_limitations: string[];
  claim_boundary: string;
}

export interface LeakageAuditFinding {
  name: string;
  status: string;
  evidence: Record<string, unknown>;
  meaning: string;
}

export interface LeakageAuditReport {
  schema_version?: string;
  status: "passed" | "failed" | "missing" | string;
  generated_at?: string;
  training_rows_path?: string;
  feature_columns?: string[];
  row_level_targets?: string[];
  known_label_proxies?: string[];
  findings: LeakageAuditFinding[];
  temporal_sub_audit?: { status: string; findings?: LeakageAuditFinding[] };
  summary: { checks_total: number; checks_passed: number; checks_failed: number };
  interpretation?: string;
  claim_boundary?: string;
  message?: string;
}

export interface EvidenceAbstentionScenario {
  scenario: string;
  stripped_modalities: string[];
  rows_evaluated: number;
  coverage_rate: number | null;
  abstention_rate: number | null;
  false_abstention_rate: number | null;
  covered_accuracy: number | null;
  decision_counts: Record<string, number>;
  covered_mean_probability: number | null;
  calibration_bins?: Array<{ range: string; count: number; mean_predicted: number | null; observed_rate: number | null }>;
}

export interface KbSourceGovernanceSource {
  source_id: string;
  title: string;
  source_url?: string | null;
  trust_level: string;
  tier: string;
  tier_rank: number;
  tier_description: string;
  allowed_use: string[];
  ingested_at?: string | null;
  staleness_status: "current" | "aging" | "needs_review" | "unknown" | string;
  staleness_days?: number | null;
  chunk_count: number;
  topics: string[];
  modalities: string[];
}

export interface KbSourceGovernanceReport {
  schema_version?: string;
  status: "strong" | "acceptable" | "needs_attention" | "missing" | "error" | string;
  generated_at?: string;
  source_count?: number;
  chunk_count?: number;
  tier_distribution?: Record<string, number>;
  allowed_use_distribution?: Record<string, number>;
  staleness_distribution?: Record<string, number>;
  tier_order?: string[];
  staleness_ttl_days?: number;
  sources?: KbSourceGovernanceSource[];
  governance_issues?: Array<{ severity: string; code: string; message: string; examples?: string[] }>;
  interpretation?: string;
  claim_boundary?: string;
  message?: string;
}

export interface SyntheticGeneratorCard {
  schema_version?: string;
  status: "passed" | "needs_attention" | "missing" | string;
  generator_card_version?: string;
  generated_at?: string;
  dataset_dir?: string;
  dataset_schema_version?: string;
  card_version_matches_dataset?: boolean;
  generation_options?: Record<string, unknown>;
  cohort?: {
    patients_created?: number;
    cycles_per_patient?: number;
    table_counts?: Record<string, number>;
    rows_fingerprint?: string | null;
  };
  supported_labels?: string[];
  feature_distribution_summary?: {
    row_count?: number;
    numeric?: Record<string, { count: number; mean: number | null; std: number | null; min: number | null; max: number | null }>;
    categorical?: Record<string, Record<string, number>>;
    positive_label_rate?: number | null;
  };
  causal_assumptions?: string[];
  known_shortcuts?: string[];
  unsupported_claims?: string[];
  realism_checks_referenced?: string[];
  claim_boundary?: string;
  message?: string;
}

export interface FailureModeEntry {
  name: string;
  category: string;
  example: string;
  risk: string;
  detection: string;
  mitigation: string;
  benchmark_coverage: string[];
  remaining_gap: string | null;
  severity: "high" | "medium" | "low" | string;
}

export interface FailureModeRegistry {
  schema_version?: string;
  status: "strong" | "acceptable" | "needs_attention" | "missing" | string;
  generated_at?: string;
  entry_count?: number;
  summary?: {
    by_severity?: Record<string, number>;
    by_category?: Record<string, number>;
    entries_with_unresolved_gap?: number;
  };
  entries: FailureModeEntry[];
  sources?: Record<string, string>;
  interpretation?: string;
  claim_boundary?: string;
  message?: string;
}

export interface ModalityRobustnessScenario {
  scenario: string;
  stripped_modalities: string[];
  rows_evaluated: number;
  force_score: {
    champion: { accuracy: number | null; brier: number | null; mean_probability: number | null };
    robust:   { accuracy: number | null; brier: number | null; mean_probability: number | null };
  };
  with_abstention: {
    champion: { coverage_rate: number | null; abstention_rate: number | null; covered_accuracy: number | null; covered_mean_probability: number | null };
    robust:   { coverage_rate: number | null; abstention_rate: number | null; covered_accuracy: number | null; covered_mean_probability: number | null };
  };
  deltas: {
    force_score_accuracy_robust_minus_champion: number | null;
    force_score_brier_robust_minus_champion: number | null;
    with_abstention_accuracy_robust_minus_champion: number | null;
  };
}

export interface ModalityRobustnessComparisonReport {
  schema_version?: string;
  status: "robust" | "acceptable" | "needs_attention" | "missing" | string;
  generated_at?: string;
  training_rows_path?: string;
  models_under_test?: Record<string, string>;
  scenarios: ModalityRobustnessScenario[];
  summary: {
    status?: string;
    scenario_count?: number;
    force_score_accuracy_wins_for_robust?: number;
    force_score_accuracy_ties?: number;
    force_score_accuracy_losses_for_robust?: number;
    full_data_accuracy_delta?: number | null;
    full_data_brier_delta?: number | null;
  };
  interpretation?: string;
  claim_boundary?: string;
  message?: string;
}

export interface ResponseConformalCalibrationReport {
  schema_version?: string;
  status: "strong" | "acceptable" | "needs_attention" | "missing" | string;
  generated_at?: string;
  nominal_coverage?: number | null;
  raw_coverage?: number | null;
  adjusted_coverage?: number | null;
  qhat_percent?: number | null;
  qhat_normalized?: number | null;
  calibration_rows?: number | null;
  model_paths?: Record<string, string | null>;
  interpretation?: string;
  claim_boundary?: string;
  message?: string;
}

export interface RobustnessStressCase {
  case?: string;
  case_id?: string;
  category: string;
  description?: string;
  expected?: string;
  expected_behavior?: string;
  passed: boolean;
  abstained_any_head?: boolean;
  decision?: string | null;
  evidence_sufficiency?: string | null;
  abstained?: boolean;
  clinician_review?: boolean;
  clinician_review_routed?: boolean;
  uncertainty_increased?: boolean;
  uncertainty_increased_or_equal?: boolean;
  safety_correct?: boolean;
  medical_claim_boundary_decision?: string | null;
  notes?: string[];
}

export interface RobustnessStressReport {
  schema_version?: string;
  status: "strong" | "acceptable" | "needs_attention" | "missing" | string;
  generated_at?: string;
  summary?: {
    case_count?: number;
    passed?: number;
    pass_rate?: number | null;
    abstention_or_review_rate?: number | null;
  };
  cases?: RobustnessStressCase[];
  interpretation?: string;
  claim_boundary?: string;
  message?: string;
}

export interface PredictionTraceRow {
  id: number;
  created_at: string | null;
  patient_id: string | null;
  request_id: string | null;
  actor_role: string | null;
  question: string;
  decision: string;
  probability: number | null;
  raw_probability: number | null;
  calibrated: boolean;
  confidence: string | null;
  evidence_sufficiency: string | null;
  abstained: boolean;
  abstain_reason: string | null;
  modalities_present: string[];
  modalities_missing: string[];
  confidence_modifier: number | null;
  model_version: string;
  feature_set_version: string | null;
  threshold_config: Record<string, number | null>;
  calibration_config: Record<string, unknown>;
  safety_triggers: string[];
  validator_decision: string | null;
  rag_source_ids: string[];
  timeline_snapshot_hash: string | null;
  notes: string | null;
}

export interface PredictionTraceResponse {
  traces: PredictionTraceRow[];
  summary: {
    total: number;
    abstention_rate: number | null;
    decision_counts: Record<string, number>;
    evidence_sufficiency_counts: Record<string, number>;
    model_versions: string[];
  };
}

export interface ClinicianPredictionTracesResponse {
  patient_id: string;
  traces: PredictionTraceRow[];
  patient_summary: {
    total: number;
    abstention_rate: number | null;
    decision_counts: Record<string, number>;
  };
  cohort_summary: PredictionTraceResponse["summary"];
  claim_boundary: string;
}

export interface EvidenceAbstentionEvalReport {
  schema_version?: string;
  status: "strong" | "acceptable" | "needs_attention" | "missing" | string;
  generated_at?: string;
  training_rows_path?: string;
  model_path?: string;
  calibrator_path?: string | null;
  label_column?: string;
  rows_evaluated?: number;
  scenarios: EvidenceAbstentionScenario[];
  summary: {
    overall_status?: string;
    full_data_coverage_rate?: number | null;
    full_data_covered_accuracy?: number | null;
    demographics_only_abstention_rate?: number | null;
    abstention_rates_by_scenario?: Record<string, number | null>;
    coverage_rates_by_scenario?: Record<string, number | null>;
    covered_accuracy_by_scenario?: Record<string, number | null>;
    scenario_count?: number;
  };
  interpretation?: string;
  claim_boundary?: string;
  message?: string;
}

export interface SystemHealth {
  schema_version: string;
  generated_at: string;
  status: string;
  backend: {
    python_version: string;
    database_url_kind: string;
    database: { status: string; message: string };
  };
  environment: {
    groq_configured: boolean;
    rag_judge_enabled: boolean;
    cors_origins_configured: boolean;
  };
  dependencies: Array<{ package: string; purpose: string; available: boolean }>;
  artifacts: Array<{
    name: string;
    path: string;
    exists: boolean;
    status: string;
    freshness: string;
    generated_at?: string | null;
    error?: string;
  }>;
  frontend: {
    react_app_present: boolean;
    production_build_present: boolean;
    dist_index?: string | null;
  };
  issues: Array<{ area: string; severity: string; message: string }>;
  next_actions: string[];
  claim_boundary: string;
}

// Public imaging / image-model readiness
export interface PublicImagingDataset {
  id: string;
  name: string;
  source_url: string;
  modality: string;
  task: string;
  available: boolean;
  local_path: string | null;
  file_count: number;
  image_count: number;
  mask_count: number;
  metadata_count: number;
  class_counts: Record<string, number>;
  readiness: string;
  claim_boundary: string;
}

export interface PublicImagingManifest {
  schema_version: string;
  generated_at: string;
  status: string;
  dataset_root: string;
  available_dataset_count: number;
  datasets: PublicImagingDataset[];
  recommended_next_task: string;
  claim_boundary: string;
  manifest_hash: string;
}

export interface UltrasoundBaselineResult {
  schema_version: string;
  generated_at: string;
  status: string;
  reason?: string;
  expected_layout?: string;
  dataset_root?: string;
  task?: string;
  model_family?: string;
  image_count?: number;
  train_count?: number;
  test_count?: number;
  label_counts?: Record<string, number>;
  models?: Record<string, {
    accuracy: number;
    balanced_accuracy: number;
    macro_f1: number;
    auroc?: number | null;
    auroc_ovr?: number | null;
    confusion_matrix: number[][];
    classes: string[];
  }>;
  best_model?: string;
  predictions_path?: string;
  claim_boundary: string;
}

export interface UltrasoundTransferBaselineResult {
  schema_version: string;
  generated_at: string;
  status: string;
  reason?: string;
  expected_layout?: string;
  dataset_root?: string;
  task?: string;
  model_family?: string;
  pretrained_requested?: boolean;
  device?: string;
  image_count?: number;
  train_count?: number;
  test_count?: number;
  epochs?: number;
  classes?: string[];
  balanced_accuracy?: number | null;
  macro_f1?: number | null;
  confusion_matrix?: number[][];
  final_train_loss?: number | null;
  claim_boundary: string;
}

export interface UltrasoundSegmentationBaselineResult {
  schema_version: string;
  generated_at: string;
  status: string;
  reason?: string;
  expected_layout?: string;
  dataset_root?: string;
  task?: string;
  model_family?: string;
  pair_count?: number;
  mean_dice?: number | null;
  median_dice?: number | null;
  mean_iou?: number | null;
  label_breakdown?: Record<string, { count: number; mean_dice: number; mean_iou: number }>;
  claim_boundary: string;
}

export interface CtLesionWorkflowReport {
  schema_version: string;
  generated_at: string;
  status: string;
  reason?: string;
  expected_layout?: string;
  dataset_root?: string;
  image_file_count?: number;
  metadata_file_count?: number;
  estimated_annotation_rows?: number | null;
  sample_image_files?: string[];
  sample_metadata_files?: string[];
  recommended_model_track?: string[];
  claim_boundary: string;
}

// ─── Safety & Evaluation Center ──────────────────────────────────────────────
export interface SafetyRedTeamCase {
  case_id: string;
  category: string;
  input_message: string;
  expected_behavior?: string | null;
  expected_route?: string | null;
  expected_refusal_type?: string | null;
  expected_safety_level?: string | null;
  expected_guardrail_status?: string | null;
  observed: {
    intent?: string | null;
    safety_level?: string | null;
    input_guardrail_status?: string | null;
    refusal_type?: string | null;
    reply_preview?: string | null;
  };
  checks: { name: string; passed: boolean; expected: unknown; observed: unknown }[];
  pass: boolean;
  reason: string | null;
  timestamp: string;
}

export interface ArtifactFreshness {
  generated_at?: string | null;
  ttl_seconds?: number | null;
  expires_at?: string | null;
  status?: "fresh" | "stale" | "unknown" | string | null;
}

export interface ReproducibilityManifest {
  git_commit?: string | null;
  git_dirty?: boolean | null;
  python_version?: string | null;
  platform?: string | null;
  dataset_fingerprints?: Record<string, string | null>;
  knowledge_base_fingerprint?: string | null;
  model_registry_fingerprint?: string | null;
  seed?: number | null;
  generated_at?: string | null;
}

export interface SafetyRedTeamArtifact {
  schema_version?: string;
  generated_at?: string;
  status?: string;
  message?: string;
  case_count?: number;
  reproducibility?: ReproducibilityManifest;
  artifact_freshness?: ArtifactFreshness;
  summary?: {
    status: string;
    pass_rate: number | null;
    total_cases: number;
    failed_cases: string[];
    category_counts: Record<string, number>;
    refusal_type_counts: Record<string, number>;
  };
  cases?: SafetyRedTeamCase[];
  limitations?: string[];
}

export interface RagEvalCase {
  case_id: string;
  input: string;
  intent?: string | null;
  expected_sources?: string[];
  requires_citations?: boolean;
  requires_refusal?: boolean;
  metrics: {
    citation_present?: boolean;
    expected_source_hit?: boolean;
    grounding_score?: number | null;
    hallucination_score?: number | null;
    retrieval_precision_at_3?: number | null;
    domain_relevance?: boolean;
    rewrite_subquery_count?: number;
    rewrite_term_hit?: boolean | null;
  };
  checks: { name: string; passed: boolean; expected: unknown; observed: unknown }[];
  pass: boolean;
  reply_preview?: string;
  timestamp: string;
}

export interface RagEvalArtifact {
  schema_version?: string;
  generated_at?: string;
  status?: string;
  message?: string;
  case_count?: number;
  reproducibility?: ReproducibilityManifest;
  artifact_freshness?: ArtifactFreshness;
  summary?: {
    status: string;
    pass_rate: number | null;
    citation_coverage_rate: number | null;
    expected_source_hit_rate: number | null;
    refusal_correct_rate: number | null;
    domain_relevance_rate: number | null;
    rewrite_term_hit_rate: number | null;
    average_grounding_score: number | null;
    average_hallucination_score: number | null;
    average_retrieval_precision_at_3: number | null;
  };
  cases?: RagEvalCase[];
  limitations?: string[];
}

export interface DriftFeatureRow {
  feature?: string;
  keyword?: string;
  baseline_mean?: number;
  current_mean?: number;
  baseline_rate?: number;
  current_rate?: number;
  standardized_shift?: number;
  shift?: number;
  status: string;
}

export interface DriftReport {
  schema_version?: string;
  generated_at?: string;
  status?: string;
  message?: string;
  reproducibility?: ReproducibilityManifest;
  artifact_freshness?: ArtifactFreshness;
  data_source?: string;
  missing_cbc_rate?: number | null;
  data_completeness_score?: number | null;
  lab_distribution_shift?: { label: string; status: string; feature_count: number; features: DriftFeatureRow[] };
  symptom_frequency_shift?: { label: string; status: string; feature_count: number; features: DriftFeatureRow[] };
  imaging_keyword_shift?: { status: string; keywords: DriftFeatureRow[] };
  model_confidence_drift?: {
    status: string;
    probability_column?: string;
    baseline_mean?: number;
    current_mean?: number;
    standardized_shift?: number;
    message?: string;
  };
  calibration_drift?: {
    status: string;
    probability_column?: string;
    baseline_ece?: number | null;
    current_ece?: number | null;
    delta_ece?: number | null;
    message?: string;
  };
  subgroup_performance_drift?: {
    status: string;
    groups: {
      group: string;
      value: string;
      baseline_positive_rate: number;
      current_positive_rate: number;
      shift: number;
      status: string;
    }[];
  };
  limitations?: string[];
}

export interface SafetyCenterCategorySummary {
  status: string;
  pass_rate: number | null;
  case_count: number;
  categories: string[];
}

export interface BenchmarkLadderSummary {
  schema_version?: string;
  generated_at?: string;
  status?: string;
  message?: string;
  artifact_freshness?: {
    status?: string;
    generated_at?: string;
    ttl_seconds?: number;
  };
  benchmarks?: {
    safety?: {
      status?: string;
      unsafe_pass_rate?: number | null;
      urgent_escalation_recall?: number | null;
      privacy_leak_rate?: number | null;
      prompt_injection_resistance?: number | null;
    };
    adversarial?: {
      status?: string;
      attack_block_rate?: number | null;
    };
    rag?: {
      status?: string;
      pass_rate?: number | null;
      citation_coverage?: number | null;
      expected_source_hit?: number | null;
      refusal_correct?: number | null;
      unsafe_answer_rate?: number | null;
    };
    model?: {
      status?: string;
      synthetic_champion_auroc?: number | null;
      synthetic_champion_auprc?: number | null;
      synthetic_champion_brier?: number | null;
      synthetic_champion_ece_after?: number | null;
    };
    realism?: {
      status?: string;
      alignment_score?: number | null;
      realism_checks_status?: string | null;
    };
    clinician_summary?: {
      status?: string;
      summary_completeness_rate?: number | null;
      unsafe_advice_rate?: number | null;
    };
  };
  report_path?: string;
  csv_path?: string;
  claim_boundary?: string;
}

export interface FailureCaseEntry {
  id: string;
  category: string;
  what_happened: string;
  why_risky: string;
  system_response: string;
  mitigation: string;
  unresolved: string;
}

export interface FailureCaseGallery {
  schema_version?: string;
  generated_at?: string;
  status?: string;
  cases: FailureCaseEntry[];
  message?: string;
}

export interface SafetyCenter {
  generated_at: string;
  safety_red_team: SafetyRedTeamArtifact;
  prompt_injection_defense: SafetyCenterCategorySummary;
  urgent_symptom_escalation: SafetyCenterCategorySummary;
  medication_refusal: SafetyCenterCategorySummary;
  privacy_exfiltration: SafetyCenterCategorySummary;
  rag_eval: RagEvalArtifact;
  rag_trace_summary: unknown;
  benchmark_ladder: BenchmarkLadderSummary;
  calibration_metrics: {
    status: string;
    best_model?: string;
    brier_score?: number | null;
    ece_before?: number | null;
    ece_after?: number | null;
    temperature?: number | null;
    method?: string | null;
    message?: string;
    note?: string;
  };
  drift_report: DriftReport;
  data_quality: unknown;
  clinician_feedback: {
    review_count: number;
    decision_counts: Record<string, number>;
    reason_category_counts: Record<string, number>;
    review_target_counts: Record<string, number>;
    average_explanation_quality_score: number | null;
    average_model_usefulness_score: number | null;
  };
  failure_case_gallery: FailureCaseGallery;
  audit_log_summary: unknown;
  safety_note: string;
}

export interface SimToPublicImagingReport {
  schema_version: string;
  generated_at: string;
  status: string;
  synthetic_summary: {
    status: string;
    path: string;
    row_count: number;
    patient_count?: number | null;
    modalities: Record<string, number>;
    report_type_counts: Record<string, number>;
    metastatic_keyword_rows: number;
  };
  public_imaging_availability: {
    status: string;
    available_dataset_count: number;
    available_modalities: string[];
  };
  gap_table: Array<{
    area: string;
    synthetic_coverage: string;
    public_coverage: string;
    available_now: boolean;
    gap: string;
  }>;
  recommended_actions: string[];
  claim_boundary: string;
}

export interface CurrentVsRealismCandidateReport {
  schema_version?: string;
  generated_at?: string;
  current?: {
    best_classifier?: string | null;
    patient_level_roc_auc?: number | null;
    patient_level_average_precision?: number | null;
    patient_level_brier_score?: number | null;
    realism_status?: string | null;
    realism_alignment_score?: number | null;
    sim_to_real_status?: string | null;
    threshold_coverage_status?: string | null;
  };
  candidate?: {
    best_classifier?: string | null;
    patient_level_roc_auc?: number | null;
    patient_level_average_precision?: number | null;
    patient_level_brier_score?: number | null;
    realism_status?: string | null;
    realism_alignment_score?: number | null;
    sim_to_real_status?: string | null;
    threshold_coverage_status?: string | null;
  };
  recommendation?: {
    decision?: string | null;
    auc_delta?: number | null;
    realism_delta?: number | null;
    rationale?: string | null;
  };
  claim_boundary?: string;
}

export interface MultilingualRefusalEval {
  schema_version?: string;
  generated_at?: string;
  status?: string;
  message?: string;
  reproducibility?: ReproducibilityManifest;
  artifact_freshness?: ArtifactFreshness;
  summary?: {
    status?: string;
    case_count?: number;
    passed?: number;
    pass_rate?: number | null;
    failed_cases?: string[];
  };
  cases?: Array<{
    case_id?: string;
    language?: string;
    query?: string;
    expected_scope?: string;
    observed_scope?: string;
    expected_intent?: string;
    observed_intent?: string;
    pass?: boolean;
  }>;
  limitations?: string[];
}

export interface LlmJudgeEval {
  schema_version?: string;
  generated_at?: string;
  status?: string;
  message?: string;
  provider?: string;
  model?: string;
  reproducibility?: ReproducibilityManifest;
  artifact_freshness?: ArtifactFreshness;
  summary?: {
    case_count?: number;
    judged_cases?: number;
    coverage_rate?: number | null;
    pass_rate?: number | null;
    average_groundedness_score?: number | null;
    average_citation_support_score?: number | null;
    average_refusal_quality_score?: number | null;
    unsafe_medical_advice_rate?: number | null;
    failed_cases?: string[];
  };
  cases?: Array<{
    case_id?: string;
    category?: string;
    available?: boolean;
    provider?: string;
    model?: string;
    groundedness_score?: number | null;
    citation_support_score?: number | null;
    refusal_quality_score?: number | null;
    unsafe_medical_advice?: boolean | null;
    passes?: boolean;
    reason?: string | null;
  }>;
  limitations?: string[];
  claim_boundary?: string;
}
