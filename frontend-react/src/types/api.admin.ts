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
  estimated_input_tokens: number | null;
  estimated_output_tokens: number | null;
  estimated_total_tokens: number | null;
  estimated_cost_usd: number | null;
  model_used: string | null;
  stage_latency: Record<string, number | null> | null;
  token_usage: {
    call_count?: number;
    provider_reported_call_count?: number;
    estimated_call_count?: number;
    actual_usage_coverage_rate?: number;
    input_tokens?: number;
    output_tokens?: number;
    total_tokens?: number;
    estimated_cost_usd?: number;
    llm_call_latency_ms?: number;
    content_retained?: boolean;
  } | null;
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
