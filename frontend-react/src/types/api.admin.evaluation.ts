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
