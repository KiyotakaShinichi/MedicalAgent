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
