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
  adversarial_generalization_v2?: Record<string, unknown>;
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
