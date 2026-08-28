import type * as Api from "../types/api";
import { get, post } from "./client.transport";

export const getAdminAnalytics = () => get<Api.AdminAnalytics>("/admin/analytics");

export const runAgentRegression = () =>
  post<{ message: string; result: Api.AgentRegressionResult }>("/admin/agent-regression");

export const runMleReadiness = () =>
  post<{ message: string; result: Api.MleReadinessSummary }>("/admin/mle-readiness");

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
  get<{ summary: import("../types/api").AgentFeedbackSummary; feedback: Api.AgentFeedbackItem[] }>(
    "/agent-feedback?limit=50"
  );

export const getRagSourceRegistry = () =>
  get<{ sources: Api.RagSourceEntry[]; metrics: unknown }>("/admin/rag-source-registry");

export const getAgentTraceLogs = (limit = 50) =>
  get<Api.AgentTraceLogsResponse>(`/admin/agent-trace-logs?limit=${limit}`);

export const getNoiseEval = () =>
  get<Api.NoiseEvalResult>("/admin/noise-eval");

export const runNoiseEval = () =>
  post<{ message: string; result: Api.NoiseEvalResult }>("/admin/noise-eval");

export const getTemporalEval = () =>
  get<Api.TemporalEvalResult>("/admin/temporal-eval");

export const runTemporalEval = () =>
  post<{ message: string; result: Api.TemporalEvalResult }>("/admin/temporal-eval");

export const getPredictionErrorTable = (limit = 100) =>
  get<Api.PredictionErrorTable>(`/admin/prediction-error-table?limit=${limit}`);

export const getRagAblation = () =>
  get<Api.RagAblationResult>("/admin/rag-ablation");

export const runRagAblation = () =>
  post<{ message: string; result: Api.RagAblationResult }>("/admin/rag-ablation");

export const getPublicDataManifest = () =>
  get<Api.PublicDataManifest>("/admin/public-data-manifest");

export const runPublicDataManifest = () =>
  post<{ message: string; result: Api.PublicDataManifest }>("/admin/public-data-manifest");

export const getPublicBiomarkerDatasetManifest = () =>
  get<Api.PublicBiomarkerDatasetManifest>("/admin/public-biomarker-dataset-manifest");

export const runPublicBiomarkerDatasetManifest = () =>
  post<{ message: string; result: Api.PublicBiomarkerDatasetManifest }>("/admin/public-biomarker-dataset-manifest");

export const getPublicBiomarkerMappingReadiness = () =>
  get<Api.PublicBiomarkerMappingReadiness>("/admin/public-biomarker-mapping-readiness");

export const runPublicBiomarkerMappingReadiness = () =>
  post<{ message: string; result: Api.PublicBiomarkerMappingReadiness }>("/admin/public-biomarker-mapping-readiness");

export const getCbioportalBiomarkerSchemaMapping = () =>
  get<Api.CbioportalBiomarkerSchemaMapping>("/admin/cbioportal-biomarker-schema-mapping");

export const runCbioportalBiomarkerSchemaMapping = (liveFetch = true) =>
  post<{ message: string; result: Api.CbioportalBiomarkerSchemaMapping }>(
    `/admin/cbioportal-biomarker-schema-mapping?live_fetch=${liveFetch ? "true" : "false"}`
  );

export const getClinicalSafetyReviewChecklist = () =>
  get<Api.ClinicalSafetyReviewChecklist>("/admin/clinical-safety-review-checklist");

export const runClinicalSafetyReviewChecklist = () =>
  post<{ message: string; result: Api.ClinicalSafetyReviewChecklist }>("/admin/clinical-safety-review-checklist");

export const getSystemHealth = () =>
  get<Api.SystemHealth>("/admin/system-health");

export const runSystemHealth = () =>
  post<{ message: string; result: Api.SystemHealth }>("/admin/system-health");

// FAST_MODE runtime toggle — emergency degradation switch when the
// Groq cloud provider is rate-limited / down.  enabled=true forces
// fast mode ON, false forces it OFF, null clears the runtime override
// and falls back to the ONCOTRACK_FAST_MODE env var.
export interface FastModeStatus {
  enabled: boolean;
  env_var_value: string | null;
  env_var_active: boolean;
  runtime_override: boolean | null;
  source: "env_var" | "runtime_override";
}

export const getAdminFastMode = () =>
  get<FastModeStatus>("/admin/fast-mode");

export const setAdminFastMode = (enabled: boolean | null) =>
  post<FastModeStatus>("/admin/fast-mode", { enabled });

// Compound-intent live probe — paste a message, see how the router
// would classify it (deterministic + LLM-merged envelope + raw LLM
// verdict).  Stateless: does NOT touch the chat DB.
export interface IntentProbeResponse {
  status: "ok" | "empty";
  message?: string;
  deterministic: import("../types/api").CompoundIntentTrace;
  merged: import("../types/api").CompoundIntentTrace;
  llm: {
    available: boolean;
    language?: string;
    llm_confidence?: number;
    provider?: string;
    model?: string;
    reason?: string;
  };
}

export const probeAdminIntent = (message: string, useLlm: boolean = true) =>
  post<IntentProbeResponse>("/admin/intent-classifier-probe", {
    message,
    use_llm: useLlm,
  });

export const getPublicImagingManifest = () =>
  get<Api.PublicImagingManifest>("/admin/public-imaging-manifest");

export const runPublicImagingManifest = () =>
  post<{ message: string; result: Api.PublicImagingManifest }>("/admin/public-imaging-manifest");

export const getUltrasoundBaseline = () =>
  get<Api.UltrasoundBaselineResult>("/admin/ultrasound-baseline");

export const runUltrasoundBaseline = () =>
  post<{ message: string; result: Api.UltrasoundBaselineResult }>("/admin/ultrasound-baseline");

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
  get<Api.CtLesionWorkflowReport>("/admin/ct-lesion-workflow");

export const runCtLesionWorkflow = () =>
  post<{ message: string; result: Api.CtLesionWorkflowReport }>("/admin/ct-lesion-workflow");

export const getSimToPublicImaging = () =>
  get<Api.SimToPublicImagingReport>("/admin/sim-to-public-imaging");

export const runSimToPublicImaging = () =>
  post<{ message: string; result: Api.SimToPublicImagingReport }>("/admin/sim-to-public-imaging");

export const getCurrentVsRealismCandidate = () =>
  get<import("../types/api").CurrentVsRealismCandidateReport>("/admin/current-vs-realism-candidate");

export const runCurrentVsRealismCandidate = () =>
  post<{ message: string; result: import("../types/api").CurrentVsRealismCandidateReport }>(
    "/admin/current-vs-realism-candidate"
  );

export const getBiomarkerFeatureBenchmark = () =>
  get<unknown>("/admin/biomarker-feature-benchmark");

export const runBiomarkerFeatureBenchmark = () =>
  post<{ message: string; result: unknown }>("/admin/biomarker-feature-benchmark");

export const getFullFeatureGroupAblation = () =>
  get<import("../types/api").FullFeatureGroupAblationReport>("/admin/full-feature-group-ablation");

export const runFullFeatureGroupAblation = () =>
  post<{ message: string; result: import("../types/api").FullFeatureGroupAblationReport }>(
    "/admin/full-feature-group-ablation",
  );

export const getLeakageAudit = () =>
  get<import("../types/api").LeakageAuditReport>("/admin/leakage-audit");

export const runLeakageAudit = () =>
  post<{ message: string; result: import("../types/api").LeakageAuditReport }>(
    "/admin/leakage-audit",
  );

export const getKbSourceGovernance = () =>
  get<import("../types/api").KbSourceGovernanceReport>("/admin/kb-source-governance");

export const runKbSourceGovernance = () =>
  post<{ message: string; result: import("../types/api").KbSourceGovernanceReport }>(
    "/admin/kb-source-governance",
  );

export const getSyntheticGeneratorCard = () =>
  get<import("../types/api").SyntheticGeneratorCard>("/admin/synthetic-generator-card");

export const runSyntheticGeneratorCard = () =>
  post<{ message: string; result: import("../types/api").SyntheticGeneratorCard }>(
    "/admin/synthetic-generator-card",
  );

export const getFailureModeRegistry = () =>
  get<import("../types/api").FailureModeRegistry>("/admin/failure-mode-registry");

export const runFailureModeRegistry = () =>
  post<{ message: string; result: import("../types/api").FailureModeRegistry }>(
    "/admin/failure-mode-registry",
  );

export const getModalityRobustnessComparison = () =>
  get<import("../types/api").ModalityRobustnessComparisonReport>(
    "/admin/modality-robustness-comparison",
  );

export const runModalityRobustnessComparison = () =>
  post<{ message: string; result: import("../types/api").ModalityRobustnessComparisonReport }>(
    "/admin/modality-robustness-comparison",
  );

export const getResponseConformalCalibration = () =>
  get<import("../types/api").ResponseConformalCalibrationReport>(
    "/admin/response-conformal-calibration",
  );

export const runResponseConformalCalibration = () =>
  post<{ message: string; result: import("../types/api").ResponseConformalCalibrationReport }>(
    "/admin/response-conformal-calibration",
  );

export const getRobustnessStress = () =>
  get<import("../types/api").RobustnessStressReport>("/admin/robustness-stress");

export const runRobustnessStress = () =>
  post<{ message: string; result: import("../types/api").RobustnessStressReport }>(
    "/admin/robustness-stress",
  );

export const getClinicianPatientPredictionTraces = (patient_id: string, params?: {
  limit?: number;
  abstained_only?: boolean;
}) => {
  const q = new URLSearchParams();
  if (params?.limit != null) q.set("limit", String(params.limit));
  if (params?.abstained_only) q.set("abstained_only", "true");
  const suffix = q.toString() ? `?${q.toString()}` : "";
  return get<import("../types/api").ClinicianPredictionTracesResponse>(
    `/clinician/patients/${encodeURIComponent(patient_id)}/prediction-traces${suffix}`,
  );
};

export const getPredictionTraces = (params?: {
  limit?: number;
  patient_id?: string;
  decision?: string;
  abstained_only?: boolean;
}) => {
  const q = new URLSearchParams();
  if (params?.limit != null) q.set("limit", String(params.limit));
  if (params?.patient_id) q.set("patient_id", params.patient_id);
  if (params?.decision) q.set("decision", params.decision);
  if (params?.abstained_only) q.set("abstained_only", "true");
  const suffix = q.toString() ? `?${q.toString()}` : "";
  return get<import("../types/api").PredictionTraceResponse>(`/admin/prediction-traces${suffix}`);
};

export const getEvidenceAbstentionEval = () =>
  get<import("../types/api").EvidenceAbstentionEvalReport>("/admin/evidence-abstention-eval");

export const runEvidenceAbstentionEval = () =>
  post<{ message: string; result: import("../types/api").EvidenceAbstentionEvalReport }>(
    "/admin/evidence-abstention-eval",
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

export const getNormalizedBenchmarkArtifact = (artifactId: string) =>
  get<import("../types/api").AdminBenchmarkResponse>(
    `/admin/benchmark-artifacts/${encodeURIComponent(artifactId)}`,
  );

export const getLiveRagEval = () =>
  get<unknown>("/admin/live-rag-eval");

export const runLiveRagEval = () =>
  post<{ message: string; result: unknown }>("/admin/live-rag-eval");

export const getClaimLevelCitationEval = () =>
  get<unknown>("/admin/claim-level-citation-eval");

export const getRagTraceReplay = (limit = 12) =>
  get<unknown>(`/admin/rag-trace-replay?limit=${limit}`);

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
