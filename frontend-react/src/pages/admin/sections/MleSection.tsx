import { useState } from "react";
import { RefreshCw, AlertTriangle, Info, ShieldCheck, ShieldAlert } from "lucide-react";
import { Badge } from "../../../components/ui/Badge";
import { statusVariant } from "../../../components/ui/badgeUtils";
import { Card, CardHeader, SectionTitle } from "../../../components/ui/Card";
import { Button } from "../../../components/ui/Button";
import { MetricCard } from "../../../components/ui/MetricCard";
import { MetricGlossary, ALL_METRIC_SPECS } from "../../../components/ui/MetricInterpretation";
import { LoadingPane, ErrorPane, EmptyPane } from "../../../components/ui/Spinner";
import { useApi } from "../../../hooks/useApi";
import { LeakageAuditCard } from "./cards/LeakageAuditCard";
import { KbSourceGovernanceCard } from "./cards/KbSourceGovernanceCard";
import { ModalityRobustnessCard } from "./cards/ModalityRobustnessCard";
import {
  runMleReadiness, getTrainingReport, getLockedHoldout,
  getExternalValidation, getModelComparison,
  getNoiseEval, getTemporalEval, getPredictionErrorTable,
  getPublicDataManifest,
  getPublicBiomarkerDatasetManifest,
  runPublicBiomarkerDatasetManifest,
  getPublicBiomarkerMappingReadiness,
  runPublicBiomarkerMappingReadiness,
  getCbioportalBiomarkerSchemaMapping,
  runCbioportalBiomarkerSchemaMapping,
  getFullFeatureGroupAblation,
  runFullFeatureGroupAblation,
  getCurrentVsRealismCandidate,
  runCurrentVsRealismCandidate,
  getLeakageAudit,
  runLeakageAudit,
  getEvidenceAbstentionEval,
  runEvidenceAbstentionEval,
  getPredictionTraces,
  getModalityRobustnessComparison,
  runModalityRobustnessComparison,
  getResponseConformalCalibration,
  runResponseConformalCalibration,
  getRobustnessStress,
  runRobustnessStress,
  getSyntheticGeneratorCard,
  runSyntheticGeneratorCard,
  getFailureModeRegistry,
  runFailureModeRegistry,
  getKbSourceGovernance,
  runKbSourceGovernance,
} from "../../../api/client";
import type {
  AdminAnalytics,
  NoiseEvalResult,
  TemporalEvalResult,
  PredictionErrorTable,
  PublicDataManifest,
  PublicBiomarkerDatasetManifest,
  PublicBiomarkerMappingReadiness,
  CbioportalBiomarkerSchemaMapping,
  FullFeatureGroupAblationReport,
  CurrentVsRealismCandidateReport,
  LeakageAuditReport,
  EvidenceAbstentionEvalReport,
  PredictionTraceResponse,
  ModalityRobustnessComparisonReport,
  ResponseConformalCalibrationReport,
  RobustnessStressReport,
  SyntheticGeneratorCard,
  FailureModeRegistry,
  KbSourceGovernanceReport,
} from "../../../types/api";

interface Props { analytics: AdminAnalytics; onRefresh: () => void }

export function MleSection({ analytics, onRefresh }: Props) {
  const mle = analytics.mle_readiness;
  const [runningMle, setRunningMle] = useState(false);
  const [showGlossary, setShowGlossary] = useState(false);

  const { data: trainingReport, status: trStatus } = useApi(getTrainingReport, []);
  const { data: holdout, status: holdoutStatus } = useApi(getLockedHoldout, []);
  const { data: extVal, status: extValStatus } = useApi(getExternalValidation, []);
  const { data: modelComp, status: modelCompStatus } = useApi(getModelComparison, []);
  const { data: noiseEval, status: noiseStatus } = useApi(getNoiseEval, []);
  const { data: temporalEval, status: temporalStatus } = useApi(getTemporalEval, []);
  const { data: errorTable, status: errorStatus } = useApi(getPredictionErrorTable, []);
  const { data: dataManifest, status: dataManifestStatus } = useApi(getPublicDataManifest, []);
  const { data: biomarkerManifest, status: biomarkerManifestStatus, refetch: refetchBiomarkerManifest } = useApi(getPublicBiomarkerDatasetManifest, []);
  const { data: biomarkerMapping, status: biomarkerMappingStatus, refetch: refetchBiomarkerMapping } = useApi(getPublicBiomarkerMappingReadiness, []);
  const { data: cbioMapping, status: cbioMappingStatus, refetch: refetchCbioMapping } = useApi(getCbioportalBiomarkerSchemaMapping, []);
  const { data: fullFeatureAblation, status: fullFeatureAblationStatus, refetch: refetchFullFeatureAblation } = useApi(getFullFeatureGroupAblation, []);
  const { data: candidateComparison, status: candidateStatus, refetch: refetchCandidate } = useApi(getCurrentVsRealismCandidate, []);
  const { data: leakageAudit, status: leakageStatus, refetch: refetchLeakageAudit } = useApi(getLeakageAudit, []);
  const { data: abstentionEval, status: abstentionStatus, refetch: refetchAbstentionEval } = useApi(getEvidenceAbstentionEval, []);
  const { data: predictionTraces, status: predictionTracesStatus, refetch: refetchPredictionTraces } = useApi(
    () => getPredictionTraces({ limit: 25 }),
    [],
  );
  const { data: modalityComparison, status: modalityComparisonStatus, refetch: refetchModalityComparison } = useApi(
    getModalityRobustnessComparison, [],
  );
  const [runningModalityComparison, setRunningModalityComparison] = useState(false);
  const { data: conformalCalibration, status: conformalStatus, refetch: refetchConformalCalibration } = useApi(
    getResponseConformalCalibration, [],
  );
  const { data: robustnessStress, status: robustnessStressStatus, refetch: refetchRobustnessStress } = useApi(
    getRobustnessStress, [],
  );
  const [runningConformalCalibration, setRunningConformalCalibration] = useState(false);
  const [runningRobustnessStress, setRunningRobustnessStress] = useState(false);

  const { data: generatorCard, status: generatorCardStatus, refetch: refetchGeneratorCard } = useApi(getSyntheticGeneratorCard, []);
  const { data: failureRegistry, status: failureRegistryStatus, refetch: refetchFailureRegistry } = useApi(getFailureModeRegistry, []);
  const { data: kbGovernance, status: kbGovernanceStatus, refetch: refetchKbGovernance } = useApi(getKbSourceGovernance, []);
  const [runningGeneratorCard, setRunningGeneratorCard] = useState(false);
  const [runningFailureRegistry, setRunningFailureRegistry] = useState(false);
  const [runningKbGovernance, setRunningKbGovernance] = useState(false);
  const [runningCandidate, setRunningCandidate] = useState(false);
  const [runningLeakageAudit, setRunningLeakageAudit] = useState(false);
  const [runningAbstentionEval, setRunningAbstentionEval] = useState(false);
  const [runningBiomarkerManifest, setRunningBiomarkerManifest] = useState(false);
  const [runningBiomarkerMapping, setRunningBiomarkerMapping] = useState(false);
  const [runningCbioMapping, setRunningCbioMapping] = useState(false);
  const [runningFullFeatureAblation, setRunningFullFeatureAblation] = useState(false);

  async function runMle() {
    setRunningMle(true);
    try { await runMleReadiness(); onRefresh(); } finally { setRunningMle(false); }
  }

  async function runCandidateComparison() {
    setRunningCandidate(true);
    try {
      await runCurrentVsRealismCandidate();
      await refetchCandidate();
    } finally {
      setRunningCandidate(false);
    }
  }

  async function refreshBiomarkerManifest() {
    setRunningBiomarkerManifest(true);
    try {
      await runPublicBiomarkerDatasetManifest();
      await refetchBiomarkerManifest();
    } finally {
      setRunningBiomarkerManifest(false);
    }
  }

  async function refreshBiomarkerMapping() {
    setRunningBiomarkerMapping(true);
    try {
      await runPublicBiomarkerMappingReadiness();
      await refetchBiomarkerMapping();
    } finally {
      setRunningBiomarkerMapping(false);
    }
  }

  async function refreshLeakageAudit() {
    setRunningLeakageAudit(true);
    try {
      await runLeakageAudit();
      await refetchLeakageAudit();
    } finally {
      setRunningLeakageAudit(false);
    }
  }

  async function refreshAbstentionEval() {
    setRunningAbstentionEval(true);
    try {
      await runEvidenceAbstentionEval();
      await refetchAbstentionEval();
    } finally {
      setRunningAbstentionEval(false);
    }
  }

  async function refreshModalityComparison() {
    setRunningModalityComparison(true);
    try {
      await runModalityRobustnessComparison();
      await refetchModalityComparison();
    } finally {
      setRunningModalityComparison(false);
    }
  }

  async function refreshConformalCalibration() {
    setRunningConformalCalibration(true);
    try {
      await runResponseConformalCalibration();
      await refetchConformalCalibration();
    } finally {
      setRunningConformalCalibration(false);
    }
  }

  async function refreshRobustnessStress() {
    setRunningRobustnessStress(true);
    try {
      await runRobustnessStress();
      await refetchRobustnessStress();
    } finally {
      setRunningRobustnessStress(false);
    }
  }

  async function refreshGeneratorCard() {
    setRunningGeneratorCard(true);
    try {
      await runSyntheticGeneratorCard();
      await refetchGeneratorCard();
    } finally {
      setRunningGeneratorCard(false);
    }
  }

  async function refreshFailureRegistry() {
    setRunningFailureRegistry(true);
    try {
      await runFailureModeRegistry();
      await refetchFailureRegistry();
    } finally {
      setRunningFailureRegistry(false);
    }
  }

  async function refreshKbGovernance() {
    setRunningKbGovernance(true);
    try {
      await runKbSourceGovernance();
      await refetchKbGovernance();
    } finally {
      setRunningKbGovernance(false);
    }
  }

  async function refreshCbioMapping() {
    setRunningCbioMapping(true);
    try {
      await runCbioportalBiomarkerSchemaMapping(true);
      await refetchCbioMapping();
    } finally {
      setRunningCbioMapping(false);
    }
  }

  async function refreshFullFeatureAblation() {
    setRunningFullFeatureAblation(true);
    try {
      await runFullFeatureGroupAblation();
      await refetchFullFeatureAblation();
    } finally {
      setRunningFullFeatureAblation(false);
    }
  }

  const tr = (trainingReport as { result?: Record<string, unknown> } | null)?.result;
  const ho = (holdout as { result?: Record<string, unknown> } | null)?.result;

  const trValues: Record<string, number | null> = tr ? {
    "AUROC": parseFloat(String(tr.auroc)) || null,
    "Brier Score": parseFloat(String(tr.brier_score)) || null,
    "ECE": parseFloat(String(tr.ece)) || null,
    "Sensitivity (Recall)": parseFloat(String(tr.sensitivity)) || null,
    "MAE (Regression)": parseFloat(String(tr.mae)) || null,
  } : {};

  return (
    <div className="flex flex-col gap-4">
      {/* Synthetic data disclaimer */}
      <div
        className="flex items-start gap-2 px-3 py-2.5 rounded-lg border text-xs"
        style={{ background: "rgba(245,158,11,0.07)", borderColor: "rgba(245,158,11,0.25)", color: "var(--amber)" }}
      >
        <AlertTriangle size={13} style={{ flexShrink: 0, marginTop: 1 }} />
        <span>
          All metrics below are computed on <strong>synthetic data</strong> unless explicitly labelled "locked holdout" or "external validation".
          Synthetic AUROC is expected to be high and does not reflect clinical validity.
          The locked holdout uses a frozen synthetic split; external validation uses BreastDCEDL/I-SPY1 tabular features.
        </span>
      </div>

      {/* Synthetic generator card — provenance + documented assumptions. */}
      <SyntheticGeneratorCardPanel
        report={generatorCard as SyntheticGeneratorCard | null}
        loading={generatorCardStatus === "loading"}
        running={runningGeneratorCard}
        onRefresh={refreshGeneratorCard}
      />

      {/* Consolidated failure-mode registry. */}
      <FailureModeRegistryCard
        report={failureRegistry as FailureModeRegistry | null}
        loading={failureRegistryStatus === "loading"}
        running={runningFailureRegistry}
        onRefresh={refreshFailureRegistry}
      />

      {/* RAG source governance — per-source tier + allowed_use + staleness. */}
      <KbSourceGovernanceCard
        report={kbGovernance as KbSourceGovernanceReport | null}
        loading={kbGovernanceStatus === "loading"}
        running={runningKbGovernance}
        onRefresh={refreshKbGovernance}
      />

      {/* Training-data leakage audit — the engineering data-hygiene gate. */}
      <LeakageAuditCard
        report={leakageAudit as LeakageAuditReport | null}
        loading={leakageStatus === "loading"}
        running={runningLeakageAudit}
        onRefresh={refreshLeakageAudit}
      />

      {/* Evidence-aware abstention eval — modality-dropout sweep showing
          coverage / accuracy / abstention rate per missing-modality scenario. */}
      <EvidenceAbstentionCard
        report={abstentionEval as EvidenceAbstentionEvalReport | null}
        loading={abstentionStatus === "loading"}
        running={runningAbstentionEval}
        onRefresh={refreshAbstentionEval}
      />

      {/* Champion vs modality-robust comparison — head-to-head over all
          modality-dropout scenarios.  Shows whether retraining with random
          modality masking actually moved the classifier or just the
          abstention rules. */}
      <ModalityRobustnessCard
        report={modalityComparison as ModalityRobustnessComparisonReport | null}
        loading={modalityComparisonStatus === "loading"}
        running={runningModalityComparison}
        onRefresh={refreshModalityComparison}
      />

      {/* Prediction traceability — one row per live evidence-aware call,
          showing decision, modalities used, model + threshold + calibration
          provenance, and validator verdict. */}
      <ResponseConformalCalibrationCard
        report={conformalCalibration as ResponseConformalCalibrationReport | null}
        loading={conformalStatus === "loading"}
        running={runningConformalCalibration}
        onRefresh={refreshConformalCalibration}
      />

      <RobustnessStressCard
        report={robustnessStress as RobustnessStressReport | null}
        loading={robustnessStressStatus === "loading"}
        running={runningRobustnessStress}
        onRefresh={refreshRobustnessStress}
      />

      <PredictionTraceCard
        response={predictionTraces as PredictionTraceResponse | null}
        loading={predictionTracesStatus === "loading"}
        onRefresh={refetchPredictionTraces}
      />

      {/* Cost-sensitive eval rationale */}
      <Card>
        <CardHeader>
          <SectionTitle>Threshold &amp; Cost-Sensitive Evaluation</SectionTitle>
          <Info size={14} style={{ color: "var(--text-faint)" }} />
        </CardHeader>
        <div className="grid sm:grid-cols-3 gap-3 mb-3">
          <CostCard
            label="False Negative Cost"
            level="HIGH"
            color="var(--rose)"
            description="Missed positive case (failed response) delays clinician intervention. Prioritise recall."
          />
          <CostCard
            label="False Positive Cost"
            level="MODERATE"
            color="var(--amber)"
            description="Unnecessary flag increases review burden. Acceptable trade-off for lower FNR."
          />
          <CostCard
            label="Operating Threshold"
            level="≤ 0.40"
            color="var(--blue)"
            description="Decision threshold set below 0.50 to bias toward sensitivity. Reviewed per-model at training time."
          />
        </div>
        <p className="text-xs" style={{ color: "var(--text-dim)" }}>
          This system uses a cost-sensitive approach: the classification threshold is chosen to minimise FNR at acceptable FPR,
          reflecting the assumption that missing a treatment non-response is more harmful than over-flagging for clinician review.
          Weighted cost = FN_weight × FN + FP_weight × FP where FN_weight = 3, FP_weight = 1 (engineering heuristic, not clinical guidance).
        </p>
      </Card>

      {/* Gate status */}
      <Card>
        <CardHeader>
          <SectionTitle>MLE Readiness Gates</SectionTitle>
          <Button
            variant="secondary" size="sm"
            loading={runningMle}
            icon={<RefreshCw size={12} />}
            onClick={() => void runMle()}
          >
            Re-run gates
          </Button>
        </CardHeader>
        <div className="flex flex-wrap items-center gap-3 mb-3">
          <Badge variant={statusVariant(mle.status)}>{mle.status}</Badge>
          <span className="text-xs" style={{ color: "var(--text-dim)" }}>
            {mle.release_recommendation.replace(/_/g, " ")}
          </span>
          <span
            className="text-xs px-2 py-0.5 rounded border"
            style={{
              background: mle.hard_gate_failures === 0 ? "rgba(16,185,129,0.08)" : "rgba(244,63,94,0.08)",
              borderColor: mle.hard_gate_failures === 0 ? "rgba(16,185,129,0.25)" : "rgba(244,63,94,0.25)",
              color: mle.hard_gate_failures === 0 ? "var(--green)" : "var(--rose)",
            }}
          >
            {mle.hard_gate_failures === 0 ? "All hard gates passed" : `${mle.hard_gate_failures} gate failures`}
          </span>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
          {Object.entries(mle.category_statuses).map(([cat, st]) => (
            <div key={cat} className="flex flex-col gap-1 p-2 rounded-md" style={{ background: "var(--surface2)" }}>
              <span className="text-xs" style={{ color: "var(--text-faint)" }}>{cat.replace(/_/g, " ")}</span>
              <Badge variant={statusVariant(st)}>{st}</Badge>
            </div>
          ))}
        </div>
      </Card>

      <Card>
        <CardHeader>
          <SectionTitle>Current vs Realism-Calibrated Candidate</SectionTitle>
          <Button
            variant="secondary"
            size="sm"
            loading={runningCandidate}
            icon={<RefreshCw size={12} />}
            onClick={() => void runCandidateComparison()}
          >
            Compare
          </Button>
        </CardHeader>
        {candidateStatus === "loading" ? <LoadingPane /> :
         candidateStatus === "error" ? <ErrorPane message="Could not load current-vs-candidate report" /> :
         !candidateComparison ? <EmptyPane label="No current-vs-candidate report available" /> : (
          <CandidateComparisonPanel data={candidateComparison} />
        )}
      </Card>

      {/* Training report */}
      <Card>
        <CardHeader>
          <SectionTitle>Synthetic Training Report</SectionTitle>
          <span className="text-xs px-2 py-0.5 rounded" style={{ background: "rgba(245,158,11,0.12)", color: "var(--amber)" }}>
            Synthetic data
          </span>
        </CardHeader>
        {trStatus === "loading" ? <LoadingPane /> :
         trStatus === "error" ? <ErrorPane message="Could not load training report" /> :
         !tr ? <EmptyPane label="No training report — run training first" /> : (
          <>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
              {[
                ["Test patients",   tr.test_patients,   null,  "muted"],
                ["Best classifier", tr.best_classifier, null,  "muted"],
                ["Best regressor",  tr.best_regressor,  null,  "muted"],
                ["AUROC",          tr.auroc,            0.85,  "green"],
                ["Brier score",     tr.brier_score,     0.10,  "green"],
                ["MAE",            tr.mae,              0.10,  "green"],
                ["RMSE",           tr.rmse,             0.15,  "green"],
              ].map(([label, val, threshold]) => {
                const num = typeof val === "number" ? val : parseFloat(String(val));
                const status: "green" | "amber" | "red" | "muted" =
                  threshold == null || isNaN(num) ? "muted" :
                  label === "Brier score" || label === "MAE" || label === "RMSE"
                    ? (num <= (threshold as number) ? "green" : num <= (threshold as number) * 2 ? "amber" : "red")
                    : (num >= (threshold as number) ? "green" : num >= (threshold as number) * 0.85 ? "amber" : "red");
                return (
                  <MetricCard
                    key={label as string}
                    label={label as string}
                    value={val != null ? String(val) : null}
                    status={status}
                  />
                );
              })}
            </div>
            <p className="text-xs" style={{ color: "var(--text-faint)" }}>
              Metrics labelled with interpretation bands below. AUROC ≥ 0.85 = strong on synthetic; Brier &lt; 0.10 = well-calibrated; MAE &lt; 0.10 = good regression fit.
            </p>
          </>
        )}
      </Card>

      {/* Locked holdout */}
      <Card>
        <CardHeader>
          <SectionTitle>Locked Holdout Evaluation</SectionTitle>
          <span className="text-xs px-2 py-0.5 rounded" style={{ background: "rgba(59,130,246,0.12)", color: "#93c5fd" }}>
            Frozen synthetic split
          </span>
        </CardHeader>
        {holdoutStatus === "loading" ? <LoadingPane /> :
         !ho ? <EmptyPane label="No holdout evaluation — run holdout first" /> : (
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {[
              ["AUROC",       ho.auroc,        0.80, "Higher = better discrimination"],
              ["Brier",       ho.brier_score,  0.12, "Lower = better calibration"],
              ["Sensitivity", ho.sensitivity,  0.75, "Higher = fewer missed positives"],
              ["MAE",         ho.mae,          0.12, "Lower = better regression fit"],
            ].map(([label, val, threshold, tip]) => {
              const num = typeof val === "number" ? val : parseFloat(String(val));
              const isLowerBetter = label === "Brier" || label === "MAE";
              const status: "green" | "amber" | "red" | "muted" = isNaN(num) ? "muted" :
                isLowerBetter
                  ? (num <= (threshold as number) ? "green" : num <= (threshold as number) * 1.6 ? "amber" : "red")
                  : (num >= (threshold as number) ? "green" : num >= (threshold as number) * 0.85 ? "amber" : "red");
              return (
                <MetricCard
                  key={label as string}
                  label={label as string}
                  value={val != null ? String(val) : null}
                  status={status}
                  sub={tip as string}
                />
              );
            })}
          </div>
        )}
      </Card>

      {/* External validation */}
      <Card>
        <CardHeader>
          <SectionTitle>External Validation Direction</SectionTitle>
          <span className="text-xs px-2 py-0.5 rounded" style={{ background: "rgba(139,92,246,0.12)", color: "#c4b5fd" }}>
            BreastDCEDL / I-SPY1
          </span>
        </CardHeader>
        {extValStatus === "loading" ? <LoadingPane /> :
         !(extVal as { result?: unknown } | null)?.result
           ? <EmptyPane label="No external validation data — run external validation first" />
           : (
            <div className="flex flex-col gap-2">
              <p className="text-xs" style={{ color: "var(--text-dim)" }}>
                External validation uses BreastDCEDL and I-SPY1 tabular MRI-derived features.
                These are real datasets (non-synthetic) used for directional validation only —
                not a clinical performance claim.
              </p>
              <p className="text-xs px-2 py-1.5 rounded-md border" style={{
                background: "rgba(139,92,246,0.07)", borderColor: "rgba(139,92,246,0.25)", color: "#c4b5fd"
              }}>
                ✓ External validation report loaded. See <code>Data/external_validation/</code> for per-dataset metrics.
              </p>
            </div>
          )}
      </Card>

      {/* Model comparison */}
      <Card>
        <CardHeader><SectionTitle>Model Comparison</SectionTitle></CardHeader>
        {modelCompStatus === "loading" ? <LoadingPane /> :
         !(modelComp as { result?: unknown } | null)?.result
           ? <EmptyPane label="No model comparison available" />
           : (
            <p className="text-xs" style={{ color: "var(--text-dim)" }}>
              Model comparison loaded. Δ AUROC, Δ Brier, Δ ECE, Δ FNR deltas available in <code>Data/model_comparison/</code>.
            </p>
          )}
      </Card>

      {/* Public data feasibility */}
      <Card>
        <CardHeader>
          <SectionTitle>Public Data Feasibility</SectionTitle>
          <span className="text-xs px-2 py-0.5 rounded" style={{ background: "rgba(59,130,246,0.12)", color: "#93c5fd" }}>
            Source-calibrated synthetic data
          </span>
        </CardHeader>
        {dataManifestStatus === "loading" ? <LoadingPane /> :
         dataManifestStatus === "error" ? <ErrorPane message="Could not load public data manifest" /> :
         !dataManifest ? <EmptyPane label="No public data manifest available" /> : (
          <PublicDataManifestPanel data={dataManifest as PublicDataManifest} />
        )}
      </Card>

      <Card>
        <CardHeader>
          <SectionTitle>Public Biomarker &amp; Tumor-Marker Sources</SectionTitle>
          <Button
            variant="secondary"
            size="sm"
            loading={runningBiomarkerManifest}
            icon={<RefreshCw size={12} />}
            onClick={() => void refreshBiomarkerManifest()}
          >
            Refresh manifest
          </Button>
        </CardHeader>
        {biomarkerManifestStatus === "loading" ? <LoadingPane /> :
         biomarkerManifestStatus === "error" ? <ErrorPane message="Could not load public biomarker manifest" /> :
         !biomarkerManifest ? <EmptyPane label="No public biomarker manifest available" /> : (
          <PublicBiomarkerManifestPanel data={biomarkerManifest as PublicBiomarkerDatasetManifest} />
        )}
      </Card>

      <Card>
        <CardHeader>
          <SectionTitle>Public Biomarker Mapping Readiness</SectionTitle>
          <Button
            variant="secondary"
            size="sm"
            loading={runningBiomarkerMapping}
            icon={<RefreshCw size={12} />}
            onClick={() => void refreshBiomarkerMapping()}
          >
            Rebuild mapping
          </Button>
        </CardHeader>
        {biomarkerMappingStatus === "loading" ? <LoadingPane /> :
         biomarkerMappingStatus === "error" ? <ErrorPane message="Could not load public biomarker mapping readiness" /> :
         !biomarkerMapping ? <EmptyPane label="No public biomarker mapping readiness report available" /> : (
          <PublicBiomarkerMappingPanel data={biomarkerMapping as PublicBiomarkerMappingReadiness} />
        )}
      </Card>

      <Card>
        <CardHeader>
          <SectionTitle>TCGA / METABRIC cBioPortal Mapping</SectionTitle>
          <Button
            variant="secondary"
            size="sm"
            loading={runningCbioMapping}
            icon={<RefreshCw size={12} />}
            onClick={() => void refreshCbioMapping()}
          >
            Fetch schema
          </Button>
        </CardHeader>
        {cbioMappingStatus === "loading" ? <LoadingPane /> :
         cbioMappingStatus === "error" ? <ErrorPane message="Could not load cBioPortal schema mapping" /> :
         !cbioMapping ? <EmptyPane label="No cBioPortal mapping available" /> : (
          <CbioPortalMappingPanel data={cbioMapping} />
        )}
      </Card>

      <Card>
        <CardHeader>
          <SectionTitle>Full Feature-Group Ablation</SectionTitle>
          <Button
            variant="secondary"
            size="sm"
            loading={runningFullFeatureAblation}
            icon={<RefreshCw size={12} />}
            onClick={() => void refreshFullFeatureAblation()}
          >
            Rerun ablation
          </Button>
        </CardHeader>
        {fullFeatureAblationStatus === "loading" ? <LoadingPane /> :
         fullFeatureAblationStatus === "error" ? <ErrorPane message="Could not load full feature-group ablation" /> :
         !fullFeatureAblation || (fullFeatureAblation as FullFeatureGroupAblationReport).status === "missing" ? (
          <EmptyPane label="No full feature-group ablation available" />
         ) : (
          <FullFeatureGroupAblationPanel data={fullFeatureAblation as FullFeatureGroupAblationReport} />
        )}
      </Card>

      {/* Noise robustness */}
      <Card>
        <CardHeader>
          <SectionTitle>Noise Robustness Evaluation</SectionTitle>
          <span className="text-xs px-2 py-0.5 rounded" style={{ background: "rgba(245,158,11,0.12)", color: "var(--amber)" }}>
            Synthetic perturbations
          </span>
        </CardHeader>
        {noiseStatus === "loading" ? <LoadingPane /> :
         noiseStatus === "error" ? <ErrorPane message="Could not load noise eval" /> :
         !noiseEval || (noiseEval as NoiseEvalResult).status === "unavailable" ? (
          <EmptyPane label="No noise eval — run POST /admin/noise-eval first" />
         ) : (
          <NoiseEvalPanel data={noiseEval as NoiseEvalResult} />
         )}
      </Card>

      {/* Temporal generalization */}
      <Card>
        <CardHeader>
          <SectionTitle>Temporal Generalization</SectionTitle>
          <span className="text-xs px-2 py-0.5 rounded" style={{ background: "rgba(59,130,246,0.12)", color: "#93c5fd" }}>
            Synthetic timeline splits
          </span>
        </CardHeader>
        {temporalStatus === "loading" ? <LoadingPane /> :
         temporalStatus === "error" ? <ErrorPane message="Could not load temporal eval" /> :
         !temporalEval || (temporalEval as TemporalEvalResult).status === "unavailable" ? (
          <EmptyPane label="No temporal eval — run POST /admin/temporal-eval first" />
         ) : (
          <TemporalEvalPanel data={temporalEval as TemporalEvalResult} />
         )}
      </Card>

      {/* Per-prediction ML error table */}
      <Card>
        <CardHeader>
          <SectionTitle>Per-Prediction Error Table</SectionTitle>
          <span className="text-xs px-2 py-0.5 rounded" style={{ background: "rgba(245,158,11,0.12)", color: "var(--amber)" }}>
            Synthetic holdout
          </span>
        </CardHeader>
        {errorStatus === "loading" ? <LoadingPane /> :
         errorStatus === "error" ? <ErrorPane message="Could not load prediction error table" /> :
         !errorTable || !(errorTable as PredictionErrorTable).rows?.length ? (
          <EmptyPane label="No predictions — run training pipeline first" />
         ) : (
          <PredictionErrorPanel data={errorTable as PredictionErrorTable} />
         )}
      </Card>

      {/* Metric glossary toggle */}
      <Card>
        <CardHeader>
          <SectionTitle>Metric Interpretation Guide</SectionTitle>
          <Button
            variant="ghost" size="sm"
            onClick={() => setShowGlossary((v) => !v)}
          >
            {showGlossary ? "Hide" : "Show"} guide
          </Button>
        </CardHeader>
        {showGlossary ? (
          <div className="flex flex-col gap-2">
            <p className="text-xs mb-2" style={{ color: "var(--text-dim)" }}>
              Each metric below shows its definition, why it matters in this context, and ideal / warning / bad interpretation bands.
              Bands reflect engineering heuristics for a cancer monitoring PoC — not clinical validation thresholds.
            </p>
            <MetricGlossary specs={ALL_METRIC_SPECS} values={trValues} />
          </div>
        ) : (
          <p className="text-xs" style={{ color: "var(--text-faint)" }}>
            Click "Show guide" to see interpretation bands for AUROC, PR-AUC, Brier, ECE, Sensitivity, Specificity, FNR, and MAE.
          </p>
        )}
      </Card>
    </div>
  );
}

function PublicDataManifestPanel({ data }: { data: PublicDataManifest }) {
  const visibleNeeds = data.feature_feasibility.slice(0, 6);
  const sourceNames = new Map(data.sources.map((source) => [source.id, source.name]));

  return (
    <div className="flex flex-col gap-3">
      <div className="rounded-md border p-3" style={{ background: "var(--surface2)", borderColor: "var(--border)" }}>
        <p className="text-xs mb-2" style={{ color: "var(--text-dim)" }}>{data.central_data_reality}</p>
        <p className="text-xs" style={{ color: "var(--text-faint)" }}>{data.recommended_strategy}</p>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr style={{ borderBottom: "1px solid var(--border)" }}>
              {["Need", "Status", "Sources", "Project action"].map((h) => (
                <th key={h} className="text-left py-2 pr-4 font-medium" style={{ color: "var(--text-faint)" }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {visibleNeeds.map((need) => (
              <tr key={need.need} style={{ borderBottom: "1px solid var(--border)" }} className="last:border-0">
                <td className="py-2 pr-4 font-medium" style={{ color: "var(--text)" }}>{need.need}</td>
                <td className="py-2 pr-4">
                  <Badge variant={
                    need.status === "covered_by_public_data" ? "green" :
                    need.status === "partially_covered" ? "amber" :
                    need.status === "future_extension" ? "blue" :
                    "red"
                  }>
                    {need.status.replace(/_/g, " ")}
                  </Badge>
                </td>
                <td className="py-2 pr-4" style={{ color: "var(--text-dim)" }}>
                  {need.sources.length ? need.sources.map((id) => sourceNames.get(id) || id).join(", ") : "No direct public source"}
                </td>
                <td className="py-2 pr-4" style={{ color: "var(--text-dim)" }}>{need.project_action}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="grid sm:grid-cols-2 gap-2">
        {data.sources.slice(0, 6).map((source) => (
          <div key={source.id} className="rounded-md border p-2" style={{ background: "var(--surface)", borderColor: "var(--border)" }}>
            <div className="flex items-center justify-between gap-2 mb-1">
              <p className="text-xs font-semibold" style={{ color: "var(--text)" }}>{source.name}</p>
              <span className="text-[10px]" style={{ color: "var(--text-faint)" }}>{source.provider}</span>
            </div>
            <p className="text-xs" style={{ color: "var(--text-dim)" }}>{source.use_in_project[0]}</p>
          </div>
        ))}
      </div>

      <p className="text-xs italic" style={{ color: "var(--text-faint)" }}>
        Manifest {data.manifest_hash}. {data.claim_boundary}
      </p>
    </div>
  );
}

function PublicBiomarkerManifestPanel({ data }: { data: PublicBiomarkerDatasetManifest }) {
  return (
    <div className="flex flex-col gap-3">
      <div className="grid sm:grid-cols-3 gap-3">
        <MetricCard label="Candidate sources" value={String(data.dataset_count)} status="green" />
        <MetricCard label="Manifest status" value={data.status.replace(/_/g, " ")} status="amber" />
        <MetricCard label="Fingerprint" value={data.manifest_hash.slice(0, 8)} status="muted" />
      </div>

      <div className="rounded-md border p-3" style={{ background: "var(--surface2)", borderColor: "var(--border)" }}>
        <p className="text-xs" style={{ color: "var(--text-dim)" }}>{data.next_step}</p>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr style={{ borderBottom: "1px solid var(--border)" }}>
              {["Source", "Predictors", "Targets", "Use"].map((h) => (
                <th key={h} className="text-left py-2 pr-4 font-medium" style={{ color: "var(--text-faint)" }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.datasets.map((source) => (
              <tr key={source.id} style={{ borderBottom: "1px solid var(--border)" }} className="last:border-0">
                <td className="py-2 pr-4">
                  <a href={source.url} target="_blank" rel="noreferrer" className="font-semibold" style={{ color: "var(--text)" }}>
                    {source.name}
                  </a>
                  <p style={{ color: "var(--text-faint)" }}>{source.provider}</p>
                </td>
                <td className="py-2 pr-4" style={{ color: "var(--text-dim)" }}>{source.predictor_fields.slice(0, 4).join(", ")}</td>
                <td className="py-2 pr-4" style={{ color: "var(--text-dim)" }}>{source.target_fields.join(", ")}</td>
                <td className="py-2 pr-4" style={{ color: "var(--text-dim)" }}>{source.use_in_project[0]}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className="text-xs italic" style={{ color: "var(--text-faint)" }}>
        {data.claim_boundary}
      </p>
    </div>
  );
}

function PublicBiomarkerMappingPanel({ data }: { data: PublicBiomarkerMappingReadiness }) {
  const breastdcedl = data.datasets.breastdcedl;
  return (
    <div className="flex flex-col gap-3">
      <div className="grid sm:grid-cols-4 gap-3">
        <MetricCard label="Mapping status" value={data.status} status={data.status === "ready" ? "green" : "amber"} />
        <MetricCard label="BreastDCEDL rows" value={breastdcedl?.rows != null ? String(breastdcedl.rows) : "missing"} status={breastdcedl?.mapped_now ? "green" : "amber"} />
        <MetricCard label="Patients" value={breastdcedl?.patients != null ? String(breastdcedl.patients) : "missing"} status="muted" />
        <MetricCard label="Hash" value={data.mapping_hash.slice(0, 8)} status="muted" />
      </div>

      <div className="rounded-md border p-3" style={{ background: "var(--surface2)", borderColor: "var(--border)" }}>
        <p className="text-xs font-semibold mb-1" style={{ color: "var(--text)" }}>Three-stage ablation</p>
        <div className="grid sm:grid-cols-3 gap-2">
          {Object.entries(data.three_stage_ablation_plan).map(([name, description]) => (
            <div key={name} className="rounded-md border p-2" style={{ background: "var(--surface)", borderColor: "var(--border)" }}>
              <p className="text-xs font-semibold" style={{ color: "var(--text)" }}>{name.replace(/_/g, " ")}</p>
              <p className="text-xs" style={{ color: "var(--text-dim)" }}>{description}</p>
            </div>
          ))}
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr style={{ borderBottom: "1px solid var(--border)" }}>
              {["Dataset", "Status", "Mapped", "Next action"].map((h) => (
                <th key={h} className="text-left py-2 pr-4 font-medium" style={{ color: "var(--text-faint)" }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {Object.entries(data.datasets).map(([id, dataset]) => (
              <tr key={id} style={{ borderBottom: "1px solid var(--border)" }} className="last:border-0">
                <td className="py-2 pr-4 font-medium" style={{ color: "var(--text)" }}>{id.replace(/_/g, " ")}</td>
                <td className="py-2 pr-4"><Badge variant={dataset.mapped_now ? "green" : dataset.status.includes("future") ? "blue" : "amber"}>{dataset.status.replace(/_/g, " ")}</Badge></td>
                <td className="py-2 pr-4" style={{ color: "var(--text-dim)" }}>{dataset.mapped_now ? "Yes" : "No"}</td>
                <td className="py-2 pr-4" style={{ color: "var(--text-dim)" }}>{dataset.next_action ?? dataset.role ?? dataset.target_to_map}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className="text-xs italic" style={{ color: "var(--text-faint)" }}>{data.tumor_marker_boundary}</p>
    </div>
  );
}

function FullFeatureGroupAblationPanel({ data }: { data: FullFeatureGroupAblationReport }) {
  const groups = Object.entries(data.feature_groups ?? {});
  const recommendation = data.recommendation;
  return (
    <div className="flex flex-col gap-3">
      <div className="grid sm:grid-cols-4 gap-3">
        <MetricCard
          label="Full vs clinical AUROC"
          value={data.deltas?.full_vs_clinical_auroc_delta != null ? formatDelta(data.deltas.full_vs_clinical_auroc_delta) : null}
          status={(data.deltas?.full_vs_clinical_auroc_delta ?? 0) >= 0 ? "green" : "amber"}
        />
        <MetricCard
          label="Full vs clinical Brier"
          value={data.deltas?.full_vs_clinical_brier_delta != null ? data.deltas.full_vs_clinical_brier_delta.toFixed(4) : null}
          status={(data.deltas?.full_vs_clinical_brier_delta ?? 1) <= 0 ? "green" : "amber"}
        />
        <MetricCard
          label="Full vs clinical ECE"
          value={data.deltas?.full_vs_clinical_ece_delta != null ? data.deltas.full_vs_clinical_ece_delta.toFixed(4) : null}
          status={(data.deltas?.full_vs_clinical_ece_delta ?? 1) <= 0.02 ? "green" : "amber"}
        />
        <MetricCard
          label="Recommended use"
          value={recommendation?.recommended_use?.replace(/_/g, " ") ?? "monitor only"}
          status={recommendation?.promote_feature_set ? "amber" : "muted"}
        />
      </div>

      {recommendation?.reason && (
        <p className="text-xs" style={{ color: "var(--text-dim)" }}>{recommendation.reason}</p>
      )}

      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr style={{ borderBottom: "1px solid var(--border)", color: "var(--text-faint)" }}>
              <th className="text-left py-2 pr-3 font-medium">Feature group</th>
              <th className="text-left py-2 pr-3 font-medium">Modalities</th>
              <th className="text-right py-2 pr-3 font-medium">AUROC</th>
              <th className="text-right py-2 pr-3 font-medium">Brier</th>
              <th className="text-right py-2 pr-3 font-medium">ECE</th>
              <th className="text-right py-2 pr-3 font-medium">FN</th>
              <th className="text-right py-2 pr-3 font-medium">Reg. MAE</th>
            </tr>
          </thead>
          <tbody>
            {groups.map(([name, group]) => (
              <tr key={name} style={{ borderBottom: "1px solid var(--border)" }} className="last:border-0">
                <td className="py-2 pr-3 font-medium" style={{ color: "var(--text)" }}>{name.replace(/_/g, " ")}</td>
                <td className="py-2 pr-3" style={{ color: "var(--text-dim)" }}>{(group.modalities ?? []).join(", ")}</td>
                <td className="py-2 pr-3 tabular-nums text-right" style={{ color: "var(--text-dim)" }}>{group.classification?.patient_level_auroc?.toFixed(3) ?? group.classification?.auroc?.toFixed(3) ?? "—"}</td>
                <td className="py-2 pr-3 tabular-nums text-right" style={{ color: "var(--text-dim)" }}>{group.classification?.brier?.toFixed(3) ?? "—"}</td>
                <td className="py-2 pr-3 tabular-nums text-right" style={{ color: "var(--text-dim)" }}>{group.classification?.ece?.toFixed(3) ?? "—"}</td>
                <td className="py-2 pr-3 tabular-nums text-right" style={{ color: "var(--text-dim)" }}>{group.classification?.false_negative_count ?? "—"}</td>
                <td className="py-2 pr-3 tabular-nums text-right" style={{ color: "var(--text-dim)" }}>{group.regression?.mae?.toFixed(3) ?? "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className="text-xs italic" style={{ color: "var(--text-faint)" }}>{data.claim_boundary}</p>
    </div>
  );
}

function CbioPortalMappingPanel({ data }: { data: CbioportalBiomarkerSchemaMapping }) {
  return (
    <div className="flex flex-col gap-3">
      <div className="grid sm:grid-cols-3 gap-3">
        <MetricCard label="Status" value={data.status.replace(/_/g, " ")} status={data.status === "ready" ? "green" : "amber"} />
        <MetricCard label="Mapped studies" value={String(data.mapped_dataset_count ?? 0)} />
        <MetricCard label="Mapping hash" value={data.mapping_hash?.slice(0, 8) ?? "n/a"} />
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr style={{ borderBottom: "1px solid var(--border)" }}>
              {["Study", "Status", "Core hits", "Mapped groups", "Next action"].map((h) => (
                <th key={h} className="text-left py-2 pr-4 font-medium" style={{ color: "var(--text-faint)" }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {Object.entries(data.datasets).map(([id, dataset]) => {
              const groupNames = Object.keys(dataset.mapped_groups ?? {});
              return (
                <tr key={id} style={{ borderBottom: "1px solid var(--border)" }} className="last:border-0">
                  <td className="py-2 pr-4 font-medium" style={{ color: "var(--text)" }}>{dataset.label}</td>
                  <td className="py-2 pr-4"><Badge variant={statusVariant(dataset.status)}>{dataset.status.replace(/_/g, " ")}</Badge></td>
                  <td className="py-2 pr-4 tabular-nums" style={{ color: "var(--text-dim)" }}>{dataset.core_biomarker_group_hits ?? 0}</td>
                  <td className="py-2 pr-4" style={{ color: "var(--text-dim)" }}>{groupNames.length ? groupNames.join(", ") : "none"}</td>
                  <td className="py-2 pr-4" style={{ color: "var(--text-dim)" }}>{dataset.next_action ?? dataset.reason ?? "Inspect schema before use."}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="text-xs italic" style={{ color: "var(--text-faint)" }}>{data.claim_boundary}</p>
    </div>
  );
}

// LeakageAuditCard extracted to ./cards/LeakageAuditCard.tsx

function EvidenceAbstentionCard({
  report,
  loading,
  running,
  onRefresh,
}: {
  report: EvidenceAbstentionEvalReport | null;
  loading: boolean;
  running: boolean;
  onRefresh: () => void;
}) {
  const status = report?.status ?? "missing";
  const summary = report?.summary ?? {};
  const scenarios = report?.scenarios ?? [];

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <ShieldCheck size={14} style={{ color: "var(--blue, #1e3a8a)" }} aria-hidden="true" />
          <SectionTitle>Evidence-aware abstention eval</SectionTitle>
          <Badge variant={statusVariant(status === "strong" ? "strong" : status === "acceptable" ? "acceptable" : status === "missing" ? "stale" : "needs_attention")}>
            {status.toUpperCase()}
          </Badge>
        </div>
        <Button onClick={onRefresh} disabled={running} icon={<RefreshCw size={13} />}>
          {running ? "Running…" : "Rerun eval"}
        </Button>
      </CardHeader>

      <p className="text-xs mb-3" style={{ color: "var(--text-dim)" }}>
        Sweeps modality-dropout scenarios over the test rows. Coverage is the
        fraction the system chose to score; the rest were routed to clinician
        review with <code>insufficient_evidence</code>. False-abstention rate
        flags rows where we refused but the underlying model would have been
        correct — high values mean the rules are too cautious.
      </p>

      {loading ? (
        <LoadingPane label="Loading abstention eval…" />
      ) : !report || status === "missing" ? (
        <EmptyPane label={report?.message ?? "Abstention eval has not been generated yet."} />
      ) : (
        <>
          <div className="grid sm:grid-cols-3 gap-3 mb-3">
            <MetricCard
              label="Full-data coverage"
              value={formatRate(summary.full_data_coverage_rate)}
              status={(summary.full_data_coverage_rate ?? 0) >= 0.95 ? "green" : "amber"}
            />
            <MetricCard
              label="Full-data accuracy"
              value={formatRate(summary.full_data_covered_accuracy)}
              status={(summary.full_data_covered_accuracy ?? 0) >= 0.80 ? "green" : "amber"}
            />
            <MetricCard
              label="Demographics-only abstention"
              value={formatRate(summary.demographics_only_abstention_rate)}
              status={(summary.demographics_only_abstention_rate ?? 0) >= 0.95 ? "green" : "amber"}
            />
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-xs" style={{ borderCollapse: "separate", borderSpacing: 0 }}>
              <thead>
                <tr style={{ color: "var(--text-faint)" }}>
                  <th className="text-left font-semibold py-1.5 pr-3" style={{ borderBottom: "1px solid var(--border)" }}>Scenario</th>
                  <th className="text-right font-semibold py-1.5 px-2" style={{ borderBottom: "1px solid var(--border)" }}>Rows</th>
                  <th className="text-right font-semibold py-1.5 px-2" style={{ borderBottom: "1px solid var(--border)" }}>Coverage</th>
                  <th className="text-right font-semibold py-1.5 px-2" style={{ borderBottom: "1px solid var(--border)" }}>Abstention</th>
                  <th className="text-right font-semibold py-1.5 px-2" style={{ borderBottom: "1px solid var(--border)" }}>Covered accuracy</th>
                  <th className="text-right font-semibold py-1.5 pl-2" style={{ borderBottom: "1px solid var(--border)" }}>False abstention</th>
                </tr>
              </thead>
              <tbody>
                {scenarios.map((s) => (
                  <tr key={s.scenario}>
                    <td className="py-1.5 pr-3" style={{ borderBottom: "1px solid var(--border-soft)", fontWeight: 600 }}>
                      {s.scenario}
                    </td>
                    <td className="py-1.5 px-2 tabular-nums text-right" style={{ borderBottom: "1px solid var(--border-soft)", color: "var(--text-dim)" }}>{s.rows_evaluated}</td>
                    <td className="py-1.5 px-2 tabular-nums text-right" style={{ borderBottom: "1px solid var(--border-soft)" }}>{formatRate(s.coverage_rate)}</td>
                    <td className="py-1.5 px-2 tabular-nums text-right" style={{ borderBottom: "1px solid var(--border-soft)" }}>{formatRate(s.abstention_rate)}</td>
                    <td className="py-1.5 px-2 tabular-nums text-right" style={{ borderBottom: "1px solid var(--border-soft)" }}>{formatRate(s.covered_accuracy)}</td>
                    <td className="py-1.5 pl-2 tabular-nums text-right" style={{ borderBottom: "1px solid var(--border-soft)" }}>{formatRate(s.false_abstention_rate)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {report.generated_at && (
            <p className="text-[0.7rem] mt-2" style={{ color: "var(--text-faint)" }}>
              Last run: {new Date(report.generated_at).toLocaleString()} · {report.rows_evaluated ?? 0} rows
            </p>
          )}
        </>
      )}
    </Card>
  );
}

function formatRate(value: number | null | undefined): string {
  if (value == null) return "—";
  return `${(value * 100).toFixed(1)}%`;
}

function SyntheticGeneratorCardPanel({
  report,
  loading,
  running,
  onRefresh,
}: {
  report: SyntheticGeneratorCard | null;
  loading: boolean;
  running: boolean;
  onRefresh: () => void;
}) {
  const status = report?.status ?? "missing";
  const cohort = report?.cohort ?? {};
  const dist = report?.feature_distribution_summary ?? {};

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <Info size={14} style={{ color: "var(--blue, #1e3a8a)" }} aria-hidden="true" />
          <SectionTitle>Synthetic generator card</SectionTitle>
          <Badge variant={statusVariant(status === "passed" ? "strong" : status === "missing" ? "stale" : "needs_attention")}>
            {status.toUpperCase()}
          </Badge>
        </div>
        <Button onClick={onRefresh} disabled={running} icon={<RefreshCw size={13} />}>
          {running ? "Rebuilding…" : "Rebuild"}
        </Button>
      </CardHeader>

      <p className="text-xs mb-3" style={{ color: "var(--text-dim)" }}>
        Documents what the synthetic dataset is, the causal assumptions baked
        into the generator, known shortcuts, and what cannot be claimed from
        these numbers. Reviewer-facing provenance.
      </p>

      {loading ? (
        <LoadingPane label="Loading generator card…" />
      ) : !report || status === "missing" ? (
        <EmptyPane label={report?.message ?? "Generator card has not been built yet."} />
      ) : (
        <>
          <div className="grid sm:grid-cols-4 gap-3 mb-3">
            <MetricCard label="Patients" value={cohort.patients_created ?? 0} status="muted" />
            <MetricCard label="Rows" value={dist.row_count ?? 0} status="muted" />
            <MetricCard
              label="Positive label rate"
              value={dist.positive_label_rate != null ? `${(dist.positive_label_rate * 100).toFixed(1)}%` : "—"}
              status="muted"
            />
            <MetricCard
              label="Card ↔ dataset"
              value={report.card_version_matches_dataset ? "in sync" : "drifted"}
              status={report.card_version_matches_dataset ? "green" : "amber"}
            />
          </div>

          {report.causal_assumptions && report.causal_assumptions.length > 0 && (
            <ProvenanceNarrative title="Causal assumptions" items={report.causal_assumptions} tone="info" />
          )}
          {report.known_shortcuts && report.known_shortcuts.length > 0 && (
            <ProvenanceNarrative title="Known shortcuts the model could exploit" items={report.known_shortcuts} tone="amber" />
          )}
          {report.unsupported_claims && report.unsupported_claims.length > 0 && (
            <ProvenanceNarrative title="What this dataset CANNOT support claiming" items={report.unsupported_claims} tone="rose" />
          )}

          <p className="text-[0.7rem] mt-2" style={{ color: "var(--text-faint)" }}>
            Schema: <code>{report.dataset_schema_version ?? "unknown"}</code>{" · "}
            Card: <code>{report.generator_card_version ?? "unknown"}</code>{" · "}
            Rows fingerprint: <code>{cohort.rows_fingerprint ?? "—"}</code>
          </p>
        </>
      )}
    </Card>
  );
}

function ProvenanceNarrative({ title, items, tone }: { title: string; items: string[]; tone: "info" | "amber" | "rose" }) {
  const palette = tone === "amber"
    ? { fg: "#92400e", bg: "rgba(245,158,11,0.06)", border: "rgba(245,158,11,0.25)" }
    : tone === "rose"
    ? { fg: "#b91c1c", bg: "rgba(244,63,94,0.05)", border: "rgba(244,63,94,0.25)" }
    : { fg: "var(--text)", bg: "var(--surface2)", border: "var(--border)" };
  return (
    <div
      className="rounded-md border p-3 mb-2"
      style={{ background: palette.bg, borderColor: palette.border }}
    >
      <p className="text-[0.72rem] uppercase font-semibold mb-1.5" style={{ color: palette.fg }}>
        {title}
      </p>
      <ul className="flex flex-col gap-1 pl-4" style={{ listStyle: "disc", color: "var(--text)" }}>
        {items.map((item, i) => (
          <li key={i} className="text-xs leading-relaxed">{item}</li>
        ))}
      </ul>
    </div>
  );
}

function FailureModeRegistryCard({
  report,
  loading,
  running,
  onRefresh,
}: {
  report: FailureModeRegistry | null;
  loading: boolean;
  running: boolean;
  onRefresh: () => void;
}) {
  const status = report?.status ?? "missing";
  const summary = report?.summary ?? {};
  const entries = report?.entries ?? [];
  const high = summary.by_severity?.high ?? 0;
  const unresolved = summary.entries_with_unresolved_gap ?? 0;

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <ShieldAlert size={14} style={{ color: "var(--amber, #92400e)" }} aria-hidden="true" />
          <SectionTitle>Failure-mode registry</SectionTitle>
          <Badge variant={statusVariant(
            status === "strong" ? "strong" :
            status === "acceptable" ? "acceptable" :
            status === "missing" ? "stale" : "needs_attention",
          )}>
            {status.toUpperCase()}
          </Badge>
        </div>
        <Button onClick={onRefresh} disabled={running} icon={<RefreshCw size={13} />}>
          {running ? "Rebuilding…" : "Rebuild"}
        </Button>
      </CardHeader>

      <p className="text-xs mb-3" style={{ color: "var(--text-dim)" }}>
        Consolidates engineering risks, failure case gallery entries, safety
        red-team failures, and drift findings into one auditable list. Each
        row carries category, severity, detection method, mitigation, and
        remaining gap.
      </p>

      {loading ? (
        <LoadingPane label="Loading failure-mode registry…" />
      ) : !report || status === "missing" ? (
        <EmptyPane label={report?.message ?? "Failure-mode registry has not been built yet."} />
      ) : (
        <>
          <div className="grid sm:grid-cols-3 gap-3 mb-3">
            <MetricCard label="Entries" value={report.entry_count ?? entries.length} status="muted" />
            <MetricCard
              label="High severity"
              value={high}
              status={high > 0 ? "amber" : "green"}
            />
            <MetricCard
              label="With unresolved gap"
              value={unresolved}
              status={unresolved > 0 ? "amber" : "green"}
            />
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-xs" style={{ borderCollapse: "separate", borderSpacing: 0 }}>
              <thead>
                <tr style={{ color: "var(--text-faint)" }}>
                  {["Name", "Category", "Severity", "Detection", "Remaining gap"].map((h) => (
                    <th key={h} className="text-left font-semibold py-1.5 px-2"
                        style={{ borderBottom: "1px solid var(--border)" }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {entries.slice(0, 15).map((e) => {
                  const severityColor = e.severity === "high"
                    ? "var(--rose, #b91c1c)"
                    : e.severity === "medium"
                    ? "var(--amber, #92400e)"
                    : "var(--text-dim)";
                  return (
                    <tr key={e.name}>
                      <td className="py-1.5 px-2 font-semibold" style={{ borderBottom: "1px solid var(--border-soft)" }}>
                        {e.name}
                      </td>
                      <td className="py-1.5 px-2" style={{ borderBottom: "1px solid var(--border-soft)", color: "var(--text-dim)" }}>
                        {e.category}
                      </td>
                      <td className="py-1.5 px-2 font-semibold uppercase" style={{ borderBottom: "1px solid var(--border-soft)", color: severityColor, fontSize: "0.68rem" }}>
                        {e.severity}
                      </td>
                      <td className="py-1.5 px-2" style={{ borderBottom: "1px solid var(--border-soft)", color: "var(--text-dim)" }}>
                        {e.detection}
                      </td>
                      <td className="py-1.5 px-2" style={{ borderBottom: "1px solid var(--border-soft)", color: e.remaining_gap ? "var(--amber, #92400e)" : "var(--text-faint)" }}>
                        {e.remaining_gap ?? "—"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {entries.length > 15 && (
            <p className="text-[0.7rem] mt-2" style={{ color: "var(--text-faint)" }}>
              Showing first 15 of {entries.length} entries.
            </p>
          )}
        </>
      )}
    </Card>
  );
}

// KbSourceGovernanceCard extracted to ./cards/KbSourceGovernanceCard.tsx
// ModalityRobustnessCard extracted to ./cards/ModalityRobustnessCard.tsx

function formatDelta(value: number | null | undefined): string {
  if (value == null) return "—";
  const sign = value > 0 ? "+" : "";
  return `${sign}${(value * 100).toFixed(2)}pp`;
}

function ResponseConformalCalibrationCard({
  report,
  loading,
  running,
  onRefresh,
}: {
  report: ResponseConformalCalibrationReport | null;
  loading: boolean;
  running: boolean;
  onRefresh: () => void;
}) {
  const status = report?.status ?? "missing";
  const adjustedMeetsNominal =
    report?.adjusted_coverage != null &&
    report?.nominal_coverage != null &&
    report.adjusted_coverage >= report.nominal_coverage - 0.01;

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <ShieldCheck size={14} style={{ color: "var(--blue, #1e3a8a)" }} aria-hidden="true" />
          <SectionTitle>Response-score conformal calibration</SectionTitle>
          <Badge variant={statusVariant(
            status === "strong" ? "strong" :
            status === "acceptable" ? "acceptable" :
            status === "missing" ? "stale" : "needs_attention",
          )}>
            {status.toUpperCase()}
          </Badge>
        </div>
        <Button onClick={onRefresh} disabled={running} icon={<RefreshCw size={13} />}>
          {running ? "Running..." : "Rerun calibration"}
        </Button>
      </CardHeader>

      <p className="text-xs mb-3" style={{ color: "var(--text-dim)" }}>
        Split-conformal adjustment for the response-score regression interval.
        The raw quantile band is widened by a held-out residual quantile so the
        interval is calibrated as an engineering reliability signal, not a
        clinical guarantee.
      </p>

      {loading ? (
        <LoadingPane label="Loading conformal calibration..." />
      ) : !report || status === "missing" ? (
        <EmptyPane label={report?.message ?? "Conformal calibration has not been generated yet."} />
      ) : (
        <>
          <div className="grid sm:grid-cols-4 gap-3 mb-3">
            <MetricCard label="Nominal coverage" value={formatRate(report.nominal_coverage)} status="muted" />
            <MetricCard
              label="Raw coverage"
              value={formatRate(report.raw_coverage)}
              status={(report.raw_coverage ?? 0) >= (report.nominal_coverage ?? 1) ? "green" : "amber"}
            />
            <MetricCard
              label="Adjusted coverage"
              value={formatRate(report.adjusted_coverage)}
              status={adjustedMeetsNominal ? "green" : "amber"}
            />
            <MetricCard
              label="qhat widen"
              value={report.qhat_percent != null ? report.qhat_percent.toFixed(3) : null}
              status="muted"
            />
          </div>
          {report.interpretation && (
            <p className="text-xs mb-2" style={{ color: "var(--text-dim)" }}>{report.interpretation}</p>
          )}
          {report.generated_at && (
            <p className="text-[0.7rem]" style={{ color: "var(--text-faint)" }}>
              Last run: {new Date(report.generated_at).toLocaleString()}
              {report.calibration_rows != null && <> Â· calibration rows: {report.calibration_rows}</>}
            </p>
          )}
        </>
      )}
    </Card>
  );
}

function RobustnessStressCard({
  report,
  loading,
  running,
  onRefresh,
}: {
  report: RobustnessStressReport | null;
  loading: boolean;
  running: boolean;
  onRefresh: () => void;
}) {
  const status = report?.status ?? "missing";
  const summary = report?.summary ?? {};
  const cases = report?.cases ?? [];

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <ShieldAlert size={14} style={{ color: "var(--amber, #92400e)" }} aria-hidden="true" />
          <SectionTitle>Synthetic robustness stress suite</SectionTitle>
          <Badge variant={statusVariant(
            status === "strong" ? "strong" :
            status === "acceptable" ? "acceptable" :
            status === "missing" ? "stale" : "needs_attention",
          )}>
            {status.toUpperCase()}
          </Badge>
        </div>
        <Button onClick={onRefresh} disabled={running} icon={<RefreshCw size={13} />}>
          {running ? "Running..." : "Rerun stress"}
        </Button>
      </CardHeader>

      <p className="text-xs mb-3" style={{ color: "var(--text-dim)" }}>
        Fault-injection suite for missing labs, missing imaging, wrong units,
        contradictory symptoms, delayed reports, noisy tumor markers,
        incomplete family history, and ambiguous biomarkers. Passing means the
        system routes to uncertainty, abstention, or clinician review instead
        of overconfident clinical claims.
      </p>

      {loading ? (
        <LoadingPane label="Loading robustness stress report..." />
      ) : !report || status === "missing" ? (
        <EmptyPane label={report?.message ?? "Robustness stress report has not been generated yet."} />
      ) : (
        <>
          <div className="grid sm:grid-cols-3 gap-3 mb-3">
            <MetricCard
              label="Pass rate"
              value={formatRate(summary.pass_rate)}
              status={(summary.pass_rate ?? 0) >= 0.95 ? "green" : "amber"}
            />
            <MetricCard label="Stress cases" value={summary.case_count ?? cases.length} status="muted" />
            <MetricCard
              label="Abstain/review route"
              value={formatRate(summary.abstention_or_review_rate)}
              status={(summary.abstention_or_review_rate ?? 0) >= 0.90 ? "green" : "amber"}
            />
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-xs" style={{ borderCollapse: "separate", borderSpacing: 0 }}>
              <thead>
                <tr style={{ color: "var(--text-faint)" }}>
                  <th className="text-left font-semibold py-1.5 pr-3" style={{ borderBottom: "1px solid var(--border)" }}>Case</th>
                  <th className="text-left font-semibold py-1.5 px-2" style={{ borderBottom: "1px solid var(--border)" }}>Category</th>
                  <th className="text-left font-semibold py-1.5 px-2" style={{ borderBottom: "1px solid var(--border)" }}>Expected</th>
                  <th className="text-right font-semibold py-1.5 px-2" style={{ borderBottom: "1px solid var(--border)" }}>Review</th>
                  <th className="text-right font-semibold py-1.5 pl-2" style={{ borderBottom: "1px solid var(--border)" }}>Result</th>
                </tr>
              </thead>
              <tbody>
                {cases.slice(0, 10).map((c, index) => {
                  const name = c.case ?? c.case_id ?? `case_${index + 1}`;
                  const review = c.clinician_review_routed ?? c.clinician_review ?? c.abstained_any_head ?? c.abstained ?? false;
                  return (
                    <tr key={name}>
                      <td className="py-1.5 pr-3" style={{ borderBottom: "1px solid var(--border-soft)", fontWeight: 600 }}>{name.replace(/_/g, " ")}</td>
                      <td className="py-1.5 px-2" style={{ borderBottom: "1px solid var(--border-soft)", color: "var(--text-dim)" }}>{c.category}</td>
                      <td className="py-1.5 px-2" style={{ borderBottom: "1px solid var(--border-soft)", color: "var(--text-dim)" }}>{(c.expected ?? c.expected_behavior ?? "safe routing").replace(/_/g, " ")}</td>
                      <td className="py-1.5 px-2 text-right" style={{ borderBottom: "1px solid var(--border-soft)" }}>{review ? "yes" : "no"}</td>
                      <td className="py-1.5 pl-2 text-right" style={{ borderBottom: "1px solid var(--border-soft)" }}>
                        <Badge variant={c.passed ? "green" : "red"}>{c.passed ? "passed" : "failed"}</Badge>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          {report.generated_at && (
            <p className="text-[0.7rem] mt-2" style={{ color: "var(--text-faint)" }}>
              Last run: {new Date(report.generated_at).toLocaleString()}
            </p>
          )}
        </>
      )}
    </Card>
  );
}

function PredictionTraceCard({
  response,
  loading,
  onRefresh,
}: {
  response: PredictionTraceResponse | null;
  loading: boolean;
  onRefresh: () => void;
}) {
  const traces = response?.traces ?? [];
  const summary = response?.summary;

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <Info size={14} style={{ color: "var(--blue, #1e3a8a)" }} aria-hidden="true" />
          <SectionTitle>Prediction trace log</SectionTitle>
          <Badge variant={statusVariant((summary?.total ?? 0) > 0 ? "strong" : "stale")}>
            {summary?.total ?? 0} TRACES
          </Badge>
        </div>
        <Button onClick={onRefresh} disabled={loading} icon={<RefreshCw size={13} />}>
          Refresh
        </Button>
      </CardHeader>

      <p className="text-xs mb-3" style={{ color: "var(--text-dim)" }}>
        One row per live evidence-aware prediction.  Each trace records the
        model + feature-set + threshold + calibration that were active, which
        modalities were present, whether the abstention layer refused to
        answer, and what the safety validator decided.
      </p>

      {loading ? (
        <LoadingPane label="Loading prediction traces…" />
      ) : traces.length === 0 ? (
        <EmptyPane label="No prediction traces recorded yet — they are written by `predict_and_trace` when live inference fires." />
      ) : (
        <>
          {summary && (
            <div className="grid sm:grid-cols-3 gap-3 mb-3">
              <MetricCard
                label="Recent traces"
                value={summary.total}
                status="muted"
              />
              <MetricCard
                label="Abstention rate"
                value={formatRate(summary.abstention_rate)}
                status={(summary.abstention_rate ?? 0) > 0.5 ? "amber" : "green"}
              />
              <MetricCard
                label="Model versions seen"
                value={summary.model_versions.length}
                status={summary.model_versions.length > 1 ? "amber" : "muted"}
              />
            </div>
          )}

          <div className="overflow-x-auto">
            <table className="w-full text-xs" style={{ borderCollapse: "separate", borderSpacing: 0 }}>
              <thead>
                <tr style={{ color: "var(--text-faint)" }}>
                  {[
                    "When", "Patient", "Question", "Decision", "Prob.",
                    "Conf.", "Evidence", "Modalities", "Validator",
                  ].map((h) => (
                    <th key={h} className="text-left font-semibold py-1.5 px-2"
                        style={{ borderBottom: "1px solid var(--border)" }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {traces.slice(0, 12).map((t) => (
                  <tr key={t.id}>
                    <td className="py-1.5 px-2" style={{ borderBottom: "1px solid var(--border-soft)", color: "var(--text-dim)" }}>
                      {t.created_at ? new Date(t.created_at).toLocaleString() : "—"}
                    </td>
                    <td className="py-1.5 px-2 font-semibold" style={{ borderBottom: "1px solid var(--border-soft)" }}>
                      {t.patient_id ?? "—"}
                    </td>
                    <td className="py-1.5 px-2" style={{ borderBottom: "1px solid var(--border-soft)" }}>
                      {t.question}
                    </td>
                    <td className="py-1.5 px-2" style={{ borderBottom: "1px solid var(--border-soft)", color: t.abstained ? "var(--amber)" : "var(--text)" }}>
                      {t.decision}
                    </td>
                    <td className="py-1.5 px-2 tabular-nums" style={{ borderBottom: "1px solid var(--border-soft)" }}>
                      {t.probability == null ? "—" : t.probability.toFixed(3)}
                    </td>
                    <td className="py-1.5 px-2" style={{ borderBottom: "1px solid var(--border-soft)" }}>
                      {t.confidence ?? "—"}
                    </td>
                    <td className="py-1.5 px-2" style={{ borderBottom: "1px solid var(--border-soft)" }}>
                      {t.evidence_sufficiency ?? "—"}
                    </td>
                    <td className="py-1.5 px-2" style={{ borderBottom: "1px solid var(--border-soft)", color: "var(--text-dim)" }}>
                      {t.modalities_present.length}/{t.modalities_present.length + t.modalities_missing.length}
                    </td>
                    <td className="py-1.5 px-2" style={{ borderBottom: "1px solid var(--border-soft)" }}>
                      {t.validator_decision ?? "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {traces.length > 12 && (
            <p className="text-[0.7rem] mt-2" style={{ color: "var(--text-faint)" }}>
              Showing first 12 of {traces.length} recent traces.
            </p>
          )}
        </>
      )}
    </Card>
  );
}

function CostCard({ label, level, color, description }: {
  label: string; level: string; color: string; description: string;
}) {
  return (
    <div className="rounded-md border p-3" style={{
      background: `${color}0d`, borderColor: `${color}30`,
    }}>
      <p className="text-xs font-semibold mb-0.5" style={{ color: "var(--text-faint)" }}>{label}</p>
      <p className="text-lg font-bold mb-1" style={{ color }}>{level}</p>
      <p className="text-xs" style={{ color: "var(--text-dim)" }}>{description}</p>
    </div>
  );
}

function CandidateComparisonPanel({ data }: { data: CurrentVsRealismCandidateReport }) {
  const current = data.current ?? {};
  const candidate = data.candidate ?? {};
  const rec = data.recommendation ?? {};
  const decision = rec.decision ?? "not_available";
  const promote = decision === "promote_candidate_after_review";
  return (
    <div className="flex flex-col gap-3">
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <MetricCard
          label="Current AUROC"
          value={current.patient_level_roc_auc != null ? current.patient_level_roc_auc.toFixed(3) : null}
          status="muted"
        />
        <MetricCard
          label="Candidate AUROC"
          value={candidate.patient_level_roc_auc != null ? candidate.patient_level_roc_auc.toFixed(3) : null}
          status={promote ? "green" : "amber"}
        />
        <MetricCard
          label="AUROC delta"
          value={rec.auc_delta != null ? `${rec.auc_delta >= 0 ? "+" : ""}${rec.auc_delta.toFixed(3)}` : null}
          status={rec.auc_delta != null && rec.auc_delta >= -0.03 ? "green" : "amber"}
        />
        <MetricCard
          label="Realism delta"
          value={rec.realism_delta != null ? `+${rec.realism_delta.toFixed(3)}` : null}
          status={rec.realism_delta != null && rec.realism_delta > 0 ? "green" : "amber"}
        />
      </div>
      <div className="grid sm:grid-cols-2 gap-3">
        <div className="rounded-md border p-3" style={{ background: "var(--surface)", borderColor: "var(--border)" }}>
          <p className="text-xs font-semibold mb-2" style={{ color: "var(--text-dim)" }}>Current champion</p>
          <Row label="Realism" value={`${current.realism_status ?? "unknown"} (${current.realism_alignment_score?.toFixed(3) ?? "n/a"})`} />
          <Row label="Sim-to-real" value={current.sim_to_real_status ?? "unknown"} />
          <Row label="Threshold coverage" value={current.threshold_coverage_status ?? "unknown"} />
        </div>
        <div className="rounded-md border p-3" style={{ background: "var(--surface)", borderColor: "var(--border)" }}>
          <p className="text-xs font-semibold mb-2" style={{ color: "var(--text-dim)" }}>Realism-v2 candidate</p>
          <Row label="Realism" value={`${candidate.realism_status ?? "unknown"} (${candidate.realism_alignment_score?.toFixed(3) ?? "n/a"})`} />
          <Row label="Sim-to-real" value={candidate.sim_to_real_status ?? "unknown"} />
          <Row label="Threshold coverage" value={candidate.threshold_coverage_status ?? "unknown"} />
        </div>
      </div>
      <div
        className="rounded-md border p-3 text-xs"
        style={{
          background: promote ? "rgba(16,185,129,0.07)" : "rgba(245,158,11,0.07)",
          borderColor: promote ? "rgba(16,185,129,0.25)" : "rgba(245,158,11,0.25)",
          color: promote ? "var(--green)" : "var(--amber)",
        }}
      >
        <strong>{decision.replace(/_/g, " ")}</strong>
        {rec.rationale ? <span style={{ color: "var(--text-dim)" }}> - {rec.rationale}</span> : null}
      </div>
      {data.claim_boundary && (
        <p className="text-xs italic" style={{ color: "var(--text-faint)" }}>{data.claim_boundary}</p>
      )}
    </div>
  );
}

function NoiseEvalPanel({ data }: { data: NoiseEvalResult }) {
  const base = data.clean_baseline;
  return (
    <div className="flex flex-col gap-3">
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <MetricCard label="Baseline AUROC"    value={base.auroc != null ? base.auroc.toFixed(3) : null}    status="green" />
        <MetricCard label="Baseline Brier"    value={base.brier_score != null ? base.brier_score.toFixed(3) : null} status="green" />
        <MetricCard label="Baseline Sensitivity" value={base.sensitivity != null ? base.sensitivity.toFixed(3) : null} status="green" />
        <MetricCard label="Baseline PR-AUC"   value={base.pr_auc != null ? base.pr_auc.toFixed(3) : null}   status="green" />
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr style={{ borderBottom: "1px solid var(--border)" }}>
              {["Noise mode", "AUROC", "Δ AUROC", "Sensitivity", "Δ Sensitivity", "Status"].map(h => (
                <th key={h} className="text-left py-2 pr-4 font-medium" style={{ color: "var(--text-faint)" }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.noise_results.map((r) => (
              <tr key={r.mode} style={{ borderBottom: "1px solid var(--border)" }} className="last:border-0">
                <td className="py-2 pr-4 font-medium" style={{ color: "var(--text)" }}>{r.mode.replace(/_/g, " ")}</td>
                <td className="py-2 pr-4 tabular-nums" style={{ color: "var(--text-dim)" }}>{r.auroc?.toFixed(3) ?? "—"}</td>
                <td className="py-2 pr-4 tabular-nums" style={{ color: r.auroc_delta != null && r.auroc_delta < -0.05 ? "var(--rose)" : "var(--text-dim)" }}>
                  {r.auroc_delta != null ? (r.auroc_delta >= 0 ? "+" : "") + r.auroc_delta.toFixed(3) : "—"}
                </td>
                <td className="py-2 pr-4 tabular-nums" style={{ color: "var(--text-dim)" }}>{r.sensitivity?.toFixed(3) ?? "—"}</td>
                <td className="py-2 pr-4 tabular-nums" style={{ color: r.sensitivity_delta != null && r.sensitivity_delta < -0.05 ? "var(--rose)" : "var(--text-dim)" }}>
                  {r.sensitivity_delta != null ? (r.sensitivity_delta >= 0 ? "+" : "") + r.sensitivity_delta.toFixed(3) : "—"}
                </td>
                <td className="py-2">
                  <Badge variant={r.status === "robust" ? "green" : r.status === "degraded" ? "amber" : "red"}>{r.status}</Badge>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {data.summary.worst_mode && (
        <p className="text-xs" style={{ color: "var(--text-faint)" }}>
          Worst mode: <strong style={{ color: "var(--text-dim)" }}>{data.summary.worst_mode.replace(/_/g, " ")}</strong>
          {data.summary.max_auroc_drop != null && ` · max AUROC drop ${data.summary.max_auroc_drop.toFixed(3)}`}
        </p>
      )}
      <p className="text-xs italic" style={{ color: "var(--text-faint)" }}>{data.claim_boundary}</p>
    </div>
  );
}

function TemporalEvalPanel({ data }: { data: TemporalEvalResult }) {
  const splits = [
    { label: "Patient timeline split", metrics: data.temporal_split },
    { label: "Cycle accumulation split", metrics: data.cycle_split },
    { label: "Random baseline", metrics: data.random_split_baseline },
  ] as const;
  return (
    <div className="flex flex-col gap-3">
      <div className="grid sm:grid-cols-3 gap-3">
        {splits.map(({ label, metrics }) => (
          <div key={label} className="rounded-md border p-3" style={{ background: "var(--surface)", borderColor: "var(--border)" }}>
            <p className="text-xs font-semibold mb-2" style={{ color: "var(--text-dim)" }}>{label}</p>
            <div className="flex flex-col gap-1">
              <Row label="AUROC"       value={metrics.auroc?.toFixed(3)} />
              <Row label="Brier"       value={metrics.brier_score?.toFixed(3)} />
              <Row label="Sensitivity" value={metrics.sensitivity?.toFixed(3)} />
              <Row label="n train"     value={String(metrics.n_train)} />
              <Row label="n eval"      value={String(metrics.n_eval)} />
            </div>
          </div>
        ))}
      </div>
      {data.generalization_gap && (
        <div className="flex gap-4">
          <p className="text-xs" style={{ color: "var(--text-faint)" }}>
            Temporal gap: <span style={{ color: "var(--text-dim)" }}>{data.generalization_gap.temporal_auroc_gap?.toFixed(3) ?? "—"}</span>
          </p>
          <p className="text-xs" style={{ color: "var(--text-faint)" }}>
            Cycle gap: <span style={{ color: "var(--text-dim)" }}>{data.generalization_gap.cycle_auroc_gap?.toFixed(3) ?? "—"}</span>
          </p>
        </div>
      )}
      {data.interpretation && (
        <p className="text-xs" style={{ color: "var(--text-dim)" }}>{data.interpretation}</p>
      )}
      <p className="text-xs italic" style={{ color: "var(--text-faint)" }}>{data.claim_boundary}</p>
    </div>
  );
}

function Row({ label, value }: { label: string; value: string | undefined }) {
  return (
    <div className="flex justify-between gap-2">
      <span className="text-xs" style={{ color: "var(--text-faint)" }}>{label}</span>
      <span className="text-xs tabular-nums font-medium" style={{ color: "var(--text-dim)" }}>{value ?? "—"}</span>
    </div>
  );
}

const CONFUSION_COLOR: Record<string, string> = {
  TP: "var(--green)", FP: "var(--amber)", TN: "var(--text-dim)", FN: "var(--rose)"
};

function PredictionErrorPanel({ data }: { data: PredictionErrorTable }) {
  const [showAll, setShowAll] = useState(false);
  const rows = showAll ? data.rows : data.rows.slice(0, 20);
  const cm = data.confusion_summary;

  return (
    <div className="flex flex-col gap-3">
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <MetricCard label="Sensitivity" value={data.sensitivity != null ? data.sensitivity.toFixed(3) : null}
          status={data.sensitivity != null && data.sensitivity >= 0.75 ? "green" : "amber"} />
        <MetricCard label="Specificity" value={data.specificity != null ? data.specificity.toFixed(3) : null} />
        <MetricCard label="MAE"         value={data.mae != null ? data.mae.toFixed(4) : null} />
        <MetricCard label="Threshold"   value={String(data.threshold)} />
      </div>
      <div className="flex gap-4">
        {(["TP", "FP", "TN", "FN"] as const).map(k => (
          <div key={k} className="flex flex-col items-center gap-0.5">
            <span className="text-lg font-bold tabular-nums" style={{ color: CONFUSION_COLOR[k] }}>{cm[k]}</span>
            <span className="text-xs" style={{ color: "var(--text-faint)" }}>{k}</span>
          </div>
        ))}
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr style={{ borderBottom: "1px solid var(--border)" }}>
              {["ID", "Actual", "Prob", "Class", "Error", "Type"].map(h => (
                <th key={h} className="text-left py-2 pr-3 font-medium" style={{ color: "var(--text-faint)" }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.patient_id} style={{ borderBottom: "1px solid var(--border)" }} className="last:border-0">
                <td className="py-1.5 pr-3" style={{ color: "var(--text-dim)" }}>{r.patient_id}</td>
                <td className="py-1.5 pr-3 tabular-nums" style={{ color: "var(--text)" }}>{r.actual_label}</td>
                <td className="py-1.5 pr-3 tabular-nums" style={{ color: "var(--text-dim)" }}>{r.predicted_probability.toFixed(3)}</td>
                <td className="py-1.5 pr-3 tabular-nums" style={{ color: "var(--text-dim)" }}>{r.predicted_class}</td>
                <td className="py-1.5 pr-3 tabular-nums" style={{ color: "var(--text-dim)" }}>{r.absolute_error.toFixed(4)}</td>
                <td className="py-1.5">
                  <span className="font-bold" style={{ color: CONFUSION_COLOR[r.confusion_type] }}>{r.confusion_type}</span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {data.rows.length > 20 && (
        <Button variant="ghost" size="sm" onClick={() => setShowAll(v => !v)}>
          {showAll ? `Show fewer` : `Show all ${data.rows.length} rows`}
        </Button>
      )}
      <p className="text-xs italic" style={{ color: "var(--text-faint)" }}>{data.claim_boundary}</p>
    </div>
  );
}
