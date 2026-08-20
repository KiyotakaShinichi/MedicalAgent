import { useCallback, useState } from "react";
import { RefreshCw, AlertTriangle, Info } from "lucide-react";
import { Badge } from "../../../components/ui/Badge";
import { statusVariant } from "../../../components/ui/badgeUtils";
import { Card, CardHeader, SectionTitle } from "../../../components/ui/Card";
import { Button } from "../../../components/ui/Button";
import { MetricCard } from "../../../components/ui/MetricCard";
import { MetricGlossary, ALL_METRIC_SPECS } from "../../../components/ui/MetricInterpretation";
import { LoadingPane, ErrorPane, EmptyPane } from "../../../components/ui/Spinner";
import { useApi } from "../../../hooks/useApi";
import { useArtifactRunner } from "../../../hooks/useArtifactRunner";
import { LeakageAuditCard } from "./cards/LeakageAuditCard";
import { EvidenceAbstentionCard } from "./cards/EvidenceAbstentionCard";
import { FailureModeRegistryCard } from "./cards/FailureModeRegistryCard";
import { KbSourceGovernanceCard } from "./cards/KbSourceGovernanceCard";
import { ModalityRobustnessCard } from "./cards/ModalityRobustnessCard";
import { PredictionTraceCard } from "./cards/PredictionTraceCard";
import { ResponseConformalCalibrationCard } from "./cards/ResponseConformalCalibrationCard";
import { RobustnessStressCard } from "./cards/RobustnessStressCard";
import { SyntheticGeneratorCardPanel } from "./cards/SyntheticGeneratorCardPanel";
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
  FullFeatureGroupAblationReport,
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

import {
  CandidateComparisonPanel,
  CbioPortalMappingPanel,
  CostCard,
  FullFeatureGroupAblationPanel,
  NoiseEvalPanel,
  PredictionErrorPanel,
  PublicBiomarkerManifestPanel,
  PublicBiomarkerMappingPanel,
  PublicDataManifestPanel,
  TemporalEvalPanel,
} from "./MleEvidencePanels";
interface Props { analytics: AdminAnalytics; onRefresh: () => void }

export function MleSection({ analytics, onRefresh }: Props) {
  const mle = analytics.mle_readiness;
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
  const { data: conformalCalibration, status: conformalStatus, refetch: refetchConformalCalibration } = useApi(
    getResponseConformalCalibration, [],
  );
  const { data: robustnessStress, status: robustnessStressStatus, refetch: refetchRobustnessStress } = useApi(
    getRobustnessStress, [],
  );

  const { data: generatorCard, status: generatorCardStatus, refetch: refetchGeneratorCard } = useApi(getSyntheticGeneratorCard, []);
  const { data: failureRegistry, status: failureRegistryStatus, refetch: refetchFailureRegistry } = useApi(getFailureModeRegistry, []);
  const { data: kbGovernance, status: kbGovernanceStatus, refetch: refetchKbGovernance } = useApi(getKbSourceGovernance, []);

  // Each of these was previously a `useState` flag plus a hand-written
  // `try/finally` with no `catch`, so a failed regeneration reset the spinner
  // and then escaped as an unhandled promise rejection with nothing shown to
  // the operator. `useArtifactRunner` keeps the spinner behaviour, captures
  // the failure, and reports it — see `runnerErrors` below for the surface.
  const { running: runningMle, run: runMle, error: mleError } =
    useArtifactRunner(runMleReadiness, onRefresh, "admin.mle.readiness");
  const { running: runningCandidate, run: runCandidateComparison, error: candidateError } =
    useArtifactRunner(runCurrentVsRealismCandidate, refetchCandidate, "admin.mle.candidateComparison");
  const { running: runningBiomarkerManifest, run: refreshBiomarkerManifest, error: biomarkerManifestError } =
    useArtifactRunner(runPublicBiomarkerDatasetManifest, refetchBiomarkerManifest, "admin.mle.biomarkerManifest");
  const { running: runningBiomarkerMapping, run: refreshBiomarkerMapping, error: biomarkerMappingError } =
    useArtifactRunner(runPublicBiomarkerMappingReadiness, refetchBiomarkerMapping, "admin.mle.biomarkerMapping");
  const { running: runningLeakageAudit, run: refreshLeakageAudit, error: leakageAuditError } =
    useArtifactRunner(runLeakageAudit, refetchLeakageAudit, "admin.mle.leakageAudit");
  const { running: runningAbstentionEval, run: refreshAbstentionEval, error: abstentionEvalError } =
    useArtifactRunner(runEvidenceAbstentionEval, refetchAbstentionEval, "admin.mle.abstentionEval");
  const { running: runningModalityComparison, run: refreshModalityComparison, error: modalityComparisonError } =
    useArtifactRunner(runModalityRobustnessComparison, refetchModalityComparison, "admin.mle.modalityComparison");
  const { running: runningConformalCalibration, run: refreshConformalCalibration, error: conformalCalibrationError } =
    useArtifactRunner(runResponseConformalCalibration, refetchConformalCalibration, "admin.mle.conformalCalibration");
  const { running: runningRobustnessStress, run: refreshRobustnessStress, error: robustnessStressError } =
    useArtifactRunner(runRobustnessStress, refetchRobustnessStress, "admin.mle.robustnessStress");
  const { running: runningGeneratorCard, run: refreshGeneratorCard, error: generatorCardError } =
    useArtifactRunner(runSyntheticGeneratorCard, refetchGeneratorCard, "admin.mle.generatorCard");
  const { running: runningFailureRegistry, run: refreshFailureRegistry, error: failureRegistryError } =
    useArtifactRunner(runFailureModeRegistry, refetchFailureRegistry, "admin.mle.failureRegistry");
  const { running: runningKbGovernance, run: refreshKbGovernance, error: kbGovernanceError } =
    useArtifactRunner(runKbSourceGovernance, refetchKbGovernance, "admin.mle.kbGovernance");
  const { running: runningCbioMapping, run: refreshCbioMapping, error: cbioMappingError } =
    useArtifactRunner(
      useCallback(() => runCbioportalBiomarkerSchemaMapping(true), []),
      refetchCbioMapping,
      "admin.mle.cbioMapping",
    );
  const { running: runningFullFeatureAblation, run: refreshFullFeatureAblation, error: fullFeatureAblationError } =
    useArtifactRunner(runFullFeatureGroupAblation, refetchFullFeatureAblation, "admin.mle.fullFeatureAblation");

  /**
   * First failure among the artifact runners. Only one banner is shown because
   * these are operator-triggered one at a time; listing every stale error would
   * be noise.
   */
  const runnerError = [
    mleError,
    candidateError,
    biomarkerManifestError,
    biomarkerMappingError,
    leakageAuditError,
    abstentionEvalError,
    modalityComparisonError,
    conformalCalibrationError,
    robustnessStressError,
    generatorCardError,
    failureRegistryError,
    kbGovernanceError,
    cbioMappingError,
    fullFeatureAblationError,
  ].find(Boolean) ?? null;

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
      {runnerError && (
        <div
          role="alert"
          className="flex items-start gap-2 px-3 py-2.5 rounded-lg border text-xs"
          style={{ background: "rgba(244,63,94,0.06)", borderColor: "rgba(244,63,94,0.28)", color: "var(--text)" }}
        >
          <AlertTriangle size={13} aria-hidden="true" style={{ flexShrink: 0, marginTop: 1 }} />
          <span>
            <strong>Artifact run failed.</strong> {runnerError} The panel below still shows the
            previous artifact.
          </span>
        </div>
      )}

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

      {/* Synthetic generator card â€” provenance + documented assumptions. */}
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

      {/* RAG source governance â€” per-source tier + allowed_use + staleness. */}
      <KbSourceGovernanceCard
        report={kbGovernance as KbSourceGovernanceReport | null}
        loading={kbGovernanceStatus === "loading"}
        running={runningKbGovernance}
        onRefresh={refreshKbGovernance}
      />

      {/* Training-data leakage audit â€” the engineering data-hygiene gate. */}
      <LeakageAuditCard
        report={leakageAudit as LeakageAuditReport | null}
        loading={leakageStatus === "loading"}
        running={runningLeakageAudit}
        onRefresh={refreshLeakageAudit}
      />

      {/* Evidence-aware abstention eval â€” modality-dropout sweep showing
          coverage / accuracy / abstention rate per missing-modality scenario. */}
      <EvidenceAbstentionCard
        report={abstentionEval as EvidenceAbstentionEvalReport | null}
        loading={abstentionStatus === "loading"}
        running={runningAbstentionEval}
        onRefresh={refreshAbstentionEval}
      />

      {/* Champion vs modality-robust comparison â€” head-to-head over all
          modality-dropout scenarios.  Shows whether retraining with random
          modality masking actually moved the classifier or just the
          abstention rules. */}
      <ModalityRobustnessCard
        report={modalityComparison as ModalityRobustnessComparisonReport | null}
        loading={modalityComparisonStatus === "loading"}
        running={runningModalityComparison}
        onRefresh={refreshModalityComparison}
      />

      {/* Prediction traceability â€” one row per live evidence-aware call,
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
            level="â‰¤ 0.40"
            color="var(--blue)"
            description="Decision threshold set below 0.50 to bias toward sensitivity. Reviewed per-model at training time."
          />
        </div>
        <p className="text-xs" style={{ color: "var(--text-dim)" }}>
          This system uses a cost-sensitive approach: the classification threshold is chosen to minimise FNR at acceptable FPR,
          reflecting the assumption that missing a treatment non-response is more harmful than over-flagging for clinician review.
          Weighted cost = FN_weight Ã— FN + FP_weight Ã— FP where FN_weight = 3, FP_weight = 1 (engineering heuristic, not clinical guidance).
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
         !tr ? <EmptyPane label="No training report â€” run training first" /> : (
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
              Metrics labelled with interpretation bands below. AUROC â‰¥ 0.85 = strong on synthetic; Brier &lt; 0.10 = well-calibrated; MAE &lt; 0.10 = good regression fit.
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
         !ho ? <EmptyPane label="No holdout evaluation â€” run holdout first" /> : (
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
           ? <EmptyPane label="No external validation data â€” run external validation first" />
           : (
            <div className="flex flex-col gap-2">
              <p className="text-xs" style={{ color: "var(--text-dim)" }}>
                External validation uses BreastDCEDL and I-SPY1 tabular MRI-derived features.
                These are real datasets (non-synthetic) used for directional validation only â€”
                not a clinical performance claim.
              </p>
              <p className="text-xs px-2 py-1.5 rounded-md border" style={{
                background: "rgba(139,92,246,0.07)", borderColor: "rgba(139,92,246,0.25)", color: "#c4b5fd"
              }}>
                âœ“ External validation report loaded. See <code>Data/external_validation/</code> for per-dataset metrics.
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
              Model comparison loaded. Î” AUROC, Î” Brier, Î” ECE, Î” FNR deltas available in <code>Data/model_comparison/</code>.
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
          <EmptyPane label="No noise eval â€” run POST /admin/noise-eval first" />
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
          <EmptyPane label="No temporal eval â€” run POST /admin/temporal-eval first" />
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
          <EmptyPane label="No predictions â€” run training pipeline first" />
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
              Bands reflect engineering heuristics for a cancer monitoring PoC â€” not clinical validation thresholds.
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
