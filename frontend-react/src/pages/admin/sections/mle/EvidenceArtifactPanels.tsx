/**
 * Self-contained containers for the nine evidence artifacts that already have
 * dedicated card components under `../cards/`.
 *
 * Each export is the complete wiring for one artifact: its GET, its
 * regeneration job, its telemetry surface, and its card. Nothing here reaches
 * into another panel's state, so a panel can be moved, removed, or tested in
 * isolation without touching `MleSection`.
 */
import { LeakageAuditCard } from "../cards/LeakageAuditCard";
import { EvidenceAbstentionCard } from "../cards/EvidenceAbstentionCard";
import { FailureModeRegistryCard } from "../cards/FailureModeRegistryCard";
import { KbSourceGovernanceCard } from "../cards/KbSourceGovernanceCard";
import { ModalityRobustnessCard } from "../cards/ModalityRobustnessCard";
import { PredictionTraceCard } from "../cards/PredictionTraceCard";
import { ResponseConformalCalibrationCard } from "../cards/ResponseConformalCalibrationCard";
import { RobustnessStressCard } from "../cards/RobustnessStressCard";
import { SyntheticGeneratorCardPanel } from "../cards/SyntheticGeneratorCardPanel";
import {
  getEvidenceAbstentionEval,
  runEvidenceAbstentionEval,
  getFailureModeRegistry,
  runFailureModeRegistry,
  getKbSourceGovernance,
  runKbSourceGovernance,
  getLeakageAudit,
  runLeakageAudit,
  getModalityRobustnessComparison,
  runModalityRobustnessComparison,
  getPredictionTraces,
  getResponseConformalCalibration,
  runResponseConformalCalibration,
  getRobustnessStress,
  runRobustnessStress,
  getSyntheticGeneratorCard,
  runSyntheticGeneratorCard,
} from "../../../../api/client";
import type {
  EvidenceAbstentionEvalReport,
  FailureModeRegistry,
  KbSourceGovernanceReport,
  LeakageAuditReport,
  ModalityRobustnessComparisonReport,
  PredictionTraceResponse,
  ResponseConformalCalibrationReport,
  RobustnessStressReport,
  SyntheticGeneratorCard,
} from "../../../../types/api";
import { useArtifactPanel } from "./useArtifactPanel";
import { PanelErrorNotice } from "./PanelErrorNotice";

/** Provenance and documented generator assumptions. */
export function SyntheticGeneratorPanel() {
  const { report, loading, running, error, onRefresh } = useArtifactPanel<SyntheticGeneratorCard>(
    getSyntheticGeneratorCard, runSyntheticGeneratorCard, "admin.mle.generatorCard",
  );
  return (
    <>
      <PanelErrorNotice panel="Synthetic generator card" error={error} />
      <SyntheticGeneratorCardPanel report={report} loading={loading} running={running} onRefresh={onRefresh} />
    </>
  );
}

/** Consolidated registry of known failure modes. */
export function FailureModeRegistryPanel() {
  const { report, loading, running, error, onRefresh } = useArtifactPanel<FailureModeRegistry>(
    getFailureModeRegistry, runFailureModeRegistry, "admin.mle.failureRegistry",
  );
  return (
    <>
      <PanelErrorNotice panel="Failure mode registry" error={error} />
      <FailureModeRegistryCard report={report} loading={loading} running={running} onRefresh={onRefresh} />
    </>
  );
}

/** Per-source retrieval tier, allowed use, and staleness. */
export function KbSourceGovernancePanel() {
  const { report, loading, running, error, onRefresh } = useArtifactPanel<KbSourceGovernanceReport>(
    getKbSourceGovernance, runKbSourceGovernance, "admin.mle.kbGovernance",
  );
  return (
    <>
      <PanelErrorNotice panel="Knowledge-base source governance" error={error} />
      <KbSourceGovernanceCard report={report} loading={loading} running={running} onRefresh={onRefresh} />
    </>
  );
}

/** Training-data leakage audit — the data-hygiene gate. */
export function LeakageAuditPanel() {
  const { report, loading, running, error, onRefresh } = useArtifactPanel<LeakageAuditReport>(
    getLeakageAudit, runLeakageAudit, "admin.mle.leakageAudit",
  );
  return (
    <>
      <PanelErrorNotice panel="Leakage audit" error={error} />
      <LeakageAuditCard report={report} loading={loading} running={running} onRefresh={onRefresh} />
    </>
  );
}

/** Modality-dropout sweep: coverage, accuracy, abstention rate. */
export function EvidenceAbstentionPanel() {
  const { report, loading, running, error, onRefresh } = useArtifactPanel<EvidenceAbstentionEvalReport>(
    getEvidenceAbstentionEval, runEvidenceAbstentionEval, "admin.mle.abstentionEval",
  );
  return (
    <>
      <PanelErrorNotice panel="Evidence-aware abstention eval" error={error} />
      <EvidenceAbstentionCard report={report} loading={loading} running={running} onRefresh={onRefresh} />
    </>
  );
}

/** Champion vs modality-robust model, head to head. */
export function ModalityRobustnessPanel() {
  const { report, loading, running, error, onRefresh } = useArtifactPanel<ModalityRobustnessComparisonReport>(
    getModalityRobustnessComparison, runModalityRobustnessComparison, "admin.mle.modalityComparison",
  );
  return (
    <>
      <PanelErrorNotice panel="Modality robustness comparison" error={error} />
      <ModalityRobustnessCard report={report} loading={loading} running={running} onRefresh={onRefresh} />
    </>
  );
}

/** Conformal calibration of response predictions. */
export function ConformalCalibrationPanel() {
  const { report, loading, running, error, onRefresh } = useArtifactPanel<ResponseConformalCalibrationReport>(
    getResponseConformalCalibration, runResponseConformalCalibration, "admin.mle.conformalCalibration",
  );
  return (
    <>
      <PanelErrorNotice panel="Response conformal calibration" error={error} />
      <ResponseConformalCalibrationCard report={report} loading={loading} running={running} onRefresh={onRefresh} />
    </>
  );
}

/** Perturbation stress sweep. */
export function RobustnessStressPanel() {
  const { report, loading, running, error, onRefresh } = useArtifactPanel<RobustnessStressReport>(
    getRobustnessStress, runRobustnessStress, "admin.mle.robustnessStress",
  );
  return (
    <>
      <PanelErrorNotice panel="Robustness stress" error={error} />
      <RobustnessStressCard report={report} loading={loading} running={running} onRefresh={onRefresh} />
    </>
  );
}

/**
 * Per-call prediction traceability. Read-only: traces are emitted by live
 * inference, so there is no regeneration job — refresh simply reloads.
 */
export function PredictionTracePanel() {
  const { report, loading, error, refetch } = useArtifactPanel<PredictionTraceResponse>(
    () => getPredictionTraces({ limit: 25 }), undefined, "admin.mle.predictionTraces",
  );
  return (
    <>
      <PanelErrorNotice panel="Prediction traces" error={error} />
      <PredictionTraceCard response={report} loading={loading} onRefresh={refetch} />
    </>
  );
}
