import { AlertTriangle } from "lucide-react";
import { useApi } from "../../../hooks/useApi";
import { getTrainingReport } from "../../../api/client";
import type { AdminAnalytics } from "../../../types/api";
import { buildGlossaryValues, toRecord } from "./mle/mleMetrics";
import { CandidateComparisonCard } from "./mle/CandidateComparisonCard";
import { CostSensitiveEvaluationCard } from "./mle/CostSensitiveEvaluationCard";
import { MetricGlossaryCard } from "./mle/MetricGlossaryCard";
import { MleReadinessGatesCard } from "./mle/MleReadinessGatesCard";
import {
  ConformalCalibrationPanel,
  EvidenceAbstentionPanel,
  FailureModeRegistryPanel,
  KbSourceGovernancePanel,
  LeakageAuditPanel,
  ModalityRobustnessPanel,
  PredictionTracePanel,
  RobustnessStressPanel,
  SyntheticGeneratorPanel,
} from "./mle/EvidenceArtifactPanels";
import {
  ExternalValidationPanel,
  LockedHoldoutPanel,
  ModelComparisonPanel,
  TrainingReportPanel,
} from "./mle/ModelPerformanceSection";
import {
  CbioPortalSchemaMappingPanel,
  FullFeatureAblationPanel,
  PublicBiomarkerMappingReadinessPanel,
  PublicBiomarkerSourcesPanel,
  PublicDataFeasibilityPanel,
} from "./mle/PublicDataSection";
import {
  NoiseRobustnessPanel,
  PredictionErrorTablePanel,
  TemporalGeneralizationPanel,
} from "./mle/RobustnessEvalSection";

interface Props {
  analytics: AdminAnalytics;
  onRefresh: () => void;
}

/**
 * The synthetic-data caveat that frames every number on this page.
 *
 * Deliberately the first thing rendered. Synthetic AUROC is expected to look
 * excellent, and a reader who scrolls into the metric panels without this
 * context will over-read them.
 */
function SyntheticDataDisclaimer() {
  return (
    <div
      className="flex items-start gap-2 px-3 py-2.5 rounded-lg border text-xs"
      style={{ background: "rgba(245,158,11,0.07)", borderColor: "rgba(245,158,11,0.25)", color: "var(--amber)" }}
    >
      <AlertTriangle size={13} aria-hidden="true" style={{ flexShrink: 0, marginTop: 1 }} />
      <span>
        All metrics below are computed on <strong>synthetic data</strong> unless explicitly labelled
        "locked holdout" or "external validation". Synthetic AUROC is expected to be high and does not
        reflect clinical validity. The locked holdout uses a frozen synthetic split; external validation
        uses BreastDCEDL/I-SPY1 tabular features.
      </span>
    </div>
  );
}

/**
 * Admin MLE evidence dashboard.
 *
 * Composition only. Every panel below owns its own fetch, regeneration job,
 * loading/error/empty handling, and telemetry surface via `useArtifactPanel`,
 * so this file states the *order and grouping* of the evidence and nothing
 * else. The previous version held 24 `useApi` calls and 14 runner hooks in one
 * body, which made it impossible to tell which state drove which panel.
 *
 * The single exception is the training report, which is fetched here as well
 * because the metric glossary annotates its bands with the current run's
 * numbers. `useApi` de-duplicates the concurrent GET with the one inside
 * `TrainingReportPanel`, so this costs no extra request.
 */
export function MleSection({ analytics, onRefresh }: Props) {
  const { data: trainingReport } = useApi<{ result?: unknown }>(getTrainingReport, []);
  const glossaryValues = buildGlossaryValues(toRecord(trainingReport?.result));

  return (
    <div className="flex flex-col gap-4">
      <SyntheticDataDisclaimer />

      {/* Provenance and governance — what the data is and where it came from. */}
      <SyntheticGeneratorPanel />
      <FailureModeRegistryPanel />
      <KbSourceGovernancePanel />
      <LeakageAuditPanel />

      {/* Evidence-aware behaviour under missing or degraded inputs. */}
      <EvidenceAbstentionPanel />
      <ModalityRobustnessPanel />
      <ConformalCalibrationPanel />
      <RobustnessStressPanel />
      <PredictionTracePanel />

      {/* Operating point, release gates, and candidate comparison. */}
      <CostSensitiveEvaluationCard />
      <MleReadinessGatesCard mle={analytics.mle_readiness} onRefresh={onRefresh} />
      <CandidateComparisonCard />

      {/* Model performance. */}
      <TrainingReportPanel />
      <LockedHoldoutPanel />
      <ExternalValidationPanel />
      <ModelComparisonPanel />

      {/* Public-data feasibility and schema mapping. */}
      <PublicDataFeasibilityPanel />
      <PublicBiomarkerSourcesPanel />
      <PublicBiomarkerMappingReadinessPanel />
      <CbioPortalSchemaMappingPanel />
      <FullFeatureAblationPanel />

      {/* Robustness sweeps. */}
      <NoiseRobustnessPanel />
      <TemporalGeneralizationPanel />
      <PredictionErrorTablePanel />

      <MetricGlossaryCard values={glossaryValues} />
    </div>
  );
}
