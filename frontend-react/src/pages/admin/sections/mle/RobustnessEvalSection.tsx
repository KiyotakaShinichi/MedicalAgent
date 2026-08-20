import {
  getNoiseEval,
  getPredictionErrorTable,
  getTemporalEval,
} from "../../../../api/client";
import type {
  NoiseEvalResult,
  PredictionErrorTable,
  TemporalEvalResult,
} from "../../../../types/api";
import { NoiseEvalPanel, PredictionErrorPanel, TemporalEvalPanel } from "../MleEvidencePanels";
import { DataPanelCard } from "./DataPanelCard";
import { useArtifactPanel } from "./useArtifactPanel";

const TAG_PERTURBATIONS = { label: "Synthetic perturbations", background: "rgba(245,158,11,0.12)", color: "var(--amber)" };
const TAG_TIMELINE = { label: "Synthetic timeline splits", background: "rgba(59,130,246,0.12)", color: "#93c5fd" };
const TAG_HOLDOUT = { label: "Synthetic holdout", background: "rgba(245,158,11,0.12)", color: "var(--amber)" };

/** Performance under injected measurement noise. */
export function NoiseRobustnessPanel() {
  const { report, loading, error } = useArtifactPanel<NoiseEvalResult>(
    getNoiseEval, undefined, "admin.mle.noiseEval",
  );

  return (
    <DataPanelCard
      title="Noise Robustness Evaluation"
      tag={TAG_PERTURBATIONS}
      loading={loading}
      error={error}
      // "unavailable" is the artifact telling us it was not produced.
      empty={!report || report.status === "unavailable"}
      emptyLabel="No noise eval — run POST /admin/noise-eval first"
      errorLabel="Could not load noise eval"
    >
      {report && <NoiseEvalPanel data={report} />}
    </DataPanelCard>
  );
}

/** Performance when train and test are split along the timeline. */
export function TemporalGeneralizationPanel() {
  const { report, loading, error } = useArtifactPanel<TemporalEvalResult>(
    getTemporalEval, undefined, "admin.mle.temporalEval",
  );

  return (
    <DataPanelCard
      title="Temporal Generalization"
      tag={TAG_TIMELINE}
      loading={loading}
      error={error}
      empty={!report || report.status === "unavailable"}
      emptyLabel="No temporal eval — run POST /admin/temporal-eval first"
      errorLabel="Could not load temporal eval"
    >
      {report && <TemporalEvalPanel data={report} />}
    </DataPanelCard>
  );
}

/** Row-level residuals for the synthetic holdout. */
export function PredictionErrorTablePanel() {
  const { report, loading, error } = useArtifactPanel<PredictionErrorTable>(
    getPredictionErrorTable, undefined, "admin.mle.predictionErrorTable",
  );

  return (
    <DataPanelCard
      title="Per-Prediction Error Table"
      tag={TAG_HOLDOUT}
      loading={loading}
      error={error}
      empty={!report?.rows?.length}
      emptyLabel="No predictions — run training pipeline first"
      errorLabel="Could not load prediction error table"
    >
      {report && <PredictionErrorPanel data={report} />}
    </DataPanelCard>
  );
}
