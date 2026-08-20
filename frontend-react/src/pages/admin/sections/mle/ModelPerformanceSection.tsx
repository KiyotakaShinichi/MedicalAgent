import { MetricCard } from "../../../../components/ui/MetricCard";
import {
  getExternalValidation,
  getLockedHoldout,
  getModelComparison,
  getTrainingReport,
} from "../../../../api/client";
import { DataPanelCard } from "./DataPanelCard";
import { useArtifactPanel } from "./useArtifactPanel";
import {
  HOLDOUT_FIELDS,
  HOLDOUT_METRICS,
  TRAINING_REPORT_FIELDS,
  TRAINING_REPORT_METRICS,
  displayMetricValue,
  gradeMetric,
  toRecord,
  type MetricSpec,
} from "./mleMetrics";

/**
 * Artifacts from these endpoints arrive in a `{ result: … }` envelope whose
 * payload the API types as `unknown` — its shape belongs to a training script,
 * not the OpenAPI schema. `toRecord` is the narrowing boundary.
 */
interface ResultEnvelope {
  result?: unknown;
}

const TAG_SYNTHETIC = { label: "Synthetic data", background: "rgba(245,158,11,0.12)", color: "var(--amber)" };
const TAG_FROZEN = { label: "Frozen synthetic split", background: "rgba(59,130,246,0.12)", color: "#93c5fd" };
const TAG_EXTERNAL = { label: "BreastDCEDL / I-SPY1", background: "rgba(139,92,246,0.12)", color: "#c4b5fd" };

function MetricGrid({
  specs,
  fields,
  result,
}: {
  specs: readonly MetricSpec[];
  fields: Record<string, string>;
  result: Record<string, unknown>;
}) {
  return (
    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
      {specs.map((spec) => {
        const raw = result[fields[spec.label]];
        return (
          <MetricCard
            key={spec.label}
            label={spec.label}
            value={displayMetricValue(raw)}
            status={gradeMetric(raw, spec)}
            sub={spec.sub}
          />
        );
      })}
    </div>
  );
}

/** Synthetic training run: discrimination, calibration, and regression fit. */
export function TrainingReportPanel() {
  const { report, loading, error } = useArtifactPanel<ResultEnvelope>(
    getTrainingReport, undefined, "admin.mle.trainingReport",
  );
  const result = toRecord(report?.result);

  return (
    <DataPanelCard
      title="Synthetic Training Report"
      tag={TAG_SYNTHETIC}
      loading={loading}
      error={error}
      empty={!result}
      emptyLabel="No training report — run training first"
      errorLabel="Could not load training report"
    >
      {result && (
        <>
          <div className="mb-4">
            <MetricGrid specs={TRAINING_REPORT_METRICS} fields={TRAINING_REPORT_FIELDS} result={result} />
          </div>
          <p className="text-xs" style={{ color: "var(--text-faint)" }}>
            Interpretation bands are engineering heuristics for synthetic data. AUROC ≥ 0.85 = strong on
            synthetic; Brier &lt; 0.10 = well-calibrated; MAE &lt; 0.10 = good regression fit.
          </p>
        </>
      )}
    </DataPanelCard>
  );
}

/** Frozen synthetic split held out from model selection. */
export function LockedHoldoutPanel() {
  const { report, loading, error } = useArtifactPanel<ResultEnvelope>(
    getLockedHoldout, undefined, "admin.mle.lockedHoldout",
  );
  const result = toRecord(report?.result);

  return (
    <DataPanelCard
      title="Locked Holdout Evaluation"
      tag={TAG_FROZEN}
      loading={loading}
      error={error}
      empty={!result}
      emptyLabel="No holdout evaluation — run holdout first"
      errorLabel="Could not load holdout evaluation"
    >
      {result && <MetricGrid specs={HOLDOUT_METRICS} fields={HOLDOUT_FIELDS} result={result} />}
    </DataPanelCard>
  );
}

/**
 * Directional validation against real tabular MRI-derived features.
 *
 * The disclaimer is load-bearing: these are real datasets, which makes it the
 * one panel a reader might mistake for a clinical performance claim.
 */
export function ExternalValidationPanel() {
  const { report, loading, error } = useArtifactPanel<ResultEnvelope>(
    getExternalValidation, undefined, "admin.mle.externalValidation",
  );

  return (
    <DataPanelCard
      title="External Validation Direction"
      tag={TAG_EXTERNAL}
      loading={loading}
      error={error}
      empty={!report?.result}
      emptyLabel="No external validation data — run external validation first"
      errorLabel="Could not load external validation"
    >
      <div className="flex flex-col gap-2">
        <p className="text-xs" style={{ color: "var(--text-dim)" }}>
          External validation uses BreastDCEDL and I-SPY1 tabular MRI-derived features. These are real
          datasets (non-synthetic) used for directional validation only — not a clinical performance claim.
        </p>
        <p
          className="text-xs px-2 py-1.5 rounded-md border"
          style={{ background: "rgba(139,92,246,0.07)", borderColor: "rgba(139,92,246,0.25)", color: "#c4b5fd" }}
        >
          External validation report loaded. See <code>Data/external_validation/</code> for per-dataset metrics.
        </p>
      </div>
    </DataPanelCard>
  );
}

/** Champion-vs-alternatives deltas. */
export function ModelComparisonPanel() {
  const { report, loading, error } = useArtifactPanel<ResultEnvelope>(
    getModelComparison, undefined, "admin.mle.modelComparison",
  );

  return (
    <DataPanelCard
      title="Model Comparison"
      loading={loading}
      error={error}
      empty={!report?.result}
      emptyLabel="No model comparison available"
      errorLabel="Could not load model comparison"
    >
      <p className="text-xs" style={{ color: "var(--text-dim)" }}>
        Model comparison loaded. Δ AUROC, Δ Brier, Δ ECE, and Δ FNR deltas are available in{" "}
        <code>Data/model_comparison/</code>.
      </p>
    </DataPanelCard>
  );
}
