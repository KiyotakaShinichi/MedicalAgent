import { useState } from "react";
import { Badge } from "../../../components/ui/Badge";
import { statusVariant } from "../../../components/ui/badgeUtils";
import { Button } from "../../../components/ui/Button";
import { MetricCard } from "../../../components/ui/MetricCard";
import type {
  NoiseEvalResult,
  TemporalEvalResult,
  PredictionErrorTable,
  PublicDataManifest,
  PublicBiomarkerDatasetManifest,
  PublicBiomarkerMappingReadiness,
  CbioportalBiomarkerSchemaMapping,
  FullFeatureGroupAblationReport,
  CurrentVsRealismCandidateReport,
} from "../../../types/api";
export function PublicDataManifestPanel({ data }: { data: PublicDataManifest }) {
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

export function PublicBiomarkerManifestPanel({ data }: { data: PublicBiomarkerDatasetManifest }) {
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

export function PublicBiomarkerMappingPanel({ data }: { data: PublicBiomarkerMappingReadiness }) {
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

export function FullFeatureGroupAblationPanel({ data }: { data: FullFeatureGroupAblationReport }) {
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
                <td className="py-2 pr-3 tabular-nums text-right" style={{ color: "var(--text-dim)" }}>{group.classification?.patient_level_auroc?.toFixed(3) ?? group.classification?.auroc?.toFixed(3) ?? "â€”"}</td>
                <td className="py-2 pr-3 tabular-nums text-right" style={{ color: "var(--text-dim)" }}>{group.classification?.brier?.toFixed(3) ?? "â€”"}</td>
                <td className="py-2 pr-3 tabular-nums text-right" style={{ color: "var(--text-dim)" }}>{group.classification?.ece?.toFixed(3) ?? "â€”"}</td>
                <td className="py-2 pr-3 tabular-nums text-right" style={{ color: "var(--text-dim)" }}>{group.classification?.false_negative_count ?? "â€”"}</td>
                <td className="py-2 pr-3 tabular-nums text-right" style={{ color: "var(--text-dim)" }}>{group.regression?.mae?.toFixed(3) ?? "â€”"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className="text-xs italic" style={{ color: "var(--text-faint)" }}>{data.claim_boundary}</p>
    </div>
  );
}

export function CbioPortalMappingPanel({ data }: { data: CbioportalBiomarkerSchemaMapping }) {
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
// KbSourceGovernanceCard extracted to ./cards/KbSourceGovernanceCard.tsx
// ModalityRobustnessCard extracted to ./cards/ModalityRobustnessCard.tsx

function formatDelta(value: number | null | undefined): string {
  if (value == null) return "â€”";
  const sign = value > 0 ? "+" : "";
  return `${sign}${(value * 100).toFixed(2)}pp`;
}

export function CostCard({ label, level, color, description }: {
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

export function CandidateComparisonPanel({ data }: { data: CurrentVsRealismCandidateReport }) {
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

export function NoiseEvalPanel({ data }: { data: NoiseEvalResult }) {
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
              {["Noise mode", "AUROC", "Î” AUROC", "Sensitivity", "Î” Sensitivity", "Status"].map(h => (
                <th key={h} className="text-left py-2 pr-4 font-medium" style={{ color: "var(--text-faint)" }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.noise_results.map((r) => (
              <tr key={r.mode} style={{ borderBottom: "1px solid var(--border)" }} className="last:border-0">
                <td className="py-2 pr-4 font-medium" style={{ color: "var(--text)" }}>{r.mode.replace(/_/g, " ")}</td>
                <td className="py-2 pr-4 tabular-nums" style={{ color: "var(--text-dim)" }}>{r.auroc?.toFixed(3) ?? "â€”"}</td>
                <td className="py-2 pr-4 tabular-nums" style={{ color: r.auroc_delta != null && r.auroc_delta < -0.05 ? "var(--rose)" : "var(--text-dim)" }}>
                  {r.auroc_delta != null ? (r.auroc_delta >= 0 ? "+" : "") + r.auroc_delta.toFixed(3) : "â€”"}
                </td>
                <td className="py-2 pr-4 tabular-nums" style={{ color: "var(--text-dim)" }}>{r.sensitivity?.toFixed(3) ?? "â€”"}</td>
                <td className="py-2 pr-4 tabular-nums" style={{ color: r.sensitivity_delta != null && r.sensitivity_delta < -0.05 ? "var(--rose)" : "var(--text-dim)" }}>
                  {r.sensitivity_delta != null ? (r.sensitivity_delta >= 0 ? "+" : "") + r.sensitivity_delta.toFixed(3) : "â€”"}
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
          {data.summary.max_auroc_drop != null && ` Â· max AUROC drop ${data.summary.max_auroc_drop.toFixed(3)}`}
        </p>
      )}
      <p className="text-xs italic" style={{ color: "var(--text-faint)" }}>{data.claim_boundary}</p>
    </div>
  );
}

export function TemporalEvalPanel({ data }: { data: TemporalEvalResult }) {
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
            Temporal gap: <span style={{ color: "var(--text-dim)" }}>{data.generalization_gap.temporal_auroc_gap?.toFixed(3) ?? "â€”"}</span>
          </p>
          <p className="text-xs" style={{ color: "var(--text-faint)" }}>
            Cycle gap: <span style={{ color: "var(--text-dim)" }}>{data.generalization_gap.cycle_auroc_gap?.toFixed(3) ?? "â€”"}</span>
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
      <span className="text-xs tabular-nums font-medium" style={{ color: "var(--text-dim)" }}>{value ?? "â€”"}</span>
    </div>
  );
}

const CONFUSION_COLOR: Record<string, string> = {
  TP: "var(--green)", FP: "var(--amber)", TN: "var(--text-dim)", FN: "var(--rose)"
};

export function PredictionErrorPanel({ data }: { data: PredictionErrorTable }) {
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
