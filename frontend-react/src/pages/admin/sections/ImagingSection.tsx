import { useState } from "react";
import { Activity, Database, Image as ImageIcon, RefreshCw } from "lucide-react";
import { Badge } from "../../../components/ui/Badge";
import { Button } from "../../../components/ui/Button";
import { Card, CardHeader, SectionTitle } from "../../../components/ui/Card";
import { MetricCard } from "../../../components/ui/MetricCard";
import { EmptyPane, ErrorPane, LoadingPane } from "../../../components/ui/Spinner";
import { useApi } from "../../../hooks/useApi";
import {
  getCtLesionWorkflow,
  getPublicImagingManifest,
  getSimToPublicImaging,
  getUltrasoundBaseline,
  runCtLesionWorkflow,
  runPublicImagingManifest,
  runSimToPublicImaging,
  runUltrasoundBaseline,
} from "../../../api/client";
import type {
  CtLesionWorkflowReport,
  PublicImagingManifest,
  SimToPublicImagingReport,
  UltrasoundBaselineResult,
} from "../../../types/api";

export function ImagingSection() {
  const manifest = useApi(getPublicImagingManifest, []);
  const ultrasound = useApi(getUltrasoundBaseline, []);
  const ct = useApi(getCtLesionWorkflow, []);
  const gap = useApi(getSimToPublicImaging, []);
  const [running, setRunning] = useState<string | null>(null);

  async function rerun(name: string, fn: () => Promise<unknown>, refresh: () => void) {
    setRunning(name);
    try {
      await fn();
      refresh();
    } finally {
      setRunning(null);
    }
  }

  return (
    <div className="flex flex-col gap-4">
      <div
        className="rounded-lg border p-3 text-xs"
        style={{ background: "rgba(59,130,246,0.07)", borderColor: "rgba(59,130,246,0.24)", color: "var(--text-dim)" }}
      >
        Imaging support is intentionally narrow: public datasets power engineering baselines and workflow readiness,
        while patient-facing CT/ultrasound behavior remains report-text extraction plus clinician-review routing.
        No panel here is a diagnostic image reader.
      </div>

      <Card>
        <CardHeader>
          <SectionTitle>Public Imaging Dataset Readiness</SectionTitle>
          <Button
            size="sm"
            variant="secondary"
            icon={<RefreshCw size={12} />}
            loading={running === "manifest"}
            onClick={() => void rerun("manifest", runPublicImagingManifest, manifest.refetch)}
          >
            Rebuild
          </Button>
        </CardHeader>
        {manifest.status === "loading" ? <LoadingPane /> :
         manifest.status === "error" ? <ErrorPane message={manifest.error ?? "Could not load manifest"} /> :
         manifest.data ? <ManifestPanel data={manifest.data as PublicImagingManifest} /> :
         <EmptyPane label="No public imaging manifest" />}
      </Card>

      <div className="grid lg:grid-cols-2 gap-4">
        <Card>
          <CardHeader>
            <SectionTitle>Ultrasound Baseline</SectionTitle>
            <Button
              size="sm"
              variant="secondary"
              icon={<Activity size={12} />}
              loading={running === "ultrasound"}
              onClick={() => void rerun("ultrasound", runUltrasoundBaseline, ultrasound.refetch)}
            >
              Run
            </Button>
          </CardHeader>
          {ultrasound.status === "loading" ? <LoadingPane /> :
           ultrasound.status === "error" ? <ErrorPane message={ultrasound.error ?? "Could not load ultrasound baseline"} /> :
           ultrasound.data ? <UltrasoundPanel data={ultrasound.data as UltrasoundBaselineResult} /> :
           <EmptyPane label="No ultrasound baseline artifact" />}
        </Card>

        <Card>
          <CardHeader>
            <SectionTitle>CT / PET-CT Workflow</SectionTitle>
            <Button
              size="sm"
              variant="secondary"
              icon={<Database size={12} />}
              loading={running === "ct"}
              onClick={() => void rerun("ct", runCtLesionWorkflow, ct.refetch)}
            >
              Rebuild
            </Button>
          </CardHeader>
          {ct.status === "loading" ? <LoadingPane /> :
           ct.status === "error" ? <ErrorPane message={ct.error ?? "Could not load CT workflow"} /> :
           ct.data ? <CtPanel data={ct.data as CtLesionWorkflowReport} /> :
           <EmptyPane label="No CT workflow artifact" />}
        </Card>
      </div>

      <Card>
        <CardHeader>
          <SectionTitle>Sim-to-Public Imaging Gap</SectionTitle>
          <Button
            size="sm"
            variant="secondary"
            icon={<RefreshCw size={12} />}
            loading={running === "gap"}
            onClick={() => void rerun("gap", runSimToPublicImaging, gap.refetch)}
          >
            Recompute
          </Button>
        </CardHeader>
        {gap.status === "loading" ? <LoadingPane /> :
         gap.status === "error" ? <ErrorPane message={gap.error ?? "Could not load gap report"} /> :
         gap.data ? <GapPanel data={gap.data as SimToPublicImagingReport} /> :
         <EmptyPane label="No sim-to-public report" />}
      </Card>
    </div>
  );
}

function ManifestPanel({ data }: { data: PublicImagingManifest }) {
  return (
    <div className="flex flex-col gap-3">
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <MetricCard label="Status" value={data.status.replace(/_/g, " ")} />
        <MetricCard label="Available Datasets" value={String(data.available_dataset_count)} />
        <MetricCard label="Dataset Root" value={data.dataset_root} />
        <MetricCard label="Manifest Hash" value={data.manifest_hash} />
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr style={{ borderBottom: "1px solid var(--border)" }}>
              {["Dataset", "Ready?", "Images", "Masks", "Labels", "Readiness"].map((header) => (
                <th key={header} className="text-left py-2 pr-4 font-medium" style={{ color: "var(--text-faint)" }}>{header}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.datasets.map((dataset) => (
              <tr key={dataset.id} style={{ borderBottom: "1px solid var(--border)" }} className="last:border-0">
                <td className="py-2 pr-4">
                  <div className="flex items-center gap-2">
                    <ImageIcon size={12} style={{ color: "var(--text-faint)" }} />
                    <div>
                      <p className="font-medium" style={{ color: "var(--text)" }}>{dataset.name}</p>
                      <p style={{ color: "var(--text-faint)" }}>{dataset.modality} - {dataset.task.replace(/_/g, " ")}</p>
                    </div>
                  </div>
                </td>
                <td className="py-2 pr-4">
                  <Badge variant={dataset.available ? "green" : "red"}>{dataset.available ? "available" : "missing"}</Badge>
                </td>
                <td className="py-2 pr-4 tabular-nums" style={{ color: "var(--text-dim)" }}>{dataset.image_count}</td>
                <td className="py-2 pr-4 tabular-nums" style={{ color: "var(--text-dim)" }}>{dataset.mask_count}</td>
                <td className="py-2 pr-4" style={{ color: "var(--text-dim)" }}>{formatCounts(dataset.class_counts)}</td>
                <td className="py-2 pr-4" style={{ color: "var(--text-dim)" }}>{dataset.readiness.replace(/_/g, " ")}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-xs" style={{ color: "var(--text-faint)" }}>{data.recommended_next_task}</p>
      <p className="text-xs italic" style={{ color: "var(--text-faint)" }}>{data.claim_boundary}</p>
    </div>
  );
}

function UltrasoundPanel({ data }: { data: UltrasoundBaselineResult }) {
  if (data.status === "unavailable") {
    return <Unavailable reason={data.reason} expected={data.expected_layout} boundary={data.claim_boundary} />;
  }
  const best = data.best_model && data.models ? data.models[data.best_model] : null;
  return (
    <div className="flex flex-col gap-3">
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <MetricCard label="Images" value={String(data.image_count ?? 0)} />
        <MetricCard label="Best Model" value={data.best_model ?? "unknown"} />
        <MetricCard label="Balanced Acc." value={best?.balanced_accuracy?.toFixed(3) ?? null} />
        <MetricCard label="Macro F1" value={best?.macro_f1?.toFixed(3) ?? null} />
      </div>
      {best && (
        <div className="rounded-md border p-3" style={{ borderColor: "var(--border)", background: "var(--surface2)" }}>
          <p className="text-xs font-semibold mb-2" style={{ color: "var(--text)" }}>Best-model confusion matrix</p>
          <p className="text-xs" style={{ color: "var(--text-dim)" }}>
            Classes: {best.classes.join(", ")}. Matrix: {JSON.stringify(best.confusion_matrix)}
          </p>
        </div>
      )}
      <p className="text-xs" style={{ color: "var(--text-faint)" }}>
        Labels: {formatCounts(data.label_counts ?? {})}. Predictions: {data.predictions_path ?? "not written"}.
      </p>
      <p className="text-xs italic" style={{ color: "var(--text-faint)" }}>{data.claim_boundary}</p>
    </div>
  );
}

function CtPanel({ data }: { data: CtLesionWorkflowReport }) {
  if (data.status === "unavailable") {
    return <Unavailable reason={data.reason} expected={data.expected_layout} boundary={data.claim_boundary} />;
  }
  return (
    <div className="flex flex-col gap-3">
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
        <MetricCard label="Status" value={data.status.replace(/_/g, " ")} />
        <MetricCard label="Image Files" value={String(data.image_file_count ?? 0)} />
        <MetricCard label="Metadata Files" value={String(data.metadata_file_count ?? 0)} />
      </div>
      <div className="rounded-md border p-3" style={{ background: "var(--surface2)", borderColor: "var(--border)" }}>
        <p className="text-xs font-semibold mb-2" style={{ color: "var(--text)" }}>Recommended model track</p>
        {(data.recommended_model_track ?? []).map((item) => (
          <p key={item} className="text-xs" style={{ color: "var(--text-dim)" }}>- {item}</p>
        ))}
      </div>
      <p className="text-xs italic" style={{ color: "var(--text-faint)" }}>{data.claim_boundary}</p>
    </div>
  );
}

function GapPanel({ data }: { data: SimToPublicImagingReport }) {
  return (
    <div className="flex flex-col gap-3">
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <MetricCard label="Synthetic MRI Rows" value={String(data.synthetic_summary.row_count)} />
        <MetricCard label="Synthetic Patients" value={String(data.synthetic_summary.patient_count ?? "unknown")} />
        <MetricCard label="Public Sets Present" value={String(data.public_imaging_availability.available_dataset_count)} />
        <MetricCard label="Metastatic Rows" value={String(data.synthetic_summary.metastatic_keyword_rows)} />
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr style={{ borderBottom: "1px solid var(--border)" }}>
              {["Area", "Synthetic", "Public", "Available", "Gap"].map((header) => (
                <th key={header} className="text-left py-2 pr-4 font-medium" style={{ color: "var(--text-faint)" }}>{header}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.gap_table.map((row) => (
              <tr key={row.area} style={{ borderBottom: "1px solid var(--border)" }} className="last:border-0">
                <td className="py-2 pr-4 font-medium" style={{ color: "var(--text)" }}>{row.area.replace(/_/g, " ")}</td>
                <td className="py-2 pr-4" style={{ color: "var(--text-dim)" }}>{row.synthetic_coverage.replace(/_/g, " ")}</td>
                <td className="py-2 pr-4" style={{ color: "var(--text-dim)" }}>{row.public_coverage.replace(/_/g, " ")}</td>
                <td className="py-2 pr-4"><Badge variant={row.available_now ? "green" : "amber"}>{row.available_now ? "yes" : "not local"}</Badge></td>
                <td className="py-2 pr-4" style={{ color: "var(--text-dim)" }}>{row.gap}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="rounded-md border p-3" style={{ background: "var(--surface2)", borderColor: "var(--border)" }}>
        <p className="text-xs font-semibold mb-2" style={{ color: "var(--text)" }}>Recommended actions</p>
        {data.recommended_actions.map((item) => (
          <p key={item} className="text-xs" style={{ color: "var(--text-dim)" }}>- {item}</p>
        ))}
      </div>
      <p className="text-xs italic" style={{ color: "var(--text-faint)" }}>{data.claim_boundary}</p>
    </div>
  );
}

function Unavailable({ reason, expected, boundary }: { reason?: string; expected?: string; boundary: string }) {
  return (
    <div className="rounded-md border p-3" style={{ background: "rgba(245,158,11,0.07)", borderColor: "rgba(245,158,11,0.25)" }}>
      <p className="text-xs font-semibold mb-1" style={{ color: "var(--amber)" }}>Artifact unavailable</p>
      <p className="text-xs" style={{ color: "var(--text-dim)" }}>{reason}</p>
      {expected && <p className="text-xs mt-1" style={{ color: "var(--text-faint)" }}>{expected}</p>}
      <p className="text-xs italic mt-2" style={{ color: "var(--text-faint)" }}>{boundary}</p>
    </div>
  );
}

function formatCounts(counts: Record<string, number>) {
  const entries = Object.entries(counts ?? {});
  return entries.length ? entries.map(([key, value]) => `${key}: ${value}`).join(", ") : "none";
}
