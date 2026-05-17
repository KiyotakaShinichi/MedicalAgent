import { useState } from "react";
import { ClipboardList, RefreshCw } from "lucide-react";
import { SectionCard } from "../../components/ui/SectionCard";
import { Badge } from "../../components/ui/Badge";
import { statusVariant } from "../../components/ui/badgeUtils";
import { Button } from "../../components/ui/Button";
import { LoadingPane, EmptyPane } from "../../components/ui/Spinner";
import type { ClinicianPredictionTracesResponse } from "../../types/api";

interface Props {
  data: ClinicianPredictionTracesResponse | null;
  loading: boolean;
  onRefresh: () => void;
}

/**
 * Clinician-facing prediction-trace log for ONE patient.  Shows the model
 * decisions made on this patient over time alongside the modalities the
 * system actually had to work with.  Designed to answer the clinician's
 * audit question: "what did the system tell my patient, when, under which
 * model version, and on what evidence?"
 */
export function PredictionTracesPanel({ data, loading, onRefresh }: Props) {
  const [showAbstainedOnly, setShowAbstainedOnly] = useState(false);
  const summary = data?.patient_summary;
  const traces = data?.traces ?? [];
  const visible = showAbstainedOnly ? traces.filter((t) => t.abstained) : traces;

  return (
    <SectionCard
      title="Prediction trace log"
      icon={ClipboardList}
      meta={
        summary ? (
          <span style={{ color: "var(--text-faint)", fontSize: "0.74rem" }}>
            {summary.total} total · {formatRate(summary.abstention_rate)} abstained
          </span>
        ) : undefined
      }
      action={
        <Button onClick={onRefresh} disabled={loading} icon={<RefreshCw size={13} />}>
          Refresh
        </Button>
      }
      footer={
        data?.claim_boundary ? (
          <span>{data.claim_boundary}</span>
        ) : undefined
      }
    >
      {loading ? (
        <LoadingPane label="Loading traces…" />
      ) : traces.length === 0 ? (
        <EmptyPane label="No predictions recorded for this patient yet." />
      ) : (
        <>
          <div className="flex items-center gap-2 mb-3">
            <Badge variant={statusVariant("strong")}>{summary?.total ?? traces.length} entries</Badge>
            <button
              type="button"
              onClick={() => setShowAbstainedOnly((v) => !v)}
              className="text-[0.74rem] font-medium"
              style={{
                color: showAbstainedOnly ? "var(--rose)" : "var(--text-dim)",
                background: "transparent",
                border: "1px solid var(--border)",
                padding: "3px 10px",
                borderRadius: 999,
                cursor: "pointer",
              }}
              aria-pressed={showAbstainedOnly}
            >
              {showAbstainedOnly ? "Show all" : "Show abstained only"}
            </button>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-xs" style={{ borderCollapse: "separate", borderSpacing: 0 }}>
              <thead>
                <tr style={{ color: "var(--text-faint)" }}>
                  {["When", "Question", "Decision", "Prob.", "Confidence", "Evidence", "Modalities used", "Model"].map((h) => (
                    <th key={h} className="text-left font-semibold py-1.5 px-2"
                        style={{ borderBottom: "1px solid var(--border)" }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {visible.slice(0, 15).map((t) => (
                  <tr key={t.id}>
                    <td className="py-1.5 px-2" style={{ borderBottom: "1px solid var(--border-soft)", color: "var(--text-dim)" }}>
                      {t.created_at ? new Date(t.created_at).toLocaleString() : "—"}
                    </td>
                    <td className="py-1.5 px-2" style={{ borderBottom: "1px solid var(--border-soft)" }}>{t.question}</td>
                    <td className="py-1.5 px-2 font-semibold"
                        style={{ borderBottom: "1px solid var(--border-soft)", color: t.abstained ? "var(--amber)" : "var(--text)" }}>
                      {t.decision}
                    </td>
                    <td className="py-1.5 px-2 tabular-nums" style={{ borderBottom: "1px solid var(--border-soft)" }}>
                      {t.probability == null ? "—" : t.probability.toFixed(3)}
                    </td>
                    <td className="py-1.5 px-2" style={{ borderBottom: "1px solid var(--border-soft)" }}>{t.confidence ?? "—"}</td>
                    <td className="py-1.5 px-2" style={{ borderBottom: "1px solid var(--border-soft)" }}>{t.evidence_sufficiency ?? "—"}</td>
                    <td className="py-1.5 px-2" style={{ borderBottom: "1px solid var(--border-soft)", color: "var(--text-dim)" }}>
                      {t.modalities_present.length}/{t.modalities_present.length + t.modalities_missing.length}
                    </td>
                    <td className="py-1.5 px-2" style={{ borderBottom: "1px solid var(--border-soft)", color: "var(--text-faint)", fontFamily: "monospace", fontSize: "0.7rem" }}>
                      {t.model_version}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {visible.length > 15 && (
            <p className="text-[0.7rem] mt-2" style={{ color: "var(--text-faint)" }}>
              Showing first 15 of {visible.length} traces.
            </p>
          )}
        </>
      )}
    </SectionCard>
  );
}

function formatRate(value: number | null | undefined): string {
  if (value == null) return "—";
  return `${(value * 100).toFixed(1)}%`;
}
