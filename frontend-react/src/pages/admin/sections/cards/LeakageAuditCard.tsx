import { RefreshCw, ShieldCheck, ShieldAlert } from "lucide-react";
import { Badge } from "../../../../components/ui/Badge";
import { statusVariant } from "../../../../components/ui/badgeUtils";
import { Card, CardHeader, SectionTitle } from "../../../../components/ui/Card";
import { Button } from "../../../../components/ui/Button";
import { MetricCard } from "../../../../components/ui/MetricCard";
import { LoadingPane, EmptyPane } from "../../../../components/ui/Spinner";
import type { LeakageAuditReport } from "../../../../types/api";

/**
 * Phase 1 — Training-data leakage audit summary.
 *
 * Extracted from `MleSection.tsx` so each hardening-arc card lives in its
 * own file.  The card is presentational only; the parent owns the
 * `useApi` + the "Rerun audit" handler.
 */
export function LeakageAuditCard({
  report,
  loading,
  running,
  onRefresh,
}: {
  report: LeakageAuditReport | null;
  loading: boolean;
  running: boolean;
  onRefresh: () => void;
}) {
  const status = report?.status ?? "missing";
  const passed = status === "passed";
  const failedFindings = (report?.findings ?? []).filter((f) => f.status !== "passed");
  const tone =
    status === "passed" ? { color: "var(--green, #047857)", icon: ShieldCheck, label: "PASSED" } :
    status === "failed" ? { color: "var(--rose, #b91c1c)", icon: ShieldAlert, label: "FAILED" } :
                          { color: "var(--text-faint)",     icon: ShieldAlert, label: "MISSING" };
  const ToneIcon = tone.icon;

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <ToneIcon size={14} style={{ color: tone.color }} aria-hidden="true" />
          <SectionTitle>Training-data leakage audit</SectionTitle>
          <Badge variant={statusVariant(passed ? "strong" : status === "missing" ? "stale" : "failed")}>
            {tone.label}
          </Badge>
        </div>
        <Button onClick={onRefresh} disabled={running} icon={<RefreshCw size={13} />}>
          {running ? "Running…" : "Rerun audit"}
        </Button>
      </CardHeader>

      <p className="text-xs mb-3" style={{ color: "var(--text-dim)" }}>
        Hard CI gate over the synthetic temporal training rows. Fails the build
        if patient IDs overlap between train/test, if any known label proxy
        appears in the feature contract, or if a feature column is numerically
        identical to a classification label.
      </p>

      {loading ? (
        <LoadingPane label="Loading leakage audit…" />
      ) : !report || status === "missing" ? (
        <EmptyPane label={report?.message ?? "Leakage audit has not been generated yet."} />
      ) : (
        <>
          <div className="grid sm:grid-cols-3 gap-3 mb-3">
            <MetricCard
              label="Checks passed"
              value={report.summary?.checks_passed ?? null}
              status={passed ? "green" : "amber"}
            />
            <MetricCard
              label="Checks failed"
              value={report.summary?.checks_failed ?? null}
              status={(report.summary?.checks_failed ?? 0) === 0 ? "green" : "red"}
            />
            <MetricCard
              label="Temporal sub-audit"
              value={report.temporal_sub_audit?.status ?? "—"}
              status={report.temporal_sub_audit?.status === "passed" ? "green" : "amber"}
            />
          </div>

          {failedFindings.length > 0 && (
            <div
              className="rounded-md border p-3 mb-2"
              style={{ background: "rgba(244,63,94,0.05)", borderColor: "rgba(244,63,94,0.25)" }}
            >
              <p className="text-xs font-semibold mb-1.5" style={{ color: "var(--rose, #b91c1c)" }}>
                Failed checks ({failedFindings.length})
              </p>
              <ul className="text-xs flex flex-col gap-1.5" style={{ color: "var(--text)" }}>
                {failedFindings.slice(0, 8).map((finding, i) => (
                  <li key={i}>
                    <code style={{ fontWeight: 600 }}>{finding.name}</code>
                    {finding.meaning && (
                      <span style={{ color: "var(--text-dim)" }}> — {finding.meaning}</span>
                    )}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {report.generated_at && (
            <p className="text-[0.7rem]" style={{ color: "var(--text-faint)" }}>
              Last run: {new Date(report.generated_at).toLocaleString()}
              {report.training_rows_path && (
                <> · source: <code>{report.training_rows_path}</code></>
              )}
            </p>
          )}
        </>
      )}
    </Card>
  );
}
