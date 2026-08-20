import { RefreshCw } from "lucide-react";
import { Badge } from "../../../../components/ui/Badge";
import { statusVariant } from "../../../../components/ui/badgeUtils";
import { Button } from "../../../../components/ui/Button";
import { Card, CardHeader, SectionTitle } from "../../../../components/ui/Card";
import { runMleReadiness } from "../../../../api/client";
import { useArtifactRunner } from "../../../../hooks/useArtifactRunner";
import type { AdminAnalytics } from "../../../../types/api";
import { PanelErrorNotice } from "./PanelErrorNotice";

type MleReadiness = AdminAnalytics["mle_readiness"];

/**
 * Release-gate status.
 *
 * Unlike the artifact panels this reads from the parent's `analytics` payload
 * rather than its own endpoint, so re-running gates calls the parent's
 * `onRefresh` instead of a local refetch.
 */
export function MleReadinessGatesCard({
  mle,
  onRefresh,
}: {
  mle: MleReadiness;
  onRefresh: () => void;
}) {
  const { running, error, run } = useArtifactRunner(runMleReadiness, onRefresh, "admin.mle.readiness");

  const allGatesPassed = mle.hard_gate_failures === 0;
  const gateTone = allGatesPassed
    ? { background: "rgba(16,185,129,0.08)", borderColor: "rgba(16,185,129,0.25)", color: "var(--green)" }
    : { background: "rgba(244,63,94,0.08)", borderColor: "rgba(244,63,94,0.25)", color: "var(--rose)" };

  return (
    <>
      <PanelErrorNotice panel="MLE readiness gates" error={error} />
      <Card>
        <CardHeader>
          <SectionTitle>MLE Readiness Gates</SectionTitle>
          <Button
            variant="secondary"
            size="sm"
            loading={running}
            icon={<RefreshCw size={12} aria-hidden="true" />}
            onClick={() => void run()}
            aria-label="Re-run MLE readiness gates"
          >
            Re-run gates
          </Button>
        </CardHeader>

        <div className="flex flex-wrap items-center gap-3 mb-3">
          <Badge variant={statusVariant(mle.status)}>{mle.status}</Badge>
          <span className="text-xs" style={{ color: "var(--text-dim)" }}>
            {mle.release_recommendation.replace(/_/g, " ")}
          </span>
          <span className="text-xs px-2 py-0.5 rounded border" style={gateTone}>
            {allGatesPassed ? "All hard gates passed" : `${mle.hard_gate_failures} gate failures`}
          </span>
        </div>

        <dl className="grid grid-cols-2 sm:grid-cols-4 gap-2">
          {Object.entries(mle.category_statuses).map(([category, status]) => (
            <div key={category} className="flex flex-col gap-1 p-2 rounded-md" style={{ background: "var(--surface2)" }}>
              <dt className="text-xs" style={{ color: "var(--text-faint)" }}>
                {category.replace(/_/g, " ")}
              </dt>
              <dd className="m-0">
                <Badge variant={statusVariant(status)}>{status}</Badge>
              </dd>
            </div>
          ))}
        </dl>
      </Card>
    </>
  );
}
