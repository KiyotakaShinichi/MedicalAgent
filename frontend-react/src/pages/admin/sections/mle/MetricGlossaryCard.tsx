import { useState } from "react";
import { Button } from "../../../../components/ui/Button";
import { Card, CardHeader, SectionTitle } from "../../../../components/ui/Card";
import { MetricGlossary, ALL_METRIC_SPECS } from "../../../../components/ui/MetricInterpretation";

/**
 * Collapsible interpretation guide.
 *
 * `values` are the current training-report numbers so each band can show where
 * this model actually sits. They are optional — the guide is still useful with
 * no run to compare against.
 */
export function MetricGlossaryCard({ values }: { values: Record<string, number | null> }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <Card>
      <CardHeader>
        <SectionTitle>Metric Interpretation Guide</SectionTitle>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setExpanded((v) => !v)}
          aria-expanded={expanded}
        >
          {expanded ? "Hide" : "Show"} guide
        </Button>
      </CardHeader>

      {expanded ? (
        <div className="flex flex-col gap-2">
          <p className="text-xs mb-2" style={{ color: "var(--text-dim)" }}>
            Each metric shows its definition, why it matters here, and ideal / warning / bad
            interpretation bands. Bands reflect engineering heuristics for a cancer monitoring PoC —
            not clinical validation thresholds.
          </p>
          <MetricGlossary specs={ALL_METRIC_SPECS} values={values} />
        </div>
      ) : (
        <p className="text-xs" style={{ color: "var(--text-faint)" }}>
          Show the guide for interpretation bands covering AUROC, PR-AUC, Brier, ECE, Sensitivity,
          Specificity, FNR, and MAE.
        </p>
      )}
    </Card>
  );
}
