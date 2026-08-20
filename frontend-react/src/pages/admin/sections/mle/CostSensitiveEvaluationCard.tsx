import { Info } from "lucide-react";
import { Card, CardHeader, SectionTitle } from "../../../../components/ui/Card";
import { CostCard } from "../MleEvidencePanels";

const COSTS = [
  {
    label: "False Negative Cost",
    level: "HIGH",
    color: "var(--rose)",
    description: "Missed positive case (failed response) delays clinician intervention. Prioritise recall.",
  },
  {
    label: "False Positive Cost",
    level: "MODERATE",
    color: "var(--amber)",
    description: "Unnecessary flag increases review burden. Acceptable trade-off for lower FNR.",
  },
  {
    label: "Operating Threshold",
    level: "≤ 0.40",
    color: "var(--blue)",
    description: "Decision threshold set below 0.50 to bias toward sensitivity. Reviewed per-model at training time.",
  },
] as const;

/**
 * Static rationale for the cost-sensitive operating point.
 *
 * Fetches nothing — it documents a modelling decision. It sits next to the
 * metric panels so a reader sees *why* the threshold is below 0.50 before
 * reading the sensitivity numbers it produces.
 */
export function CostSensitiveEvaluationCard() {
  return (
    <Card>
      <CardHeader>
        <SectionTitle>Threshold &amp; Cost-Sensitive Evaluation</SectionTitle>
        <Info size={14} aria-hidden="true" style={{ color: "var(--text-faint)" }} />
      </CardHeader>
      <div className="grid sm:grid-cols-3 gap-3 mb-3">
        {COSTS.map((cost) => (
          <CostCard
            key={cost.label}
            label={cost.label}
            level={cost.level}
            color={cost.color}
            description={cost.description}
          />
        ))}
      </div>
      <p className="text-xs" style={{ color: "var(--text-dim)" }}>
        This system uses a cost-sensitive approach: the classification threshold is chosen to minimise FNR at
        an acceptable FPR, reflecting the assumption that missing a treatment non-response is more harmful
        than over-flagging for clinician review. Weighted cost = FN_weight × FN + FP_weight × FP where
        FN_weight = 3 and FP_weight = 1 (engineering heuristic, not clinical guidance).
      </p>
    </Card>
  );
}
