import type { ConfusionMatrix } from "../../types/api";

interface Props { cm: ConfusionMatrix }

function Cell({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div
      className="flex flex-col items-center justify-center rounded-md p-3 gap-1"
      style={{ background: color }}
    >
      <span className="text-2xl font-bold tabular-nums" style={{ color: "var(--text)" }}>{value}</span>
      <span className="text-xs" style={{ color: "var(--text-dim)" }}>{label}</span>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: number | null }) {
  return (
    <div className="flex justify-between text-xs py-1 border-b" style={{ borderColor: "var(--border)" }}>
      <span style={{ color: "var(--text-dim)" }}>{label}</span>
      <span className="font-medium tabular-nums">{value != null ? `${(value * 100).toFixed(1)}%` : "-"}</span>
    </div>
  );
}

export function ConfusionMatrixPanel({ cm }: Props) {
  return (
    <div className="flex flex-col gap-4">
      <div className="grid grid-cols-2 gap-2">
        <Cell label="True Positive"  value={cm.tp} color="rgba(16,185,129,0.12)" />
        <Cell label="False Positive" value={cm.fp} color="rgba(244,63,94,0.10)" />
        <Cell label="False Negative" value={cm.fn} color="rgba(245,158,11,0.10)" />
        <Cell label="True Negative"  value={cm.tn} color="rgba(16,185,129,0.08)" />
      </div>
      <div>
        <Stat label="Sensitivity (Recall)" value={cm.sensitivity} />
        <Stat label="Specificity"          value={cm.specificity} />
        <Stat label="Precision (PPV)"      value={cm.precision} />
        <Stat label="False Negative Rate"  value={cm.fnr} />
      </div>
    </div>
  );
}
