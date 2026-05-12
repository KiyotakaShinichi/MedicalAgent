import { Card, CardHeader, SectionTitle } from "../../components/ui/Card";
import { EmptyPane } from "../../components/ui/Spinner";
import type { Symptom } from "../../types/api";

interface Props { symptoms: Symptom[] }

function SeverityBar({ value }: { value: number }) {
  const color = value >= 7 ? "var(--rose)" : value >= 4 ? "var(--amber)" : "var(--green)";
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 rounded-full h-1.5" style={{ background: "var(--border)" }}>
        <div
          className="h-1.5 rounded-full"
          style={{ width: `${(value / 10) * 100}%`, background: color }}
        />
      </div>
      <span className="text-xs tabular-nums w-4 text-right" style={{ color: "var(--text-dim)" }}>{value}</span>
    </div>
  );
}

export function SymptomsTable({ symptoms }: Props) {
  const sorted = [...(symptoms ?? [])].sort((a, b) => b.date.localeCompare(a.date));
  return (
    <Card>
      <CardHeader>
        <SectionTitle>Symptom Log</SectionTitle>
      </CardHeader>
      {sorted.length === 0 ? (
        <EmptyPane label="No symptoms recorded" />
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr style={{ borderBottom: "1px solid var(--border)" }}>
                {["Date", "Symptom", "Severity", "Notes"].map((h) => (
                  <th key={h} className="text-left py-2 pr-3 font-medium" style={{ color: "var(--text-faint)" }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {sorted.map((s, i) => (
                <tr key={i} style={{ borderBottom: "1px solid var(--border)" }} className="last:border-0">
                  <td className="py-2 pr-3 tabular-nums" style={{ color: "var(--text-dim)" }}>{s.date?.slice(0, 10)}</td>
                  <td className="py-2 pr-3 font-medium" style={{ color: "var(--text)" }}>{s.symptom}</td>
                  <td className="py-2 pr-4" style={{ minWidth: 100 }}><SeverityBar value={s.severity} /></td>
                  <td className="py-2" style={{ color: "var(--text-dim)" }}>{s.notes || "-"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}
