import { FlaskConical } from "lucide-react";
import { Card, CardHeader, SectionTitle } from "../../components/ui/Card";
import { MetricCard } from "../../components/ui/MetricCard";
import { LabTrendsChart } from "../../components/charts/LabTrendsChart";
import type { PatientReport } from "../../types/api";

function labStatus(val: number | null, key: string): "green" | "amber" | "red" | "muted" {
  if (val == null) return "muted";
  if (key === "wbc")        return val < 2 ? "red" : val < 4 ? "amber" : "green";
  if (key === "hemoglobin") return val < 8 ? "red" : val < 11 ? "amber" : "green";
  if (key === "platelets")  return val < 50 ? "red" : val < 100 ? "amber" : "green";
  return "muted";
}

interface Props { report: PatientReport }

export function LabsPanel({ report }: Props) {
  const { latest_labs: labs, lab_history } = report;
  return (
    <Card>
      <CardHeader>
        <SectionTitle>Lab Values (CBC)</SectionTitle>
        <FlaskConical size={14} style={{ color: "var(--text-faint)" }} />
      </CardHeader>
      <div className="grid grid-cols-3 gap-3 mb-4">
        <MetricCard
          label="WBC"
          value={labs?.wbc != null ? labs.wbc.toFixed(1) : null}
          unit="K/uL"
          status={labStatus(labs?.wbc ?? null, "wbc")}
        />
        <MetricCard
          label="Hemoglobin"
          value={labs?.hemoglobin != null ? labs.hemoglobin.toFixed(1) : null}
          unit="g/dL"
          status={labStatus(labs?.hemoglobin ?? null, "hemoglobin")}
        />
        <MetricCard
          label="Platelets"
          value={labs?.platelets != null ? Math.round(labs.platelets) : null}
          unit="K/uL"
          status={labStatus(labs?.platelets ?? null, "platelets")}
        />
      </div>
      <LabTrendsChart data={lab_history ?? []} />
    </Card>
  );
}
