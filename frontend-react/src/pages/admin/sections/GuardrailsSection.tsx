import { ShieldCheck, ShieldX } from "lucide-react";
import { MetricCard } from "../../../components/ui/MetricCard";
import { Card, CardHeader, SectionTitle } from "../../../components/ui/Card";
import type { AdminAnalytics } from "../../../types/api";

interface Props { analytics: AdminAnalytics }

export function GuardrailsSection({ analytics }: Props) {
  const g = analytics.guardrails;

  return (
    <div className="flex flex-col gap-4">
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <MetricCard
          label="Input blocks"
          value={g.input_blocks}
          icon={ShieldX}
          status={g.input_blocks === 0 ? "green" : g.input_blocks < 5 ? "amber" : "red"}
        />
        <MetricCard
          label="Output blocks"
          value={g.output_blocks}
          icon={ShieldX}
          status={g.output_blocks === 0 ? "green" : "amber"}
        />
        <MetricCard
          label="Attack block rate"
          value={g.attack_block_rate != null ? `${(g.attack_block_rate * 100).toFixed(0)}%` : null}
          icon={ShieldCheck}
          status={g.attack_block_rate === 1 ? "green" : g.attack_block_rate != null && g.attack_block_rate >= 0.9 ? "amber" : "red"}
        />
        <MetricCard
          label="Pass rate"
          value={g.pass_rate != null ? `${(g.pass_rate * 100).toFixed(0)}%` : null}
          status={g.pass_rate != null && g.pass_rate >= 0.95 ? "green" : "amber"}
        />
      </div>

      <Card>
        <CardHeader><SectionTitle>Guardrail Policy Summary</SectionTitle></CardHeader>
        <div className="flex flex-col gap-3">
          <PolicyRow
            title="Input guardrails"
            items={[
              "Prompt injection detection (English + multilingual patterns)",
              "PHI access boundary - own data only",
              "Unsafe clinical intent (treatment decisions, diagnoses)",
              "Jailbreak and role-override attempts",
            ]}
            icon={<ShieldCheck size={14} style={{ color: "var(--green)" }} />}
          />
          <PolicyRow
            title="Output guardrails"
            items={[
              "Diagnosis or survival-rate generation",
              "Unsafe treatment directives",
              "PII/PHI leakage in responses",
              "Unsupported clinical claims",
            ]}
            icon={<ShieldCheck size={14} style={{ color: "var(--blue)" }} />}
          />
        </div>
      </Card>
    </div>
  );
}

function PolicyRow({ title, items, icon }: { title: string; items: string[]; icon: React.ReactNode }) {
  return (
    <div
      className="rounded-md p-3 border"
      style={{ background: "var(--surface2)", borderColor: "var(--border)" }}
    >
      <div className="flex items-center gap-2 mb-2">
        {icon}
        <p className="text-xs font-semibold" style={{ color: "var(--text)" }}>{title}</p>
      </div>
      <ul className="flex flex-col gap-1">
        {items.map((item, i) => (
          <li key={i} className="text-xs flex items-start gap-1.5" style={{ color: "var(--text-dim)" }}>
            <span style={{ color: "var(--text-faint)", marginTop: 2 }}>-</span>
            {item}
          </li>
        ))}
      </ul>
    </div>
  );
}
