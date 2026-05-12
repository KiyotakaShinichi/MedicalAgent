import { useState } from "react";
import { Play, CheckCircle, XCircle } from "lucide-react";
import { Button } from "../../../components/ui/Button";
import { Badge } from "../../../components/ui/Badge";
import { Card, CardHeader, SectionTitle } from "../../../components/ui/Card";
import { MetricCard } from "../../../components/ui/MetricCard";
import { LoadingPane, EmptyPane } from "../../../components/ui/Spinner";
import { runAgentRegression } from "../../../api/client";
import type { AgentRegressionResult } from "../../../types/api";

export function RegressionSection() {
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<AgentRegressionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function run() {
    setRunning(true);
    setError(null);
    try {
      const res = await runAgentRegression();
      setResult(res.result);
    } catch (e: unknown) {
      setError((e as Error).message);
    } finally {
      setRunning(false);
    }
  }

  const cases = result?.cases ?? [];
  const passed = cases.filter((c) => c.status === "passed");
  const failed = cases.filter((c) => c.status === "failed");

  return (
    <div className="flex flex-col gap-4">
      <Card>
        <CardHeader>
          <SectionTitle>Agent Regression Suite</SectionTitle>
          <Button
            variant="primary" size="sm"
            loading={running}
            icon={<Play size={12} />}
            onClick={() => void run()}
          >
            Run suite
          </Button>
        </CardHeader>

        {running && <LoadingPane label="Running regression suite..." />}

        {error && (
          <p className="text-xs px-3 py-2 rounded-md" style={{ background: "rgba(244,63,94,0.08)", color: "var(--rose)" }}>
            {error}
          </p>
        )}

        {result && !running && (
          <div className="flex flex-col gap-4">
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <MetricCard
                label="Status"
                value={result.status}
                status={result.status === "strong" ? "green" : result.status === "acceptable" ? "amber" : "red"}
              />
              <MetricCard
                label="Pass rate"
                value={`${(result.pass_rate * 100).toFixed(0)}%`}
                status={result.pass_rate >= 0.95 ? "green" : result.pass_rate >= 0.9 ? "amber" : "red"}
                sub={`${passed.length}/${result.case_count} passed`}
              />
              <MetricCard
                label="Attack block"
                value={`${(result.attack_block_rate * 100).toFixed(0)}%`}
                status={result.attack_block_rate === 1 ? "green" : "amber"}
              />
              <MetricCard
                label="Source hit"
                value={`${(result.expected_source_hit_rate * 100).toFixed(0)}%`}
                status={result.expected_source_hit_rate >= 0.95 ? "green" : "amber"}
              />
            </div>

            {/* Failed cases */}
            {failed.length > 0 && (
              <div>
                <p className="text-xs font-semibold mb-2" style={{ color: "var(--rose)" }}>
                  {failed.length} failed case{failed.length !== 1 ? "s" : ""}
                </p>
                {failed.map((c) => (
                  <div
                    key={c.id}
                    className="flex items-start gap-2 py-2 border-b last:border-0"
                    style={{ borderColor: "var(--border)" }}
                  >
                    <XCircle size={13} style={{ color: "var(--rose)", flexShrink: 0, marginTop: 1 }} />
                    <div className="flex-1 min-w-0">
                      <p className="text-xs font-medium" style={{ color: "var(--text)" }}>{c.id}</p>
                      <p className="text-xs" style={{ color: "var(--text-dim)" }}>{c.query}</p>
                      {c.checks.filter((ch) => !ch.passed).map((ch, i) => (
                        <p key={i} className="text-xs" style={{ color: "var(--rose)" }}>
                          Failed: {ch.name}: expected {JSON.stringify(ch.expected)}, got {JSON.stringify(ch.observed)}
                        </p>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* Passed cases summary */}
            {passed.length > 0 && (
              <div>
                <p className="text-xs font-semibold mb-2" style={{ color: "var(--green)" }}>
                  {passed.length} passed
                </p>
                <div className="flex flex-wrap gap-1">
                  {passed.map((c) => (
                    <span
                      key={c.id}
                      className="flex items-center gap-1 text-xs px-2 py-0.5 rounded border"
                      style={{ background: "rgba(16,185,129,0.06)", borderColor: "rgba(16,185,129,0.2)", color: "var(--green)" }}
                    >
                      <CheckCircle size={10} />
                      {c.id}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {!result && !running && <EmptyPane label="Click 'Run suite' to evaluate agent behavior" />}
      </Card>

      {/* Category breakdown */}
      {result?.cases && (
        <Card>
          <CardHeader><SectionTitle>Results by Category</SectionTitle></CardHeader>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
            {["education", "portal_help", "clinical_safety", "security", "conversation", "boundary"].map((cat) => {
              const catCases = result.cases!.filter((c) => c.category === cat);
              if (!catCases.length) return null;
              const catPassed = catCases.filter((c) => c.status === "passed").length;
              return (
                <div
                  key={cat}
                  className="p-3 rounded-md border"
                  style={{ background: "var(--surface2)", borderColor: "var(--border)" }}
                >
                  <p className="text-xs font-medium mb-1.5" style={{ color: "var(--text)" }}>
                    {cat.replace(/_/g, " ")}
                  </p>
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-bold tabular-nums" style={{
                      color: catPassed === catCases.length ? "var(--green)" : "var(--amber)"
                    }}>
                      {catPassed}/{catCases.length}
                    </span>
                    <Badge variant={catPassed === catCases.length ? "green" : "amber"}>
                      {catPassed === catCases.length ? "pass" : "partial"}
                    </Badge>
                  </div>
                </div>
              );
            })}
          </div>
        </Card>
      )}
    </div>
  );
}
