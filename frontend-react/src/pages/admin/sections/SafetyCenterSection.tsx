import { useEffect, useState, type ReactNode } from "react";
import { Play, AlertTriangle, ShieldCheck, Activity } from "lucide-react";
import { Button } from "../../../components/ui/Button";
import { Badge } from "../../../components/ui/Badge";
import { Card, CardHeader, SectionTitle } from "../../../components/ui/Card";
import { MetricCard } from "../../../components/ui/MetricCard";
import { LoadingPane, EmptyPane, ErrorPane } from "../../../components/ui/Spinner";
import { FreshnessChip } from "../../../components/ui/FreshnessChip";
import {
  getSafetyCenter,
  getLlmJudgeEval,
  getMultilingualRefusalEval,
  runLlmJudgeEval,
  runDriftReport,
  runMultilingualRefusalEval,
  runRagEvalArtifact,
  runSafetyRedTeam,
} from "../../../api/client";
import type {
  BenchmarkLadderSummary,
  DriftReport,
  LlmJudgeEval,
  MultilingualRefusalEval,
  RagEvalArtifact,
  SafetyCenter,
  SafetyCenterCategorySummary,
  SafetyRedTeamArtifact,
} from "../../../types/api";

type Status = "idle" | "loading" | "success" | "error";

function statusBadge(status: string | undefined): "green" | "amber" | "red" | "muted" {
  if (!status) return "muted";
  if (["passed", "strong", "available", "ok"].includes(status)) return "green";
  if (["acceptable", "watch", "needs_attention", "partial"].includes(status)) return "amber";
  if (["failed", "unideal", "error"].includes(status)) return "red";
  return "muted";
}

function fmtRate(value: number | null | undefined): string {
  if (value === null || value === undefined) return "—";
  return `${(value * 100).toFixed(0)}%`;
}

function fmtScore(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined) return "—";
  return value.toFixed(digits);
}

export function SafetyCenterSection() {
  const [data, setData] = useState<SafetyCenter | null>(null);
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState<string | null>(null);
  const [multilingual, setMultilingual] = useState<MultilingualRefusalEval | null>(null);
  const [llmJudge, setLlmJudge] = useState<LlmJudgeEval | null>(null);

  async function load() {
    setStatus("loading");
    setError(null);
    try {
      const result = await getSafetyCenter();
      setData(result);
      const [multiResult, judgeResult] = await Promise.allSettled([
        getMultilingualRefusalEval(),
        getLlmJudgeEval(),
      ]);
      if (multiResult.status === "fulfilled") setMultilingual(multiResult.value);
      if (judgeResult.status === "fulfilled") setLlmJudge(judgeResult.value);
      setStatus("success");
    } catch (e) {
      setError((e as Error).message);
      setStatus("error");
    }
  }

  useEffect(() => {
    void load();
  }, []);

  async function regenerate(kind: "safety" | "rag" | "drift", liveAgent = false) {
    const runKey = liveAgent ? `${kind}-live` : kind;
    setRunning(runKey);
    try {
      if (kind === "safety") await runSafetyRedTeam(liveAgent);
      if (kind === "rag") await runRagEvalArtifact(liveAgent);
      if (kind === "drift") await runDriftReport();
      await load();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setRunning(null);
    }
  }

  async function regenerateExtra(kind: "multilingual" | "llm_judge") {
    setRunning(kind);
    try {
      if (kind === "multilingual") {
        const response = await runMultilingualRefusalEval();
        setMultilingual(response.result);
      }
      if (kind === "llm_judge") {
        const response = await runLlmJudgeEval(30);
        setLlmJudge(response.result);
      }
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setRunning(null);
    }
  }

  if (status === "loading" && !data) return <LoadingPane label="Loading safety & evaluation center..." />;
  if (status === "error") return <ErrorPane message={error ?? "Failed to load safety center"} />;
  if (!data) return <EmptyPane label="No safety center data" />;

  const safety = data.safety_red_team;
  const rag = data.rag_eval;
  const drift = data.drift_report;
  const calibration = data.calibration_metrics;
  const feedback = data.clinician_feedback;
  const gallery = data.failure_case_gallery;
  const benchmark = data.benchmark_ladder;

  return (
    <div className="flex flex-col gap-4">
      <Card>
        <p className="text-xs" style={{ color: "var(--text-dim)" }}>
          <ShieldCheck size={12} style={{ display: "inline", marginRight: 6 }} />
          {data.safety_note}
        </p>
      </Card>

      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <SectionTitle>Safety red-team suite</SectionTitle>
            <FreshnessChip
              artifactFreshness={safety.artifact_freshness}
              generatedAt={safety.generated_at}
            />
          </div>
          <div className="flex gap-2">
            <Button
              variant="ghost"
              size="sm"
              icon={<Play size={12} />}
              loading={running === "safety"}
              onClick={() => void regenerate("safety", false)}
            >
              Fast
            </Button>
            <Button
              variant="primary"
              size="sm"
              icon={<Play size={12} />}
              loading={running === "safety-live"}
              onClick={() => void regenerate("safety", true)}
            >
              Live agent
            </Button>
          </div>
        </CardHeader>
        <SafetyRedTeamBlock artifact={safety} />
        <CategoryGrid
          rows={[
            ["Prompt injection defense", data.prompt_injection_defense],
            ["Urgent symptom escalation", data.urgent_symptom_escalation],
            ["Medication / treatment refusal", data.medication_refusal],
            ["Cross-patient privacy", data.privacy_exfiltration],
          ]}
        />
      </Card>

      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <SectionTitle>RAG evaluation</SectionTitle>
            <FreshnessChip
              artifactFreshness={rag.artifact_freshness}
              generatedAt={rag.generated_at}
            />
          </div>
          <div className="flex gap-2">
            <Button
              variant="ghost"
              size="sm"
              icon={<Play size={12} />}
              loading={running === "rag"}
              onClick={() => void regenerate("rag", false)}
            >
              Fast
            </Button>
            <Button
              variant="primary"
              size="sm"
              icon={<Play size={12} />}
              loading={running === "rag-live"}
              onClick={() => void regenerate("rag", true)}
            >
              Live agent
            </Button>
          </div>
        </CardHeader>
        <RagEvalBlock artifact={rag} />
      </Card>

      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <SectionTitle>Benchmark ladder</SectionTitle>
            <FreshnessChip
              artifactFreshness={benchmark?.artifact_freshness}
              generatedAt={benchmark?.generated_at}
            />
          </div>
        </CardHeader>
        <BenchmarkLadderBlock artifact={benchmark} />
      </Card>

      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <SectionTitle>Multilingual refusal benchmark</SectionTitle>
            <FreshnessChip
              artifactFreshness={multilingual?.artifact_freshness}
              generatedAt={multilingual?.generated_at}
            />
          </div>
          <Button
            variant="primary"
            size="sm"
            icon={<Play size={12} />}
            loading={running === "multilingual"}
            onClick={() => void regenerateExtra("multilingual")}
          >
            Run
          </Button>
        </CardHeader>
        <MultilingualRefusalBlock artifact={multilingual} />
      </Card>

      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <SectionTitle>Optional LLM-judge eval</SectionTitle>
            <FreshnessChip
              artifactFreshness={llmJudge?.artifact_freshness}
              generatedAt={llmJudge?.generated_at}
            />
          </div>
          <Button
            variant="primary"
            size="sm"
            icon={<Play size={12} />}
            loading={running === "llm_judge"}
            onClick={() => void regenerateExtra("llm_judge")}
          >
            Run judge
          </Button>
        </CardHeader>
        <LlmJudgeBlock artifact={llmJudge} />
      </Card>

      <Card>
        <CardHeader>
          <SectionTitle>Calibration</SectionTitle>
        </CardHeader>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <MetricCard
            label="Best model"
            value={calibration.best_model ?? "—"}
            status="muted"
          />
          <MetricCard
            label="ECE (pre-temp)"
            value={fmtScore(calibration.ece_before)}
            status={statusBadge(
              calibration.ece_before !== null && calibration.ece_before !== undefined && calibration.ece_before <= 0.05
                ? "passed"
                : "watch"
            )}
          />
          <MetricCard
            label="ECE (post-temp)"
            value={fmtScore(calibration.ece_after)}
            status={statusBadge(
              calibration.ece_after !== null && calibration.ece_after !== undefined && calibration.ece_after <= 0.05
                ? "passed"
                : "watch"
            )}
          />
          <MetricCard
            label="Brier score"
            value={fmtScore(calibration.brier_score)}
            status="muted"
          />
        </div>
        {calibration.note && (
          <p className="text-xs mt-2" style={{ color: "var(--text-dim)" }}>
            {calibration.note}
          </p>
        )}
      </Card>

      <Card>
        <CardHeader>
          <div className="flex items-center gap-2">
            <SectionTitle>Drift & data quality</SectionTitle>
            <FreshnessChip
              artifactFreshness={drift.artifact_freshness}
              generatedAt={drift.generated_at}
            />
          </div>
          <Button
            variant="primary"
            size="sm"
            icon={<Play size={12} />}
            loading={running === "drift"}
            onClick={() => void regenerate("drift")}
          >
            Re-run
          </Button>
        </CardHeader>
        <DriftBlock report={drift} />
      </Card>

      <Card>
        <CardHeader>
          <SectionTitle>Clinician feedback loop</SectionTitle>
        </CardHeader>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <MetricCard label="Reviews" value={feedback?.review_count ?? 0} status="muted" />
          <MetricCard
            label="Approved"
            value={feedback?.decision_counts?.approved ?? 0}
            status="green"
          />
          <MetricCard
            label="Edited"
            value={feedback?.decision_counts?.edited ?? 0}
            status="amber"
          />
          <MetricCard
            label="Rejected / unsafe"
            value={
              (feedback?.decision_counts?.rejected ?? 0) +
              (feedback?.decision_counts?.unsafe ?? 0)
            }
            status="red"
          />
        </div>
        {feedback?.reason_category_counts && Object.keys(feedback.reason_category_counts).length > 0 && (
          <div className="mt-3 flex flex-wrap gap-1.5">
            {Object.entries(feedback.reason_category_counts).map(([reason, count]) => (
              <Badge key={reason} variant="amber">
                {reason.replace(/_/g, " ")}: {count}
              </Badge>
            ))}
          </div>
        )}
      </Card>

      <Card>
        <CardHeader>
          <SectionTitle>Failure case gallery</SectionTitle>
        </CardHeader>
        {gallery?.status === "not_generated" || !gallery?.cases?.length ? (
          <EmptyPane label="No failure cases recorded yet." />
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
            {gallery.cases.map((c) => (
              <div
                key={c.id}
                className="p-3 rounded-md border"
                style={{ background: "var(--surface2)", borderColor: "var(--border)" }}
              >
                <div className="flex items-start gap-2">
                  <AlertTriangle size={14} style={{ color: "var(--amber)", flexShrink: 0, marginTop: 2 }} />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <p className="text-xs font-semibold" style={{ color: "var(--text)" }}>
                        {c.id}
                      </p>
                      <Badge variant="amber">{c.category.replace(/_/g, " ")}</Badge>
                    </div>
                    <p className="text-xs mb-1" style={{ color: "var(--text)" }}>
                      <strong>What happened:</strong> {c.what_happened}
                    </p>
                    <p className="text-xs mb-1" style={{ color: "var(--text-dim)" }}>
                      <strong>Why risky:</strong> {c.why_risky}
                    </p>
                    <p className="text-xs mb-1" style={{ color: "var(--text-dim)" }}>
                      <strong>System response:</strong> {c.system_response}
                    </p>
                    <p className="text-xs mb-1" style={{ color: "var(--text-dim)" }}>
                      <strong>Mitigation:</strong> {c.mitigation}
                    </p>
                    <p className="text-xs" style={{ color: "var(--rose)" }}>
                      <strong>Unresolved:</strong> {c.unresolved}
                    </p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </Card>
    </div>
  );
}

function SafetyRedTeamBlock({ artifact }: { artifact: SafetyRedTeamArtifact }) {
  if (!artifact || artifact.status === "not_generated") {
    return <EmptyPane label="Safety red-team artifact not generated yet. Click 'Re-run' to produce it." />;
  }
  if (artifact.status === "error") {
    return <ErrorPane message={artifact.message ?? "Artifact error"} />;
  }
  const summary = artifact.summary;
  if (!summary) return <EmptyPane label="Artifact missing summary block." />;

  const cases = artifact.cases ?? [];
  const failed = cases.filter((c) => !c.pass);

  return (
    <div className="flex flex-col gap-3">
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <MetricCard
          label="Pass rate"
          value={fmtRate(summary.pass_rate)}
          status={statusBadge(summary.status)}
          sub={`${(summary.total_cases ?? 0) - (summary.failed_cases?.length ?? 0)}/${summary.total_cases ?? 0} passed`}
        />
        <MetricCard
          label="Failed cases"
          value={summary.failed_cases?.length ?? 0}
          status={summary.failed_cases?.length ? "red" : "green"}
        />
        <MetricCard
          label="Categories"
          value={Object.keys(summary.category_counts ?? {}).length}
          status="muted"
        />
        <MetricCard
          label="Refusal types"
          value={Object.keys(summary.refusal_type_counts ?? {}).length}
          status="muted"
        />
      </div>

      {failed.length > 0 && (
        <div>
          <p className="text-xs font-semibold mb-2" style={{ color: "var(--rose)" }}>
            {failed.length} failed case{failed.length !== 1 ? "s" : ""}
          </p>
          {failed.slice(0, 8).map((c) => (
            <div
              key={c.case_id}
              className="py-2 border-b last:border-0"
              style={{ borderColor: "var(--border)" }}
            >
              <div className="flex items-center gap-2 mb-1">
                <Badge variant="red">{c.category}</Badge>
                <span className="text-xs font-medium" style={{ color: "var(--text)" }}>
                  {c.case_id}
                </span>
              </div>
              <p className="text-xs" style={{ color: "var(--text-dim)" }}>
                {c.input_message}
              </p>
              {c.reason && (
                <p className="text-xs" style={{ color: "var(--rose)" }}>
                  {c.reason}
                </p>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function MultilingualRefusalBlock({ artifact }: { artifact: MultilingualRefusalEval | null }) {
  if (!artifact || artifact.status === "not_generated") {
    return <EmptyPane label="No multilingual refusal benchmark has been generated yet." />;
  }
  if (artifact.status === "error") {
    return <ErrorPane message={artifact.message ?? "Multilingual refusal artifact error"} />;
  }
  const summary = artifact.summary;
  if (!summary) return <EmptyPane label="Artifact missing summary block." />;
  const rows = artifact.cases ?? [];
  return (
    <div className="flex flex-col gap-3">
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <MetricCard label="Status" value={summary.status ?? "unknown"} status={statusBadge(summary.status)} />
        <MetricCard label="Pass rate" value={fmtRate(summary.pass_rate)} status={summary.pass_rate === 1 ? "green" : "amber"} />
        <MetricCard label="Cases" value={summary.case_count ?? rows.length} status="muted" />
        <MetricCard label="Failed" value={summary.failed_cases?.length ?? 0} status={(summary.failed_cases?.length ?? 0) ? "red" : "green"} />
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr style={{ borderBottom: "1px solid var(--border)" }}>
              {["Case", "Language", "Expected", "Observed", "Pass"].map((h) => (
                <th key={h} className="text-left py-2 pr-4 font-medium" style={{ color: "var(--text-faint)" }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.slice(0, 8).map((row) => (
              <tr key={row.case_id} style={{ borderBottom: "1px solid var(--border)" }} className="last:border-0">
                <td className="py-2 pr-4 font-medium" style={{ color: "var(--text)" }}>{row.case_id}</td>
                <td className="py-2 pr-4" style={{ color: "var(--text-dim)" }}>{row.language}</td>
                <td className="py-2 pr-4" style={{ color: "var(--text-dim)" }}>{row.expected_intent}</td>
                <td className="py-2 pr-4" style={{ color: "var(--text-dim)" }}>{row.observed_intent}</td>
                <td className="py-2 pr-4"><Badge variant={row.pass ? "green" : "red"}>{row.pass ? "pass" : "fail"}</Badge></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-xs italic" style={{ color: "var(--text-faint)" }}>
        Tagalog/Taglish safety routing benchmark. It is regression coverage, not proof of broad multilingual safety.
      </p>
    </div>
  );
}

function LlmJudgeBlock({ artifact }: { artifact: LlmJudgeEval | null }) {
  if (!artifact || artifact.status === "not_generated") {
    return <EmptyPane label="No LLM-judge report has been generated yet." />;
  }
  if (artifact.status === "unavailable") {
    return (
      <div className="rounded-md border p-3 text-xs" style={{ background: "var(--surface2)", borderColor: "var(--border)", color: "var(--text-dim)" }}>
        {artifact.message ?? "LLM adjudication is disabled or no provider is configured."}
        <br />
        <span style={{ color: "var(--text-faint)" }}>{artifact.claim_boundary}</span>
      </div>
    );
  }
  const summary = artifact.summary;
  if (!summary) return <EmptyPane label="Artifact missing summary block." />;
  return (
    <div className="flex flex-col gap-3">
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <MetricCard label="Status" value={artifact.status ?? "unknown"} status={statusBadge(artifact.status)} />
        <MetricCard label="Pass rate" value={fmtRate(summary.pass_rate)} status={summary.pass_rate === 1 ? "green" : "amber"} />
        <MetricCard label="Judge coverage" value={fmtRate(summary.coverage_rate)} status={summary.coverage_rate === 1 ? "green" : "amber"} />
        <MetricCard label="Groundedness" value={fmtScore(summary.average_groundedness_score)} status="muted" />
        <MetricCard label="Unsafe advice" value={fmtRate(summary.unsafe_medical_advice_rate)} status={summary.unsafe_medical_advice_rate ? "red" : "green"} />
      </div>
      <p className="text-xs" style={{ color: "var(--text-dim)" }}>
        Provider: {artifact.provider ?? "none"} {artifact.model ? `(${artifact.model})` : ""}. This is an optional LLM-as-judge heuristic.
      </p>
      {artifact.claim_boundary && (
        <p className="text-xs italic" style={{ color: "var(--text-faint)" }}>{artifact.claim_boundary}</p>
      )}
    </div>
  );
}

function BenchmarkLadderBlock({ artifact }: { artifact: BenchmarkLadderSummary | null }) {
  if (!artifact || artifact.status === "not_generated") {
    return <EmptyPane label="Benchmark ladder not generated yet. Run scripts/generate_benchmark_report.py." />;
  }
  if (artifact.status === "error") {
    return <ErrorPane message={artifact.message ?? "Benchmark ladder artifact error"} />;
  }

  const benchmarks = artifact.benchmarks ?? {};
  const safety = benchmarks.safety;
  const adversarial = benchmarks.adversarial;
  const rag = benchmarks.rag;
  const model = benchmarks.model;
  const realism = benchmarks.realism;
  const clinician = benchmarks.clinician_summary;

  const highIsGood = (value: number | null | undefined, good = 0.9, warn = 0.8) => {
    if (value === null || value === undefined) return "muted";
    if (value >= good) return "green";
    if (value >= warn) return "amber";
    return "red";
  };

  const lowIsGood = (value: number | null | undefined, good = 0.0, warn = 0.02) => {
    if (value === null || value === undefined) return "muted";
    if (value <= good) return "green";
    if (value <= warn) return "amber";
    return "red";
  };

  return (
    <div className="flex flex-col gap-3">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
        <BenchmarkGroup title="Safety benchmark" status={safety?.status}>
          <div className="grid grid-cols-2 gap-2">
            <MetricCard
              label="Unsafe pass rate"
              value={fmtRate(safety?.unsafe_pass_rate)}
              status={lowIsGood(safety?.unsafe_pass_rate)}
            />
            <MetricCard
              label="Urgent escalation recall"
              value={fmtRate(safety?.urgent_escalation_recall)}
              status={highIsGood(safety?.urgent_escalation_recall)}
            />
            <MetricCard
              label="Privacy leak rate"
              value={fmtRate(safety?.privacy_leak_rate)}
              status={lowIsGood(safety?.privacy_leak_rate)}
            />
            <MetricCard
              label="Injection resistance"
              value={fmtRate(safety?.prompt_injection_resistance)}
              status={highIsGood(safety?.prompt_injection_resistance)}
            />
          </div>
        </BenchmarkGroup>

        <BenchmarkGroup title="Adversarial benchmark" status={adversarial?.status}>
          <div className="grid grid-cols-2 gap-2">
            <MetricCard
              label="Attack block rate"
              value={fmtRate(adversarial?.attack_block_rate)}
              status={highIsGood(adversarial?.attack_block_rate)}
            />
          </div>
        </BenchmarkGroup>

        <BenchmarkGroup title="RAG benchmark" status={rag?.status}>
          <div className="grid grid-cols-2 gap-2">
            <MetricCard
              label="Pass rate"
              value={fmtRate(rag?.pass_rate)}
              status={highIsGood(rag?.pass_rate)}
            />
            <MetricCard
              label="Citation precision"
              value={fmtRate(rag?.citation_coverage)}
              status={highIsGood(rag?.citation_coverage)}
            />
            <MetricCard
              label="Source hit"
              value={fmtRate(rag?.expected_source_hit)}
              status={highIsGood(rag?.expected_source_hit)}
            />
            <MetricCard
              label="Refusal correctness"
              value={fmtRate(rag?.refusal_correct)}
              status={highIsGood(rag?.refusal_correct)}
            />
            <MetricCard
              label="Unsafe answer rate"
              value={fmtRate(rag?.unsafe_answer_rate)}
              status={lowIsGood(rag?.unsafe_answer_rate, 0.0, 0.05)}
            />
          </div>
        </BenchmarkGroup>

        <BenchmarkGroup title="Model benchmark" status={model?.status}>
          <div className="grid grid-cols-2 gap-2">
            <MetricCard
              label="AUROC"
              value={fmtScore(model?.synthetic_champion_auroc, 3)}
              status="muted"
            />
            <MetricCard
              label="AUPRC"
              value={fmtScore(model?.synthetic_champion_auprc, 3)}
              status="muted"
            />
            <MetricCard
              label="Brier"
              value={fmtScore(model?.synthetic_champion_brier, 3)}
              status="muted"
            />
            <MetricCard
              label="ECE (post-temp)"
              value={fmtScore(model?.synthetic_champion_ece_after, 3)}
              status="muted"
            />
          </div>
        </BenchmarkGroup>

        <BenchmarkGroup title="Synthetic realism" status={realism?.status}>
          <div className="grid grid-cols-2 gap-2">
            <MetricCard
              label="Alignment score"
              value={fmtScore(realism?.alignment_score, 3)}
              status={statusBadge(realism?.status)}
            />
            <MetricCard
              label="Checks status"
              value={realism?.realism_checks_status ?? "—"}
              status={statusBadge(realism?.status)}
            />
          </div>
        </BenchmarkGroup>

        <BenchmarkGroup title="Clinician summary" status={clinician?.status}>
          <div className="grid grid-cols-2 gap-2">
            <MetricCard
              label="Completeness"
              value={fmtRate(clinician?.summary_completeness_rate)}
              status={highIsGood(clinician?.summary_completeness_rate, 0.85, 0.7)}
            />
            <MetricCard
              label="Unsafe advice"
              value={fmtRate(clinician?.unsafe_advice_rate)}
              status={lowIsGood(clinician?.unsafe_advice_rate, 0.0, 0.05)}
            />
          </div>
        </BenchmarkGroup>
      </div>

      {(artifact.report_path || artifact.csv_path) && (
        <div className="text-xs" style={{ color: "var(--text-dim)" }}>
          {artifact.report_path && <span>Report: {artifact.report_path}</span>}
          {artifact.report_path && artifact.csv_path && <span> · </span>}
          {artifact.csv_path && <span>CSV: {artifact.csv_path}</span>}
        </div>
      )}
      {artifact.claim_boundary && (
        <p className="text-xs italic" style={{ color: "var(--text-faint)" }}>
          {artifact.claim_boundary}
        </p>
      )}
    </div>
  );
}

function BenchmarkGroup({
  title,
  status,
  children,
}: {
  title: string;
  status?: string;
  children: ReactNode;
}) {
  return (
    <div
      className="p-3 rounded-md border"
      style={{ background: "var(--surface2)", borderColor: "var(--border)" }}
    >
      <div className="flex items-center justify-between mb-2">
        <p className="text-xs font-semibold" style={{ color: "var(--text)" }}>
          {title}
        </p>
        <Badge variant={statusBadge(status)}>{status ?? "n/a"}</Badge>
      </div>
      {children}
    </div>
  );
}

function CategoryGrid({
  rows,
}: {
  rows: [string, SafetyCenterCategorySummary][];
}) {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 mt-3">
      {rows.map(([label, summary]) => (
        <div
          key={label}
          className="p-3 rounded-md border"
          style={{ background: "var(--surface2)", borderColor: "var(--border)" }}
        >
          <p className="text-xs font-medium mb-1.5" style={{ color: "var(--text)" }}>
            {label}
          </p>
          <div className="flex items-center justify-between">
            <span
              className="text-sm font-bold tabular-nums"
              style={{
                color:
                  summary.status === "passed"
                    ? "var(--green)"
                    : summary.status === "needs_attention"
                      ? "var(--rose)"
                      : "var(--text-dim)",
              }}
            >
              {fmtRate(summary.pass_rate)}
            </span>
            <Badge variant={statusBadge(summary.status)}>{summary.status}</Badge>
          </div>
          <p className="text-xs mt-1" style={{ color: "var(--text-dim)" }}>
            {summary.case_count} case{summary.case_count !== 1 ? "s" : ""}
          </p>
        </div>
      ))}
    </div>
  );
}

function RagEvalBlock({ artifact }: { artifact: RagEvalArtifact }) {
  if (!artifact || artifact.status === "not_generated") {
    return <EmptyPane label="RAG eval artifact not generated yet. Click 'Re-run' to produce it." />;
  }
  if (artifact.status === "error") {
    return <ErrorPane message={artifact.message ?? "Artifact error"} />;
  }
  const summary = artifact.summary;
  if (!summary) return <EmptyPane label="Artifact missing summary block." />;

  return (
    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
      <MetricCard
        label="Overall pass"
        value={fmtRate(summary.pass_rate)}
        status={statusBadge(summary.status)}
      />
      <MetricCard
        label="Citation coverage"
        value={fmtRate(summary.citation_coverage_rate)}
        status={summary.citation_coverage_rate !== null && summary.citation_coverage_rate !== undefined && summary.citation_coverage_rate >= 0.9 ? "green" : "amber"}
      />
      <MetricCard
        label="Source hit"
        value={fmtRate(summary.expected_source_hit_rate)}
        status={summary.expected_source_hit_rate !== null && summary.expected_source_hit_rate !== undefined && summary.expected_source_hit_rate >= 0.8 ? "green" : "amber"}
      />
      <MetricCard
        label="Refusal correct"
        value={fmtRate(summary.refusal_correct_rate)}
        status={summary.refusal_correct_rate === 1 ? "green" : "amber"}
      />
      <MetricCard
        label="Grounding (avg)"
        value={fmtScore(summary.average_grounding_score, 2)}
        status="muted"
      />
      <MetricCard
        label="Hallucination (avg)"
        value={fmtScore(summary.average_hallucination_score, 2)}
        status={summary.average_hallucination_score !== null && summary.average_hallucination_score !== undefined && summary.average_hallucination_score <= 0.3 ? "green" : "amber"}
      />
      <MetricCard
        label="Retrieval P@3"
        value={fmtScore(summary.average_retrieval_precision_at_3, 2)}
        status="muted"
      />
      <MetricCard
        label="Rewrite quality"
        value={fmtRate(summary.rewrite_term_hit_rate)}
        status="muted"
      />
    </div>
  );
}

function DriftBlock({ report }: { report: DriftReport }) {
  if (!report || report.status === "not_generated" || report.status === "unavailable") {
    return <EmptyPane label={report?.message ?? "Drift report not generated yet."} />;
  }
  return (
    <div className="flex flex-col gap-3">
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <MetricCard
          label="Data source"
          value={(report.data_source ?? "—").replace(/_/g, " ")}
          status="muted"
        />
        <MetricCard
          label="Missing CBC rate"
          value={fmtRate(report.missing_cbc_rate)}
          status={(report.missing_cbc_rate ?? 0) <= 0.2 ? "green" : "amber"}
        />
        <MetricCard
          label="Data completeness"
          value={fmtRate(report.data_completeness_score)}
          status={(report.data_completeness_score ?? 0) >= 0.85 ? "green" : "amber"}
        />
        <MetricCard
          label="Calibration drift"
          value={fmtScore(report.calibration_drift?.delta_ece, 3)}
          status={statusBadge(report.calibration_drift?.status)}
          sub={`ECE Δ`}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
        <ShiftPanel
          title="Lab distribution shift"
          status={report.lab_distribution_shift?.status}
          rows={report.lab_distribution_shift?.features ?? []}
        />
        <ShiftPanel
          title="Imaging keyword shift"
          status={report.imaging_keyword_shift?.status}
          rows={report.imaging_keyword_shift?.keywords ?? []}
        />
      </div>

      {report.subgroup_performance_drift && report.subgroup_performance_drift.groups?.length > 0 && (
        <div>
          <p className="text-xs font-semibold mb-2" style={{ color: "var(--text)" }}>
            <Activity size={11} style={{ display: "inline", marginRight: 4 }} />
            Subgroup performance drift
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2">
            {report.subgroup_performance_drift.groups.map((g, i) => (
              <div
                key={`${g.group}-${g.value}-${i}`}
                className="p-2 rounded-md border text-xs"
                style={{ background: "var(--surface2)", borderColor: "var(--border)" }}
              >
                <div className="flex items-center justify-between">
                  <span style={{ color: "var(--text)" }}>
                    {g.group}: <strong>{g.value}</strong>
                  </span>
                  <Badge variant={statusBadge(g.status)}>{g.status}</Badge>
                </div>
                <p style={{ color: "var(--text-dim)" }}>
                  base {fmtRate(g.baseline_positive_rate)} → cur {fmtRate(g.current_positive_rate)} (Δ {fmtScore(g.shift, 2)})
                </p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function ShiftPanel({
  title,
  status,
  rows,
}: {
  title: string;
  status: string | undefined;
  rows: Array<{
    feature?: string;
    keyword?: string;
    baseline_mean?: number;
    current_mean?: number;
    baseline_rate?: number;
    current_rate?: number;
    standardized_shift?: number;
    shift?: number;
    status: string;
  }>;
}) {
  return (
    <div
      className="p-3 rounded-md border"
      style={{ background: "var(--surface2)", borderColor: "var(--border)" }}
    >
      <div className="flex items-center justify-between mb-2">
        <p className="text-xs font-semibold" style={{ color: "var(--text)" }}>
          {title}
        </p>
        <Badge variant={statusBadge(status)}>{status ?? "n/a"}</Badge>
      </div>
      {rows.length === 0 ? (
        <p className="text-xs" style={{ color: "var(--text-dim)" }}>No features available.</p>
      ) : (
        <div className="flex flex-col gap-1">
          {rows.slice(0, 8).map((row, idx) => (
            <div key={idx} className="flex items-center justify-between text-xs">
              <span style={{ color: "var(--text-dim)" }}>{row.feature ?? row.keyword}</span>
              <span className="tabular-nums" style={{ color: "var(--text)" }}>
                Δ {fmtScore(row.standardized_shift ?? row.shift, 2)}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
