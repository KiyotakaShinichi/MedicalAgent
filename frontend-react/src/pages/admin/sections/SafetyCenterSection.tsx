import type { ComponentProps, ReactNode } from "react";
import { Play, ShieldCheck, X } from "lucide-react";
import { Button } from "../../../components/ui/Button";
import { Badge } from "../../../components/ui/Badge";
import { Card, CardHeader, SectionTitle } from "../../../components/ui/Card";
import { LoadingPane, EmptyPane, ErrorPane } from "../../../components/ui/Spinner";
import { FreshnessChip } from "../../../components/ui/FreshnessChip";
import { AdversarialGeneralizationBlock } from "./safety/AdversarialGeneralizationBlock";
import { BenchmarkLadderBlock } from "./safety/BenchmarkLadderBlock";
import { CalibrationBlock } from "./safety/CalibrationBlock";
import { CategoryGrid } from "./safety/CategoryGrid";
import { ClinicianFeedbackBlock } from "./safety/ClinicianFeedbackBlock";
import { DriftBlock } from "./safety/DriftBlock";
import { FailureCaseGallery } from "./safety/FailureCaseGallery";
import { LlmJudgeBlock } from "./safety/LlmJudgeBlock";
import { MultilingualRefusalBlock } from "./safety/MultilingualRefusalBlock";
import { RagEvalBlock } from "./safety/RagEvalBlock";
import { SafetyRedTeamBlock } from "./safety/SafetyRedTeamBlock";
import { statusBadge, readString } from "./safety/safetyFormat";
import { useSafetyCenter } from "./safety/useSafetyCenter";

/**
 * Derived from FreshnessChip rather than redeclared, so the several artifact
 * freshness shapes the API returns stay assignable without a cast.
 */
type FreshnessInput = ComponentProps<typeof FreshnessChip>["artifactFreshness"];

/**
 * Card header carrying a title, an artifact-freshness chip, and the actions
 * that regenerate that artifact. Every panel on this page shares this shape,
 * so it is expressed once rather than twelve times.
 */
function ArtifactPanel({
  title,
  freshness,
  generatedAt,
  actions,
  children,
}: {
  title: string;
  freshness?: FreshnessInput;
  generatedAt?: string | null;
  actions?: ReactNode;
  children: ReactNode;
}) {
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <SectionTitle>{title}</SectionTitle>
          <FreshnessChip artifactFreshness={freshness} generatedAt={generatedAt} />
        </div>
        {actions && <div className="flex gap-2">{actions}</div>}
      </CardHeader>
      {children}
    </Card>
  );
}

/**
 * Non-fatal banner for a failed regeneration.
 *
 * Previously a failed re-run wrote to the same state as the fatal load error,
 * which was only rendered in the `status === "error"` branch — so the message
 * was set but never shown and the button simply stopped spinning. Surfacing it
 * separately keeps the loaded artifacts visible while still reporting that the
 * requested run did not happen.
 */
function ActionErrorBanner({ message, onDismiss }: { message: string; onDismiss: () => void }) {
  return (
    <div
      role="alert"
      className="flex items-start gap-2 p-3 rounded-md border text-xs"
      style={{ background: "rgba(244, 63, 94, 0.06)", borderColor: "rgba(244, 63, 94, 0.28)" }}
    >
      <span className="flex-1" style={{ color: "var(--text)" }}>
        <strong>Run failed.</strong> {message}
      </span>
      <button
        type="button"
        onClick={onDismiss}
        aria-label="Dismiss run error"
        className="shrink-0"
        style={{ color: "var(--text-dim)", background: "none", border: "none", cursor: "pointer" }}
      >
        <X size={14} aria-hidden="true" />
      </button>
    </div>
  );
}

/**
 * Safety & Evaluation Center.
 *
 * This component is composition only: `useSafetyCenter` owns all data and
 * actions, and each panel body is an independently testable block under
 * `./safety/`. Keeping it declarative is what makes the twelve panels
 * reviewable — the previous single-file version made it hard to tell which
 * artifact drove which board.
 */
export function SafetyCenterSection() {
  const {
    data,
    multilingual,
    llmJudge,
    status,
    error,
    actionError,
    dismissActionError,
    running,
    regenerate,
    runExtraEval,
  } = useSafetyCenter();

  // "idle" is grouped with "loading" so a pre-fetch frame never renders as an
  // empty result. On a safety surface, "we have not asked yet" and "there is
  // nothing to report" must not look the same.
  if ((status === "loading" || status === "idle") && !data) {
    return <LoadingPane label="Loading safety & evaluation center..." />;
  }
  if (status === "error") return <ErrorPane message={error ?? "Failed to load safety center"} />;
  if (!data) return <EmptyPane label="No safety center data" />;

  const { safety_red_team: safety, rag_eval: rag, drift_report: drift, benchmark_ladder: benchmark } = data;

  return (
    <div className="flex flex-col gap-4">
      {actionError && <ActionErrorBanner message={actionError} onDismiss={dismissActionError} />}

      <Card>
        <p className="text-xs" style={{ color: "var(--text-dim)" }}>
          <ShieldCheck size={12} aria-hidden="true" style={{ display: "inline", marginRight: 6 }} />
          {data.safety_note}
        </p>
      </Card>

      <ArtifactPanel
        title="Safety red-team suite"
        freshness={safety.artifact_freshness}
        generatedAt={safety.generated_at}
        actions={
          <>
            <Button variant="ghost" size="sm" icon={<Play size={12} />} loading={running === "safety"} onClick={() => void regenerate("safety", false)} aria-label="Run safety red-team suite (fast)">
              Fast
            </Button>
            <Button variant="primary" size="sm" icon={<Play size={12} />} loading={running === "safety-live"} onClick={() => void regenerate("safety", true)} aria-label="Run safety red-team suite (live agent)">
              Live agent
            </Button>
          </>
        }
      >
        <SafetyRedTeamBlock artifact={safety} />
        <CategoryGrid
          rows={[
            { label: "Prompt injection defense", summary: data.prompt_injection_defense },
            { label: "Urgent symptom escalation", summary: data.urgent_symptom_escalation },
            { label: "Medication / treatment refusal", summary: data.medication_refusal },
            { label: "Cross-patient privacy", summary: data.privacy_exfiltration },
          ]}
        />
      </ArtifactPanel>

      <ArtifactPanel
        title="RAG evaluation"
        freshness={rag.artifact_freshness}
        generatedAt={rag.generated_at}
        actions={
          <>
            <Button variant="ghost" size="sm" icon={<Play size={12} />} loading={running === "rag"} onClick={() => void regenerate("rag", false)} aria-label="Run RAG evaluation (fast)">
              Fast
            </Button>
            <Button variant="primary" size="sm" icon={<Play size={12} />} loading={running === "rag-live"} onClick={() => void regenerate("rag", true)} aria-label="Run RAG evaluation (live agent)">
              Live agent
            </Button>
          </>
        }
      >
        <RagEvalBlock artifact={rag} />
      </ArtifactPanel>

      <ArtifactPanel
        title="Benchmark ladder"
        freshness={benchmark?.artifact_freshness}
        generatedAt={benchmark?.generated_at}
      >
        <BenchmarkLadderBlock artifact={benchmark} />
      </ArtifactPanel>

      <Card>
        <CardHeader>
          <SectionTitle>Adversarial Generalization</SectionTitle>
          <Badge variant={statusBadge(readString(data.adversarial_generalization_v2, "status"))}>
            {readString(data.adversarial_generalization_v2, "status") ?? "n/a"}
          </Badge>
        </CardHeader>
        <AdversarialGeneralizationBlock artifact={data.adversarial_generalization_v2} />
      </Card>

      <ArtifactPanel
        title="Multilingual refusal benchmark"
        freshness={multilingual?.artifact_freshness}
        generatedAt={multilingual?.generated_at}
        actions={
          <Button variant="primary" size="sm" icon={<Play size={12} />} loading={running === "multilingual"} onClick={() => void runExtraEval("multilingual")} aria-label="Run multilingual refusal benchmark">
            Run
          </Button>
        }
      >
        <MultilingualRefusalBlock artifact={multilingual} />
      </ArtifactPanel>

      <ArtifactPanel
        title="Optional LLM-judge eval"
        freshness={llmJudge?.artifact_freshness}
        generatedAt={llmJudge?.generated_at}
        actions={
          <Button variant="primary" size="sm" icon={<Play size={12} />} loading={running === "llm_judge"} onClick={() => void runExtraEval("llm_judge")} aria-label="Run optional LLM-judge evaluation">
            Run judge
          </Button>
        }
      >
        <LlmJudgeBlock artifact={llmJudge} />
      </ArtifactPanel>

      <Card>
        <CardHeader>
          <SectionTitle>Calibration</SectionTitle>
        </CardHeader>
        <CalibrationBlock calibration={data.calibration_metrics} />
      </Card>

      <ArtifactPanel
        title="Drift & data quality"
        freshness={drift.artifact_freshness}
        generatedAt={drift.generated_at}
        actions={
          <Button variant="primary" size="sm" icon={<Play size={12} />} loading={running === "drift"} onClick={() => void regenerate("drift")} aria-label="Re-run drift and data quality report">
            Re-run
          </Button>
        }
      >
        <DriftBlock report={drift} />
      </ArtifactPanel>

      <Card>
        <CardHeader>
          <SectionTitle>Clinician feedback loop</SectionTitle>
        </CardHeader>
        <ClinicianFeedbackBlock feedback={data.clinician_feedback} />
      </Card>

      <Card>
        <CardHeader>
          <SectionTitle>Failure case gallery</SectionTitle>
        </CardHeader>
        <FailureCaseGallery gallery={data.failure_case_gallery} />
      </Card>
    </div>
  );
}
