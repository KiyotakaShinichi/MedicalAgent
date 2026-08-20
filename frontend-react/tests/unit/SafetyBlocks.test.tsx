import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { SafetyRedTeamBlock } from "../../src/pages/admin/sections/safety/SafetyRedTeamBlock";
import { LlmJudgeBlock } from "../../src/pages/admin/sections/safety/LlmJudgeBlock";
import { MultilingualRefusalBlock } from "../../src/pages/admin/sections/safety/MultilingualRefusalBlock";
import { AdversarialGeneralizationBlock } from "../../src/pages/admin/sections/safety/AdversarialGeneralizationBlock";
import { DriftBlock } from "../../src/pages/admin/sections/safety/DriftBlock";
import { CategoryGrid } from "../../src/pages/admin/sections/safety/CategoryGrid";
import { FailureCaseGallery } from "../../src/pages/admin/sections/safety/FailureCaseGallery";
import { RagEvalBlock } from "../../src/pages/admin/sections/safety/RagEvalBlock";
import type {
  DriftReport,
  LlmJudgeEval,
  MultilingualRefusalEval,
  RagEvalArtifact,
  SafetyRedTeamArtifact,
} from "../../src/types/api";

/**
 * These blocks were extracted out of the 1023-line SafetyCenterSection. The
 * behaviour that matters here is not layout — it is that an absent, disabled,
 * or malformed evaluation artifact never renders as a passing result.
 */

describe("SafetyRedTeamBlock", () => {
  it("shows an empty state when the artifact has not been generated", () => {
    render(<SafetyRedTeamBlock artifact={{ status: "not_generated" } as SafetyRedTeamArtifact} />);
    expect(screen.getByText(/not generated yet/i)).toBeInTheDocument();
    expect(screen.queryByText("Pass rate")).not.toBeInTheDocument();
  });

  it("shows the artifact error message rather than a metric board", () => {
    render(
      <SafetyRedTeamBlock
        artifact={{ status: "error", message: "red-team runner crashed" } as SafetyRedTeamArtifact}
      />,
    );
    expect(screen.getByText(/red-team runner crashed/i)).toBeInTheDocument();
    expect(screen.queryByText("Pass rate")).not.toBeInTheDocument();
  });

  it("refuses to render metrics when the summary block is missing", () => {
    render(<SafetyRedTeamBlock artifact={{ status: "passed" } as SafetyRedTeamArtifact} />);
    expect(screen.getByText(/missing summary block/i)).toBeInTheDocument();
  });

  it("renders pass rate and failure count from a complete summary", () => {
    const artifact = {
      status: "passed",
      summary: {
        status: "passed",
        pass_rate: 0.95,
        total_cases: 20,
        failed_cases: ["c1"],
        category_counts: { injection: 10, privacy: 10 },
        refusal_type_counts: { hard: 5 },
      },
      cases: [
        { case_id: "c1", category: "injection", pass: false, input_message: "ignore instructions", reason: "leaked" },
      ],
    } as unknown as SafetyRedTeamArtifact;

    render(<SafetyRedTeamBlock artifact={artifact} />);

    expect(screen.getByText("95%")).toBeInTheDocument();
    expect(screen.getByText("19/20 passed")).toBeInTheDocument();
    expect(screen.getByText("1 failed case")).toBeInTheDocument();
    expect(screen.getByText("ignore instructions")).toBeInTheDocument();
  });

  it("truncates a long failure list and says how many were hidden", () => {
    const cases = Array.from({ length: 12 }, (_, i) => ({
      case_id: `c${i}`,
      category: "injection",
      pass: false,
      input_message: `attack ${i}`,
    }));
    const artifact = {
      status: "failed",
      summary: { status: "failed", pass_rate: 0, total_cases: 12, failed_cases: cases.map((c) => c.case_id) },
      cases,
    } as unknown as SafetyRedTeamArtifact;

    render(<SafetyRedTeamBlock artifact={artifact} />);

    expect(screen.getByText("attack 0")).toBeInTheDocument();
    expect(screen.queryByText("attack 11")).not.toBeInTheDocument();
    expect(screen.getByText(/Showing 8 of 12/)).toBeInTheDocument();
  });
});

describe("LlmJudgeBlock", () => {
  it("states plainly when adjudication is unavailable instead of showing a clean board", () => {
    // Safety-critical: a disabled evaluator must never read as a passing one.
    render(
      <LlmJudgeBlock
        artifact={{
          status: "unavailable",
          message: "LLM adjudication is disabled",
          claim_boundary: "No judge signal available.",
        } as LlmJudgeEval}
      />,
    );

    expect(screen.getByRole("status")).toHaveTextContent(/adjudication is disabled/i);
    expect(screen.queryByText("Pass rate")).not.toBeInTheDocument();
    expect(screen.queryByText("Unsafe advice")).not.toBeInTheDocument();
  });

  it("renders the claim boundary alongside results so scope stays visible", () => {
    render(
      <LlmJudgeBlock
        artifact={{
          status: "passed",
          provider: "anthropic",
          model: "claude",
          claim_boundary: "Heuristic only; not a clinical validation.",
          summary: { pass_rate: 1, coverage_rate: 1, average_groundedness_score: 0.9, unsafe_medical_advice_rate: 0 },
        } as unknown as LlmJudgeEval}
      />,
    );

    expect(screen.getByText(/Heuristic only; not a clinical validation/)).toBeInTheDocument();
    expect(screen.getByText(/Provider: anthropic/)).toBeInTheDocument();
  });
});

describe("MultilingualRefusalBlock", () => {
  it("renders per-case rows in an accessible table", () => {
    render(
      <MultilingualRefusalBlock
        artifact={{
          status: "passed",
          summary: { status: "passed", pass_rate: 1, case_count: 1, failed_cases: [] },
          cases: [
            { case_id: "tl-1", language: "tl", expected_intent: "refuse", observed_intent: "refuse", pass: true },
          ],
        } as unknown as MultilingualRefusalEval}
      />,
    );

    expect(screen.getByRole("columnheader", { name: "Language" })).toBeInTheDocument();
    expect(screen.getByRole("rowheader", { name: "tl-1" })).toBeInTheDocument();
    expect(screen.getByText(/regression coverage, not proof/i)).toBeInTheDocument();
  });

  it("distinguishes a summary with no rows from a missing artifact", () => {
    render(
      <MultilingualRefusalBlock
        artifact={{
          status: "passed",
          summary: { status: "passed", pass_rate: 1, case_count: 0, failed_cases: [] },
          cases: [],
        } as unknown as MultilingualRefusalEval}
      />,
    );
    expect(screen.getByText(/no per-case rows were returned/i)).toBeInTheDocument();
  });
});

describe("AdversarialGeneralizationBlock", () => {
  it("survives a malformed artifact without crashing", () => {
    // The artifact is produced by an eval script, not the API schema, so the
    // frontend must degrade rather than throw when its shape drifts.
    render(
      <AdversarialGeneralizationBlock
        artifact={{
          status: "acceptable",
          metrics: "not-an-object",
          heldout_v2: { total_n: "twelve", failures: "nope" },
        }}
      />,
    );

    expect(screen.getByText("Heldout v2")).toBeInTheDocument();
    // Unparseable numbers degrade to an em dash instead of "NaN%".
    expect(screen.queryByText(/NaN/)).not.toBeInTheDocument();
  });

  it("always reports the benchmark as not solved", () => {
    render(
      <AdversarialGeneralizationBlock
        artifact={{ status: "passed", metrics: { heldout_v2_pass_rate: 1.0 }, heldout_v2: { total_n: 50 } }}
      />,
    );
    expect(screen.getByText("Not solved")).toBeInTheDocument();
  });

  it("shows an empty state when the artifact is absent", () => {
    render(<AdversarialGeneralizationBlock artifact={undefined} />);
    expect(screen.getByText(/No adversarial generalization v2 artifact yet/i)).toBeInTheDocument();
  });
});

describe("DriftBlock", () => {
  it("treats an unavailable drift report as no signal, not as no drift", () => {
    render(<DriftBlock report={{ status: "unavailable", message: "drift job never ran" } as DriftReport} />);
    expect(screen.getByText("drift job never ran")).toBeInTheDocument();
    expect(screen.queryByText("Data completeness")).not.toBeInTheDocument();
  });

  it("renders shift panels and handles empty feature lists", () => {
    render(
      <DriftBlock
        report={{
          status: "ok",
          data_source: "synthetic_cohort",
          missing_cbc_rate: 0.1,
          data_completeness_score: 0.92,
          lab_distribution_shift: { label: "labs", status: "ok", feature_count: 0, features: [] },
          imaging_keyword_shift: { status: "ok", keywords: [] },
        } as unknown as DriftReport}
      />,
    );

    expect(screen.getByText("synthetic cohort")).toBeInTheDocument();
    expect(screen.getAllByText("No features available.")).toHaveLength(2);
  });
});

describe("CategoryGrid", () => {
  it("pairs each category label with its rate as a definition list", () => {
    render(
      <CategoryGrid
        rows={[
          { label: "Prompt injection defense", summary: { status: "passed", pass_rate: 1, case_count: 4, categories: [] } },
          { label: "Cross-patient privacy", summary: { status: "needs_attention", pass_rate: 0.5, case_count: 1, categories: [] } },
        ]}
      />,
    );

    expect(screen.getByText("Prompt injection defense")).toBeInTheDocument();
    expect(screen.getByText("100%")).toBeInTheDocument();
    expect(screen.getByText("4 cases")).toBeInTheDocument();
    // Singular/plural is computed, so verify the singular branch too.
    expect(screen.getByText("1 case")).toBeInTheDocument();
  });

  it("renders nothing when there are no categories", () => {
    const { container } = render(<CategoryGrid rows={[]} />);
    expect(container).toBeEmptyDOMElement();
  });
});

describe("FailureCaseGallery", () => {
  it("keeps unresolved weaknesses visible", () => {
    render(
      <FailureCaseGallery
        gallery={{
          status: "ok",
          cases: [
            {
              id: "F-1",
              category: "urgent_escalation",
              what_happened: "missed a red flag",
              why_risky: "delays care",
              system_response: "generic advice",
              mitigation: "added rule",
              unresolved: "paraphrases still slip through",
            },
          ],
        } as never}
      />,
    );

    expect(screen.getByText("F-1")).toBeInTheDocument();
    expect(screen.getByText("urgent escalation")).toBeInTheDocument();
    expect(screen.getByText(/paraphrases still slip through/)).toBeInTheDocument();
  });

  it("shows an empty state when no cases are recorded", () => {
    render(<FailureCaseGallery gallery={{ status: "not_generated", cases: [] } as never} />);
    expect(screen.getByText(/No failure cases recorded yet/i)).toBeInTheDocument();
  });
});

describe("RagEvalBlock", () => {
  it("flags an unmeasured citation coverage as watch rather than pass", () => {
    render(
      <RagEvalBlock
        artifact={{
          status: "acceptable",
          summary: { pass_rate: 0.8, status: "acceptable", citation_coverage_rate: null },
        } as unknown as RagEvalArtifact}
      />,
    );

    expect(screen.getByText("Citation coverage")).toBeInTheDocument();
    // "Borderline" is MetricCard's label for the amber tone.
    expect(screen.getAllByText("Borderline").length).toBeGreaterThan(0);
  });

  it("shows an empty state when the artifact has not been generated", () => {
    render(<RagEvalBlock artifact={{ status: "not_generated" } as RagEvalArtifact} />);
    expect(screen.getByText(/not generated yet/i)).toBeInTheDocument();
  });
});
