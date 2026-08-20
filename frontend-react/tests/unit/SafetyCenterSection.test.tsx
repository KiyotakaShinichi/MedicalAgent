import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

vi.mock("../../src/api/client", () => ({
  API_BASE: "http://test.local",
  getSafetyCenter: vi.fn(),
  getMultilingualRefusalEval: vi.fn(),
  getLlmJudgeEval: vi.fn(),
  runSafetyRedTeam: vi.fn(),
  runRagEvalArtifact: vi.fn(),
  runDriftReport: vi.fn(),
  runMultilingualRefusalEval: vi.fn(),
  runLlmJudgeEval: vi.fn(),
}));

import * as api from "../../src/api/client";
import { SafetyCenterSection } from "../../src/pages/admin/sections/SafetyCenterSection";
import { resetTelemetrySinks } from "../../src/lib/telemetry";

const mocked = vi.mocked(api);

const emptyCategory = { status: "passed", pass_rate: 1, case_count: 2, categories: [] };

/** Minimal payload that exercises every panel's "artifact absent" branch. */
const CENTER = {
  generated_at: "2026-01-01T00:00:00Z",
  safety_note: "Synthetic data only. Not a diagnostic device.",
  safety_red_team: { status: "not_generated" },
  prompt_injection_defense: emptyCategory,
  urgent_symptom_escalation: emptyCategory,
  medication_refusal: emptyCategory,
  privacy_exfiltration: emptyCategory,
  rag_eval: { status: "not_generated" },
  rag_trace_summary: null,
  benchmark_ladder: { status: "not_generated" },
  adversarial_generalization_v2: { status: "not_generated" },
  calibration_metrics: { status: "ok", best_model: "logreg", ece_before: 0.08, ece_after: 0.03, brier_score: 0.12 },
  drift_report: { status: "not_generated" },
  data_quality: null,
  clinician_feedback: { review_count: 0, decision_counts: {}, reason_category_counts: {}, review_target_counts: {}, average_explanation_quality_score: null, average_model_usefulness_score: null },
  failure_case_gallery: { status: "not_generated", cases: [] },
};

describe("SafetyCenterSection", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    resetTelemetrySinks();
    vi.spyOn(console, "error").mockImplementation(() => {});
    vi.spyOn(console, "warn").mockImplementation(() => {});
    mocked.getMultilingualRefusalEval.mockResolvedValue({ status: "not_generated" } as never);
    mocked.getLlmJudgeEval.mockResolvedValue({ status: "not_generated" } as never);
  });

  afterEach(() => {
    resetTelemetrySinks();
    vi.restoreAllMocks();
  });

  it("shows a loading pane on the very first frame, never an empty state", () => {
    // Regression: the fetch is deferred to a macrotask, so the first render
    // happened while status was still "idle" and fell through to the "No
    // safety center data" empty pane — reading as "nothing to report" when the
    // request had not yet been made.
    mocked.getSafetyCenter.mockReturnValue(new Promise(() => {}) as never);
    render(<SafetyCenterSection />);
    expect(screen.getByText(/Loading safety & evaluation center/i)).toBeInTheDocument();
    expect(screen.queryByText(/No safety center data/i)).not.toBeInTheDocument();
  });

  it("shows an error pane when the payload fails to load", async () => {
    mocked.getSafetyCenter.mockRejectedValue(new Error("safety center unreachable"));
    render(<SafetyCenterSection />);
    await waitFor(() => expect(screen.getByText("safety center unreachable")).toBeInTheDocument());
  });

  it("renders every panel with its empty state when no artifacts exist", async () => {
    mocked.getSafetyCenter.mockResolvedValue(CENTER as never);
    render(<SafetyCenterSection />);

    await waitFor(() => expect(screen.getByText(/Synthetic data only/)).toBeInTheDocument());

    for (const title of [
      "Safety red-team suite",
      "RAG evaluation",
      "Benchmark ladder",
      "Adversarial Generalization",
      "Multilingual refusal benchmark",
      "Optional LLM-judge eval",
      "Calibration",
      "Drift & data quality",
      "Clinician feedback loop",
      "Failure case gallery",
    ]) {
      expect(screen.getByText(title)).toBeInTheDocument();
    }

    // Absent artifacts must not produce a metric board.
    expect(screen.queryByText("Pass rate")).not.toBeInTheDocument();
  });

  it("gives every run control a distinct accessible name", async () => {
    // "Fast" and "Live agent" appear on two different panels; without an
    // explicit label a screen-reader user cannot tell which artifact a button
    // regenerates.
    mocked.getSafetyCenter.mockResolvedValue(CENTER as never);
    render(<SafetyCenterSection />);
    await waitFor(() => expect(screen.getByText("Safety red-team suite")).toBeInTheDocument());

    const names = screen
      .getAllByRole("button")
      .map((b) => b.getAttribute("aria-label") ?? b.textContent?.trim() ?? "");
    expect(new Set(names).size).toBe(names.length);
  });

  it("surfaces a failed re-run in a dismissible alert while keeping the data visible", async () => {
    // Regression: this banner did not exist. A failed regeneration wrote to the
    // fatal-error state, which is only rendered in the error branch, so the
    // failure was invisible to the operator.
    const user = userEvent.setup();
    mocked.getSafetyCenter.mockResolvedValue(CENTER as never);
    mocked.runDriftReport.mockRejectedValue(new Error("drift job failed to start"));

    render(<SafetyCenterSection />);
    await waitFor(() => expect(screen.getByText("Drift & data quality")).toBeInTheDocument());

    await user.click(screen.getByRole("button", { name: /Re-run drift and data quality report/i }));

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("drift job failed to start");
    // The loaded content is still on screen.
    expect(screen.getByText("Drift & data quality")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /dismiss run error/i }));
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("reloads the payload after a successful regeneration", async () => {
    const user = userEvent.setup();
    mocked.getSafetyCenter.mockResolvedValue(CENTER as never);
    mocked.runSafetyRedTeam.mockResolvedValue({ ok: true } as never);

    render(<SafetyCenterSection />);
    await waitFor(() => expect(screen.getByText("Safety red-team suite")).toBeInTheDocument());
    expect(mocked.getSafetyCenter).toHaveBeenCalledTimes(1);

    await user.click(screen.getByRole("button", { name: /Run safety red-team suite \(live agent\)/i }));

    await waitFor(() => expect(mocked.getSafetyCenter).toHaveBeenCalledTimes(2));
    expect(mocked.runSafetyRedTeam).toHaveBeenCalledWith(true);
  });
});
