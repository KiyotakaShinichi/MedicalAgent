import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import {
  CandidateComparisonPanel,
  FullFeatureGroupAblationPanel,
  NoiseEvalPanel,
  PredictionErrorPanel,
  TemporalEvalPanel,
} from "../../src/pages/admin/sections/MleEvidencePanels";

describe("MLE evidence panels", () => {
  it("keeps synthetic candidate decisions and claim boundaries visible", () => {
    render(<CandidateComparisonPanel data={{
      current: { patient_level_roc_auc: 0.8, realism_status: "synthetic_only", realism_alignment_score: 0.4 },
      candidate: { patient_level_roc_auc: 0.82, realism_status: "ab_test_only", realism_alignment_score: 0.5 },
      recommendation: {
        decision: "hold_candidate",
        auc_delta: 0.02,
        realism_delta: 0.1,
        rationale: "No external validation.",
      },
      claim_boundary: "Synthetic engineering comparison, not clinical evidence.",
    } as never} />);

    expect(screen.getByText("hold candidate")).toBeInTheDocument();
    expect(screen.getByText(/not clinical evidence/i)).toBeInTheDocument();
    expect(screen.getByText(/No external validation/i)).toBeInTheDocument();
  });

  it("renders clean and degraded noise scenarios without hiding the worst mode", () => {
    render(<NoiseEvalPanel data={{
      clean_baseline: { auroc: 0.8, brier_score: 0.2, sensitivity: 0.75, pr_auc: 0.7 },
      noise_results: [{ mode: "missing_imaging", auroc: 0.7, auroc_delta: -0.1, sensitivity: 0.6, sensitivity_delta: -0.15, status: "degraded" }],
      summary: { worst_mode: "missing_imaging", max_auroc_drop: 0.1 },
      claim_boundary: "Synthetic-only robustness test.",
    } as never} />);
    expect(screen.getAllByText("missing imaging").length).toBeGreaterThan(0);
    expect(screen.getByText("degraded")).toBeInTheDocument();
    expect(screen.getByText(/Synthetic-only robustness/i)).toBeInTheDocument();
  });

  it("shows patient-level temporal splits and their stated limitation", () => {
    const split = { auroc: 0.78, brier_score: 0.21, sensitivity: 0.7, n_train: 60, n_eval: 20 };
    render(<TemporalEvalPanel data={{
      temporal_split: split,
      cycle_split: split,
      random_split_baseline: split,
      generalization_gap: { temporal_auroc_gap: 0.04, cycle_auroc_gap: 0.03 },
      interpretation: "Random rows can overstate performance.",
      claim_boundary: "Synthetic temporal evidence only.",
    } as never} />);
    expect(screen.getByText("Patient timeline split")).toBeInTheDocument();
    expect(screen.getByText(/overstate performance/i)).toBeInTheDocument();
    expect(screen.getByText(/Synthetic temporal evidence/i)).toBeInTheDocument();
  });

  it("shows feature-group performance without promoting contextual features", () => {
    render(<FullFeatureGroupAblationPanel data={{
      feature_groups: {
        clinical_only: { modalities: ["labs"], classification: { auroc: 0.75, brier: 0.2, ece: 0.03, false_negative_count: 4 }, regression: { mae: 0.2 } },
      },
      deltas: { full_vs_clinical_auroc_delta: 0.01, full_vs_clinical_brier_delta: 0, full_vs_clinical_ece_delta: 0 },
      recommendation: { promote_feature_set: false, recommended_use: "context_only", reason: "Synthetic ablation only." },
      claim_boundary: "No treatment recommendation.",
    } as never} />);
    expect(screen.getByText("context only")).toBeInTheDocument();
    expect(screen.getByText(/No treatment recommendation/i)).toBeInTheDocument();
  });

  it("expands row-level error evidence on explicit operator action", async () => {
    const user = userEvent.setup();
    const rows = Array.from({ length: 21 }, (_, index) => ({
      patient_id: `P${index}`,
      actual_label: index % 2,
      predicted_probability: 0.5,
      predicted_class: 1,
      absolute_error: 0.5,
      confusion_type: index % 2 ? "TP" : "FP",
    }));
    render(<PredictionErrorPanel data={{
      sensitivity: 0.8,
      specificity: 0.7,
      mae: 0.2,
      threshold: 0.5,
      confusion_summary: { TP: 10, FP: 5, TN: 4, FN: 2 },
      rows,
      claim_boundary: "Synthetic row evidence.",
    } as never} />);

    expect(screen.queryByText("P20")).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /show all 21 rows/i }));
    expect(screen.getByText("P20")).toBeInTheDocument();
  });
});
