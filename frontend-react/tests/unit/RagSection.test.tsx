import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

const api = vi.hoisted(() => ({
  getNormalizedBenchmarkArtifact: vi.fn(),
  getRagAblation: vi.fn(),
  getRagSourceRegistry: vi.fn(),
  getRagTraceReplay: vi.fn(),
  runLiveRagEval: vi.fn(),
}));

vi.mock("../../src/api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../src/api/client")>();
  return { ...actual, ...api };
});

import { RagSection } from "../../src/pages/admin/sections/RagSection";

const artifacts: Record<string, unknown> = {
  rag_baseline_comparison: {
    status: "needs_attention",
    metrics: { improvement_proven_vs_bm25: false },
    rows: [{
      configuration: "bm25",
      label: "BM25 only",
      recall_at_10: 0.8,
      mrr: 0.7,
      citation_precision: 0.4,
      unsupported_context_rate: 0.1,
      source_tier_correctness: 0.5,
      latency_p95_ms: 42,
      failure_count: 1,
      failure_examples: [{ case_id: "rag-1", query: "Explain VUS", failure_reasons: ["missing_source"] }],
    }],
  },
  live_rag_eval: {
    status: "acceptable",
    metrics: { pass_rate: 0.9, claim_support_rate: 0.8, source_tier_correctness: 1, unsafe_answer_rate: 0 },
    claim_boundary: "Internal engineering evidence only.",
  },
};

describe("RagSection operator states", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    api.getRagSourceRegistry.mockResolvedValue({
      sources: [{ id: "nci", source_name: "NCI patient guide", trust_level: "high", chunk_count: 8, topics: ["VUS"] }],
    });
    api.getRagAblation.mockResolvedValue({
      strategies: {
        bm25_only: { case_count: 10, pass_rate: 0.7, expected_source_hit_rate: 0.8 },
        dense_faiss_bm25_rrf_reranked: { case_count: 10, pass_rate: 0.9, expected_source_hit_rate: 0.9 },
      },
      comparison: { notes: ["Reranking remains internal evidence."] },
      limitations: ["No clinical validation."],
      claim_boundary: "Engineering comparison only.",
    });
    api.getNormalizedBenchmarkArtifact.mockImplementation(async (id: string) => artifacts[id] ?? null);
    api.getRagTraceReplay.mockResolvedValue({ traces: [] });
    api.runLiveRagEval.mockResolvedValue({});
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("renders current metrics, governed sources, and honest negative RAG evidence", async () => {
    render(<RagSection analytics={{
      rag_evaluation: {
        evaluations: 74,
        grounding_score: 0.82,
        hallucination_score: 0.04,
        precision_at_3: 0.71,
        estimated_cost_usd: 0.012,
        input_tokens: 500,
        output_tokens: 120,
        p95_latency_ms: 900,
      },
    } as never} />);

    expect(await screen.findByText("NCI patient guide")).toBeInTheDocument();
    expect(screen.getByText("not proven")).toBeInTheDocument();
    expect(screen.getAllByText("BM25 only").length).toBeGreaterThan(1);
    expect(screen.getAllByText(/clinical validation: false/i).length).toBeGreaterThan(0);
    expect(screen.getByText(/No clinical validation/i)).toBeInTheDocument();
    expect(screen.getByText("74")).toBeInTheDocument();
  });

  it("shows a deterministic operator error when a rerun fails", async () => {
    vi.spyOn(console, "error").mockImplementation(() => undefined);
    api.runLiveRagEval.mockRejectedValue(new Error("runner unavailable"));
    const user = userEvent.setup();
    render(<RagSection />);

    const rerun = await screen.findByRole("button", { name: /rerun/i });
    await user.click(rerun);
    expect(await screen.findByRole("alert")).toHaveTextContent("runner unavailable");
  });
});
