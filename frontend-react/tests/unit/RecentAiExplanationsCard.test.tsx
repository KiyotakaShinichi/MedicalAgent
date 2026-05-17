import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { RecentAiExplanationsCard } from "../../src/pages/patient/RecentAiExplanationsCard";
import type { ChatMessage } from "../../src/types/api";

function renderWithRouter(ui: React.ReactElement) {
  return render(<MemoryRouter>{ui}</MemoryRouter>);
}

describe("RecentAiExplanationsCard", () => {
  it("shows an empty state when there are no messages", () => {
    renderWithRouter(<RecentAiExplanationsCard messages={[]} />);
    expect(screen.getByText(/No recent assistant replies/i)).toBeInTheDocument();
  });

  it("ignores user messages and only renders assistant replies", () => {
    const messages: ChatMessage[] = [
      { role: "user",      message: "How are my labs?" },
      { role: "assistant", message: "Your WBC trends are steady this cycle." },
      { role: "user",      message: "Thanks." },
    ];
    renderWithRouter(<RecentAiExplanationsCard messages={messages} />);
    expect(screen.getByText(/WBC trends are steady/)).toBeInTheDocument();
    expect(screen.queryByText("How are my labs?")).toBeNull();
  });

  it("shows at most 3 assistant turns in reverse-chronological order", () => {
    const messages: ChatMessage[] = [
      { role: "assistant", message: "reply one — oldest" },
      { role: "assistant", message: "reply two" },
      { role: "assistant", message: "reply three" },
      { role: "assistant", message: "reply four — newest" },
    ];
    renderWithRouter(<RecentAiExplanationsCard messages={messages} />);
    const items = screen.getAllByRole("listitem");
    expect(items).toHaveLength(3);
    // Newest first
    expect(items[0]).toHaveTextContent("reply four — newest");
    expect(items[2]).toHaveTextContent("reply two");
    // Oldest one is dropped
    expect(screen.queryByText(/reply one — oldest/)).toBeNull();
  });

  it("tags each rendered reply with an AI badge for clarity", () => {
    const messages: ChatMessage[] = [
      { role: "assistant", message: "anything" },
    ];
    renderWithRouter(<RecentAiExplanationsCard messages={messages} />);
    // AI badge appears next to each rendered reply
    expect(screen.getAllByLabelText(/AI-generated/i).length).toBeGreaterThanOrEqual(1);
  });

  it("includes a 'verify with care team' safety footnote when there are replies", () => {
    const messages: ChatMessage[] = [
      { role: "assistant", message: "x" },
    ];
    renderWithRouter(<RecentAiExplanationsCard messages={messages} />);
    expect(screen.getByText(/Verify anything actionable with your care team/i)).toBeInTheDocument();
  });

  it("survives a malformed message list (defensive null + missing fields)", () => {
    // The component types say messages: ChatMessage[], but the report payload
    // sometimes arrives with nulls or partial rows. We render past it instead
    // of crashing.
    const messages = [
      null,
      { role: "assistant" },
      { role: "assistant", message: "real reply" },
    ] as unknown as ChatMessage[];
    renderWithRouter(<RecentAiExplanationsCard messages={messages} />);
    expect(screen.getByText("real reply")).toBeInTheDocument();
  });
});
