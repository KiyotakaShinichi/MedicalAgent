import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { ChatPanel } from "../../src/components/ui/ChatPanel";

const emptyMessages: never[] = [];

describe("ChatPanel behavioral contract", () => {
  beforeEach(() => vi.spyOn(console, "error").mockImplementation(() => {}));
  afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  it("exposes an accessible empty state and disables empty submissions", () => {
    render(<ChatPanel messages={emptyMessages} onSend={vi.fn()} />);
    expect(screen.getByRole("heading", { name: /how can i support/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Send message" })).toBeDisabled();
    expect(screen.getByPlaceholderText(/tell me how you are feeling/i)).toBeEnabled();
  });

  it("renders a source-backed answer, citations, and a saved-action status", async () => {
    const user = userEvent.setup();
    const onSend = vi.fn().mockResolvedValue({
      reply: "This is educational context, not a diagnosis.",
      citations: ["NCI patient guide"],
      saved_actions: [{ type: "saved_symptom", symptom: "nausea", severity: 3 }],
    });
    render(<ChatPanel messages={emptyMessages} onSend={onSend} />);

    await user.type(screen.getByRole("textbox"), "What does nausea mean?");
    await user.click(screen.getByRole("button", { name: "Send message" }));

    expect(await screen.findByText(/educational context/i)).toBeInTheDocument();
    expect(screen.getByText(/Sources: NCI patient guide/i)).toBeInTheDocument();
    expect(screen.getByText("Symptom saved")).toBeInTheDocument();
    expect(onSend).toHaveBeenCalledWith("What does nausea mean?");
  });

  it("shows refusal and escalation text exactly as returned without fabricating citations", () => {
    render(<ChatPanel messages={[{
      id: "a1",
      role: "assistant",
      content: "I cannot recommend a dose change. Contact your oncology team now.",
      citations: [],
      saved_actions: [],
    }]} onSend={vi.fn()} />);
    expect(screen.getByText(/cannot recommend a dose change/i)).toBeInTheDocument();
    expect(screen.queryByText(/Sources:/i)).not.toBeInTheDocument();
  });

  it("streams stage and delta updates, then replaces them with the canonical answer", async () => {
    const user = userEvent.setup();
    let release: (() => void) | undefined;
    const onSendStream = vi.fn().mockImplementation(async (_text, handlers) => {
      handlers.onStage("Checking safety");
      handlers.onDelta("Partial answer");
      await new Promise<void>((resolve) => { release = resolve; });
      return { reply: "Final safe answer", citations: ["ASCO"], saved_actions: [] };
    });
    render(<ChatPanel messages={emptyMessages} onSend={vi.fn()} onSendStream={onSendStream} />);

    await user.type(screen.getByRole("textbox"), "Explain this");
    await user.click(screen.getByRole("button", { name: "Send message" }));
    expect(await screen.findByText("Checking safety")).toBeInTheDocument();
    expect(screen.getByText("Partial answer")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Send message" })).toBeDisabled();

    release?.();
    expect(await screen.findByText("Final safe answer")).toBeInTheDocument();
    expect(screen.getByText(/Sources: ASCO/i)).toBeInTheDocument();
  });

  it("prevents a second submission while the first request is pending", async () => {
    const user = userEvent.setup();
    const onSend = vi.fn().mockImplementation(() => new Promise(() => {}));
    render(<ChatPanel messages={emptyMessages} onSend={onSend} />);
    const input = screen.getByRole("textbox");

    await user.type(input, "one{enter}");
    expect(onSend).toHaveBeenCalledTimes(1);
    expect(input).toBeDisabled();
    await user.keyboard("{Enter}");
    expect(onSend).toHaveBeenCalledTimes(1);
  });

  it("shows a dismissible inline error and removes the failed optimistic turn", async () => {
    const user = userEvent.setup();
    render(<ChatPanel messages={emptyMessages} onSend={vi.fn().mockRejectedValue(new Error("Support is unavailable"))} />);

    await user.type(screen.getByRole("textbox"), "hello{enter}");
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("Support is unavailable");
    expect(screen.queryByText("hello")).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Dismiss error" }));
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("requires explicit confirmation before a pending record action can be sent", async () => {
    const user = userEvent.setup();
    const onSend = vi.fn().mockResolvedValue({ reply: "Cancelled", saved_actions: [] });
    render(<ChatPanel messages={[{
      id: "a1",
      role: "assistant",
      content: "Please review this extraction.",
      citations: [],
      saved_actions: [{ type: "pending_record_confirmation", preview: "Nausea severity 4/10" }],
    }]} onSend={onSend} />);

    expect(screen.getByRole("group", { name: /confirm patient record preview/i })).toHaveTextContent(
      /nothing is saved until you confirm/i,
    );
    await user.click(screen.getByRole("button", { name: /confirm save/i }));
    await waitFor(() => expect(onSend).toHaveBeenCalledWith("Confirm save"));
  });

  it("undoes a confirmed record action once and reports completion", async () => {
    const user = userEvent.setup();
    const onUndoAction = vi.fn().mockResolvedValue(undefined);
    render(<ChatPanel messages={[{
      id: "a1",
      role: "assistant",
      content: "Saved after confirmation.",
      citations: [],
      saved_actions: [{ type: "saved_symptom", symptom: "fatigue", severity: 5, undo_available: true, audit_action_id: 9 }],
    }]} onSend={vi.fn()} onUndoAction={onUndoAction} />);

    await user.click(screen.getByRole("button", { name: /undo/i }));
    expect(await screen.findByText("Entry removed")).toBeInTheDocument();
    expect(onUndoAction).toHaveBeenCalledWith(9);
    expect(screen.queryByRole("button", { name: /undo/i })).not.toBeInTheDocument();
  });
});
