import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { sendMyChatStream } from "../../src/api/client.patient";

function streamResponse(chunks: string[]): Response {
  const encoder = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      chunks.forEach((chunk) => controller.enqueue(encoder.encode(chunk)));
      controller.close();
    },
  });
  return new Response(stream, { status: 200, headers: { "Content-Type": "text/event-stream" } });
}

describe("patient chat streaming transport", () => {
  beforeEach(() => {
    sessionStorage.clear();
    sessionStorage.setItem("patientPortalAccessToken", "patient-token");
    vi.useRealTimers();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.useRealTimers();
    sessionStorage.clear();
  });

  it("parses split SSE events and normalizes citation objects", async () => {
    const fetchMock = vi.fn().mockResolvedValue(streamResponse([
      'event: pipeline_stage\ndata: {"label":"Checking sources"}\n\n',
      'event: answer_delta\ndata: {"text":"Source-"}\n\n',
      'event: answer_delta\ndata: {"text":"backed"}\n\n',
      'event: answer\ndata: {"reply":"Source-backed","saved_actions":[],"citations":[{"title":"NCI guide"},"ASCO"]}\n\n',
    ]));
    vi.stubGlobal("fetch", fetchMock);
    const onStage = vi.fn();
    const onDelta = vi.fn();

    await expect(sendMyChatStream("What does this mean?", { onStage, onDelta })).resolves.toEqual({
      reply: "Source-backed",
      saved_actions: [],
      citations: ["NCI guide", "ASCO"],
      assistant_message_id: undefined,
    });
    expect(onStage).toHaveBeenCalledWith("Checking sources");
    expect(onDelta.mock.calls.flat()).toEqual(["Source-", "backed"]);
    expect(fetchMock).toHaveBeenCalledWith(
      "http://127.0.0.1:8017/me/chat/stream",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ message: "What does this mean?" }),
        headers: expect.objectContaining({ Authorization: "Bearer patient-token" }),
      }),
    );
  });

  it("ignores malformed SSE frames but requires a final answer", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(streamResponse([
      "not-an-event\n\n",
      "event: answer\ndata: not-json\n\n",
    ])));
    await expect(sendMyChatStream("hello")).rejects.toThrow("ended without an answer");
  });

  it("surfaces explicit server error events", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(streamResponse([
      'event: error\ndata: {"error":"The support service is unavailable"}\n\n',
    ])));
    await expect(sendMyChatStream("hello")).rejects.toThrow("support service is unavailable");
  });

  it("converts aborts into a patient-safe timeout message", async () => {
    vi.useFakeTimers();
    vi.stubGlobal("fetch", vi.fn().mockImplementation((_url, options: RequestInit) => (
      new Promise<Response>((_resolve, reject) => {
        options.signal?.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")));
      })
    )));

    const pending = sendMyChatStream("hello");
    const rejection = expect(pending).rejects.toThrow(/timed out.*no record was saved/i);
    await vi.advanceTimersByTimeAsync(45_000);
    await rejection;
  });
});
