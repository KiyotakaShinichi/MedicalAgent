import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { sendMyChatStream } from "../../src/api/client";


describe("streaming API boundary headers", () => {
  beforeEach(() => {
    sessionStorage.clear();
    sessionStorage.setItem("patientPortalAccessToken", "test-token");
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    sessionStorage.clear();
  });

  it("marks patient chat streams as synthetic data in the staging profile", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(
      'event: answer\ndata: {"reply":"ok","saved_actions":[],"citations":[]}\n\n',
      {
        status: 200,
        headers: { "Content-Type": "text/event-stream" },
      },
    ));
    vi.stubGlobal("fetch", fetchMock);

    const result = await sendMyChatStream("hello");

    expect(result.reply).toBe("ok");
    expect(fetchMock).toHaveBeenCalledOnce();
    const requestInit = fetchMock.mock.calls[0][1] as RequestInit;
    const headers = new Headers(requestInit.headers);
    expect(headers.get("X-NLCare-Data-Class")).toBe("synthetic");
    expect(headers.get("Authorization")).toBe("Bearer test-token");
    expect(headers.get("Accept")).toBe("text/event-stream");
  });
});
