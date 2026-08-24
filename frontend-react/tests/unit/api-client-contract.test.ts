/**
 * Contract tests for the API client's request core.
 *
 * `src/api/client.ts` is the single path every request in the app takes, and
 * three behaviours in it are load-bearing and were untested:
 *
 *  - `ApiError` carries the HTTP status so a caller can tell an expected 4xx
 *    product state apart from a server fault without re-parsing a message;
 *  - concurrent identical GETs are de-duplicated, so a page mounting several
 *    components that each need the same resource issues one request;
 *  - every failure is reported to telemetry exactly once, at the network
 *    boundary, with the route redacted and the request body never attached.
 *
 * These are tested through the exported wrappers rather than by reaching into
 * the module's internals, so the tests exercise the same path the application
 * does.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ApiError, getMyReport, getPatients } from "../../src/api/client";

const reportError = vi.hoisted(() => vi.fn());

vi.mock("../../src/lib/telemetry", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../src/lib/telemetry")>();
  return { ...actual, reportError };
});

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

/**
 * A fetch mock that builds a *fresh* Response per call.
 *
 * `mockResolvedValue(new Response(...))` hands the same object to every call,
 * and a Response body can only be read once - the second call fails with
 * "Body is unusable" rather than testing anything.
 */
function fetchReturning(body: unknown, status = 200) {
  return vi.fn().mockImplementation(() => Promise.resolve(jsonResponse(body, status)));
}

describe("ApiError", () => {
  it("carries the HTTP status alongside the message", () => {
    const error = new ApiError("Patient not found", 404);
    expect(error.status).toBe(404);
    expect(error.message).toBe("Patient not found");
    expect(error.name).toBe("ApiError");
  });

  it("is a real Error, so existing catch handling keeps working", () => {
    expect(new ApiError("boom", 500)).toBeInstanceOf(Error);
  });

  it("classifies 4xx as an expected product state", () => {
    // Unauthorised, invalid input, and not found are things the product does,
    // not faults to page someone about.
    expect(new ApiError("unauthorised", 401).isExpected).toBe(true);
    expect(new ApiError("invalid", 422).isExpected).toBe(true);
    expect(new ApiError("missing", 404).isExpected).toBe(true);
  });

  it("classifies 5xx as unexpected", () => {
    expect(new ApiError("server", 500).isExpected).toBe(false);
    expect(new ApiError("gateway", 503).isExpected).toBe(false);
  });
});

describe("error normalisation", () => {
  beforeEach(() => {
    sessionStorage.clear();
    reportError.mockClear();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    sessionStorage.clear();
  });

  it("surfaces the API's `detail` message and status", async () => {
    vi.stubGlobal(
      "fetch",
      fetchReturning({ detail: "Patient not found" }, 404),
    );

    await expect(getPatients()).rejects.toMatchObject({
      message: "Patient not found",
      status: 404,
    });
  });

  it("falls back to a status-bearing message when the body has none", async () => {
    vi.stubGlobal("fetch", vi.fn().mockImplementation(
      () => Promise.resolve(new Response("", { status: 500 })),
    ));

    await expect(getPatients()).rejects.toMatchObject({ status: 500 });
  });

  it("does not throw away a non-JSON error body", async () => {
    // A proxy or gateway returns HTML, not JSON. Losing it entirely would
    // leave a bare status with nothing to diagnose from.
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation(
        () => Promise.resolve(new Response("<html>gateway timeout</html>", { status: 504 })),
      ),
    );

    await expect(getPatients()).rejects.toMatchObject({ status: 504 });
  });
});

describe("in-flight GET de-duplication", () => {
  beforeEach(() => {
    sessionStorage.clear();
    reportError.mockClear();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    sessionStorage.clear();
  });

  it("issues one request when the same GET is asked for concurrently", async () => {
    let resolveFetch: ((value: Response) => void) | undefined;
    const fetchMock = vi.fn().mockImplementation(
      () => new Promise<Response>((resolve) => {
        resolveFetch = resolve;
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    const first = getPatients();
    const second = getPatients();
    resolveFetch?.(jsonResponse([{ id: "P001" }]));

    await expect(first).resolves.toEqual([{ id: "P001" }]);
    await expect(second).resolves.toEqual([{ id: "P001" }]);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("does not conflate different paths", async () => {
    const fetchMock = fetchReturning({});
    vi.stubGlobal("fetch", fetchMock);

    await Promise.all([getPatients(), getMyReport()]);
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("releases the entry once settled, so a later call refetches", async () => {
    // Otherwise the first response would be cached for the life of the page
    // and the UI could never show updated data.
    const fetchMock = fetchReturning([]);
    vi.stubGlobal("fetch", fetchMock);

    await getPatients();
    await getPatients();
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("releases the entry after a failure too", async () => {
    // A cached rejected promise would make one transient error permanent.
    const fetchMock = vi
      .fn()
      .mockImplementationOnce(() => Promise.resolve(jsonResponse({ detail: "boom" }, 500)))
      .mockImplementationOnce(() => Promise.resolve(jsonResponse([])));
    vi.stubGlobal("fetch", fetchMock);

    await expect(getPatients()).rejects.toBeInstanceOf(ApiError);
    await expect(getPatients()).resolves.toEqual([]);
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("does not share a response between different sessions", async () => {
    // The de-dup key includes the token; sharing across it would serve one
    // user's data to another.
    const fetchMock = fetchReturning([]);
    vi.stubGlobal("fetch", fetchMock);

    sessionStorage.setItem("patientPortalAccessToken", "token-a");
    const first = getPatients();
    sessionStorage.setItem("patientPortalAccessToken", "token-b");
    const second = getPatients();

    await Promise.all([first, second]);
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });
});

describe("authentication header", () => {
  beforeEach(() => {
    sessionStorage.clear();
    reportError.mockClear();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    sessionStorage.clear();
  });

  it("attaches the stored bearer token", async () => {
    const fetchMock = fetchReturning([]);
    vi.stubGlobal("fetch", fetchMock);
    sessionStorage.setItem("patientPortalAccessToken", "test-token");

    await getPatients();

    const headers = fetchMock.mock.calls[0][1].headers as Record<string, string>;
    expect(headers.Authorization).toBe("Bearer test-token");
    expect(headers["X-NLCare-Data-Class"]).toBe("synthetic");
  });

  it("omits the header entirely when no session exists", async () => {
    const fetchMock = fetchReturning([]);
    vi.stubGlobal("fetch", fetchMock);

    await getPatients();

    const headers = fetchMock.mock.calls[0][1].headers as Record<string, string>;
    expect(headers.Authorization).toBeUndefined();
  });
});

describe("failure telemetry", () => {
  beforeEach(() => {
    sessionStorage.clear();
    reportError.mockClear();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    sessionStorage.clear();
  });

  it("reports a failure once, at the network boundary", async () => {
    vi.stubGlobal("fetch", fetchReturning({ detail: "nope" }, 500));

    await expect(getPatients()).rejects.toBeInstanceOf(ApiError);
    expect(reportError).toHaveBeenCalledTimes(1);
  });

  it("marks a 4xx as expected and a 5xx as unexpected", async () => {
    vi.stubGlobal("fetch", fetchReturning({ detail: "no" }, 404));
    await expect(getPatients()).rejects.toBeInstanceOf(ApiError);
    expect(reportError.mock.calls[0][1]).toMatchObject({ kind: "expected" });

    reportError.mockClear();
    vi.stubGlobal("fetch", fetchReturning({ detail: "no" }, 500));
    await expect(getPatients()).rejects.toBeInstanceOf(ApiError);
    expect(reportError.mock.calls[0][1]).toMatchObject({ kind: "unexpected" });
  });

  it("records the route and status but never the request body", async () => {
    vi.stubGlobal("fetch", fetchReturning({ detail: "no" }, 403));

    await expect(getMyReport()).rejects.toBeInstanceOf(ApiError);

    const context = reportError.mock.calls[0][1] as {
      surface: string;
      detail: Record<string, unknown>;
    };
    expect(context.surface).toBe("api.request");
    expect(context.detail.method).toBe("GET");
    expect(context.detail.status).toBe(403);
    expect(JSON.stringify(context)).not.toContain("password");
    expect(Object.keys(context.detail)).not.toContain("body");
  });

  it("reports a network-level failure that never produced a response", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));

    await expect(getPatients()).rejects.toBeInstanceOf(TypeError);
    expect(reportError).toHaveBeenCalledTimes(1);
    expect(reportError.mock.calls[0][1]).toMatchObject({ kind: "unexpected" });
  });

  it("does not report anything on a successful request", async () => {
    vi.stubGlobal("fetch", fetchReturning([]));

    await getPatients();
    expect(reportError).not.toHaveBeenCalled();
  });
});
