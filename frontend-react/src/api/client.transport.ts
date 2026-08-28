import { redactUrlPath, reportError } from "../lib/telemetry";

/**
 * Backend base URL.  Resolved from (in order):
 *   1. Vite `VITE_API_BASE` env var (set in .env.local for non-default hosts)
 *   2. `http://127.0.0.1:8017` fallback for the local dev profile
 *
 * Exported so the ErrorPane + tool trace drawer can show the actual host
 * the frontend is trying to talk to.
 */
export const API_BASE: string =
  (import.meta as unknown as { env?: { VITE_API_BASE?: string } }).env?.VITE_API_BASE
    ?? "http://127.0.0.1:8017";

export const BASE = API_BASE;
const inFlightGetRequests = new Map<string, Promise<unknown>>();

/**
 * An API failure carrying the HTTP status, so callers (and telemetry) can tell
 * an expected 4xx apart from a server fault without re-parsing the message.
 */
export class ApiError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }

  /** 4xx responses are product states: unauthorised, invalid input, not found. */
  get isExpected(): boolean {
    return this.status >= 400 && this.status < 500;
  }
}

export async function responseError(response: Response): Promise<ApiError> {
  const raw = await response.text().catch(() => "");
  if (raw) {
    try {
      const payload = JSON.parse(raw) as { detail?: unknown; message?: unknown };
      const message =
        typeof payload.detail === "string"
          ? payload.detail
          : typeof payload.message === "string"
            ? payload.message
            : null;
      if (message) return new ApiError(message, response.status);
    } catch {
      return new ApiError(`Request failed (${response.status}): ${raw}`, response.status);
    }
  }
  return new ApiError(`Request failed (${response.status})`, response.status);
}

export function getToken(): string | null {
  return (
    sessionStorage.getItem("patientPortalAccessToken") ||
    sessionStorage.getItem("clinicianAccessToken") ||
    sessionStorage.getItem("adminAccessToken")
  );
}

export async function request<T>(
  method: string,
  path: string,
  body?: unknown,
  extraHeaders: Record<string, string> = {},
): Promise<T> {
  const token = getToken();
  const cacheKey = method === "GET" && body === undefined ? `${token ?? "anon"}:${path}` : null;
  if (cacheKey && inFlightGetRequests.has(cacheKey)) {
    return inFlightGetRequests.get(cacheKey) as Promise<T>;
  }

  const promise = fetch(`${BASE}${path}`, {
    method,
    headers: {
      "Content-Type": "application/json",
      "X-NLCare-Data-Class": "synthetic",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...extraHeaders,
    },
    ...(body !== undefined ? { body: JSON.stringify(body) } : {}),
  })
    .then(async (res) => {
      if (!res.ok) {
        throw await responseError(res);
      }
      return res.json() as Promise<T>;
    })
    .catch((error: unknown) => {
      // Report once, here, at the network boundary. `path` and `method` are
      // safe to record; the request body is not and is never attached.
      reportError(error, {
        surface: "api.request",
        kind: error instanceof ApiError && error.isExpected ? "expected" : "unexpected",
        detail: {
          method,
          route: redactUrlPath(path),
          status: error instanceof ApiError ? error.status : undefined,
        },
      });
      throw error;
    })
    .finally(() => {
      if (cacheKey) inFlightGetRequests.delete(cacheKey);
    });

  if (cacheKey) inFlightGetRequests.set(cacheKey, promise);
  return promise;
}

export const get = <T>(path: string) => request<T>("GET", path);
export const post = <T>(path: string, body?: unknown) => request<T>("POST", path, body);
export const del = <T>(path: string) => request<T>("DELETE", path);

// Synthetic AI assurance SaaS control plane
