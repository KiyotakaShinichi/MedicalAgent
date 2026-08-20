/**
 * Centralised frontend error reporting.
 *
 * Why this exists
 * ---------------
 * Before this module, failures were reported by ad-hoc `console.error` calls
 * scattered across components. That has three problems in a clinical product:
 *
 *   1. Expected failures (a 401, a validation rejection, an artifact that has
 *      not been generated yet) were logged at the same severity as genuine
 *      application bugs, so the signal was unusable.
 *   2. Raw error payloads from a medical backend can carry patient free-text,
 *      identifiers, or bearer tokens. Anything we log is a disclosure surface.
 *   3. There was no seam to attach a real monitoring provider later.
 *
 * Design
 * ------
 * `reportError` classifies, redacts, then fans out to registered sinks. The
 * console sink is always installed; a hosted provider (Sentry/Datadog/…) can
 * be added at bootstrap via `registerTelemetrySink` without touching call
 * sites. Sinks are fully isolated — a throwing or missing sink must never
 * escalate into a user-visible failure, so every dispatch is guarded.
 *
 * This module deliberately does NOT bundle an external SDK. Observability
 * plumbing that a deployment may not want is a dependency and a data-egress
 * decision, not a default.
 */

/**
 * `expected` — the application behaved correctly in the face of a known
 * failure mode: an API 4xx, a rejected login, a not-yet-generated artifact,
 * a user input violation. These are product states, not defects. They are
 * recorded at `warn` and are expected to be noisy.
 *
 * `unexpected` — an invariant broke: a render crash, a TypeError, a 5xx, a
 * malformed response that got past its parser. These are defects and should
 * page someone in a real deployment.
 */
export type ErrorKind = "expected" | "unexpected";

export interface TelemetryContext {
  /** Where it happened, e.g. "SafetyCenterSection.load" or "api.request". */
  surface: string;
  kind?: ErrorKind;
  /** Structured, NON-sensitive detail. Values are redacted before dispatch. */
  detail?: Record<string, unknown>;
}

export interface TelemetryEvent {
  surface: string;
  kind: ErrorKind;
  message: string;
  /** Error.name when available — "TypeError", "AbortError", … */
  name: string;
  stack?: string;
  detail: Record<string, unknown>;
  timestamp: string;
}

export type TelemetrySink = (event: TelemetryEvent) => void;

// ── Redaction ────────────────────────────────────────────────────────────────

/**
 * Keys whose values are never safe to emit. Matched case-insensitively as a
 * substring, so `access_token`, `Authorization`, and `patientNotes` all hit.
 */
const SENSITIVE_KEY_PATTERN =
  /(token|authorization|password|secret|api[-_]?key|cookie|session|bearer|patient_id|patientid|mrn|ssn|dob|birth|email|phone|address|note|message_text|free_text|content)/i;

/** Bearer tokens, JWTs, and long opaque credentials appearing inside strings. */
const TOKEN_LIKE_PATTERN =
  /\b(?:Bearer\s+)?[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b|\bBearer\s+[A-Za-z0-9._-]{12,}\b/gi;

const EMAIL_PATTERN = /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b/g;

/**
 * Free-text from a clinical backend may be a patient message or a note. We do
 * not attempt to detect PHI — we cap length instead, on the assumption that
 * anything long is prose and anything prose-shaped is a disclosure risk.
 */
const MAX_STRING_LENGTH = 200;

export function redactString(value: string): string {
  const scrubbed = value
    .replace(TOKEN_LIKE_PATTERN, "[redacted-token]")
    .replace(EMAIL_PATTERN, "[redacted-email]");
  return scrubbed.length > MAX_STRING_LENGTH
    ? `${scrubbed.slice(0, MAX_STRING_LENGTH)}…[truncated]`
    : scrubbed;
}

/**
 * Replace identifier-shaped segments of a URL path with `:id`.
 *
 * Request paths are the most useful thing to log and also the most likely
 * place for a record identifier to leak (`/patients/P001/labs`). Templating
 * them keeps the route groupable while dropping the subject.
 */
export function redactUrlPath(path: string): string {
  const [pathname, query] = path.split("?", 2);
  const templated = pathname
    .split("/")
    .map((segment) => {
      if (!segment) return segment;
      // Numeric ids, UUIDs, and the P001-style synthetic patient ids.
      if (/^\d+$/.test(segment)) return ":id";
      if (/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(segment)) return ":id";
      if (/^[A-Z]{1,3}\d{2,}$/.test(segment)) return ":id";
      return segment;
    })
    .join("/");
  // Query strings routinely carry filters keyed by subject — drop the values.
  return query ? `${templated}?[redacted-query]` : templated;
}

/**
 * Recursively strip sensitive values from a structured payload. Depth is
 * bounded because telemetry must never be the thing that hangs a render.
 */
export function redactValue(value: unknown, depth = 0): unknown {
  if (value === null || value === undefined) return value;
  if (depth > 4) return "[redacted-depth]";
  if (typeof value === "string") return redactString(value);
  if (typeof value === "number" || typeof value === "boolean") return value;
  if (Array.isArray(value)) {
    return value.slice(0, 20).map((entry) => redactValue(entry, depth + 1));
  }
  if (typeof value === "object") {
    const out: Record<string, unknown> = {};
    for (const [key, entry] of Object.entries(value as Record<string, unknown>)) {
      out[key] = SENSITIVE_KEY_PATTERN.test(key)
        ? "[redacted]"
        : redactValue(entry, depth + 1);
    }
    return out;
  }
  return "[redacted-unserialisable]";
}

function redactDetail(detail: Record<string, unknown> | undefined): Record<string, unknown> {
  if (!detail) return {};
  return redactValue(detail) as Record<string, unknown>;
}

// ── Sinks ────────────────────────────────────────────────────────────────────

const sinks = new Set<TelemetrySink>();

/**
 * Attach an additional sink (e.g. a hosted monitoring provider installed at
 * bootstrap). Returns an unsubscribe function so tests can clean up.
 */
export function registerTelemetrySink(sink: TelemetrySink): () => void {
  sinks.add(sink);
  return () => {
    sinks.delete(sink);
  };
}

/** Test seam — drops every registered sink. */
export function resetTelemetrySinks(): void {
  sinks.clear();
}

function consoleSink(event: TelemetryEvent): void {
  const line = `[nlcare:${event.kind}] ${event.surface}: ${event.message}`;
  if (event.kind === "unexpected") {
    console.error(line, event.detail);
  } else {
    console.warn(line, event.detail);
  }
}

// ── Public API ───────────────────────────────────────────────────────────────

/**
 * Normalise anything throwable into a message + name pair. Non-Error throws
 * are common when a rejected fetch surfaces a string or a response body.
 */
function describe(error: unknown): { message: string; name: string; stack?: string } {
  if (error instanceof Error) {
    return { message: error.message || error.name, name: error.name, stack: error.stack };
  }
  if (typeof error === "string") return { message: error, name: "StringError" };
  return { message: String(error), name: "UnknownError" };
}

/**
 * Report a failure. Safe to call from any layer — it never throws and never
 * rejects, so it can sit inside a `catch` without a nested guard.
 */
export function reportError(error: unknown, context: TelemetryContext): TelemetryEvent {
  const described = describe(error);
  const event: TelemetryEvent = {
    surface: context.surface,
    kind: context.kind ?? "unexpected",
    message: redactString(described.message),
    name: described.name,
    stack: described.stack ? redactString(described.stack) : undefined,
    detail: redactDetail(context.detail),
    timestamp: new Date().toISOString(),
  };

  // Telemetry is best-effort by contract. A broken sink degrades observability;
  // it must not degrade the application.
  try {
    consoleSink(event);
  } catch {
    /* console unavailable — nothing further we can do */
  }
  for (const sink of sinks) {
    try {
      sink(event);
    } catch {
      /* a failing sink must not affect other sinks or the caller */
    }
  }
  return event;
}

/** Convenience wrapper for known, non-defect failures. */
export function reportExpectedError(error: unknown, surface: string, detail?: Record<string, unknown>) {
  return reportError(error, { surface, kind: "expected", detail });
}

/**
 * Attach handlers for failures that escape React entirely — unhandled promise
 * rejections and top-level `window.onerror`. Called once from `main.tsx`.
 * Returns a teardown function.
 */
export function installGlobalErrorHandlers(target: Window = window): () => void {
  const onRejection = (event: PromiseRejectionEvent) => {
    reportError(event.reason, { surface: "window.unhandledrejection" });
  };
  const onError = (event: ErrorEvent) => {
    reportError(event.error ?? event.message, {
      surface: "window.onerror",
      detail: { filename: event.filename, lineno: event.lineno },
    });
  };
  target.addEventListener("unhandledrejection", onRejection as EventListener);
  target.addEventListener("error", onError as EventListener);
  return () => {
    target.removeEventListener("unhandledrejection", onRejection as EventListener);
    target.removeEventListener("error", onError as EventListener);
  };
}
