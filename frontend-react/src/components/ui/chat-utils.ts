import type { SavedAction } from "../../types/api";

/**
 * Normalised in-memory message shape used by ChatPanel.  Always carries
 * an `id` so React keys are stable across re-renders and streaming deltas
 * can target the exact bubble they belong to.
 */
export interface NormalisedMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  saved_actions: SavedAction[];
  citations: string[];
  /** Tagged when this assistant bubble is the live target of an in-flight stream. */
  streamId?: string;
}

/**
 * Defensive parser: the backend may store either a raw SavedAction[] or
 * an object wrapper `{saved_actions: [...], tool_plan, agent_pipeline}`.
 * Returns [] for anything that isn't recognisable.
 */
export function parseSavedActions(raw?: unknown): SavedAction[] {
  if (!raw) return [];
  let parsed: unknown = raw;
  if (typeof raw === "string") {
    try { parsed = JSON.parse(raw); } catch { return []; }
  }
  if (Array.isArray(parsed)) {
    return parsed.filter((a) => a && typeof (a as SavedAction).type === "string") as SavedAction[];
  }
  if (parsed && typeof parsed === "object") {
    const wrapped = (parsed as { saved_actions?: unknown }).saved_actions;
    if (Array.isArray(wrapped)) {
      return wrapped.filter((a) => a && typeof (a as SavedAction).type === "string") as SavedAction[];
    }
  }
  return [];
}

export function parseCitations(raw: unknown): string[] {
  if (!Array.isArray(raw)) return [];
  return raw
    .map((c) => (typeof c === "string" ? c : String(((c as { id?: unknown }).id) ?? "")))
    .filter((s) => Boolean(s));
}

/**
 * Normalise any value that might be a message into our strict shape.
 * Accepts ``message`` or ``content`` field names so a backend schema rename
 * doesn't crash the UI.  Returns ``null`` for unsalvageable input.
 */
export function normaliseMessage(raw: unknown, fallbackIndex: number): NormalisedMessage | null {
  if (!raw || typeof raw !== "object") return null;
  const m = raw as Record<string, unknown>;
  const role = m.role === "assistant" ? "assistant" : m.role === "user" ? "user" : null;
  if (!role) return null;
  const rawContent =
    typeof m.message === "string" ? m.message :
    typeof m.content === "string" ? m.content :
    "";
  const id =
    typeof m.id === "string" || typeof m.id === "number"
      ? String(m.id)
      : `msg_${role}_${fallbackIndex}`;
  return {
    id,
    role,
    content: rawContent,
    saved_actions: parseSavedActions(m.saved_actions_json ?? m.saved_actions),
    citations: parseCitations(m.citations),
  };
}

export function normaliseMessages(input: unknown): NormalisedMessage[] {
  if (!Array.isArray(input)) return [];
  const out: NormalisedMessage[] = [];
  for (let i = 0; i < input.length; i++) {
    const n = normaliseMessage(input[i], i);
    if (n) out.push(n);
  }
  return out;
}
