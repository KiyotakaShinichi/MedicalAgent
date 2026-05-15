import { useState } from "react";
import { ChevronUp, ChevronDown, Cpu, BookOpen, Wrench, Route as RouteIcon, Gauge, ShieldCheck } from "lucide-react";
import type { ChatMessage, SavedAction } from "../../types/api";

interface ChatTrace {
  saved_actions: SavedAction[];
  tool_plan?: {
    selected_tools?: string[];
    intent?: string;
    rationale?: string;
  };
  agent_pipeline?: {
    intent?: string;
    safety?: { level?: string; scope?: string };
    citations?: Array<{ id?: string; title?: string }>;
    cache?: { status?: string; cacheable?: boolean };
    rag_evaluation?: {
      cost_latency?: { latency_ms?: number };
      answer_grounding?: { score?: number };
      hallucination?: { score?: number; risk?: string };
    };
  };
}

function parseTrace(raw?: string): ChatTrace | null {
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw);
    // New shape: { saved_actions, tool_plan, agent_pipeline }
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      return parsed as ChatTrace;
    }
    // Legacy shape: raw SavedAction[]
    if (Array.isArray(parsed)) return { saved_actions: parsed as SavedAction[] };
  } catch {
    /* malformed legacy payload — silently ignore */
  }
  return null;
}

/** Returns the latest assistant message that carries trace metadata, or null. */
// eslint-disable-next-line react-refresh/only-export-components
export function extractLatestTrace(messages: ChatMessage[]): ChatTrace | null {
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i];
    if (m.role !== "assistant") continue;
    const trace = parseTrace(m.saved_actions_json);
    if (trace) return trace;
  }
  return null;
}

interface ToolTraceDrawerProps {
  messages: ChatMessage[];
}

/**
 * Bottom-sheet drawer with the latest assistant response's pipeline trace —
 * intent route, tools selected, save actions, latency, and citations.
 * Collapsed by default to stay out of the way.
 */
export function ToolTraceDrawer({ messages }: ToolTraceDrawerProps) {
  const [drawerState, setDrawerState] = useState({
    open: false,
    messageCount: messages.length,
  });
  const trace = extractLatestTrace(messages);

  if (!trace) return null;

  const open = drawerState.messageCount === messages.length ? drawerState.open : false;
  const intent = trace.agent_pipeline?.intent ?? trace.tool_plan?.intent ?? "unknown";
  const safety = trace.agent_pipeline?.safety ?? {};
  const selectedTools = trace.tool_plan?.selected_tools ?? [];
  const saved = trace.saved_actions ?? [];
  const citations = trace.agent_pipeline?.citations ?? [];
  const latencyMs = trace.agent_pipeline?.rag_evaluation?.cost_latency?.latency_ms;
  const cacheStatus = trace.agent_pipeline?.cache?.status;

  return (
    <div className="tool-trace-drawer" data-open={open}>
      <button
        type="button"
        className="tool-trace-toggle"
        onClick={() => setDrawerState({ open: !open, messageCount: messages.length })}
        aria-expanded={open}
      >
        <Cpu size={13} aria-hidden="true" />
        <span>Last action trace</span>
        <span className="tool-trace-route">{intent}</span>
        {typeof latencyMs === "number" && (
          <span className="tool-trace-latency">{Math.round(latencyMs)} ms</span>
        )}
        {open ? <ChevronDown size={13} /> : <ChevronUp size={13} />}
      </button>
      {open && (
        <div className="tool-trace-body">
          <TraceRow icon={RouteIcon} label="Intent route">
            <code>{intent}</code>
            {safety.level && safety.level !== "low_risk" && (
              <span className="tool-trace-pill tool-trace-pill--warn">
                <ShieldCheck size={11} /> {safety.level}
              </span>
            )}
          </TraceRow>

          <TraceRow icon={Wrench} label="Tools selected">
            {selectedTools.length === 0 ? (
              <span className="tool-trace-muted">— none —</span>
            ) : (
              selectedTools.map((t) => (
                <code key={t}>{t}</code>
              ))
            )}
          </TraceRow>

          <TraceRow icon={ShieldCheck} label="Saved actions">
            {saved.length === 0 ? (
              <span className="tool-trace-muted">— none persisted —</span>
            ) : (
              saved.map((a, i) => (
                <code key={i}>{a.type}</code>
              ))
            )}
          </TraceRow>

          {(typeof latencyMs === "number" || cacheStatus) && (
            <TraceRow icon={Gauge} label="Latency">
              {typeof latencyMs === "number" && (
                <code>{Math.round(latencyMs)} ms</code>
              )}
              {cacheStatus && <code>cache: {cacheStatus}</code>}
            </TraceRow>
          )}

          {citations.length > 0 && (
            <TraceRow icon={BookOpen} label="Citations">
              {citations.slice(0, 4).map((c, i) => (
                <code key={i}>{c.id ?? c.title ?? `source ${i}`}</code>
              ))}
              {citations.length > 4 && (
                <span className="tool-trace-muted">+ {citations.length - 4} more</span>
              )}
            </TraceRow>
          )}
        </div>
      )}
    </div>
  );
}

function TraceRow({
  icon: Icon,
  label,
  children,
}: {
  icon: typeof Cpu;
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div className="tool-trace-row">
      <span className="tool-trace-row-label">
        <Icon size={12} />
        {label}
      </span>
      <div className="tool-trace-row-value">{children}</div>
    </div>
  );
}
