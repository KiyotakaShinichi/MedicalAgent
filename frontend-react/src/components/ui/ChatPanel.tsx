import { useState, useRef, useEffect, useCallback } from "react";
import { Send, Bot, User, AlertCircle } from "lucide-react";
import { clsx } from "clsx";
import { Spinner } from "./Spinner";
import type { ChatMessage, SavedAction } from "../../types/api";

const PIPELINE_STAGES = [
  { label: "Checking safety gate...", delay: 0 },
  { label: "Routing intent...", delay: 300 },
  { label: "Retrieving context...", delay: 700 },
  { label: "Generating response...", delay: 1500 },
];

function usePipelineStatus(active: boolean) {
  const [timing, setTiming] = useState<{ startedAt: number; now: number } | null>(null);

  useEffect(() => {
    let cancelled = false;

    if (!active) {
      window.setTimeout(() => {
        if (!cancelled) setTiming(null);
      }, 0);
      return () => {
        cancelled = true;
      };
    }

    const startedAt = Date.now();
    window.setTimeout(() => {
      if (!cancelled) setTiming({ startedAt, now: startedAt });
    }, 0);

    const interval = window.setInterval(() => {
      setTiming((current) => current ? { ...current, now: Date.now() } : current);
    }, 150);

    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [active]);

  if (!active || !timing) {
    return PIPELINE_STAGES[0].label;
  }

  const elapsedMs = Math.max(0, timing.now - timing.startedAt);
  const stageIndex = PIPELINE_STAGES.findLastIndex((step) => elapsedMs >= step.delay);
  return PIPELINE_STAGES[Math.max(0, stageIndex)].label;
}

interface Props {
  messages: ChatMessage[];
  onSend: (text: string) => Promise<{ reply: string; saved_actions?: SavedAction[]; citations?: string[] }>;
  loading?: boolean;
  disabled?: boolean;
  placeholder?: string;
}

function ActionChip({ action }: { action: SavedAction }) {
  const labels: Record<string, string> = {
    save_symptom: "Symptom saved",
    save_lab: "Lab saved",
    save_medication: "Medication saved",
    save_mri: "MRI note saved",
  };
  return (
    <span
      className="inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded border"
      style={{
        background: "rgba(16,185,129,0.08)",
        borderColor: "rgba(16,185,129,0.2)",
        color: "var(--green)",
      }}
    >
      Saved: {labels[action.type] ?? action.type}
    </span>
  );
}

export function ChatPanel({ messages: initialMessages, onSend, disabled, placeholder }: Props) {
  const [messages, setMessages] = useState<ChatMessage[]>(initialMessages);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const pipelineLabel = usePipelineStatus(sending);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, sending]);

  // Auto-resize the textarea up to a max so the chat input grows
  // with the message but never balloons the layout.
  useEffect(() => {
    const node = textareaRef.current;
    if (!node) return;
    node.style.height = "auto";
    const max = 160;
    node.style.height = `${Math.min(max, node.scrollHeight)}px`;
  }, [input]);

  const send = useCallback(async () => {
    const text = input.trim();
    if (!text || sending) return;
    setInput("");
    setError(null);
    setSending(true);
    setMessages((prev) => [...prev, { role: "user", message: text }]);
    try {
      const res = await onSend(text);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          message: res.reply,
          saved_actions_json: res.saved_actions?.length
            ? JSON.stringify(res.saved_actions)
            : undefined,
          citations: res.citations,
        },
      ]);
    } catch (e: unknown) {
      setError((e as Error).message);
    } finally {
      setSending(false);
    }
  }, [input, sending, onSend]);

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void send();
    }
  }

  return (
    <div
      className="flex flex-col"
      style={{ height: "100%", minHeight: 0, flex: 1, width: "100%" }}
    >
      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-4 flex flex-col gap-3" style={{ minHeight: 0 }}>
        {messages.length === 0 && (
          <div
            className="flex flex-col items-center gap-3 pt-12 pb-4 text-center mx-auto"
            style={{ color: "var(--text-faint)", maxWidth: 460 }}
          >
            <span
              className="inline-flex items-center justify-center"
              style={{
                width: 48,
                height: 48,
                borderRadius: 16,
                background: "rgba(244,63,94,0.12)",
                color: "var(--rose)",
              }}
            >
              <Bot size={22} aria-hidden="true" />
            </span>
            <p className="text-sm" style={{ color: "var(--text-dim)" }}>
              Ask about your monitoring data, symptoms, medications, or how this portal works.
            </p>
            <p className="text-xs" style={{ color: "var(--text-faint)" }}>
              This assistant is for monitoring support only and does not diagnose or recommend treatment.
            </p>
          </div>
        )}
        {messages.map((msg, i) => {
          const isUser = msg.role === "user";
          let actions: SavedAction[] = [];
          try {
            if (msg.saved_actions_json) actions = JSON.parse(msg.saved_actions_json);
          } catch { /* */ }
          return (
            <div
              key={i}
              className={clsx("flex gap-2", isUser ? "justify-end" : "justify-start")}
            >
              {!isUser && (
                <span
                  className="flex-shrink-0 w-6 h-6 rounded-full flex items-center justify-center mt-0.5"
                  style={{ background: "rgba(244,63,94,0.15)" }}
                >
                  <Bot size={12} style={{ color: "var(--rose)" }} />
                </span>
              )}
              <div className={clsx("flex flex-col gap-1.5", isUser ? "items-end" : "items-start", "max-w-[85%]")}>
                <div
                  className="px-3 py-2 rounded-lg text-sm whitespace-pre-wrap"
                  style={{
                    background: isUser ? "var(--rose-dim)" : "var(--surface2)",
                    color: isUser ? "#fff" : "var(--text)",
                    borderRadius: isUser ? "12px 12px 2px 12px" : "2px 12px 12px 12px",
                  }}
                >
                  {msg.message}
                </div>
                {actions.length > 0 && (
                  <div className="flex flex-wrap gap-1">
                    {actions.map((a, j) => <ActionChip key={j} action={a} />)}
                  </div>
                )}
                {msg.citations && msg.citations.length > 0 && (
                  <p className="text-xs" style={{ color: "var(--text-faint)" }}>
                    Sources: {msg.citations.slice(0, 3).join(", ")}
                  </p>
                )}
              </div>
              {isUser && (
                <span
                  className="flex-shrink-0 w-6 h-6 rounded-full flex items-center justify-center mt-0.5"
                  style={{ background: "var(--surface2)" }}
                >
                  <User size={12} style={{ color: "var(--text-dim)" }} />
                </span>
              )}
            </div>
          );
        })}
        {sending && (
          <div className="flex gap-2 items-center" style={{ color: "var(--text-dim)" }}>
            <span
              className="w-6 h-6 rounded-full flex items-center justify-center flex-shrink-0"
              style={{ background: "rgba(244,63,94,0.15)" }}
            >
              <Bot size={12} style={{ color: "var(--rose)" }} />
            </span>
            <div className="flex gap-1.5 items-center px-3 py-2 rounded-lg" style={{ background: "var(--surface2)" }}>
              <Spinner size={12} />
              <span className="text-xs transition-all duration-300" style={{ color: "var(--text-dim)" }}>
                {pipelineLabel}
              </span>
            </div>
          </div>
        )}
        {error && (
          <div
            className="flex items-center gap-2 text-xs px-3 py-2 rounded-lg border"
            style={{
              background: "rgba(244,63,94,0.07)",
              borderColor: "rgba(244,63,94,0.25)",
              color: "var(--rose)",
            }}
          >
            <AlertCircle size={12} />
            {error}
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Disclaimer */}
      <p
        className="text-xs text-center px-3 py-1 border-t"
        style={{ borderColor: "var(--border)", color: "var(--text-faint)" }}
      >
        Not a substitute for clinical advice. Always consult your care team.
      </p>

      {/* Input */}
      <div
        className="flex items-end gap-2 p-3 border-t"
        style={{ borderColor: "var(--border)", background: "var(--surface)" }}
      >
        <textarea
          ref={textareaRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder ?? "Ask about your monitoring data... (Enter to send, Shift+Enter for newline)"}
          disabled={disabled || sending}
          rows={1}
          className="flex-1 resize-none rounded-lg px-3 py-2.5 text-sm outline-none focus:ring-1"
          style={{
            background: "var(--surface2)",
            border: "1px solid var(--border)",
            color: "var(--text)",
            maxHeight: 160,
            lineHeight: 1.5,
          }}
        />
        <button
          onClick={send}
          disabled={!input.trim() || sending || disabled}
          className="rounded-lg transition-opacity hover:opacity-90 disabled:opacity-30"
          style={{
            background: "var(--rose)",
            color: "#fff",
            flexShrink: 0,
            height: 40,
            width: 40,
            display: "inline-flex",
            alignItems: "center",
            justifyContent: "center",
          }}
          aria-label="Send message"
        >
          <Send size={16} />
        </button>
      </div>
    </div>
  );
}
