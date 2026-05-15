import { useState, useRef, useEffect, useCallback } from "react";
import {
  Send,
  Sparkles,
  User,
  AlertCircle,
  Activity,
  FlaskConical,
  ScanLine,
  HelpCircle,
  ShieldCheck,
  CheckCircle2,
  BookOpen,
  Pill,
  AlertTriangle,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { clsx } from "clsx";
import { Spinner } from "./Spinner";
import { MarkdownMessage } from "./MarkdownMessage";
import type { ChatMessage as ChatMessageType, ChatStreamHandlers, SavedAction } from "../../types/api";

const PIPELINE_STAGES = [
  { label: "Checking safety gate...", delay: 0 },
  { label: "Routing intent...",       delay: 300 },
  { label: "Retrieving context...",   delay: 700 },
  { label: "Generating response...",  delay: 1500 },
];

function usePipelineStatus(active: boolean) {
  const [timing, setTiming] = useState<{ startedAt: number; now: number } | null>(null);

  useEffect(() => {
    let cancelled = false;
    if (!active) {
      window.setTimeout(() => { if (!cancelled) setTiming(null); }, 0);
      return () => { cancelled = true; };
    }
    const startedAt = Date.now();
    window.setTimeout(() => { if (!cancelled) setTiming({ startedAt, now: startedAt }); }, 0);
    const interval = window.setInterval(() => {
      setTiming((current) => current ? { ...current, now: Date.now() } : current);
    }, 150);
    return () => { cancelled = true; window.clearInterval(interval); };
  }, [active]);

  if (!active || !timing) return PIPELINE_STAGES[0].label;
  const elapsedMs = Math.max(0, timing.now - timing.startedAt);
  const idx = PIPELINE_STAGES.findLastIndex((step) => elapsedMs >= step.delay);
  return PIPELINE_STAGES[Math.max(0, idx)].label;
}

interface ChatPanelProps {
  messages: ChatMessageType[];
  onSend: (text: string) => Promise<{ reply: string; saved_actions?: SavedAction[]; citations?: string[] }>;
  onSendStream?: (text: string, handlers: ChatStreamHandlers) => Promise<{ reply: string; saved_actions?: SavedAction[]; citations?: string[] }>;
  /** Called whenever a response includes one or more saved_actions (used by parent to refetch state). */
  onSavedActions?: (actions: SavedAction[]) => void;
  loading?: boolean;
  disabled?: boolean;
  placeholder?: string;
}

const QUICK_PROMPTS: { icon: typeof Activity; label: string; prompt: string }[] = [
  { icon: Activity,     label: "Log a symptom",        prompt: "I would like to log a new symptom. Help me record it." },
  { icon: FlaskConical, label: "Save lab result",      prompt: "I want to save my latest CBC lab results." },
  { icon: ScanLine,     label: "Save MRI report",      prompt: "I want to save a new MRI report I received." },
  { icon: HelpCircle,   label: "Ask about the portal", prompt: "How does this portal work and what can I do here?" },
];

type ChipTone = "success" | "warning";

interface ChipDescriptor {
  label: string;
  Icon: LucideIcon;
  tone: ChipTone;
}

function chipDescriptor(action: SavedAction): ChipDescriptor {
  switch (action.type) {
    case "saved_symptom":
    case "save_symptom":
      return { label: "Symptom saved", Icon: Activity, tone: "success" };
    case "saved_labs":
    case "save_lab":
      return { label: "CBC saved", Icon: FlaskConical, tone: "success" };
    case "saved_medication":
    case "save_medication":
      return { label: "Medication saved", Icon: Pill, tone: "success" };
    case "saved_imaging_report":
    case "save_mri": {
      const modality = String((action.data as { modality?: unknown })?.modality ?? "").toLowerCase();
      const label =
        modality.includes("mri")        ? "MRI report saved" :
        modality.includes("ct")         ? "CT report saved" :
        modality.includes("ultrasound") ? "Ultrasound report saved" :
                                          "Imaging report saved";
      return { label, Icon: ScanLine, tone: "success" };
    }
    case "possible_metastatic_indicator":
      return { label: "Review flag added", Icon: AlertTriangle, tone: "warning" };
    default:
      return { label: action.type, Icon: CheckCircle2, tone: "success" };
  }
}

const CHIP_STYLE: Record<ChipTone, { bg: string; fg: string; border: string }> = {
  success: { bg: "#ecfdf5", fg: "#047857", border: "#a7f3d0" },
  warning: { bg: "#fffbeb", fg: "#92400e", border: "#fde68a" },
};

// eslint-disable-next-line react-refresh/only-export-components
export function describeSavedAction(action: SavedAction): { label: string; tone: ChipTone } {
  const { label, tone } = chipDescriptor(action);
  return { label, tone };
}

function parseSavedActions(raw?: string): SavedAction[] {
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw);
    if (Array.isArray(parsed)) return parsed;
    if (Array.isArray(parsed?.saved_actions)) return parsed.saved_actions;
  } catch { /* ignore malformed legacy payloads */ }
  return [];
}

function ActionChip({ action }: { action: SavedAction }) {
  const { label, Icon, tone } = chipDescriptor(action);
  const style = CHIP_STYLE[tone];
  return (
    <span
      className="inline-flex items-center gap-1.5 text-[0.72rem] px-2 py-0.5 rounded-full border font-medium"
      style={{ background: style.bg, borderColor: style.border, color: style.fg }}
    >
      <Icon size={11} />
      {label}
    </span>
  );
}

interface MessageProps {
  message: ChatMessageType;
  isLatestAssistant?: boolean;
  registerNode?: (node: HTMLDivElement | null) => void;
}

function ChatBubble({ message, isLatestAssistant, registerNode }: MessageProps) {
  const isUser = message.role === "user";
  const actions = parseSavedActions(message.saved_actions_json);

  return (
    <div
      ref={isLatestAssistant ? registerNode : undefined}
      className={clsx("flex gap-3 items-start", isUser ? "flex-row-reverse" : "flex-row")}
    >
      <span
        className="flex-shrink-0 inline-flex items-center justify-center"
        style={{
          width: 32,
          height: 32,
          borderRadius: 10,
          background: isUser ? "var(--surface2)" : "var(--rose-pale)",
          color: isUser ? "var(--text-dim)" : "var(--rose-deep)",
          border: "1px solid var(--border)",
        }}
        aria-hidden="true"
      >
        {isUser ? <User size={14} /> : <Sparkles size={14} />}
      </span>

      <div className={clsx("flex flex-col gap-1.5 min-w-0", isUser ? "items-end" : "items-start")} style={{ maxWidth: "78%" }}>
        <div className="text-[0.72rem] font-medium" style={{ color: "var(--text-faint)" }}>
          {isUser ? "You" : "Support assistant"}
        </div>
        <div
          className="text-[0.92rem]"
          style={{
            padding: "10px 14px",
            borderRadius: isUser ? "14px 14px 4px 14px" : "14px 14px 14px 4px",
            background: isUser ? "var(--rose)" : "var(--surface)",
            color: isUser ? "#fff" : "var(--text)",
            border: isUser ? "none" : "1px solid var(--border)",
            boxShadow: isUser
              ? "0 2px 8px rgba(236,72,153,0.18)"
              : "0 1px 2px rgba(17,24,39,0.04)",
            wordBreak: "break-word",
          }}
        >
          {isUser ? (
            <span style={{ whiteSpace: "pre-wrap", lineHeight: 1.55 }}>{message.message}</span>
          ) : (
            <MarkdownMessage text={message.message} />
          )}
        </div>
        {actions.length > 0 && (
          <div className="flex flex-wrap gap-1.5 mt-0.5">
            {actions.map((a, j) => <ActionChip key={j} action={a} />)}
          </div>
        )}
        {message.citations && message.citations.length > 0 && (
          <div
            className="flex items-center gap-1.5 text-[0.72rem] mt-0.5"
            style={{ color: "var(--text-faint)" }}
          >
            <BookOpen size={11} aria-hidden="true" />
            Sources: {message.citations.slice(0, 3).join(", ")}
          </div>
        )}
      </div>
    </div>
  );
}

export function ChatPanel({ messages: initialMessages, onSend, onSendStream, onSavedActions, disabled, placeholder }: ChatPanelProps) {
  const [messages, setMessages] = useState<ChatMessageType[]>(initialMessages);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [streamStage, setStreamStage] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const latestAssistantRef = useRef<HTMLDivElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const timedPipelineLabel = usePipelineStatus(sending);
  const pipelineLabel = streamStage || timedPipelineLabel;

  /**
   * Scroll behavior:
   *   • while sending → keep the bottom in view so the user sees their own
   *     message and the typing indicator.
   *   • new assistant message arrives → scroll its TOP into view so the user
   *     reads from the beginning of a long answer (long responses no longer
   *     get cut off at the top).
   */
  useEffect(() => {
    if (sending) {
      bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
      return;
    }
    const lastMessage = messages[messages.length - 1];
    if (!lastMessage) return;
    if (lastMessage.role === "assistant" && latestAssistantRef.current) {
      latestAssistantRef.current.scrollIntoView({ behavior: "smooth", block: "start" });
    } else {
      bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
    }
  }, [messages, sending]);

  // Auto-grow textarea, but never let the content area shrink below one full line.
  useEffect(() => {
    const node = textareaRef.current;
    if (!node) return;
    node.style.height = "auto";
    const max = 140;
    node.style.height = `${Math.min(max, node.scrollHeight)}px`;
  }, [input]);

  const send = useCallback(async (text?: string) => {
    const value = (text ?? input).trim();
    if (!value || sending) return;
    setInput("");
    setError(null);
    setStreamStage(null);
    setSending(true);
    setMessages((prev) => [...prev, { role: "user", message: value }]);
    try {
      let assistantIndex = -1;
      const res = onSendStream
        ? await onSendStream(value, {
          onStage: (label) => { if (label) setStreamStage(label); },
          onDelta: (delta) => {
            if (!delta) return;
            setMessages((prev) => {
              const next = [...prev];
              if (assistantIndex < 0) {
                assistantIndex = next.length;
                next.push({ role: "assistant", message: delta });
              } else {
                next[assistantIndex] = {
                  ...next[assistantIndex],
                  message: `${next[assistantIndex].message}${delta}`,
                };
              }
              return next;
            });
          },
        })
        : await onSend(value);
      const savedActions = res.saved_actions ?? [];
      setMessages((prev) => {
        const finalMessage = {
          role: "assistant" as const,
          message: res.reply,
          saved_actions_json: savedActions.length ? JSON.stringify(savedActions) : undefined,
          citations: res.citations,
        };
        if (assistantIndex >= 0 && assistantIndex < prev.length) {
          const next = [...prev];
          next[assistantIndex] = finalMessage;
          return next;
        }
        return [...prev, finalMessage];
      });
      if (savedActions.length && onSavedActions) {
        // Let the parent refetch report-level state (symptom log, labs, etc.)
        // so anything saved by the assistant shows up immediately.
        onSavedActions(savedActions);
      }
    } catch (e: unknown) {
      setError((e as Error).message);
    } finally {
      setStreamStage(null);
      setSending(false);
    }
  }, [input, sending, onSend, onSendStream, onSavedActions]);

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void send();
    }
  }

  const isEmpty = messages.length === 0;
  // Identify the index of the latest assistant message so we can attach the
  // scroll-target ref to exactly that bubble.
  const lastAssistantIndex = (() => {
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === "assistant") return i;
    }
    return -1;
  })();

  return (
    <div className="chat-shell">
      {/* Scrollable message area */}
      <div ref={scrollRef} className="chat-scroll">
        <div className="chat-stream">
          {isEmpty && (
            <div className="chat-empty">
              <span className="chat-empty-icon" aria-hidden="true">
                <Sparkles size={22} />
              </span>
              <div>
                <h3 className="chat-empty-title">How can I support you today?</h3>
                <p className="chat-empty-body">
                  Ask about your monitoring data, log a symptom, save a lab or imaging report, or
                  ask how the portal works. I can also explain general oncology terms.
                </p>
              </div>
              <div className="chat-quick-actions">
                {QUICK_PROMPTS.map(({ icon: Icon, label, prompt }) => (
                  <button
                    key={label}
                    type="button"
                    onClick={() => { void send(prompt); }}
                    className="chat-quick-action"
                  >
                    <Icon size={13} />
                    {label}
                  </button>
                ))}
              </div>
              <p className="chat-empty-disclaimer">
                Not medical advice. Your care team makes treatment decisions.
              </p>
            </div>
          )}

          {messages.map((msg, i) => (
            <ChatBubble
              key={i}
              message={msg}
              isLatestAssistant={i === lastAssistantIndex}
              registerNode={(node) => { latestAssistantRef.current = node; }}
            />
          ))}

          {sending && (
            <div className="flex gap-3 items-start">
              <span
                className="flex-shrink-0 inline-flex items-center justify-center"
                style={{
                  width: 32, height: 32, borderRadius: 10,
                  background: "var(--rose-pale)", color: "var(--rose-deep)",
                  border: "1px solid var(--border)",
                }}
                aria-hidden="true"
              >
                <Sparkles size={14} />
              </span>
              <div
                className="flex gap-2 items-center text-sm"
                style={{
                  padding: "10px 14px",
                  borderRadius: "14px 14px 14px 4px",
                  background: "var(--surface)",
                  border: "1px solid var(--border)",
                  color: "var(--text-dim)",
                }}
              >
                <Spinner size={13} />
                <span>{pipelineLabel}</span>
              </div>
            </div>
          )}

          {error && (
            <div
              className="flex items-center gap-2 text-xs rounded-lg border px-3 py-2"
              style={{ background: "#fef2f2", borderColor: "#fecaca", color: "#b91c1c" }}
            >
              <AlertCircle size={13} />
              {error}
            </div>
          )}

          <div ref={bottomRef} />
        </div>
      </div>

      {/* Composer */}
      <div className="chat-composer">
        <div className="chat-composer-inner">
          <div className="chat-composer-box">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={placeholder ?? "Tell me how you are feeling, or ask a question..."}
              disabled={disabled || sending}
              rows={1}
              className="chat-composer-textarea"
            />
            <button
              type="button"
              onClick={() => { void send(); }}
              disabled={!input.trim() || sending || disabled}
              className="chat-composer-send"
              aria-label="Send message"
            >
              <Send size={15} />
            </button>
          </div>

          <p className="chat-composer-disclaimer">
            <ShieldCheck size={11} aria-hidden="true" />
            Not medical advice. Your care team makes treatment decisions.
          </p>
        </div>
      </div>
    </div>
  );
}
