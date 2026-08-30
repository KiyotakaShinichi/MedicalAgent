import { useState, useRef, useEffect, useCallback, useMemo } from "react";
import {
  Send,
  Sparkles,
  AlertCircle,
  Activity,
  FlaskConical,
  ScanLine,
  HelpCircle,
  ShieldCheck,
} from "lucide-react";
import { Spinner } from "./Spinner";
import { ErrorBoundary } from "./ErrorBoundary";
import {
  normaliseMessages,
  parseCitations,
  type NormalisedMessage,
} from "./chat-utils";
import type { ChatMessage as ChatMessageType, ChatStreamHandlers, SavedAction } from "../../types/api";
import { ChatMessageBubble } from "./chat/ChatMessageBubble";
import { usePipelineStatus } from "./chat/usePipelineStatus";

interface ChatPanelProps {
  messages: ChatMessageType[];
  onSend: (text: string) => Promise<{ reply: string; saved_actions?: SavedAction[]; citations?: string[] }>;
  onSendStream?: (text: string, handlers: ChatStreamHandlers) => Promise<{ reply: string; saved_actions?: SavedAction[]; citations?: string[] }>;
  /** Called whenever a response includes one or more saved_actions (used by parent to refetch state). */
  onSavedActions?: (actions: SavedAction[]) => void;
  /** Undo a provenance-stamped write created by this patient. */
  onUndoAction?: (auditId: number) => Promise<void>;
  loading?: boolean;
  disabled?: boolean;
  placeholder?: string;
  /** Optional element rendered on the LEFT of the composer (next to the textarea).
   *  Used by the patient surface to inject the "+" attachment popover so the
   *  composer reads as: [+] [textarea] [send].  Keeping it as a slot lets the
   *  clinician chat reuse this component without the patient-only tools. */
  composerLeading?: React.ReactNode;
}

const QUICK_PROMPTS: { icon: typeof Activity; label: string; prompt: string }[] = [
  { icon: Activity,     label: "Log a symptom",        prompt: "I would like to log a new symptom. Help me record it." },
  { icon: FlaskConical, label: "Save lab result",      prompt: "I want to save my latest CBC lab results." },
  { icon: ScanLine,     label: "Save MRI report",      prompt: "I want to save a new MRI report I received." },
  { icon: HelpCircle,   label: "Ask about the portal", prompt: "How does this portal work and what can I do here?" },
];


/**
 * How close to the bottom (in px) counts as "user is reading the latest
 * messages" — within this distance we auto-scroll on new content, beyond it
 * we leave the user where they parked the scroll.
 */
const AUTO_SCROLL_FUDGE_PX = 80;

function ChatPanelInner({ messages: initialMessages, onSend, onSendStream, onSavedActions, onUndoAction, disabled, placeholder, composerLeading }: ChatPanelProps) {
  // ── Normalise the parent prop once per change.
  const normalisedParentMessages = useMemo(
    () => normaliseMessages(initialMessages),
    [initialMessages],
  );

  // We use **derived state** instead of syncing props into state via a
  // useEffect (which `react-hooks/set-state-in-effect` correctly flags as an
  // anti-pattern).  Display = parent messages + any local messages the
  // current send turn produced.  Local state holds only the in-flight turn;
  // it is cleared as soon as the parent's refetched history catches up.
  const [localMessages, setLocalMessages] = useState<NormalisedMessage[]>([]);
  const [sending, setSending] = useState(false);

  // Final display list: parent (canonical) + any locally-buffered messages
  // for the live turn, deduplicated by id.
  const messages = useMemo<NormalisedMessage[]>(() => {
    if (localMessages.length === 0) return normalisedParentMessages;
    const seen = new Set(normalisedParentMessages.map((m) => m.id));
    const parentContent = new Set(
      normalisedParentMessages.map((m) => `${m.role}\u0000${m.content}`),
    );
    const extras = localMessages.filter(
      (m) => !seen.has(m.id) && !parentContent.has(`${m.role}\u0000${m.content}`),
    );
    return [...normalisedParentMessages, ...extras];
  }, [normalisedParentMessages, localMessages]);

  // Once the parent's chat history catches up (post-refetch), we drop the
  // local turn buffer.  This effect is reacting to an external system (the
  // parent's refetched message list arriving via props) — exactly the case
  // the rule docs describe as a legitimate exception.  It short-circuits on
  // both early returns so the cascade is at most one render per turn.
  useEffect(() => {
    if (sending) return;
    if (localMessages.length === 0) return;
    if (localMessages.some((m) => m.streamId)) return;
    const parentContent = new Set(
      normalisedParentMessages.map((m) => `${m.role}\u0000${m.content}`),
    );
    const parentHasCompletedTurn = localMessages.every(
      (m) => parentContent.has(`${m.role}\u0000${m.content}`),
    );
    if (!parentHasCompletedTurn) return;
    // Schedule the clear so it runs after React commits, not synchronously
    // within the effect body — avoids the cascading-render anti-pattern.
    const id = window.setTimeout(() => setLocalMessages([]), 0);
    return () => window.clearTimeout(id);
  }, [sending, localMessages, normalisedParentMessages]);

  const [input, setInput] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [streamStage, setStreamStage] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const latestAssistantRef = useRef<HTMLDivElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  // True when the user is parked near the bottom; we only auto-scroll then.
  const stickToBottomRef = useRef<boolean>(true);
  const timedPipelineLabel = usePipelineStatus(sending);
  const pipelineLabel = streamStage || timedPipelineLabel;

  // Track whether the user has scrolled away from the bottom.
  useEffect(() => {
    const node = scrollRef.current;
    if (!node) return;
    const onScroll = () => {
      const distance = node.scrollHeight - node.scrollTop - node.clientHeight;
      stickToBottomRef.current = distance < AUTO_SCROLL_FUDGE_PX;
    };
    node.addEventListener("scroll", onScroll, { passive: true });
    return () => node.removeEventListener("scroll", onScroll);
  }, []);

  // Auto-scroll only when the user is already near the bottom, and use the
  // scroll container's own scrollTop instead of scrollIntoView so the outer
  // page never jumps.
  useEffect(() => {
    if (!stickToBottomRef.current) return;
    const node = scrollRef.current;
    if (!node) return;
    // requestAnimationFrame: wait until React has laid out the new content
    // before scrolling, so the height we read is correct.
    const frame = window.requestAnimationFrame(() => {
      node.scrollTop = node.scrollHeight;
    });
    return () => window.cancelAnimationFrame(frame);
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

    // ── Generate a unique stream id for THIS turn.  This id is the stable
    //    handle the streaming callbacks use to target the in-flight assistant
    //    bubble — never a captured-closure index.
    const streamId = `stream_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
    const userId   = `local_user_${streamId}`;
    const assistantId = `local_assistant_${streamId}`;

    // Optimistic insert into the LOCAL turn buffer.
    setLocalMessages((prev) => [
      ...prev,
      { id: userId, role: "user", content: value, saved_actions: [], citations: [] },
      { id: assistantId, role: "assistant", content: "", saved_actions: [], citations: [], streamId },
    ]);
    stickToBottomRef.current = true;

    try {
      const res = onSendStream
        ? await onSendStream(value, {
            onStage: (label) => { if (label) setStreamStage(label); },
            onDelta: (delta) => {
              if (!delta) return;
              // PURE updater keyed on a stable streamId — safe under
              // StrictMode's double-invoke, no outer-scope mutation.
              setLocalMessages((prev) => prev.map((m) =>
                m.streamId === streamId
                  ? { ...m, content: `${m.content}${delta}` }
                  : m,
              ));
            },
          })
        : await onSend(value);

      // Replace the streaming bubble with the final, server-canonical message.
      // We keep it in local state until the parent's refetched history catches
      // up — the dedupe effect above drops it cleanly when that happens.
      const reply = typeof res?.reply === "string" ? res.reply : "";
      const savedActions = Array.isArray(res?.saved_actions) ? res.saved_actions : [];
      const citations = parseCitations(res?.citations);
      setLocalMessages((prev) => prev.map((m) =>
        m.streamId === streamId
          ? {
              id: m.id,
              role: "assistant",
              content: reply,
              saved_actions: savedActions,
              citations,
            }
          : m,
      ));
      if (onSavedActions) {
        // Defer to next tick so we finish committing the bubble before the
        // parent fires refetches that may swap props underneath us.
        window.setTimeout(() => onSavedActions(savedActions), 0);
      }
    } catch (e: unknown) {
      // Pull the failed turn back out of the buffer and surface a friendly
      // inline error.  The underlying error is logged in dev for diagnosis.
      const friendlyMessage = (e instanceof Error && e.message) ? e.message : "Something went wrong sending that message.";
      if (import.meta.env?.DEV) {
        console.error("[ChatPanel] send failed:", e);
      }
      setError(friendlyMessage);
      setLocalMessages((prev) => prev.filter((m) => m.id !== userId && m.id !== assistantId));
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
  // Identify the latest assistant message so we can attach the scroll-target ref.
  let lastAssistantIndex = -1;
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === "assistant") { lastAssistantIndex = i; break; }
  }

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
            <ChatMessageBubble
              key={msg.id}
              message={msg}
              isLatestAssistant={i === lastAssistantIndex}
              registerNode={(node) => { latestAssistantRef.current = node; }}
              onQuickReply={(message) => { void send(message); }}
              onUndoAction={onUndoAction}
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
              className="flex items-start gap-2 text-xs rounded-lg border px-3 py-2"
              style={{ background: "#fef2f2", borderColor: "#fecaca", color: "#b91c1c" }}
              role="alert"
            >
              <AlertCircle size={13} style={{ marginTop: 1, flexShrink: 0 }} />
              <span style={{ flex: 1 }}>{error}</span>
              <button
                type="button"
                onClick={() => setError(null)}
                aria-label="Dismiss error"
                style={{
                  background: "transparent",
                  border: 0,
                  color: "#b91c1c",
                  opacity: 0.7,
                  cursor: "pointer",
                  fontSize: "0.74rem",
                  padding: "0 4px",
                }}
              >
                ×
              </button>
            </div>
          )}

          <div ref={bottomRef} />
        </div>
      </div>

      {/* Composer */}
      <div className="chat-composer">
        <div className="chat-composer-inner">
          <div className="chat-composer-box">
            {composerLeading && (
              <div className="chat-composer-leading">{composerLeading}</div>
            )}
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

/**
 * Public wrapper: keeps the chat sandboxed in its own ErrorBoundary so a
 * render-time exception (e.g. a malformed message arriving from the backend)
 * shows a recoverable inline state instead of taking the whole patient route
 * down to the App-level boundary.
 */
export function ChatPanel(props: ChatPanelProps) {
  return (
    <ErrorBoundary surface="the support chat">
      <ChatPanelInner {...props} />
    </ErrorBoundary>
  );
}
