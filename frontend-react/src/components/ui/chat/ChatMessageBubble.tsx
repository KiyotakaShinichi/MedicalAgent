import { useState } from "react";
import { clsx } from "clsx";
import { BookOpen, CheckCircle2, ShieldCheck, Sparkles, Undo2, User, X } from "lucide-react";

import type { SavedAction } from "../../../types/api";
import type { NormalisedMessage } from "../chat-utils";
import { MarkdownMessage } from "../MarkdownMessage";
import { savedActionDescriptor, type ChipTone } from "./savedActionPresentation";

const CHIP_STYLE: Record<ChipTone, { bg: string; fg: string; border: string }> = {
  success: { bg: "#ecfdf5", fg: "#047857", border: "#a7f3d0" },
  warning: { bg: "#fffbeb", fg: "#92400e", border: "#fde68a" },
  info: { bg: "#f0f7ff", fg: "#24527a", border: "#bfd8ee" },
};

function ActionChip({
  action,
  canConfirm,
  onQuickReply,
  onUndoAction,
}: {
  action: SavedAction;
  canConfirm?: boolean;
  onQuickReply?: (message: string) => void;
  onUndoAction?: (auditId: number) => Promise<void>;
}) {
  const [undoState, setUndoState] = useState<"idle" | "working" | "done">("idle");
  const { label, Icon, tone } = savedActionDescriptor(action);
  const style = CHIP_STYLE[tone];
  if (action.type === "pending_record_confirmation") {
    return (
      <div className="chat-confirmation-card" role="group" aria-label="Confirm patient record preview">
        <div className="chat-confirmation-title">
          <ShieldCheck size={14} aria-hidden="true" />
          Review before saving
        </div>
        <p>{String(action.preview ?? "Review the extracted record values.")}</p>
        <span>Nothing is saved until you confirm.</span>
        {canConfirm && onQuickReply && (
          <div className="chat-confirmation-actions">
            <button type="button" onClick={() => onQuickReply("Confirm save")}>
              <CheckCircle2 size={13} /> Confirm save
            </button>
            <button type="button" className="secondary" onClick={() => onQuickReply("Cancel save")}>
              <X size={13} /> Cancel
            </button>
          </div>
        )}
      </div>
    );
  }
  const auditId = typeof action.audit_action_id === "number" ? action.audit_action_id : null;
  const canUndo = Boolean(action.undo_available && auditId != null && onUndoAction && undoState !== "done");
  return (
    <span className="chat-action-chip-wrap">
      <span
        className="inline-flex items-center gap-1.5 text-[0.72rem] px-2 py-0.5 rounded-full border font-medium"
        style={{ background: style.bg, borderColor: style.border, color: style.fg }}
      >
        <Icon size={11} />
        {label}
      </span>
      {canUndo && auditId != null && (
        <button
          type="button"
          className="chat-action-undo"
          disabled={undoState === "working"}
          onClick={async () => {
            if (!onUndoAction || undoState !== "idle") return;
            setUndoState("working");
            try {
              await onUndoAction(auditId);
              setUndoState("done");
            } catch {
              setUndoState("idle");
            }
          }}
        >
          <Undo2 size={11} /> {undoState === "working" ? "Undoing..." : "Undo"}
        </button>
      )}
      {undoState === "done" && <span className="chat-action-undone">Entry removed</span>}
    </span>
  );
}

interface ChatMessageBubbleProps {
  message: NormalisedMessage;
  isLatestAssistant?: boolean;
  registerNode?: (node: HTMLDivElement | null) => void;
  onQuickReply?: (message: string) => void;
  onUndoAction?: (auditId: number) => Promise<void>;
}

export function ChatMessageBubble({
  message,
  isLatestAssistant,
  registerNode,
  onQuickReply,
  onUndoAction,
}: ChatMessageBubbleProps) {
  const isUser = message.role === "user";
  const content = message.content || (isUser ? "" : "...");

  return (
    <div
      ref={isLatestAssistant ? registerNode : undefined}
      data-testid={isUser ? "user-message" : "assistant-message"}
      data-message-ready={isUser || Boolean(message.content) ? "true" : "false"}
      className={clsx("chat-message-row flex gap-3 items-start", isUser ? "flex-row-reverse" : "flex-row")}
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

      <div className={clsx("chat-message-body flex flex-col gap-1.5 min-w-0", isUser ? "items-end" : "items-start")}>
        <div className="text-[0.72rem] font-medium" style={{ color: "var(--text-faint)" }}>
          {isUser ? "You" : "NLCare assistant"}
        </div>
        <div
          className={clsx("chat-message-bubble text-[0.92rem]", isUser ? "is-user" : "is-assistant")}
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
            <span style={{ whiteSpace: "pre-wrap", lineHeight: 1.55 }}>{content}</span>
          ) : (
            <MarkdownMessage text={content} />
          )}
        </div>
        {message.saved_actions.length > 0 && (
          <div className="flex flex-wrap gap-1.5 mt-0.5">
            {message.saved_actions.map((action, index) => (
              <ActionChip
                key={index}
                action={action}
                canConfirm={isLatestAssistant}
                onQuickReply={onQuickReply}
                onUndoAction={onUndoAction}
              />
            ))}
          </div>
        )}
        {message.citations.length > 0 && (
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
