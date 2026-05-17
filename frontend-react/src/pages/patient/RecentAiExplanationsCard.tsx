import { useMemo } from "react";
import { MessageSquare, Sparkles, Info } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { SectionCard } from "../../components/ui/SectionCard";
import { EmptyPane } from "../../components/ui/Spinner";
import type { ChatMessage } from "../../types/api";

interface Props { messages: ChatMessage[] }

function trim(text: string, max = 180): string {
  if (!text) return "";
  const single = text.replace(/\s+/g, " ").trim();
  return single.length > max ? `${single.slice(0, max - 1)}…` : single;
}

/**
 * Shows the most recent 3 assistant replies as a quick recap of what the
 * patient has been asking the AI about.  Clicking the card jumps to the
 * full chat surface where citations + saved actions are visible.
 */
export function RecentAiExplanationsCard({ messages }: Props) {
  const navigate = useNavigate();
  const recent = useMemo(() => {
    const assistantTurns = (messages ?? []).filter((m) => m?.role === "assistant" && m?.message);
    return assistantTurns.slice(-3).reverse();
  }, [messages]);

  return (
    <SectionCard
      title="Recent AI explanations"
      icon={MessageSquare}
      action={
        <button
          type="button"
          className="text-[0.75rem] font-medium"
          style={{ color: "var(--accent)" }}
          onClick={() => navigate("/patient/chat")}
        >
          Open chat →
        </button>
      }
      footer={
        recent.length > 0 ? (
          <span className="inline-flex items-center gap-1.5">
            <Info size={11} aria-hidden="true" />
            AI-generated replies. Verify anything actionable with your care team.
          </span>
        ) : undefined
      }
    >
      {recent.length === 0 ? (
        <EmptyPane label="No recent assistant replies yet — ask a question in the support chat." />
      ) : (
        <ul className="flex flex-col gap-2.5">
          {recent.map((m, i) => (
            <li
              key={i}
              className="rounded-md px-3 py-2"
              style={{
                background: "var(--surface2)",
                border: "1px solid var(--border-soft)",
                fontSize: "0.78rem",
                lineHeight: 1.5,
                color: "var(--text)",
              }}
            >
              <span
                className="inline-flex items-center gap-1"
                style={{
                  fontSize: "0.66rem",
                  fontWeight: 600,
                  textTransform: "uppercase",
                  letterSpacing: "0.06em",
                  color: "var(--rose-deep)",
                  marginRight: 6,
                }}
                aria-label="AI-generated"
              >
                <Sparkles size={10} aria-hidden="true" /> AI
              </span>
              {trim(m.message)}
            </li>
          ))}
        </ul>
      )}
    </SectionCard>
  );
}
