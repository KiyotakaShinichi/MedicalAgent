import { LayoutDashboard, FlaskConical, Activity, Clock, MessageSquare, Pill, Dna } from "lucide-react";
import { useLocation, useNavigate } from "react-router-dom";
import { useEffect } from "react";
import { AppShell } from "../../components/layout/AppShell";
import { useApi } from "../../hooks/useApi";
import { getMyReport, sendMyChat, sendMyChatStream, getMyChatHistory } from "../../api/client";
import { ErrorPane } from "../../components/ui/Spinner";
import { SkeletonDashboard } from "../../components/ui/Skeleton";
import { PatientBanner } from "./PatientBanner";
import { AiSummaryPanel } from "./AiSummaryPanel";
import { LabsPanel } from "./LabsPanel";
import { TimelinePanel } from "./TimelinePanel";
import { SymptomsTable } from "./SymptomsTable";
import { ModelSignalPanel } from "./ModelSignalPanel";
import { GeneticCounselingPanel } from "./GeneticCounselingPanel";
import { SectionCard } from "../../components/ui/SectionCard";
import { EmptyPane } from "../../components/ui/Spinner";
import { ChatPanel, describeSavedAction } from "../../components/ui/ChatPanel";
import { ToolTraceDrawer } from "../../components/ui/ToolTraceDrawer";
import { useToast } from "../../components/ui/Toast";
import { useAuth } from "../../hooks/useAuth";
import type { ChatMessage, MedicationLog, SavedAction } from "../../types/api";

const NAV = [
  { to: "/patient",          label: "Overview",  icon: LayoutDashboard },
  { to: "/patient#labs",     label: "Labs",      icon: FlaskConical },
  { to: "/patient#signals",  label: "Signals",   icon: Activity },
  { to: "/patient#timeline", label: "Timeline",  icon: Clock },
  { to: "/patient#genetics", label: "Family & Genetics", icon: Dna },
  { to: "/patient/chat",     label: "Support",   icon: MessageSquare },
];

type Tab = "overview" | "chat";

export default function PatientDashboard() {
  const { patientId } = useAuth();
  const location = useLocation();
  const navigate = useNavigate();
  const toast = useToast();
  const { data: report, status, error, refetch: refetchReport, lastFetchedAt: reportFetchedAt } = useApi(getMyReport, [patientId]);
  const { data: chatData, refetch: refetchChat } = useApi(getMyChatHistory, [patientId]);
  const tab: Tab = location.pathname.includes("/chat") ? "chat" : "overview";
  const chatMessages = chatData?.messages ?? report?.chat_history ?? [];
  const chatKey = buildChatKey(patientId ?? "patient", chatMessages);

  const tabs: { id: Tab; label: string }[] = [
    { id: "overview", label: "Overview" },
    { id: "chat",     label: "Support chat" },
  ];

  useEffect(() => {
    if (location.pathname !== "/patient") return;
    if (!location.hash) {
      window.scrollTo({ top: 0, behavior: "smooth" });
      const main = document.querySelector(".app-main");
      if (main) main.scrollTo({ top: 0, behavior: "smooth" });
      return;
    }
    const id = location.hash.slice(1);
    const timer = window.setTimeout(() => {
      const target = document.querySelector(`[data-section="${id}"]`);
      if (target) target.scrollIntoView({ behavior: "smooth", block: "start" });
    }, 60);
    return () => window.clearTimeout(timer);
  }, [location.pathname, location.hash, tab]);

  return (
    <AppShell
      navItems={NAV}
      title="Patient portal"
      subtitle={report?.patient_name ?? patientId ?? ""}
    >
      {status === "loading" && <SkeletonDashboard label="Loading your records..." />}
      {status === "error"   && <ErrorPane message={error ?? "Failed to load"} onRetry={refetchReport} />}
      {status === "success" && report && (
        <div className="dashboard-page">
          <PatientBanner report={report} />

          <div className="dashboard-tabbar">
            <div className="dashboard-tabbar-inner">
              {tabs.map(({ id, label }) => (
                <button
                  key={id}
                  onClick={() => navigate(id === "chat" ? "/patient/chat" : "/patient")}
                  className={tab === id ? "is-active" : undefined}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>

          {tab === "overview" && (
            <div className="dashboard-content dashboard-grid">
              {/* Row 1 — patient-facing summary + compact symptom log beside it */}
              <div className="col-8" data-section="summary">
                <AiSummaryPanel summary={report.ai_summary ?? null} />
              </div>
              <div className="col-4" data-section="symptoms">
                <SymptomsTable symptoms={report.symptoms ?? []} compact lastFetchedAt={reportFetchedAt} />
              </div>
              {/* Row 2 — top insight area: labs + model signal */}
              <div className="col-7" data-section="labs">
                <LabsPanel report={report} lastFetchedAt={reportFetchedAt} />
              </div>
              <div className="col-5" data-section="signals">
                <ModelSignalPanel report={report} />
              </div>
              {/* Row 3 — treatment timeline full-width below */}
              <div className="dashboard-grid-full" data-section="timeline">
                <TimelinePanel events={report.timeline ?? []} lastFetchedAt={reportFetchedAt} />
              </div>
              {/* Row 4 — medications full-width */}
              <div className="dashboard-grid-full" data-section="genetics">
                <GeneticCounselingPanel
                  readiness={report.genetic_counseling_readiness ?? null}
                  onSaved={refetchReport}
                />
              </div>
              <div className="dashboard-grid-full" data-section="medications">
                <MedLogPanel meds={report.medication_logs ?? []} />
              </div>
            </div>
          )}

          {tab === "chat" && (
            <div className="dashboard-content chat-workspace">
              <div className="chat-card-shell">
                <ChatPanel
                  key={chatKey}
                  messages={chatMessages}
                  onSend={async (text) => {
                    const res = await sendMyChat(text);
                    return {
                      reply: res.reply,
                      saved_actions: res.saved_actions,
                      citations: res.citations,
                    };
                  }}
                  onSendStream={async (text, handlers) => {
                    const res = await sendMyChatStream(text, handlers);
                    return {
                      reply: res.reply,
                      saved_actions: res.saved_actions,
                      citations: res.citations,
                    };
                  }}
                  onSavedActions={(actions: SavedAction[]) => {
                    // Any save action means the patient record changed — refetch
                    // so the symptom log / labs / timeline reflect it immediately.
                    const touchesReport = actions.some((a) =>
                      [
                        "saved_symptom",
                        "saved_labs",
                        "saved_medication",
                        "saved_imaging_report",
                        "save_symptom",
                        "save_lab",
                        "save_medication",
                        "save_mri",
                        "save_imaging_report",
                      ].includes(a.type),
                    );
                    if (touchesReport) refetchReport();
                    refetchChat();

                    // Surface a toast per save action with the right tone +
                    // human-readable label.  "possible_metastatic_indicator"
                    // becomes a warning toast, not a success.
                    for (const action of actions) {
                      const { label, tone } = describeSavedAction(action);
                      toast.push({
                        tone,
                        title: tone === "warning" ? label : `${label}.`,
                        description:
                          tone === "warning"
                            ? "Flagged for clinician review."
                            : "Patient record refreshed.",
                      });
                    }
                  }}
                />
                <ToolTraceDrawer messages={chatMessages} />
              </div>
            </div>
          )}
        </div>
      )}
    </AppShell>
  );
}

/**
 * Stable chat key — keyed only on the patient id, NOT on the message list.
 *
 * The previous implementation embedded `messages.length` and the last
 * message's content, which forced ChatPanel to remount every time a message
 * was sent or the history was refetched.  That remount dropped in-flight
 * streaming state (the optimistic assistant bubble + its streamId), and
 * combined with React StrictMode it caused the streaming setMessages
 * updater to read a stale closure index and crash with
 * `Cannot read properties of undefined (reading 'message')`.
 *
 * ChatPanel now syncs from props internally (only when not sending), so
 * we never need to remount the panel just because the message list changed.
 */
function buildChatKey(scope: string, _messages: ChatMessage[]) {
  void _messages; // intentionally unused — keyed only on patient/session scope
  return `chat:${scope}`;
}

function MedLogPanel({ meds }: { meds: MedicationLog[] }) {
  const sorted = [...meds].sort((a, b) => b.date.localeCompare(a.date));
  return (
    <SectionCard
      title="Medication log"
      icon={Pill}
      meta={sorted.length > 0 ? `${sorted.length} entries` : undefined}
    >
      {sorted.length === 0 ? (
        <EmptyPane label="No medications recorded — your care team can add these from the clinician portal." />
      ) : (
        <div className="overflow-x-auto">
          <table
            className="w-full text-[0.86rem]"
            style={{
              borderCollapse: "separate",
              borderSpacing: 0,
              minWidth: 520, /* triggers horizontal scroll on narrow screens
                                 rather than crushing columns. */
            }}
          >
            <thead>
              <tr>
                {["Date", "Medication", "Dose", "Frequency"].map((h, i) => (
                  <th
                    key={h}
                    className="text-left font-semibold py-2 px-3"
                    style={{
                      color: "var(--text-faint)",
                      fontSize: "0.72rem",
                      textTransform: "uppercase",
                      letterSpacing: "0.06em",
                      borderBottom: "1px solid var(--border)",
                      width: i === 0 ? 110 : i === 2 ? 90 : undefined,
                    }}
                  >
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {sorted.slice(0, 12).map((m, i) => (
                <tr
                  key={i}
                  className="transition-colors"
                  onMouseEnter={(e) => { e.currentTarget.style.background = "var(--surface2)"; }}
                  onMouseLeave={(e) => { e.currentTarget.style.background = "transparent"; }}
                >
                  <td
                    className="py-2.5 px-3 tabular-nums"
                    style={{ color: "var(--text-dim)", borderBottom: "1px solid var(--border-soft)" }}
                  >
                    {m.date?.slice(0, 10)}
                  </td>
                  <td
                    className="py-2.5 px-3 font-semibold"
                    style={{ color: "var(--text-strong)", borderBottom: "1px solid var(--border-soft)" }}
                  >
                    {m.medication}
                  </td>
                  <td
                    className="py-2.5 px-3 tabular-nums"
                    style={{ color: "var(--text)", borderBottom: "1px solid var(--border-soft)" }}
                  >
                    {m.dose}
                  </td>
                  <td
                    className="py-2.5 px-3"
                    style={{ color: "var(--text-dim)", borderBottom: "1px solid var(--border-soft)" }}
                  >
                    {m.frequency}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {sorted.length > 12 && (
            <p className="text-[0.74rem] mt-2 px-3" style={{ color: "var(--text-faint)" }}>
              + {sorted.length - 12} more in patient record
            </p>
          )}
        </div>
      )}
    </SectionCard>
  );
}
