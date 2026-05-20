import { LayoutDashboard, FlaskConical, Activity, Clock, MessageSquare, Pill, Dna, ShieldCheck } from "lucide-react";
import { useLocation, useNavigate } from "react-router-dom";
import { useEffect, useState } from "react";
import { AppShell } from "../../components/layout/AppShell";
import { useApi } from "../../hooks/useApi";
import { getMyReport, sendMyChat, sendMyChatStream, getMyChatHistory, uploadFile } from "../../api/client";
import { ErrorPane } from "../../components/ui/Spinner";
import { SkeletonDashboard } from "../../components/ui/Skeleton";
import { PatientBanner } from "./PatientBanner";
import { KeySignalsCard } from "./KeySignalsCard";
import { ReviewWithCareTeamCard } from "./ReviewWithCareTeamCard";
import { NextReviewItemsCard } from "./NextReviewItemsCard";
import { RecentAiExplanationsCard } from "./RecentAiExplanationsCard";
import { LabsPanel } from "./LabsPanel";
import { TimelinePanel } from "./TimelinePanel";
import { SymptomsTable } from "./SymptomsTable";
import { ModelSignalPanel } from "./ModelSignalPanel";
import { EvidenceAwarePredictionCard } from "./EvidenceAwarePredictionCard";
import { HybridPredictionCard } from "./HybridPredictionCard";
import { GeneticCounselingPanel } from "./GeneticCounselingPanel";
import { SectionCard } from "../../components/ui/SectionCard";
import { EmptyPane } from "../../components/ui/Spinner";
import { ChatPanel, describeSavedAction } from "../../components/ui/ChatPanel";
import { ToolTraceDrawer } from "../../components/ui/ToolTraceDrawer";
import { useToast } from "../../components/ui/Toast";
import { SymptomForm } from "./tools/SymptomForm";
import { CBCForm } from "./tools/CBCForm";
import { ImagingForm } from "./tools/ImagingForm";
import { MedicationForm } from "./tools/MedicationForm";
import { TreatmentNoteForm } from "./tools/TreatmentNoteForm";
import { ToolTray, type ToolKey } from "./tools/ToolTray";
import { useAuth } from "../../hooks/useAuth";
import { ClinicalBoundaryBanner } from "../../components/ui/ClinicalBoundaryBanner";
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

  // Tool tray modal state. One key at a time — opening a new tool closes
  // any other open form so the user never sees two modals stacked.
  const [activeTool, setActiveTool] = useState<ToolKey | null>(null);
  const closeTool = () => setActiveTool(null);

  // File-to-base64 helper for the upload buttons.  Keeps the full data URL
  // off the wire (we strip the data:*/*;base64, prefix) so the existing
  // /me/uploads endpoint receives just the base64 payload it expects.
  async function readFileAsBase64(file: File): Promise<string> {
    const buffer = await file.arrayBuffer();
    const bytes = new Uint8Array(buffer);
    let binary = "";
    const chunk = 0x8000;
    for (let i = 0; i < bytes.length; i += chunk) {
      binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
    }
    return btoa(binary);
  }

  async function handleToolSelect(key: ToolKey, file?: File) {
    if (key === "education") {
      // Just route the user into the chat surface; the chat composer is
      // the right place for free-form questions and the agent's RAG layer
      // already handles education intent.  No new endpoint needed.
      navigate("/patient/chat");
      toast.push({
        tone: "info",
        title: "Ask away",
        description: "Type your question in the chat below — the assistant will cite sources where it can.",
      });
      return;
    }

    if (key === "upload_cbc" || key === "upload_imaging") {
      if (!file) return;
      const uploadType = key === "upload_cbc" ? "lab_report" : "imaging_report";
      const labelForToast = key === "upload_cbc" ? "CBC image" : "Imaging file";
      try {
        const base64 = await readFileAsBase64(file);
        await uploadFile({
          upload_type: uploadType,
          file_name: file.name,
          content_type: file.type || "application/octet-stream",
          content_base64: base64,
          notes: `Patient-uploaded ${labelForToast.toLowerCase()} via tool tray.`,
        });
        toast.push({
          tone: "success",
          title: `${labelForToast} uploaded`,
          description: file.name,
        });
        refetchReport();
      } catch (err) {
        const message = err instanceof Error ? err.message : "Could not upload the file.";
        toast.push({
          tone: "warning",
          title: "Upload failed",
          description: message,
        });
      }
      return;
    }

    setActiveTool(key);
  }

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
      <div className="dashboard-page" style={{ paddingBottom: 0 }}>
        <div className="dashboard-content" style={{ paddingBottom: 0 }}>
          <ClinicalBoundaryBanner />
        </div>
      </div>
      {status === "loading" && tab !== "chat" && <SkeletonDashboard label="Loading your records..." />}
      {status === "error"   && <ErrorPane message={error ?? "Failed to load"} onRetry={refetchReport} />}
      {tab === "chat" && status === "loading" && (
        <div className="dashboard-page">
          <div className="dashboard-content chat-workspace">
            <ToolTray onSelect={handleToolSelect} />
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
        </div>
      )}
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
              {/* Row 1 — at-a-glance: key signals · review with team · recent symptoms */}
              <div className="col-5" data-section="summary">
                <KeySignalsCard summary={report.ai_summary ?? null} />
              </div>
              <div className="col-4" data-section="review">
                <ReviewWithCareTeamCard summary={report.ai_summary ?? null} />
              </div>
              <div className="col-3" data-section="symptoms">
                <SymptomsTable symptoms={report.symptoms ?? []} compact lastFetchedAt={reportFetchedAt} />
              </div>
              {/* Row 2 — CBC + treatment timeline side-by-side */}
              <div className="col-7" data-section="labs">
                <LabsPanel report={report} lastFetchedAt={reportFetchedAt} />
              </div>
              <div className="col-5" data-section="timeline">
                <TimelinePanel events={report.timeline ?? []} lastFetchedAt={reportFetchedAt} />
              </div>
              {/* Row 3 — model signal · evidence-aware prediction · next review items */}
              <div className="col-5" data-section="signals">
                <ModelSignalPanel report={report} />
              </div>
              <div className="col-4" data-section="evidence">
                {report.hybrid_prediction
                  ? <HybridPredictionCard hybrid={report.hybrid_prediction} />
                  : <EvidenceAwarePredictionCard prediction={report.evidence_aware_prediction} />
                }
              </div>
              <div className="col-3" data-section="next-review">
                <NextReviewItemsCard report={report} />
              </div>

              {/* Row 4 — chat recap full-width below the modelling row */}
              <div className="dashboard-grid-full" data-section="ai-explanations">
                <RecentAiExplanationsCard messages={chatMessages} />
              </div>
              {/* Row 4 — genetics full-width */}
              <div className="dashboard-grid-full" data-section="genetics">
                <GeneticCounselingPanel
                  readiness={report.genetic_counseling_readiness ?? null}
                  onSaved={refetchReport}
                />
              </div>
              {/* Row 5 — medication log full-width */}
              <div className="dashboard-grid-full" data-section="medications">
                <MedLogPanel meds={report.medication_logs ?? []} />
              </div>

              {/* Page-level safety footnote — reinforces the topbar pill so the
                  PoC boundary is visible after the user scrolls the dashboard. */}
              <div className="dashboard-grid-full">
                <p
                  className="inline-flex items-start gap-2 text-[0.74rem] leading-relaxed"
                  style={{
                    color: "var(--text-faint)",
                    margin: "8px 0 0",
                    padding: "10px 12px",
                    border: "1px solid var(--border)",
                    borderRadius: 8,
                    background: "var(--surface2)",
                  }}
                >
                  <ShieldCheck
                    size={13}
                    style={{ color: "var(--rose-deep)", flexShrink: 0, marginTop: 1 }}
                    aria-hidden="true"
                  />
                  <span>
                    <strong style={{ color: "var(--text-dim)" }}>Proof-of-concept</strong> — this dashboard summarises
                    what you have logged and what the AI noticed. It does not diagnose, recommend treatment, or
                    replace clinician judgement. If something feels wrong, contact your care team.
                  </span>
                </p>
              </div>
            </div>
          )}

          {tab === "chat" && (
            <div className="dashboard-content chat-workspace">
              {/* Persistent tool tray — 8 chips: 5 form openers, 2 file
                  uploads, and an education shortcut. */}
              <ToolTray onSelect={handleToolSelect} />
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

      {/* All five tool forms are mounted at the dashboard level so opening
          one never remounts the chat or its in-flight stream. */}
      <SymptomForm
        open={activeTool === "symptom"}
        onClose={closeTool}
        onSaved={({ symptom, severity, urgent_flag }) => {
          refetchReport();
          toast.push({
            tone: urgent_flag ? "warning" : "success",
            title: urgent_flag ? "Symptom saved and flagged for review" : "Symptom saved",
            description: `${symptom} · severity ${severity}/10`,
          });
        }}
      />
      <CBCForm
        open={activeTool === "cbc"}
        onClose={closeTool}
        onSaved={({ hasAbnormal }) => {
          refetchReport();
          toast.push({
            tone: hasAbnormal ? "warning" : "success",
            title: hasAbnormal ? "CBC saved — values flagged for review" : "CBC saved",
            description: "Lab card updated",
          });
        }}
      />
      <ImagingForm
        open={activeTool === "imaging"}
        onClose={closeTool}
        onSaved={({ modality }) => {
          refetchReport();
          toast.push({
            tone: "success",
            title: `${modality} report saved`,
            description: "Timeline updated",
          });
        }}
      />
      <MedicationForm
        open={activeTool === "medication"}
        onClose={closeTool}
        onSaved={({ medication }) => {
          refetchReport();
          toast.push({
            tone: "success",
            title: "Medication logged",
            description: medication,
          });
        }}
      />
      <TreatmentNoteForm
        open={activeTool === "treatment"}
        onClose={closeTool}
        onSaved={({ drug }) => {
          refetchReport();
          toast.push({
            tone: "success",
            title: "Treatment note saved",
            description: drug,
          });
        }}
      />
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
