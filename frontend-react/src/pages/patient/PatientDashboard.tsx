import { LayoutDashboard, FlaskConical, Activity, Clock, MessageSquare, Dna, ShieldCheck } from "lucide-react";
import { useLocation, useNavigate } from "react-router-dom";
import { useCallback, useEffect, useMemo, useState } from "react";
import { AppShell } from "../../components/layout/AppShell";
import { useApi } from "../../hooks/useApi";
import { useReportEnrichment } from "../../hooks/useReportEnrichment";
import { readFileAsBase64 } from "../../lib/fileEncoding";
import { getMyReportCore, sendMyChat, sendMyChatStream, getMyChatHistory, undoMyConfirmedRecordWrite, uploadFile } from "../../api/client";
import { ErrorPane } from "../../components/ui/Spinner";
import { SkeletonDashboard } from "../../components/ui/Skeleton";
import { PatientBanner } from "./PatientBanner";
import { PatientKpiStrip } from "./PatientKpiStrip";
import { KeySignalsCard } from "./KeySignalsCard";
import { ReviewWithCareTeamCard } from "./ReviewWithCareTeamCard";
import { RecentAiExplanationsCard } from "./RecentAiExplanationsCard";
import { LabsPanel } from "./LabsPanel";
import { TimelinePanel } from "./TimelinePanel";
import { SymptomsTable } from "./SymptomsTable";
import { ModelSignalPanel } from "./ModelSignalPanel";
import { EvidenceAwarePredictionCard } from "./EvidenceAwarePredictionCard";
import { HybridPredictionCard } from "./HybridPredictionCard";
import { GeneticCounselingPanel } from "./GeneticCounselingPanel";
import { MedLogPanel } from "./MedLogPanel";
import { touchesPatientReport } from "./savedActions";
import { ChatPanel, describeSavedAction } from "../../components/ui/ChatPanel";
import { ToolTraceDrawer } from "../../components/ui/ToolTraceDrawer";
import { useToast } from "../../components/ui/Toast";
import { SymptomForm } from "./tools/SymptomForm";
import { CBCForm } from "./tools/CBCForm";
import { ImagingForm } from "./tools/ImagingForm";
import { MedicationForm } from "./tools/MedicationForm";
import { TreatmentNoteForm } from "./tools/TreatmentNoteForm";
import { ComposerToolButton, type ToolKey } from "./tools/ComposerToolButton";
import { useAuth } from "../../hooks/useAuth";
import { ClinicalBoundaryBanner } from "../../components/ui/ClinicalBoundaryBanner";
import type { ChatMessage, SavedAction } from "../../types/api";

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
  const tab: Tab = location.pathname.includes("/chat") ? "chat" : "overview";
  const {
    data: coreReport,
    status,
    error,
    refetch: refetchCoreReport,
    lastFetchedAt: coreFetchedAt,
  } = useApi(getMyReportCore, [patientId]);

  // The slow half of the report arrives via a background job; the polling
  // state machine lives in its own hook so it is testable in isolation.
  const {
    enrichment,
    status: enrichmentStatus,
    fetchedAt: enrichmentFetchedAt,
    start: loadEnrichment,
    reset: resetEnrichment,
  } = useReportEnrichment<Partial<NonNullable<typeof coreReport>>>();

  const { data: chatData, refetch: refetchChat } = useApi(getMyChatHistory, [patientId]);
  const report = useMemo(
    () => coreReport ? { ...coreReport, ...(enrichment ?? {}) } : null,
    [coreReport, enrichment],
  );
  const reportFetchedAt = enrichmentFetchedAt ?? coreFetchedAt;

  const refetchReport = useCallback(() => {
    resetEnrichment();
    refetchCoreReport();
  }, [refetchCoreReport, resetEnrichment]);

  const chatMessages = chatData?.messages ?? report?.chat_history ?? [];
  const chatKey = buildChatKey(patientId ?? "patient", chatMessages);

  // Tool tray modal state. One key at a time â€” opening a new tool closes
  // any other open form so the user never sees two modals stacked.
  const [activeTool, setActiveTool] = useState<ToolKey | null>(null);
  const closeTool = () => setActiveTool(null);

  async function handleToolSelect(key: ToolKey, file?: File) {
    if (key === "education") {
      // Just route the user into the chat surface; the chat composer is
      // the right place for free-form questions and the agent's RAG layer
      // already handles education intent.  No new endpoint needed.
      navigate("/patient/chat");
      toast.push({
        tone: "info",
        title: "Ask away",
        description: "Type your question in the chat below â€” the assistant will cite sources where it can.",
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

  async function handleUndoAction(auditId: number) {
    try {
      const result = await undoMyConfirmedRecordWrite(auditId);
      refetchReport();
      refetchChat();
      toast.push({
        tone: "info",
        title: "Portal entry removed",
        description: result.message,
      });
    } catch (err) {
      toast.push({
        tone: "warning",
        title: "Undo failed",
        description: err instanceof Error ? err.message : "The entry could not be removed.",
      });
      throw err;
    }
  }

  const tabs: { id: Tab; label: string }[] = [
    { id: "overview", label: "Overview" },
    { id: "chat",     label: "Support chat" },
  ];

  useEffect(() => {
    if (tab === "overview" && status === "success" && enrichmentStatus === "idle") {
      loadEnrichment();
    }
  }, [enrichmentStatus, loadEnrichment, status, tab]);

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
      <div className="dashboard-content patient-boundary-strip">
        <ClinicalBoundaryBanner />
      </div>
      {status === "loading" && <SkeletonDashboard label="Loading your records..." />}
      {status === "error"   && <ErrorPane message={error ?? "Failed to load"} onRetry={refetchReport} />}
      {status === "success" && report && (
        <div className="dashboard-page">
          <PatientBanner report={report} compact={tab === "chat"} />

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
              {/* Row 1 â€” KPI strip across the top: at-a-glance numbers that the
                  rest of the dashboard then breaks down.  Mirrors a standard
                  analytics-dashboard top-row pattern so the most important
                  signals are above the fold without scrolling. */}
              <div className="dashboard-grid-full" data-section="kpi">
                <PatientKpiStrip report={report} />
                {enrichmentStatus === "loading" && (
                  <p className="patient-progressive-load" role="status">
                    Patient records are ready. Loading synthetic engineering details separately...
                  </p>
                )}
                {enrichmentStatus === "error" && (
                  <p className="patient-progressive-load is-warning" role="status">
                    Patient records are available, but synthetic engineering details could not be loaded.
                  </p>
                )}
              </div>

              {/* Row 2 â€” Hybrid monitoring signal (detail) sits beside the
                  Lab trend chart so the at-a-glance KPI summary AND the
                  detailed signal slots are both in the first viewport. */}
              <div className="col-7" data-section="labs">
                <LabsPanel report={report} lastFetchedAt={reportFetchedAt} />
              </div>
              <div className="col-5" data-section="evidence">
                {report.hybrid_prediction
                  ? <HybridPredictionCard hybrid={report.hybrid_prediction} />
                  : <EvidenceAwarePredictionCard prediction={report.evidence_aware_prediction} />
                }
              </div>

              {/* Row 3 â€” Today's summary Â· Review queue (merged with Next review)
                  Â· Model signal.  Review queue is now ONE strong card so the
                  WBC/platelet warning doesn't show twice on the dashboard. */}
              <div className="col-4" data-section="summary">
                <KeySignalsCard
                  summary={report.ai_summary ?? null}
                  report={report}
                  lastFetchedAt={reportFetchedAt}
                />
              </div>
              <div className="col-4" data-section="review">
                <ReviewWithCareTeamCard summary={report.ai_summary ?? null} report={report} />
              </div>
              <div className="col-4" data-section="signals">
                <ModelSignalPanel report={report} />
              </div>

              {/* Row 4 â€” Symptoms Â· Timeline Â· Recent AI explanations.  All
                  three are compact peer cards; none of them is allowed to
                  stretch into a tall scroll monolith on its own. */}
              <div className="col-4" data-section="symptoms">
                <SymptomsTable symptoms={report.symptoms ?? []} compact lastFetchedAt={reportFetchedAt} />
              </div>
              <div className="col-4" data-section="timeline">
                <TimelinePanel events={report.timeline ?? []} lastFetchedAt={reportFetchedAt} />
              </div>
              <div className="col-4" data-section="ai-explanations">
                <RecentAiExplanationsCard messages={chatMessages} />
              </div>

              {/* Row 5 â€” Family & Genetics readiness summary.  The detailed
                  forms (family history, genetic test, biomarker, tumor
                  marker) live behind the GeneticCounselingPanel's own
                  collapsible body so the overview never opens as a giant
                  inline form page. */}
              <div className="dashboard-grid-full" data-section="genetics">
                <GeneticCounselingPanel
                  readiness={report.genetic_counseling_readiness ?? null}
                  onSaved={refetchReport}
                  defaultCollapsed
                />
              </div>
              {/* Row 6 â€” medication log full-width (compact table). */}
              <div className="dashboard-grid-full" data-section="medications">
                <MedLogPanel meds={report.medication_logs ?? []} />
              </div>

              {/* Page-level safety footnote â€” reinforces the topbar pill so the
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
                    <strong style={{ color: "var(--text-dim)" }}>Proof-of-concept</strong> â€” this dashboard summarises
                    what you have logged and what the AI noticed. It does not diagnose, recommend treatment, or
                    replace clinician judgement. If something feels wrong, contact your care team.
                  </span>
                </p>
              </div>
            </div>
          )}

          {tab === "chat" && (
            <div className="dashboard-content chat-workspace">
              {/* "+" attachment trigger now lives INSIDE the chat composer
                  (passed via composerLeading below).  Layout:
                    [ + ] [ textarea ] [ send ]
                  No more top-right tool toolbar. */}
              <div className="chat-card-shell">
                <ChatPanel
                  key={chatKey}
                  messages={chatMessages}
                  composerLeading={<ComposerToolButton onSelect={handleToolSelect} />}
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
                    // A write to the clinical record must refetch, or the
                    // patient is told "saved" while still looking at pre-save data.
                    if (touchesPatientReport(actions)) refetchReport();
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
                  onUndoAction={handleUndoAction}
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
            description: `${symptom} Â· severity ${severity}/10`,
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
            title: hasAbnormal ? "CBC saved â€” values flagged for review" : "CBC saved",
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
 * Stable chat key â€” keyed only on the patient id, NOT on the message list.
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
  void _messages; // intentionally unused â€” keyed only on patient/session scope
  return `chat:${scope}`;
}

