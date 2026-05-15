import { useCallback, useState } from "react";
import { LayoutDashboard, MessageSquare, Users } from "lucide-react";
import { AppShell } from "../../components/layout/AppShell";
import { useApi } from "../../hooks/useApi";
import {
  getPatientReport,
  getReviewQueue,
  getSummaryReviews,
  sendClinicianChat,
} from "../../api/client";
import { EmptyPane, ErrorPane, LoadingPane } from "../../components/ui/Spinner";
import { Badge } from "../../components/ui/Badge";
import { statusVariant } from "../../components/ui/badgeUtils";
import { Card, CardHeader, SectionTitle } from "../../components/ui/Card";
import { LabsPanel } from "../patient/LabsPanel";
import { TimelinePanel } from "../patient/TimelinePanel";
import { AiSummaryPanel } from "../patient/AiSummaryPanel";
import { ReviewQueue } from "./ReviewQueue";
import { ReviewPanel } from "./ReviewPanel";
import { ChatPanel } from "../../components/ui/ChatPanel";
import type { ChatMessage, PatientReport } from "../../types/api";

const NAV = [
  { to: "/clinician", label: "Review Queue", icon: Users },
  { to: "/clinician/panel", label: "Dashboard", icon: LayoutDashboard },
  { to: "/clinician/chat", label: "Patient Chat", icon: MessageSquare },
];

export default function ClinicianDashboard() {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [reviewKey, setReviewKey] = useState(0);

  const { data: queueData, status: queueStatus, error: queueError } = useApi(getReviewQueue, []);
  const queue = queueData?.queue ?? [];
  const activePatientId = selectedId ?? queue[0]?.patient_id ?? null;
  const {
    data: patientReport,
    status: reportStatus,
    error: reportError,
    refetch: refetchReport,
  } = useApi(
    useCallback(
      () => activePatientId ? getPatientReport(activePatientId) : Promise.resolve(null as unknown as PatientReport),
      [activePatientId]
    ),
    [activePatientId]
  );

  const { data: reviewsData } = useApi(
    useCallback(() => {
      void reviewKey;
      return activePatientId ? getSummaryReviews(activePatientId) : Promise.resolve(null as unknown as { summary_reviews: [] });
    }, [activePatientId, reviewKey]),
    [activePatientId, reviewKey]
  );
  const activeChatMessages = patientReport?.chat_history ?? [];
  const activeChatKey = buildChatKey(activePatientId ?? "none", activeChatMessages);

  return (
    <AppShell
      navItems={NAV}
      title="Clinician Dashboard"
      subtitle="Clinician-in-the-loop review"
    >
      <div className="clinician-workspace">
        <aside className="clinician-queue-panel">
          <div className="panel-title-row">
            <div>
              <p className="app-topbar-kicker">Review queue</p>
              <h2>Patients needing review</h2>
            </div>
            <span>{queue.length}</span>
          </div>
          {queueStatus === "loading" && <LoadingPane label="Loading queue..." />}
          {queueStatus === "error" && <ErrorPane message={queueError ?? ""} />}
          {queueStatus === "success" && (
            <ReviewQueue
              queue={queue}
              selectedId={activePatientId}
              onSelect={(id) => setSelectedId(id)}
            />
          )}
        </aside>

        <section className="clinician-review-panel">
          {!activePatientId && <EmptyPane label="Select a patient from the queue to begin review" />}

          {activePatientId && reportStatus === "loading" && <LoadingPane label="Loading patient..." />}
          {activePatientId && reportStatus === "error" && <ErrorPane message={reportError ?? ""} />}

          {activePatientId && reportStatus === "success" && patientReport && (
            <div className="dashboard-content clinician-detail">
              <div className="review-patient-header">
                <div
                  className="w-10 h-10 rounded-full flex items-center justify-center font-bold text-sm"
                  style={{ background: "rgba(244,63,94,0.15)", color: "var(--rose)" }}
                >
                  {patientReport.patient_name?.[0] ?? "P"}
                </div>
                <div>
                  <p className="font-semibold text-sm" style={{ color: "var(--text)" }}>
                    {patientReport.patient_name}
                  </p>
                  <p className="text-xs" style={{ color: "var(--text-dim)" }}>
                    {patientReport.diagnosis}
                  </p>
                </div>
                <Badge variant={statusVariant(patientReport.overall_status ?? "")} className="ml-auto">
                  {patientReport.overall_status}
                </Badge>
              </div>

              {patientReport.breast_cancer_profile && (
                <BreastProfileCard profile={patientReport.breast_cancer_profile} />
              )}

              <AiSummaryPanel summary={patientReport.ai_summary ?? null} />
              <LabsPanel report={patientReport} />
              <TimelinePanel events={patientReport.timeline ?? []} />

              {(reviewsData?.summary_reviews ?? []).length > 0 && (
                <AuditTrail reviews={reviewsData!.summary_reviews} />
              )}

              <ReviewPanel
                patientId={activePatientId}
                currentSummary={
                  Array.isArray(patientReport.ai_summary?.patient_explanation)
                    ? patientReport.ai_summary!.patient_explanation.join(" ")
                    : (patientReport.ai_summary?.patient_explanation as string) ?? ""
                }
                onReviewed={() => {
                  setReviewKey((k) => k + 1);
                  refetchReport();
                }}
              />

              <div className="chat-card-shell clinician-chat-shell">
                <div className="chat-card-title">Patient chat</div>
                <div style={{ flex: 1, minHeight: 0, display: "flex" }}>
                  <ChatPanel
                    key={activeChatKey}
                    messages={activeChatMessages}
                    onSend={async (text) => {
                      const res = await sendClinicianChat(activePatientId, text);
                      return {
                        reply: res.reply,
                        saved_actions: res.saved_actions,
                        citations: res.citations,
                      };
                    }}
                  />
                </div>
              </div>
            </div>
          )}
        </section>
      </div>
    </AppShell>
  );
}

function buildChatKey(scope: string, messages: ChatMessage[]) {
  const last = messages.at(-1);
  return [scope, messages.length, last?.role ?? "none", (last?.message ?? "").slice(0, 40)].join(":");
}

function BreastProfileCard({ profile }: { profile: NonNullable<PatientReport["breast_cancer_profile"]> }) {
  const items = [
    { label: "ER", value: profile.er_status },
    { label: "PR", value: profile.pr_status },
    { label: "HER2", value: profile.her2_status },
    { label: "Stage", value: profile.cancer_stage },
    { label: "Subtype", value: profile.molecular_subtype },
    { label: "Intent", value: profile.treatment_intent },
  ];
  return (
    <Card>
      <CardHeader><SectionTitle>Breast Cancer Profile</SectionTitle></CardHeader>
      <div className="profile-stat-grid">
        {items.map(({ label, value }) => (
          <div key={label} className="profile-stat">
            <p>{label}</p>
            <strong>{value ?? "-"}</strong>
          </div>
        ))}
      </div>
    </Card>
  );
}

function AuditTrail({ reviews }: { reviews: import("../../types/api").SummaryReview[] }) {
  return (
    <Card>
      <CardHeader><SectionTitle>Audit Trail</SectionTitle></CardHeader>
      <div className="flex flex-col gap-2">
        {reviews.slice(0, 5).map((r) => (
          <div key={r.id} className="audit-row">
            <Badge variant={statusVariant(r.decision)}>{r.decision}</Badge>
            <div className="flex-1 min-w-0">
              {r.clinician_notes && <p>{r.clinician_notes}</p>}
              <span>{r.created_at?.slice(0, 16)}</span>
            </div>
          </div>
        ))}
      </div>
    </Card>
  );
}
