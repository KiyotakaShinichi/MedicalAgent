import { LayoutDashboard, FlaskConical, Activity, Clock, MessageSquare, Pill } from "lucide-react";
import { useLocation, useNavigate } from "react-router-dom";
import { AppShell } from "../../components/layout/AppShell";
import { useApi } from "../../hooks/useApi";
import { getMyReport, sendMyChat, getMyChatHistory } from "../../api/client";
import { ErrorPane } from "../../components/ui/Spinner";
import { SkeletonDashboard } from "../../components/ui/Skeleton";
import { PatientBanner } from "./PatientBanner";
import { AiSummaryPanel } from "./AiSummaryPanel";
import { LabsPanel } from "./LabsPanel";
import { TimelinePanel } from "./TimelinePanel";
import { SymptomsTable } from "./SymptomsTable";
import { ModelSignalPanel } from "./ModelSignalPanel";
import { Card, CardHeader, SectionTitle } from "../../components/ui/Card";
import { EmptyPane } from "../../components/ui/Spinner";
import { ChatPanel } from "../../components/ui/ChatPanel";
import { useAuth } from "../../hooks/useAuth";
import { useState } from "react";
import type { MedicationLog } from "../../types/api";

const NAV = [
  { to: "/patient",          label: "Overview",    icon: LayoutDashboard },
  { to: "/patient/labs",     label: "Labs",        icon: FlaskConical },
  { to: "/patient/signals",  label: "Signals",     icon: Activity },
  { to: "/patient/timeline", label: "Timeline",    icon: Clock },
  { to: "/patient/chat",     label: "Support",     icon: MessageSquare },
];

type Tab = "overview" | "chat";

export default function PatientDashboard() {
  const { patientId } = useAuth();
  const location = useLocation();
  const navigate = useNavigate();
  const { data: report, status, error } = useApi(getMyReport, [patientId]);
  const { data: chatData } = useApi(getMyChatHistory, [patientId]);
  const [manualTab, setManualTab] = useState<Tab | null>(null);
  const tab: Tab = manualTab ?? (location.pathname.includes("/chat") ? "chat" : "overview");

  const tabs: { id: Tab; label: string }[] = [
    { id: "overview", label: "Overview" },
    { id: "chat",     label: "Support Chat" },
  ];

  return (
    <AppShell
      navItems={NAV}
      title="Patient Portal"
      subtitle={report?.patient_name ?? patientId ?? ""}
    >
      {status === "loading" && <SkeletonDashboard label="Loading your records..." />}
      {status === "error"   && <ErrorPane message={error ?? "Failed to load"} />}
      {status === "success" && report && (
        <div className="dashboard-page">
          <PatientBanner report={report} />

          <div className="dashboard-tabbar">
            <div className="dashboard-tabbar-inner">
              {tabs.map(({ id, label }) => (
                <button
                  key={id}
                  onClick={() => {
                    setManualTab(id);
                    navigate(id === "chat" ? "/patient/chat" : "/patient");
                  }}
                  className={tab === id ? "is-active" : undefined}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>

          {tab === "overview" && (
            <div className="dashboard-content dashboard-grid">
              <div className="dashboard-grid-full">
                <AiSummaryPanel summary={report.ai_summary ?? null} />
              </div>
              <LabsPanel report={report} />
              <ModelSignalPanel report={report} />
              <TimelinePanel events={report.timeline ?? []} />
              <SymptomsTable symptoms={report.symptoms ?? []} />
              <div className="dashboard-grid-full">
                <MedLogPanel meds={report.medication_logs ?? []} />
              </div>
            </div>
          )}

          {tab === "chat" && (
            <div className="dashboard-content chat-workspace">
              <div className="chat-card-shell">
                <ChatPanel
                  messages={chatData?.messages ?? report.chat_history ?? []}
                  onSend={async (text) => {
                    const res = await sendMyChat(text);
                    return {
                      reply: res.reply,
                      saved_actions: res.saved_actions,
                      citations: res.citations,
                    };
                  }}
                />
              </div>
            </div>
          )}
        </div>
      )}
    </AppShell>
  );
}

function MedLogPanel({ meds }: { meds: MedicationLog[] }) {
  const sorted = [...meds].sort((a, b) => b.date.localeCompare(a.date));
  return (
    <Card>
      <CardHeader>
        <SectionTitle>Medication Log</SectionTitle>
        <Pill size={14} style={{ color: "var(--text-faint)" }} />
      </CardHeader>
      {sorted.length === 0 ? (
        <EmptyPane label="No medications recorded" />
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr style={{ borderBottom: "1px solid var(--border)" }}>
                {["Date", "Medication", "Dose", "Frequency"].map((h) => (
                  <th key={h} className="text-left py-2 pr-3 font-medium" style={{ color: "var(--text-faint)" }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {sorted.slice(0, 12).map((m, i) => (
                <tr key={i} style={{ borderBottom: "1px solid var(--border)" }} className="last:border-0">
                  <td className="py-2 pr-3 tabular-nums" style={{ color: "var(--text-dim)" }}>{m.date?.slice(0, 10)}</td>
                  <td className="py-2 pr-3 font-medium" style={{ color: "var(--text)" }}>{m.medication}</td>
                  <td className="py-2 pr-3" style={{ color: "var(--text-dim)" }}>{m.dose}</td>
                  <td className="py-2" style={{ color: "var(--text-dim)" }}>{m.frequency}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}
