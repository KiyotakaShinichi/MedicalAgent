import { useState } from "react";
import {
  BarChart2,
  Cpu,
  Database,
  FlaskConical,
  Image,
  LayoutDashboard,
  ShieldCheck,
  Star,
} from "lucide-react";
import { AppShell } from "../../components/layout/AppShell";
import { useApi } from "../../hooks/useApi";
import { getAdminAnalytics } from "../../api/client";
import { ErrorPane, LoadingPane } from "../../components/ui/Spinner";
import { FeedbackSection } from "./sections/FeedbackSection";
import { GuardrailsSection } from "./sections/GuardrailsSection";
import { MleSection } from "./sections/MleSection";
import { OverviewSection } from "./sections/OverviewSection";
import { RagSection } from "./sections/RagSection";
import { RegressionSection } from "./sections/RegressionSection";
import { AgentTraceSection } from "./sections/AgentTraceSection";
import { ImagingSection } from "./sections/ImagingSection";

const NAV = [
  { to: "/admin", label: "Overview", icon: LayoutDashboard },
  { to: "/admin/rag", label: "RAG / Cost", icon: Database },
  { to: "/admin/guardrails", label: "Guardrails", icon: ShieldCheck },
  { to: "/admin/mle", label: "MLE Gates", icon: Cpu },
  { to: "/admin/imaging", label: "Imaging MLE", icon: Image },
  { to: "/admin/regression", label: "Regression", icon: FlaskConical },
  { to: "/admin/feedback", label: "Feedback", icon: Star },
  { to: "/admin/model", label: "Model", icon: BarChart2 },
];

type Section = "overview" | "rag" | "guardrails" | "mle" | "imaging" | "regression" | "feedback" | "trace";

const SECTIONS: { id: Section; label: string }[] = [
  { id: "overview",   label: "Overview" },
  { id: "rag",        label: "RAG / Cost" },
  { id: "guardrails", label: "Guardrails" },
  { id: "mle",        label: "MLE Gates" },
  { id: "imaging",    label: "Imaging MLE" },
  { id: "regression", label: "Regression" },
  { id: "trace",      label: "Agent Trace" },
  { id: "feedback",   label: "Feedback" },
];

export default function AdminDashboard() {
  const { data, status, error, refetch } = useApi(getAdminAnalytics, []);
  const [section, setSection] = useState<Section>("overview");

  return (
    <AppShell
      navItems={NAV}
      title="Admin / MLE Dashboard"
      subtitle="Evaluation, safety, retrieval, and model governance"
    >
      <div className="dashboard-page">
        <div className="dashboard-tabbar">
          {SECTIONS.map(({ id, label }) => (
            <button
              key={id}
              type="button"
              onClick={() => setSection(id)}
              className={section === id ? "is-active" : undefined}
            >
              {label}
            </button>
          ))}
        </div>

        <div className="dashboard-content">
          {status === "loading" && <LoadingPane label="Loading analytics..." />}
          {status === "error" && <ErrorPane message={error ?? "Failed to load"} />}
          {status === "success" && data && (
            <>
              {section === "overview" && <OverviewSection analytics={data} />}
              {section === "rag" && <RagSection analytics={data} />}
              {section === "guardrails" && <GuardrailsSection analytics={data} />}
              {section === "mle" && <MleSection analytics={data} onRefresh={refetch} />}
              {section === "imaging" && <ImagingSection />}
              {section === "regression" && <RegressionSection />}
              {section === "trace"      && <AgentTraceSection />}
              {section === "feedback"   && <FeedbackSection />}
            </>
          )}
        </div>
      </div>
    </AppShell>
  );
}
