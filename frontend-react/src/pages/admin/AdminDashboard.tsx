import { useEffect, useState } from "react";
import { useLocation } from "react-router-dom";
import {
  BarChart2,
  Cpu,
  Database,
  FlaskConical,
  Image,
  LayoutDashboard,
  Activity,
  ShieldCheck,
  Star,
} from "lucide-react";
import { AppShell } from "../../components/layout/AppShell";
import { useApi } from "../../hooks/useApi";
import { getAdminAnalytics } from "../../api/client";
import { ErrorPane, LoadingPane } from "../../components/ui/Spinner";
import { ErrorBoundary } from "../../components/ui/ErrorBoundary";
import { FeedbackSection } from "./sections/FeedbackSection";
import { GuardrailsSection } from "./sections/GuardrailsSection";
import { MleSection } from "./sections/MleSection";
import { OverviewSection } from "./sections/OverviewSection";
import { RagSection } from "./sections/RagSection";
import { RegressionSection } from "./sections/RegressionSection";
import { AgentTraceSection } from "./sections/AgentTraceSection";
import { ImagingSection } from "./sections/ImagingSection";
import { SafetyCenterSection } from "./sections/SafetyCenterSection";
import { ToolActionBenchmarkSection } from "./sections/ToolActionBenchmarkSection";
import { SystemHealthSection } from "./sections/SystemHealthSection";
import { ClinicalBoundaryBanner } from "../../components/ui/ClinicalBoundaryBanner";

const NAV = [
  { to: "/admin", label: "Overview", icon: LayoutDashboard },
  { to: "/admin/rag", label: "RAG / Cost", icon: Database },
  { to: "/admin/guardrails", label: "Guardrails", icon: ShieldCheck },
  { to: "/admin/mle", label: "MLE Gates", icon: Cpu },
  { to: "/admin/imaging", label: "Imaging MLE", icon: Image },
  { to: "/admin/regression", label: "Regression", icon: FlaskConical },
  { to: "/admin/feedback", label: "Feedback", icon: Star },
  { to: "/admin/model", label: "Model", icon: BarChart2 },
  { to: "/admin/health", label: "System Health", icon: Activity },
];

type Section =
  | "overview"
  | "safety_center"
  | "rag"
  | "guardrails"
  | "mle"
  | "imaging"
  | "regression"
  | "tool_actions"
  | "system_health"
  | "feedback"
  | "trace";

const SECTIONS: { id: Section; label: string }[] = [
  { id: "overview",      label: "Overview" },
  { id: "safety_center", label: "Safety & Eval" },
  { id: "rag",           label: "RAG / Cost" },
  { id: "guardrails",    label: "Guardrails" },
  { id: "mle",           label: "MLE Gates" },
  { id: "imaging",       label: "Imaging MLE" },
  { id: "regression",    label: "Regression" },
  { id: "tool_actions",  label: "Tool Actions" },
  { id: "trace",         label: "Agent Trace" },
  { id: "system_health", label: "System Health" },
  { id: "feedback",      label: "Feedback" },
];

export default function AdminDashboard() {
  const { data, status, error, refetch } = useApi(getAdminAnalytics, []);
  const [section, setSection] = useState<Section>("overview");
  const location = useLocation();

  // Sync the active section to the URL path.  Deferred to a setTimeout(0)
  // so the setState is asynchronous, avoiding the cascading-render anti-
  // pattern flagged by `react-hooks/set-state-in-effect`.
  useEffect(() => {
    const pathSection = pathToSection(location.pathname);
    if (!pathSection) return;
    const id = window.setTimeout(() => setSection(pathSection), 0);
    return () => window.clearTimeout(id);
  }, [location.pathname]);

  return (
    <AppShell
      navItems={NAV}
      title="Admin / MLE Dashboard"
      subtitle="Evaluation, safety, retrieval, and model governance"
    >
      <div className="dashboard-page">
        <div className="dashboard-content" style={{ paddingBottom: 0 }}>
          <ClinicalBoundaryBanner />
        </div>
        <div className="dashboard-tabbar">
          <div className="dashboard-tabbar-inner">
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
        </div>

        <div className="dashboard-content">
          {section === "system_health" && (
            <ErrorBoundary surface="the system health section">
              <SystemHealthSection />
            </ErrorBoundary>
          )}
          {section === "rag" && status === "loading" && (
            <ErrorBoundary surface="the rag section">
              <RagSection />
            </ErrorBoundary>
          )}
          {section !== "system_health" && section !== "rag" && status === "loading" && <LoadingPane label="Loading analytics..." />}
          {section !== "system_health" && status === "error" && <ErrorPane message={error ?? "Failed to load"} />}
          {section !== "system_health" && status === "success" && data && (
            <ErrorBoundary surface={`the ${section.replace(/_/g, " ")} section`}>
              {section === "overview" && <OverviewSection analytics={data} />}
              {section === "safety_center" && <SafetyCenterSection />}
              {section === "rag" && <RagSection analytics={data} />}
              {section === "guardrails" && <GuardrailsSection analytics={data} />}
              {section === "mle" && <MleSection analytics={data} onRefresh={refetch} />}
              {section === "imaging" && <ImagingSection />}
              {section === "regression"   && <RegressionSection />}
              {section === "tool_actions" && <ToolActionBenchmarkSection />}
              {section === "trace"        && <AgentTraceSection />}
              {section === "feedback"     && <FeedbackSection />}
            </ErrorBoundary>
          )}
        </div>
      </div>
    </AppShell>
  );
}

function pathToSection(pathname: string): Section | null {
  if (pathname.includes("/safety")) return "safety_center";
  if (pathname.includes("/rag")) return "rag";
  if (pathname.includes("/guardrails")) return "guardrails";
  if (pathname.includes("/mle")) return "mle";
  if (pathname.includes("/imaging")) return "imaging";
  if (pathname.includes("/regression")) return "regression";
  if (pathname.includes("/tool")) return "tool_actions";
  if (pathname.includes("/trace")) return "trace";
  if (pathname.includes("/health")) return "system_health";
  if (pathname.includes("/feedback")) return "feedback";
  if (pathname.includes("/model")) return "mle";
  return null;
}
