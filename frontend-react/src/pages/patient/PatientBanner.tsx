import { Activity, MessageCircle, Sparkles, ShieldCheck } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { StatusBadge } from "../../components/ui/StatusBadge";
import { QuickActionChip } from "../../components/ui/QuickActionChip";
import type { PatientReport } from "../../types/api";

interface Props {
  report: PatientReport;
  compact?: boolean;
}

function timeOfDayGreeting(): string {
  const h = new Date().getHours();
  if (h < 12) return "Good morning";
  if (h < 18) return "Good afternoon";
  return "Good evening";
}

function statusTone(s: string | undefined): "success" | "warning" | "danger" | "neutral" {
  const v = (s ?? "").toLowerCase();
  if (v.includes("stable") || v.includes("normal") || v.includes("low_risk") || v.includes("approv")) return "success";
  if (v.includes("watch") || v.includes("review") || v.includes("warn") || v.includes("acceptable")) return "warning";
  if (v.includes("urgent") || v.includes("high") || v.includes("fail") || v.includes("unsafe"))      return "danger";
  return "neutral";
}

export function PatientBanner({ report, compact = false }: Props) {
  const navigate = useNavigate();
  const profile = report.breast_cancer_profile;
  const firstName = (report.patient_name || "").split(" ")[0] || report.patient_name || "there";
  const initials = (report.patient_name || "P")
    .split(/\s+/).filter(Boolean).slice(0, 2).map((p) => p[0]?.toUpperCase()).join("") || "P";

  return (
    <section className={`patient-hero${compact ? " patient-hero--compact" : ""}`}>
      <div className="patient-hero-main">
        <span className="patient-avatar" aria-hidden="true">{initials}</span>
        <div style={{ minWidth: 0, flex: 1 }}>
          <p className="patient-eyebrow">{timeOfDayGreeting()}</p>
          <h2>{firstName}</h2>
          <p className="patient-hero-copy">
            {compact
              ? "Support chat and health-update tools stay in this workspace."
              : "Here is your latest monitoring overview. The four tiles below summarise today's signals — open a section for the full detail."}
          </p>

          <div className="patient-meta-row">
            {report.diagnosis && <StatusBadge tone="neutral" size="sm">{report.diagnosis}</StatusBadge>}
            {profile?.cancer_stage && <StatusBadge tone="neutral" size="sm">{profile.cancer_stage}</StatusBadge>}
            {profile?.molecular_subtype && <StatusBadge tone="accent" size="sm">{profile.molecular_subtype}</StatusBadge>}
            {report.overall_status && (
              <StatusBadge tone={statusTone(report.overall_status)} size="sm">{report.overall_status}</StatusBadge>
            )}
          </div>

          {!compact && (
            <div className="patient-hero-actions">
              <QuickActionChip
                icon={MessageCircle}
                label="Open support chat"
                onClick={() => navigate("/patient/chat")}
                primary
              />
              <QuickActionChip
                icon={Sparkles}
                label="Today's signals"
                onClick={() => navigate("/patient#signals")}
              />
              <QuickActionChip
                icon={Activity}
                label="Lab trends"
                onClick={() => navigate("/patient#labs")}
              />
            </div>
          )}
        </div>
      </div>

      <div className="patient-hero-side">
        <div
          className="patient-context-card"
          style={{
            padding: "12px 14px",
            background: "var(--surface)",
            borderColor: "var(--border)",
          }}
        >
          <ShieldCheck size={14} style={{ color: "var(--rose-deep)", flexShrink: 0 }} aria-hidden="true" />
          <span style={{ fontSize: "0.78rem", lineHeight: 1.4 }}>
            Shared with your care team for review.
          </span>
        </div>
      </div>
    </section>
  );
}
