import { Activity, MessageCircle, Sparkles, ShieldCheck } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { StatusBadge } from "../../components/ui/StatusBadge";
import { QuickActionChip } from "../../components/ui/QuickActionChip";
import type { PatientReport } from "../../types/api";

interface Props {
  report: PatientReport;
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

export function PatientBanner({ report }: Props) {
  const navigate = useNavigate();
  const score = report.monitoring_score;
  const scoreColor =
    score == null  ? "var(--text-faint)" :
    score >= 70    ? "#059669" :
    score >= 40    ? "#d97706" : "#dc2626";
  const scoreCaption =
    score == null ? "Insufficient data" :
    score >= 70   ? "Stable today" :
    score >= 40   ? "Some signals to watch" : "Clinician review suggested";

  const profile = report.breast_cancer_profile;
  const firstName = (report.patient_name || "").split(" ")[0] || report.patient_name || "there";
  const initials = (report.patient_name || "P")
    .split(/\s+/).filter(Boolean).slice(0, 2).map((p) => p[0]?.toUpperCase()).join("") || "P";

  return (
    <section className="patient-hero">
      <div className="patient-hero-main">
        <span className="patient-avatar" aria-hidden="true">{initials}</span>
        <div style={{ minWidth: 0, flex: 1 }}>
          <p className="patient-eyebrow">{timeOfDayGreeting()}</p>
          <h2>{firstName}</h2>
          <p
            style={{
              marginTop: 6,
              color: "var(--text-dim)",
              fontSize: "0.88rem",
              lineHeight: 1.5,
              maxWidth: 540,
            }}
          >
            Here is your latest monitoring overview. Let your support assistant know how you are feeling today.
          </p>

          <div
            className="patient-meta-row"
            style={{ marginTop: 12 }}
          >
            {report.diagnosis && <StatusBadge tone="neutral" size="sm">{report.diagnosis}</StatusBadge>}
            {profile?.cancer_stage && <StatusBadge tone="neutral" size="sm">{profile.cancer_stage}</StatusBadge>}
            {profile?.molecular_subtype && <StatusBadge tone="accent" size="sm">{profile.molecular_subtype}</StatusBadge>}
            {report.overall_status && (
              <StatusBadge tone={statusTone(report.overall_status)} size="sm">{report.overall_status}</StatusBadge>
            )}
          </div>

          <div
            style={{
              marginTop: 16,
              display: "flex",
              flexWrap: "wrap",
              gap: 8,
            }}
          >
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
        </div>
      </div>

      <div className="patient-hero-side">
        <div
          className="patient-score-card"
          style={{
            padding: "16px 18px",
            background: "var(--surface)",
            borderColor: "var(--border)",
          }}
        >
          <div style={{ flex: 1, minWidth: 0 }}>
            <span
              style={{
                display: "block",
                fontSize: "0.7rem",
                textTransform: "uppercase",
                letterSpacing: "0.08em",
                color: "var(--text-faint)",
                fontWeight: 600,
              }}
            >
              Monitoring score
            </span>
            <div style={{ display: "flex", alignItems: "baseline", gap: 6, marginTop: 4 }}>
              <strong style={{ color: scoreColor, fontSize: "1.9rem", lineHeight: 1 }}>
                {score != null ? score : "—"}
              </strong>
              <span style={{ color: "var(--text-faint)", fontSize: "0.78rem" }}>/ 100</span>
            </div>
            <span style={{ display: "block", marginTop: 4, fontSize: "0.78rem", color: "var(--text-dim)" }}>
              {scoreCaption}
            </span>
          </div>
          <Activity size={20} style={{ color: scoreColor, flexShrink: 0 }} aria-hidden="true" />
        </div>

        <div
          className="patient-context-card"
          style={{
            padding: "10px 12px",
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
