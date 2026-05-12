import { Activity, FileText, ShieldCheck } from "lucide-react";
import { Badge } from "../../components/ui/Badge";
import { statusVariant } from "../../components/ui/badgeUtils";
import type { PatientReport } from "../../types/api";

interface Props {
  report: PatientReport;
}

export function PatientBanner({ report }: Props) {
  const score = report.monitoring_score;
  const scoreColor =
    score == null ? "var(--text-faint)" :
    score >= 70 ? "var(--green)" :
    score >= 40 ? "var(--amber)" : "var(--rose)";

  const profile = report.breast_cancer_profile;

  return (
    <section className="patient-hero">
      <div className="patient-hero-main">
        <span className="patient-avatar">{report.patient_name?.[0] ?? "P"}</span>
        <div>
          <p className="patient-eyebrow">Patient workspace</p>
          <h2>{report.patient_name}</h2>
          <div className="patient-meta-row">
            <Badge variant="muted">{report.diagnosis}</Badge>
            {profile?.cancer_stage && <Badge variant="muted">{profile.cancer_stage}</Badge>}
            {profile?.molecular_subtype && <Badge variant="green">{profile.molecular_subtype}</Badge>}
            <Badge variant={statusVariant(report.overall_status ?? "")}>
              {report.overall_status ?? "Unknown"}
            </Badge>
          </div>
        </div>
      </div>

      <div className="patient-hero-side">
        <div className="patient-score-card">
          <Activity size={18} style={{ color: scoreColor }} aria-hidden="true" />
          <div>
            <span>Monitoring score</span>
            <strong style={{ color: scoreColor }}>{score != null ? score : "-"}</strong>
          </div>
        </div>
        <div className="patient-context-card">
          <FileText size={17} aria-hidden="true" />
          <span>Timeline, labs, symptoms, imaging, and AI summaries are synced for review.</span>
        </div>
        <div className="patient-context-card">
          <ShieldCheck size={17} aria-hidden="true" />
          <span>Support only. Clinicians remain responsible for care decisions.</span>
        </div>
      </div>
    </section>
  );
}
