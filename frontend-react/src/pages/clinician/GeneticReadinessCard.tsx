import { useState } from "react";
import { Dna } from "lucide-react";
import { SectionCard } from "../../components/ui/SectionCard";
import { Badge } from "../../components/ui/Badge";
import { Button } from "../../components/ui/Button";
import { submitGeneticCounselingReview } from "../../api/client";
import type { GeneticCounselingReadiness } from "../../types/api";

export function GeneticReadinessCard({
  patientId,
  readiness,
  onReviewed,
}: {
  patientId: string;
  readiness?: GeneticCounselingReadiness | null;
  onReviewed?: () => void;
}) {
  const [notes, setNotes] = useState("");
  const [saving, setSaving] = useState<string | null>(null);
  const counts = {
    family: readiness?.family_history?.length ?? 0,
    genetic: readiness?.genetic_test_records?.length ?? 0,
    biomarkers: readiness?.biomarker_records?.length ?? 0,
    markers: readiness?.tumor_marker_records?.length ?? 0,
  };

  async function review(decision: string) {
    setSaving(decision);
    try {
      await submitGeneticCounselingReview(patientId, { decision, notes: notes || null });
      setNotes("");
      onReviewed?.();
    } finally {
      setSaving(null);
    }
  }

  return (
    <SectionCard
      title="Genetic Counseling Readiness"
      icon={Dna}
      meta={readiness?.readiness_status?.replaceAll("_", " ") ?? "not loaded"}
      footer={readiness?.boundary_note ?? "Information organization only. Not genetic counseling or treatment advice."}
    >
      <div className="genetics-panel">
        <div className="genetics-summary-grid">
          <Metric label="Family history" value={counts.family} />
          <Metric label="Genetic tests" value={counts.genetic} />
          <Metric label="Biomarkers" value={counts.biomarkers} />
          <Metric label="Tumor markers" value={counts.markers} />
        </div>

        {(readiness?.flags?.length ?? 0) > 0 && (
          <div className="flex flex-wrap gap-2">
            {readiness!.flags.map((flag) => (
              <Badge key={flag} variant="amber">{flag.replaceAll("_", " ")}</Badge>
            ))}
          </div>
        )}

        <div className="genetics-lists">
          <List title="Missing data" items={readiness?.missing_data ?? []} />
          <List title="Patient questions" items={readiness?.questions_to_ask ?? []} />
        </div>

        <div className="genetics-record-list">
          <h3>Clinician / genetics review note</h3>
          <textarea
            value={notes}
            onChange={(event) => setNotes(event.target.value)}
            rows={3}
            placeholder="Add review note or clarification request..."
            style={{
              width: "100%",
              marginTop: 10,
              border: "1px solid var(--border)",
              borderRadius: 8,
              padding: 10,
              color: "var(--text)",
              background: "var(--surface)",
            }}
          />
          <div className="flex flex-wrap gap-2 mt-3">
            <Button size="sm" variant="primary" loading={saving === "accepted"} onClick={() => void review("accepted")}>Accept</Button>
            <Button size="sm" loading={saving === "needs_clarification"} onClick={() => void review("needs_clarification")}>Needs clarification</Button>
            <Button size="sm" variant="danger" loading={saving === "not_actionable"} onClick={() => void review("not_actionable")}>Not actionable</Button>
          </div>
        </div>
      </div>
    </SectionCard>
  );
}

function Metric({ label, value }: { label: string; value: number }) {
  return (
    <div className="genetics-summary-card">
      <span>{label}</span>
      <strong style={{ color: "var(--text-strong)", fontSize: "1.45rem" }}>{value}</strong>
    </div>
  );
}

function List({ title, items }: { title: string; items: string[] }) {
  return (
    <div className="genetics-record-list">
      <h3>{title}</h3>
      {items.length === 0 ? (
        <p className="text-sm mt-2" style={{ color: "var(--text-dim)" }}>None listed.</p>
      ) : (
        <ul>
          {items.slice(0, 6).map((item) => <li key={item}>{item}</li>)}
        </ul>
      )}
    </div>
  );
}
