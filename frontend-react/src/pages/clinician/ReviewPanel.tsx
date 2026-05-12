import { useState } from "react";
import { AlertTriangle, CheckCircle, Edit3, HelpCircle, Star, XCircle } from "lucide-react";
import { Button } from "../../components/ui/Button";
import { Card, CardHeader, SectionTitle } from "../../components/ui/Card";
import { submitSummaryReview } from "../../api/client";

interface Props {
  patientId: string;
  currentSummary?: string;
  onReviewed?: () => void;
}

type Decision = "approved" | "edited" | "rejected" | "marked_unsafe" | "marked_incomplete";

const DECISIONS: {
  id: Decision;
  label: string;
  icon: React.FC<{ size?: number }>;
  variant: "primary" | "secondary" | "danger" | "ghost";
}[] = [
  { id: "approved",          label: "Approve",         icon: CheckCircle,   variant: "primary" },
  { id: "edited",            label: "Save Edit",       icon: Edit3,         variant: "secondary" },
  { id: "rejected",          label: "Reject",          icon: XCircle,       variant: "ghost" },
  { id: "marked_unsafe",     label: "Mark Unsafe",     icon: AlertTriangle, variant: "danger" },
  { id: "marked_incomplete", label: "Mark Incomplete", icon: HelpCircle,    variant: "ghost" },
];

function StarRow({ label, value, onChange }: { label: string; value: number; onChange: (n: number) => void }) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-xs w-36" style={{ color: "var(--text-dim)" }}>{label}</span>
      <div className="flex gap-0.5">
        {[1, 2, 3, 4, 5].map((n) => (
          <button
            key={n}
            onClick={() => onChange(n)}
            className="p-0.5"
            style={{ color: n <= value ? "var(--amber)" : "var(--border)" }}
            aria-label={`${label} ${n}`}
          >
            <Star size={14} fill={n <= value ? "currentColor" : "none"} />
          </button>
        ))}
      </div>
      <span className="text-xs tabular-nums" style={{ color: "var(--text-faint)" }}>{value}/5</span>
    </div>
  );
}

export function ReviewPanel({ patientId, currentSummary, onReviewed }: Props) {
  const [notes, setNotes] = useState("");
  const [editedSummary, setEditedSummary] = useState(currentSummary ?? "");
  const [qualityScore, setQualityScore] = useState<number>(3);
  const [usefulnessScore, setUsefulnessScore] = useState<number>(3);
  const [loading, setLoading] = useState<Decision | null>(null);
  const [done, setDone] = useState<string | null>(null);

  async function submit(decision: Decision) {
    setLoading(decision);
    try {
      await submitSummaryReview(patientId, {
        decision,
        clinician_notes: notes,
        edited_patient_summary: decision === "edited" ? editedSummary : undefined,
        explanation_quality_score: qualityScore,
        model_usefulness_score: usefulnessScore,
      });
      setDone(decision);
      onReviewed?.();
    } finally {
      setLoading(null);
    }
  }

  return (
    <Card>
      <CardHeader>
        <SectionTitle>Clinician Review</SectionTitle>
      </CardHeader>

      {done && (
        <div
          className="mb-3 text-sm rounded-md px-3 py-2 border"
          style={{ background: "rgba(16,185,129,0.08)", borderColor: "rgba(16,185,129,0.25)", color: "var(--green)" }}
        >
          Review submitted: {done}
        </div>
      )}

      <div className="flex flex-col gap-3">
        <div className="flex flex-col gap-1">
          <label className="text-xs font-medium" style={{ color: "var(--text-dim)" }}>Clinician notes</label>
          <textarea
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            rows={3}
            placeholder="Add clinical context, corrections, or concerns..."
            className="w-full px-3 py-2 text-xs rounded-md resize-none outline-none"
            style={{ background: "var(--surface2)", border: "1px solid var(--border)", color: "var(--text)" }}
          />
        </div>

        <div className="flex flex-col gap-1">
          <label className="text-xs font-medium" style={{ color: "var(--text-dim)" }}>
            Edited patient summary (used when decision = "edited")
          </label>
          <textarea
            value={editedSummary}
            onChange={(e) => setEditedSummary(e.target.value)}
            rows={2}
            className="w-full px-3 py-2 text-xs rounded-md resize-none outline-none"
            style={{ background: "var(--surface2)", border: "1px solid var(--border)", color: "var(--text)" }}
          />
        </div>

        <StarRow label="Explanation quality" value={qualityScore} onChange={setQualityScore} />
        <StarRow label="Model usefulness" value={usefulnessScore} onChange={setUsefulnessScore} />

        <div className="flex flex-wrap gap-2 pt-1">
          {DECISIONS.map(({ id, label, icon: Icon, variant }) => (
            <Button
              key={id}
              variant={variant}
              size="sm"
              loading={loading === id}
              icon={<Icon size={12} />}
              onClick={() => void submit(id)}
            >
              {label}
            </Button>
          ))}
        </div>
      </div>
    </Card>
  );
}
