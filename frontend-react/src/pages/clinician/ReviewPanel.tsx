import { useState } from "react";
import {
  AlertTriangle,
  CheckCircle,
  Edit3,
  FileQuestion,
  HelpCircle,
  Star,
  TrendingDown,
  XCircle,
} from "lucide-react";
import { Button } from "../../components/ui/Button";
import { Card, CardHeader, SectionTitle } from "../../components/ui/Card";
import { submitSummaryReview } from "../../api/client";
import {
  REVIEW_DECISION_LABELS,
  REVIEW_REASON_CATEGORIES,
  REVIEW_REASON_LABELS,
  type ReviewDecision,
  type ReviewReasonCategory,
} from "../../lib/constants";

interface Props {
  patientId: string;
  currentSummary?: string;
  modelVersion?: string;
  ragVersion?: string;
  onReviewed?: () => void;
}

type DecisionDef = {
  id: ReviewDecision;
  icon: React.FC<{ size?: number }>;
  variant: "primary" | "secondary" | "danger" | "ghost";
};

const DECISIONS: DecisionDef[] = [
  { id: "approved",         icon: CheckCircle,   variant: "primary"   },
  { id: "edited",           icon: Edit3,         variant: "secondary" },
  { id: "needs_followup",   icon: HelpCircle,    variant: "ghost"     },
  { id: "missing_evidence", icon: FileQuestion,  variant: "ghost"     },
  { id: "wrong_escalation", icon: TrendingDown,  variant: "ghost"     },
  { id: "rejected",         icon: XCircle,       variant: "ghost"     },
  { id: "unsafe",           icon: AlertTriangle, variant: "danger"    },
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

export function ReviewPanel({
  patientId,
  currentSummary,
  modelVersion,
  ragVersion,
  onReviewed,
}: Props) {
  const [notes, setNotes] = useState("");
  const [editedSummary, setEditedSummary] = useState(currentSummary ?? "");
  const [reasonCategory, setReasonCategory] = useState<ReviewReasonCategory | "">("");
  const [qualityScore, setQualityScore] = useState<number>(3);
  const [usefulnessScore, setUsefulnessScore] = useState<number>(3);
  const [loading, setLoading] = useState<ReviewDecision | null>(null);
  const [done, setDone] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function submit(decision: ReviewDecision) {
    setLoading(decision);
    setError(null);
    try {
      await submitSummaryReview(patientId, {
        decision,
        clinician_notes: notes,
        edited_patient_summary: decision === "edited" ? editedSummary : undefined,
        explanation_quality_score: qualityScore,
        model_usefulness_score: usefulnessScore,
        reason_category: reasonCategory || undefined,
        model_version: modelVersion,
        rag_version: ragVersion,
      });
      setDone(decision);
      onReviewed?.();
    } catch (err) {
      setError((err as Error).message);
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
          Review submitted: {REVIEW_DECISION_LABELS[done as ReviewDecision] ?? done}
        </div>
      )}

      {error && (
        <div
          className="mb-3 text-sm rounded-md px-3 py-2 border"
          style={{ background: "rgba(244,63,94,0.08)", borderColor: "rgba(244,63,94,0.25)", color: "var(--rose)" }}
        >
          {error}
        </div>
      )}

      <div className="flex flex-col gap-3">
        <div className="flex flex-col gap-1">
          <label className="text-xs font-medium" style={{ color: "var(--text-dim)" }}>
            Clinician notes
          </label>
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
            Edited patient summary (used when decision is "Save edit")
          </label>
          <textarea
            value={editedSummary}
            onChange={(e) => setEditedSummary(e.target.value)}
            rows={2}
            className="w-full px-3 py-2 text-xs rounded-md resize-none outline-none"
            style={{ background: "var(--surface2)", border: "1px solid var(--border)", color: "var(--text)" }}
          />
        </div>

        <div className="flex flex-col gap-1">
          <label className="text-xs font-medium" style={{ color: "var(--text-dim)" }}>
            Reason category (used for rejected / unsafe / missing-evidence / wrong-escalation)
          </label>
          <select
            value={reasonCategory}
            onChange={(e) => setReasonCategory(e.target.value as ReviewReasonCategory | "")}
            className="w-full px-3 py-2 text-xs rounded-md outline-none"
            style={{ background: "var(--surface2)", border: "1px solid var(--border)", color: "var(--text)" }}
          >
            <option value="">— select if applicable —</option>
            {REVIEW_REASON_CATEGORIES.map((reason) => (
              <option key={reason} value={reason}>
                {REVIEW_REASON_LABELS[reason]}
              </option>
            ))}
          </select>
        </div>

        <StarRow label="Explanation quality" value={qualityScore} onChange={setQualityScore} />
        <StarRow label="Model usefulness" value={usefulnessScore} onChange={setUsefulnessScore} />

        <div className="flex flex-wrap gap-2 pt-1">
          {DECISIONS.map(({ id, icon: Icon, variant }) => (
            <Button
              key={id}
              variant={variant}
              size="sm"
              loading={loading === id}
              icon={<Icon size={12} />}
              onClick={() => void submit(id)}
            >
              {REVIEW_DECISION_LABELS[id]}
            </Button>
          ))}
        </div>

        <p className="text-xs" style={{ color: "var(--text-faint)" }}>
          Reviews are audit data. They do not change the patient record automatically.
        </p>
      </div>
    </Card>
  );
}
