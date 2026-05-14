import { Sparkles, Clock, Shield } from "lucide-react";
import { Badge } from "./Badge";
import { ConfidenceBadge } from "./ConfidenceBadge";
import {
  AI_GENERATED_LABEL,
  CLINICIAN_REVIEW_REQUIRED_LABEL,
  type ConfidenceLevel,
} from "../../lib/constants";

interface AIGeneratedLabelProps {
  /** Confidence band — "low" | "moderate" | "high". Omit to hide the badge. */
  confidence?: ConfidenceLevel | string | null;
  /** Short text describing why confidence is what it is. */
  uncertaintyReason?: string | null;
  /** When true, surfaces the "Clinician review required" amber badge. */
  clinicianReviewRequired?: boolean | null;
  /** ISO timestamp of when this AI output was produced. */
  timestamp?: string | null;
  /** Free-text source/evidence reference, e.g. "RAG", "risk_engine", "summary_quality_v2". */
  source?: string | null;
  /** Model identifier, e.g. "gradient_boosting@2026-05" or LLM provider tag. */
  modelVersion?: string | null;
  className?: string;
}

/**
 * A single chip-row that consistently labels any AI/model output across
 * the app. It pins down the safe vocabulary so callers cannot accidentally
 * present an AI inference as a clinical conclusion.
 *
 * The component renders four pieces of information when supplied:
 *
 * 1. `AI-generated` chip (always present).
 * 2. `Confidence: ...` (uses ConfidenceBadge).
 * 3. `Clinician review required` (when explicitly true).
 * 4. Source / model / timestamp metadata as faint text.
 *
 * Uncertainty reason is shown as a small explanatory line under the chips.
 */
export function AIGeneratedLabel({
  confidence,
  uncertaintyReason,
  clinicianReviewRequired,
  timestamp,
  source,
  modelVersion,
  className,
}: AIGeneratedLabelProps) {
  const meta = [source, modelVersion].filter(Boolean).join(" · ");
  return (
    <div className={className}>
      <div className="flex flex-wrap items-center gap-1.5">
        <Badge variant="purple">
          <Sparkles size={11} aria-hidden="true" /> {AI_GENERATED_LABEL}
        </Badge>
        {confidence && <ConfidenceBadge level={confidence} />}
        {clinicianReviewRequired ? (
          <Badge variant="amber">
            <Shield size={11} aria-hidden="true" /> {CLINICIAN_REVIEW_REQUIRED_LABEL}
          </Badge>
        ) : null}
      </div>
      {(meta || timestamp) && (
        <p
          className="text-xs mt-1 flex items-center gap-1.5"
          style={{ color: "var(--text-faint)" }}
        >
          {timestamp && (
            <>
              <Clock size={10} aria-hidden="true" />
              <span>{timestamp.slice(0, 19).replace("T", " ")}</span>
            </>
          )}
          {meta && (
            <>
              {timestamp && <span aria-hidden="true">·</span>}
              <span>{meta}</span>
            </>
          )}
        </p>
      )}
      {uncertaintyReason && (
        <p className="text-xs mt-1 italic" style={{ color: "var(--text-dim)" }}>
          {uncertaintyReason}
        </p>
      )}
    </div>
  );
}
