import { useState } from "react";
import { AlertTriangle, ChevronDown } from "lucide-react";

const BOUNDARY_HEADLINE =
  "Engineering prototype only. Not clinically validated. Synthetic-only ML signals.";
const BOUNDARY_DETAIL =
  "Not for real patient care. No clinician approval. Outputs must not be used for diagnosis, treatment, prognosis, genetic-risk interpretation, tumor-marker interpretation, or medication decisions.";

/**
 * Slim clinical-boundary alert strip.
 *
 * Visually quieter than a full banner: single line + headline, expandable
 * to the full disclaimer.  The full disclaimer text is always present in
 * the DOM (in the expanded slot) so screen readers still announce it.
 *
 * Safety rule: the wording itself is not abbreviated — clicking the
 * chevron reveals the full disclaimer.  The headline only summarises
 * what the patient sees collapsed.
 */
export function ClinicalBoundaryBanner({ compact = false }: { compact?: boolean }) {
  // Default to collapsed even in compact mode; the full text is one keypress away.
  const [open, setOpen] = useState(false);
  return (
    <div
      className={`clinical-strip${compact ? " clinical-strip--compact" : ""}`}
      role="note"
      aria-label="Clinical boundary"
    >
      <button
        type="button"
        className="clinical-strip-head"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        aria-controls="clinical-boundary-detail"
      >
        <AlertTriangle size={14} className="clinical-strip-icon" aria-hidden="true" />
        <span className="clinical-strip-eyebrow">Clinical boundary</span>
        <span className="clinical-strip-text">{BOUNDARY_HEADLINE}</span>
        <ChevronDown
          size={14}
          className="clinical-strip-chevron"
          aria-hidden="true"
          style={{ transform: open ? "rotate(180deg)" : "rotate(0deg)" }}
        />
      </button>
      <p
        id="clinical-boundary-detail"
        className="clinical-strip-detail"
        hidden={!open}
      >
        {BOUNDARY_DETAIL}
      </p>
    </div>
  );
}

