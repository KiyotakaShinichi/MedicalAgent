import { SafetyBanner } from "./SafetyBanner";

const BOUNDARY_TEXT =
  "Engineering prototype only. Not clinically validated. Not for real patient care. No clinician approval. Synthetic-only ML signals. Outputs must not be used for diagnosis, treatment, prognosis, genetic-risk interpretation, tumor-marker interpretation, or medication decisions.";

export function ClinicalBoundaryBanner({ compact = false }: { compact?: boolean }) {
  return (
    <SafetyBanner tone="danger" title="Clinical boundary" compact={compact}>
      {BOUNDARY_TEXT}
    </SafetyBanner>
  );
}

