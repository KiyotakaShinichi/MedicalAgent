import {
  getCurrentVsRealismCandidate,
  runCurrentVsRealismCandidate,
} from "../../../../api/client";
import { CandidateComparisonPanel } from "../MleEvidencePanels";
import { DataPanelCard } from "./DataPanelCard";
import { useArtifactPanel } from "./useArtifactPanel";

/** Champion model against the realism-calibrated retraining candidate. */
export function CandidateComparisonCard() {
  const { report, loading, running, error, onRefresh } = useArtifactPanel<
    Parameters<typeof CandidateComparisonPanel>[0]["data"]
  >(getCurrentVsRealismCandidate, runCurrentVsRealismCandidate, "admin.mle.candidateComparison");

  return (
    <DataPanelCard
      title="Current vs Realism-Calibrated Candidate"
      action={{ label: "Compare", onClick: onRefresh, running }}
      loading={loading}
      error={error}
      empty={!report}
      emptyLabel="No current-vs-candidate report available"
      errorLabel="Could not load current-vs-candidate report"
    >
      {report && <CandidateComparisonPanel data={report} />}
    </DataPanelCard>
  );
}
