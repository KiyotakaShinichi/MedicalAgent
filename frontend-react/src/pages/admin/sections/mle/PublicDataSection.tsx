import {
  getCbioportalBiomarkerSchemaMapping,
  runCbioportalBiomarkerSchemaMapping,
  getFullFeatureGroupAblation,
  runFullFeatureGroupAblation,
  getPublicBiomarkerDatasetManifest,
  runPublicBiomarkerDatasetManifest,
  getPublicBiomarkerMappingReadiness,
  runPublicBiomarkerMappingReadiness,
  getPublicDataManifest,
} from "../../../../api/client";
import type {
  CbioportalBiomarkerSchemaMapping,
  FullFeatureGroupAblationReport,
  PublicBiomarkerDatasetManifest,
  PublicBiomarkerMappingReadiness,
  PublicDataManifest,
} from "../../../../types/api";
import {
  CbioPortalMappingPanel,
  FullFeatureGroupAblationPanel,
  PublicBiomarkerManifestPanel,
  PublicBiomarkerMappingPanel,
  PublicDataManifestPanel,
} from "../MleEvidencePanels";
import { DataPanelCard } from "./DataPanelCard";
import { useArtifactPanel } from "./useArtifactPanel";

const TAG_SOURCE_CALIBRATED = {
  label: "Source-calibrated synthetic data",
  background: "rgba(59,130,246,0.12)",
  color: "#93c5fd",
};

/** Which public datasets the synthetic generator was calibrated against. */
export function PublicDataFeasibilityPanel() {
  const { report, loading, error } = useArtifactPanel<PublicDataManifest>(
    getPublicDataManifest, undefined, "admin.mle.publicDataManifest",
  );

  return (
    <DataPanelCard
      title="Public Data Feasibility"
      tag={TAG_SOURCE_CALIBRATED}
      loading={loading}
      error={error}
      empty={!report}
      emptyLabel="No public data manifest available"
      errorLabel="Could not load public data manifest"
    >
      {report && <PublicDataManifestPanel data={report} />}
    </DataPanelCard>
  );
}

/** Catalogue of public biomarker and tumor-marker sources. */
export function PublicBiomarkerSourcesPanel() {
  const { report, loading, running, error, onRefresh } = useArtifactPanel<PublicBiomarkerDatasetManifest>(
    getPublicBiomarkerDatasetManifest, runPublicBiomarkerDatasetManifest, "admin.mle.biomarkerManifest",
  );

  return (
    <DataPanelCard
      title="Public Biomarker & Tumor-Marker Sources"
      action={{ label: "Refresh manifest", onClick: onRefresh, running }}
      loading={loading}
      error={error}
      empty={!report}
      emptyLabel="No public biomarker manifest available"
      errorLabel="Could not load public biomarker manifest"
    >
      {report && <PublicBiomarkerManifestPanel data={report} />}
    </DataPanelCard>
  );
}

/** Whether public biomarker fields can be mapped onto the internal schema. */
export function PublicBiomarkerMappingReadinessPanel() {
  const { report, loading, running, error, onRefresh } = useArtifactPanel<PublicBiomarkerMappingReadiness>(
    getPublicBiomarkerMappingReadiness, runPublicBiomarkerMappingReadiness, "admin.mle.biomarkerMapping",
  );

  return (
    <DataPanelCard
      title="Public Biomarker Mapping Readiness"
      action={{ label: "Rebuild mapping", onClick: onRefresh, running }}
      loading={loading}
      error={error}
      empty={!report}
      emptyLabel="No public biomarker mapping readiness report available"
      errorLabel="Could not load public biomarker mapping readiness"
    >
      {report && <PublicBiomarkerMappingPanel data={report} />}
    </DataPanelCard>
  );
}

/** TCGA / METABRIC schema mapping fetched from cBioPortal. */
export function CbioPortalSchemaMappingPanel() {
  // `true` forces a live fetch rather than reusing the cached schema.
  const { report, loading, running, error, onRefresh } = useArtifactPanel<CbioportalBiomarkerSchemaMapping>(
    getCbioportalBiomarkerSchemaMapping,
    () => runCbioportalBiomarkerSchemaMapping(true),
    "admin.mle.cbioMapping",
  );

  return (
    <DataPanelCard
      title="TCGA / METABRIC cBioPortal Mapping"
      action={{ label: "Fetch schema", onClick: onRefresh, running }}
      loading={loading}
      error={error}
      empty={!report}
      emptyLabel="No cBioPortal mapping available"
      errorLabel="Could not load cBioPortal schema mapping"
    >
      {report && <CbioPortalMappingPanel data={report} />}
    </DataPanelCard>
  );
}

/** Contribution of each feature group, measured by leave-one-group-out. */
export function FullFeatureAblationPanel() {
  const { report, loading, running, error, onRefresh } = useArtifactPanel<FullFeatureGroupAblationReport>(
    getFullFeatureGroupAblation, runFullFeatureGroupAblation, "admin.mle.fullFeatureAblation",
  );

  return (
    <DataPanelCard
      title="Full Feature-Group Ablation"
      action={{ label: "Rerun ablation", onClick: onRefresh, running }}
      loading={loading}
      error={error}
      // A "missing" status is a present artifact reporting that it has no data.
      empty={!report || report.status === "missing"}
      emptyLabel="No full feature-group ablation available"
      errorLabel="Could not load full feature-group ablation"
    >
      {report && <FullFeatureGroupAblationPanel data={report} />}
    </DataPanelCard>
  );
}
