import { useState } from "react";
import { Dna, ShieldCheck } from "lucide-react";
import { SectionCard } from "../../components/ui/SectionCard";
import { Badge } from "../../components/ui/Badge";
import { Button } from "../../components/ui/Button";
import { EmptyPane } from "../../components/ui/Spinner";
import {
  addMyBiomarkerRecord,
  addMyFamilyHistory,
  addMyGeneticTestRecord,
  addMyTumorMarkerRecord,
} from "../../api/client";
import type { GeneticCounselingReadiness } from "../../types/api";

interface GeneticCounselingPanelProps {
  readiness?: GeneticCounselingReadiness | null;
  onSaved?: () => void;
}

export function GeneticCounselingPanel({ readiness, onSaved }: GeneticCounselingPanelProps) {
  const [saving, setSaving] = useState<string | null>(null);
  const [family, setFamily] = useState({
    relationship: "mother",
    family_side: "maternal",
    cancer_type: "breast",
    age_at_diagnosis: "",
    notes: "",
  });
  const [genetic, setGenetic] = useState({
    test_type: "germline",
    sample_type: "blood",
    gene: "BRCA1",
    classification: "unknown",
    report_date: "",
    notes: "",
  });
  const [biomarker, setBiomarker] = useState({
    source: "biopsy",
    er_status: "unknown",
    pr_status: "unknown",
    her2_status: "unknown",
    ki67_percent: "",
    report_date: "",
    report_text: "",
  });
  const [marker, setMarker] = useState({
    marker: "CA 15-3",
    value: "",
    unit: "U/mL",
    date_collected: "",
    notes: "",
  });

  const summary = [
    { label: "Family history", value: (readiness?.family_history?.length ?? 0) > 0 },
    { label: "Genetic test record", value: (readiness?.genetic_test_records?.length ?? 0) > 0 },
    { label: "Biomarker/pathology", value: (readiness?.biomarker_records?.length ?? 0) > 0 },
    { label: "Tumor marker trends", value: (readiness?.tumor_marker_records?.length ?? 0) > 0 },
  ];

  async function saveFamily() {
    setSaving("family");
    try {
      await addMyFamilyHistory({
        ...family,
        age_at_diagnosis: family.age_at_diagnosis ? Number(family.age_at_diagnosis) : null,
        relative_status: "unknown",
        multiple_relatives_affected: "unknown",
        male_breast_cancer: "unknown",
        known_familial_mutation: "unknown",
      });
      setFamily((f) => ({ ...f, age_at_diagnosis: "", notes: "" }));
      onSaved?.();
    } finally {
      setSaving(null);
    }
  }

  async function saveGenetic() {
    setSaving("genetic");
    try {
      await addMyGeneticTestRecord({
        ...genetic,
        report_date: genetic.report_date || null,
        reviewed_by_genetic_counselor: "unknown",
      });
      setGenetic((g) => ({ ...g, report_date: "", notes: "" }));
      onSaved?.();
    } finally {
      setSaving(null);
    }
  }

  async function saveBiomarker() {
    setSaving("biomarker");
    try {
      await addMyBiomarkerRecord({
        ...biomarker,
        ki67_percent: biomarker.ki67_percent ? Number(biomarker.ki67_percent) : null,
        report_date: biomarker.report_date || null,
      });
      setBiomarker((b) => ({ ...b, ki67_percent: "", report_date: "", report_text: "" }));
      onSaved?.();
    } finally {
      setSaving(null);
    }
  }

  async function saveMarker() {
    setSaving("marker");
    try {
      await addMyTumorMarkerRecord({
        ...marker,
        value: Number(marker.value),
        date_collected: marker.date_collected || null,
        trend_direction: "unknown",
      });
      setMarker((m) => ({ ...m, value: "", date_collected: "", notes: "" }));
      onSaved?.();
    } finally {
      setSaving(null);
    }
  }

  return (
    <SectionCard
      title="Family and genetic counseling"
      icon={Dna}
      meta={readiness?.readiness_status?.replaceAll("_", " ") ?? "organize records"}
      footer="This organizes family history, genetic-test, biomarker, and tumor-marker records for review. It is not genetic counseling, diagnosis, or treatment advice."
    >
      <div className="genetics-panel">
        <div className="genetics-summary-grid">
          {summary.map((item) => (
            <div className="genetics-summary-card" key={item.label}>
              <span>{item.label}</span>
              <Badge variant={item.value ? "green" : "muted"}>{item.value ? "recorded" : "not recorded"}</Badge>
            </div>
          ))}
        </div>

        {(readiness?.flags?.length ?? 0) > 0 && (
          <div className="genetics-callout">
            <ShieldCheck size={16} />
            <div>
              <strong>Review flags</strong>
              <p>{readiness!.flags.map((flag) => flag.replaceAll("_", " ")).join("; ")}</p>
            </div>
          </div>
        )}

        <div className="genetics-form-grid">
          <MiniForm title="Add family history" onSubmit={saveFamily} loading={saving === "family"}>
            <Select label="Relative" value={family.relationship} onChange={(v) => setFamily({ ...family, relationship: v })} options={["mother", "father", "sibling", "aunt", "uncle", "grandmother", "grandfather", "cousin", "other"]} />
            <Select label="Side" value={family.family_side} onChange={(v) => setFamily({ ...family, family_side: v })} options={["maternal", "paternal", "unknown"]} />
            <Select label="Cancer type" value={family.cancer_type} onChange={(v) => setFamily({ ...family, cancer_type: v })} options={["breast", "ovarian", "pancreatic", "prostate", "colon", "melanoma", "other", "unknown"]} />
            <Input label="Age at diagnosis" type="number" value={family.age_at_diagnosis} onChange={(v) => setFamily({ ...family, age_at_diagnosis: v })} />
            <Textarea label="Notes" value={family.notes} onChange={(v) => setFamily({ ...family, notes: v })} />
          </MiniForm>

          <MiniForm title="Add genetic test record" onSubmit={saveGenetic} loading={saving === "genetic"}>
            <Select label="Test type" value={genetic.test_type} onChange={(v) => setGenetic({ ...genetic, test_type: v })} options={["germline", "somatic", "tumor sequencing", "multigene panel", "BRCA-only", "unknown"]} />
            <Select label="Sample" value={genetic.sample_type} onChange={(v) => setGenetic({ ...genetic, sample_type: v })} options={["blood", "saliva", "tumor tissue", "unknown"]} />
            <Select label="Gene" value={genetic.gene} onChange={(v) => setGenetic({ ...genetic, gene: v })} options={["BRCA1", "BRCA2", "PALB2", "TP53", "PTEN", "CHEK2", "ATM", "other", "unknown"]} />
            <Select label="Classification" value={genetic.classification} onChange={(v) => setGenetic({ ...genetic, classification: v })} options={["pathogenic", "likely pathogenic", "VUS", "likely benign", "benign", "unknown"]} />
            <Input label="Report date" type="date" value={genetic.report_date} onChange={(v) => setGenetic({ ...genetic, report_date: v })} />
          </MiniForm>

          <MiniForm title="Add biomarker/pathology" onSubmit={saveBiomarker} loading={saving === "biomarker"}>
            <Select label="Source" value={biomarker.source} onChange={(v) => setBiomarker({ ...biomarker, source: v })} options={["biopsy", "surgery pathology", "tumor sequencing", "liquid biopsy", "unknown"]} />
            <Select label="ER" value={biomarker.er_status} onChange={(v) => setBiomarker({ ...biomarker, er_status: v })} options={["positive", "negative", "low positive", "unknown"]} />
            <Select label="PR" value={biomarker.pr_status} onChange={(v) => setBiomarker({ ...biomarker, pr_status: v })} options={["positive", "negative", "low positive", "unknown"]} />
            <Select label="HER2" value={biomarker.her2_status} onChange={(v) => setBiomarker({ ...biomarker, her2_status: v })} options={["positive", "negative", "equivocal", "IHC 0", "IHC 1+", "IHC 2+", "IHC 3+", "FISH amplified", "FISH not amplified", "unknown"]} />
            <Input label="Ki-67 %" type="number" value={biomarker.ki67_percent} onChange={(v) => setBiomarker({ ...biomarker, ki67_percent: v })} />
          </MiniForm>

          <MiniForm title="Add tumor marker" onSubmit={saveMarker} loading={saving === "marker"} disabled={!marker.value}>
            <Select label="Marker" value={marker.marker} onChange={(v) => setMarker({ ...marker, marker: v })} options={["CA 15-3", "CA 27.29", "CEA", "other"]} />
            <Input label="Value" type="number" value={marker.value} onChange={(v) => setMarker({ ...marker, value: v })} />
            <Input label="Unit" value={marker.unit} onChange={(v) => setMarker({ ...marker, unit: v })} />
            <Input label="Date collected" type="date" value={marker.date_collected} onChange={(v) => setMarker({ ...marker, date_collected: v })} />
            <Textarea label="Notes" value={marker.notes} onChange={(v) => setMarker({ ...marker, notes: v })} />
          </MiniForm>
        </div>

        <div className="genetics-lists">
          <RecordList
            title="Questions to ask your care team"
            items={readiness?.questions_to_ask ?? []}
            empty="No suggested questions yet."
          />
          <RecordList
            title="Missing data"
            items={readiness?.missing_data ?? []}
            empty="No missing-data warnings."
          />
        </div>
      </div>
    </SectionCard>
  );
}

function MiniForm({
  title,
  children,
  onSubmit,
  loading,
  disabled,
}: {
  title: string;
  children: React.ReactNode;
  onSubmit: () => Promise<void>;
  loading?: boolean;
  disabled?: boolean;
}) {
  return (
    <form
      className="genetics-mini-form"
      onSubmit={(event) => {
        event.preventDefault();
        void onSubmit();
      }}
    >
      <h3>{title}</h3>
      {children}
      <Button size="sm" variant="primary" loading={loading} disabled={disabled}>
        Save for review
      </Button>
    </form>
  );
}

function Input({ label, value, onChange, type = "text" }: { label: string; value: string; onChange: (value: string) => void; type?: string }) {
  return (
    <label className="genetics-field">
      <span>{label}</span>
      <input type={type} value={value} onChange={(event) => onChange(event.target.value)} />
    </label>
  );
}

function Select({ label, value, onChange, options }: { label: string; value: string; onChange: (value: string) => void; options: string[] }) {
  return (
    <label className="genetics-field">
      <span>{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {options.map((option) => (
          <option key={option} value={option}>{option}</option>
        ))}
      </select>
    </label>
  );
}

function Textarea({ label, value, onChange }: { label: string; value: string; onChange: (value: string) => void }) {
  return (
    <label className="genetics-field genetics-field-wide">
      <span>{label}</span>
      <textarea value={value} onChange={(event) => onChange(event.target.value)} rows={2} />
    </label>
  );
}

function RecordList({ title, items, empty }: { title: string; items: string[]; empty: string }) {
  return (
    <div className="genetics-record-list">
      <h3>{title}</h3>
      {items.length === 0 ? (
        <EmptyPane label={empty} />
      ) : (
        <ul>
          {items.slice(0, 6).map((item) => <li key={item}>{item}</li>)}
        </ul>
      )}
    </div>
  );
}
