import { useMemo, useState } from "react";
import { Drawer } from "../../../components/ui/Drawer";
import {
  Field,
  NumberInput,
  TextInput,
  TextArea,
  FormFooter,
  FormError,
  FormGrid,
} from "../../../components/ui/Form";
import { SafetyBanner } from "../../../components/ui/SafetyBanner";
import { useToolForm } from "../../../hooks/useToolForm";
import { addMyLab } from "../../../api/client";
import {
  LAB_REFERENCE_RANGES,
  classifyLabValue,
  labStatusLabel,
  labStatusTone,
  NON_DIAGNOSTIC_DISCLAIMER,
  type LabKey,
} from "../../../lib/clinical-constants";

interface CBCFormProps {
  open: boolean;
  onClose: () => void;
  onSaved?: (summary: { hasAbnormal: boolean }) => void;
}

interface FormState {
  date: string;       // yyyy-mm-dd
  wbc: string;        // strings so empty input doesn't render as "0"
  hemoglobin: string;
  platelets: string;
  anc: string;
  labSource: string;
  notes: string;
}

const INITIAL_STATE: FormState = {
  date: new Date().toISOString().slice(0, 10),
  wbc: "",
  hemoglobin: "",
  platelets: "",
  anc: "",
  labSource: "",
  notes: "",
};

function parseOptionalFloat(raw: string): number | null {
  if (!raw.trim()) return null;
  const n = Number(raw);
  return Number.isFinite(n) ? n : NaN;
}

const TONE_COLOR: Record<ReturnType<typeof labStatusTone>, string> = {
  success: "#047857",
  warning: "#92400e",
  danger:  "#b91c1c",
  neutral: "var(--text-faint)",
};

/** Inline status preview shown to the right of each lab field as the user types. */
function LiveStatusHint({ labKey, value }: { labKey: LabKey; value: number | null }) {
  const range = LAB_REFERENCE_RANGES[labKey];
  if (value === null) {
    return (
      <span style={{ color: "var(--text-faint)" }}>
        ref {range.refLow}-{range.refHigh} {range.unit}
      </span>
    );
  }
  const status = classifyLabValue(labKey, value);
  const tone = labStatusTone(status);
  return (
    <span style={{ color: TONE_COLOR[tone], fontWeight: 700 }}>
      {labStatusLabel(status)}
      <span style={{ color: "var(--text-faint)", fontWeight: 500, marginLeft: 6 }}>
        · ref {range.refLow}-{range.refHigh}
      </span>
    </span>
  );
}

/**
 * Patient-facing CBC entry form.  Lives in a right-side drawer because four
 * numeric inputs + source + notes feel cramped in a centered modal at 560px.
 *
 * Live reference-range hints sit on each field — every numeric field shows
 * "in range / borderline / low / high / critical" against the same constants
 * the lab cards on the dashboard use.  The labels are decorative; backend
 * stays the source of truth for validation.
 */
export function CBCForm({ open, onClose, onSaved }: CBCFormProps) {
  const [state, setState] = useState<FormState>(INITIAL_STATE);

  const parsed = useMemo(() => ({
    wbc:        parseOptionalFloat(state.wbc),
    hemoglobin: parseOptionalFloat(state.hemoglobin),
    platelets:  parseOptionalFloat(state.platelets),
    anc:        parseOptionalFloat(state.anc),
  }), [state.wbc, state.hemoglobin, state.platelets, state.anc]);

  const fieldErrors = useMemo(() => {
    const errors: Partial<Record<keyof FormState, string>> = {};
    if (!state.date) errors.date = "Pick a date.";
    if (parsed.wbc === null || Number.isNaN(parsed.wbc)) errors.wbc = "Enter a number.";
    else if (parsed.wbc < 0 || parsed.wbc > 200) errors.wbc = "Out of plausible range.";
    if (parsed.hemoglobin === null || Number.isNaN(parsed.hemoglobin)) errors.hemoglobin = "Enter a number.";
    else if (parsed.hemoglobin < 0 || parsed.hemoglobin > 25) errors.hemoglobin = "Out of plausible range.";
    if (parsed.platelets === null || Number.isNaN(parsed.platelets)) errors.platelets = "Enter a number.";
    else if (parsed.platelets < 0 || parsed.platelets > 2000) errors.platelets = "Out of plausible range.";
    if (state.anc.trim() && Number.isNaN(parsed.anc)) errors.anc = "Enter a number or leave empty.";
    return errors;
  }, [state.date, state.anc, parsed]);

  const isValid = Object.keys(fieldErrors).length === 0;

  const { submitting, submitError, submit, reset } = useToolForm(
    async () => {
      // Guaranteed by isValid above, but TS doesn't know that.
      if (parsed.wbc === null || parsed.hemoglobin === null || parsed.platelets === null) {
        throw new Error("Fill in WBC, hemoglobin, and platelets.");
      }
      const result = await addMyLab({
        date: state.date,
        wbc: parsed.wbc,
        hemoglobin: parsed.hemoglobin,
        platelets: parsed.platelets,
        anc: parsed.anc != null && !Number.isNaN(parsed.anc) ? parsed.anc : undefined,
        lab_source: state.labSource.trim() || undefined,
        notes: state.notes.trim() || undefined,
      });
      return result;
    },
    {
      onSuccess: () => {
        // Flag any out-of-range value so the parent toast can switch to a
        // warning tone.  Only checks against the central reference ranges —
        // backend validation already vetted plausibility.
        const hasAbnormal = (["wbc", "hemoglobin", "platelets"] as const).some((key) => {
          const value = parsed[key];
          if (value === null) return false;
          return classifyLabValue(key, value) !== "in_range";
        });
        onSaved?.({ hasAbnormal });
        setState(INITIAL_STATE);
        onClose();
      },
    },
  );

  function handleClose() {
    if (submitting) return;
    setState(INITIAL_STATE);
    reset();
    onClose();
  }

  return (
    <Drawer
      open={open}
      onClose={handleClose}
      title="Save CBC / lab values"
      description="Log a complete blood count entry from your latest lab report."
      size="md"
      dismissable={!submitting}
      footer={
        <FormFooter
          onCancel={handleClose}
          submitLabel="Save CBC"
          submitting={submitting}
          disabled={!isValid}
          hint="Saved to your patient record"
        />
      }
    >
      <form
        onSubmit={submit}
        noValidate
        style={{ display: "flex", flexDirection: "column", gap: 14 }}
      >
        <FormError message={submitError} />

        <Field label="Date collected" htmlFor="cbc-date" required error={fieldErrors.date}>
          <TextInput
            id="cbc-date"
            type="date"
            value={state.date}
            onChange={(e) => setState((s) => ({ ...s, date: e.target.value }))}
            max={new Date().toISOString().slice(0, 10)}
            invalid={Boolean(fieldErrors.date)}
          />
        </Field>

        <FormGrid>
          <Field
            label="WBC"
            htmlFor="cbc-wbc"
            required
            error={fieldErrors.wbc}
            hint={<LiveStatusHint labKey="wbc" value={parsed.wbc !== null && !Number.isNaN(parsed.wbc) ? parsed.wbc : null} />}
          >
            <NumberInput
              id="cbc-wbc"
              step="0.1"
              min="0"
              value={state.wbc}
              onChange={(e) => setState((s) => ({ ...s, wbc: e.target.value }))}
              unit={LAB_REFERENCE_RANGES.wbc.unit}
              invalid={Boolean(fieldErrors.wbc)}
              placeholder="e.g. 4.8"
            />
          </Field>

          <Field
            label="Hemoglobin"
            htmlFor="cbc-hgb"
            required
            error={fieldErrors.hemoglobin}
            hint={<LiveStatusHint labKey="hemoglobin" value={parsed.hemoglobin !== null && !Number.isNaN(parsed.hemoglobin) ? parsed.hemoglobin : null} />}
          >
            <NumberInput
              id="cbc-hgb"
              step="0.1"
              min="0"
              value={state.hemoglobin}
              onChange={(e) => setState((s) => ({ ...s, hemoglobin: e.target.value }))}
              unit={LAB_REFERENCE_RANGES.hemoglobin.unit}
              invalid={Boolean(fieldErrors.hemoglobin)}
              placeholder="e.g. 12.4"
            />
          </Field>

          <Field
            label="Platelets"
            htmlFor="cbc-plt"
            required
            error={fieldErrors.platelets}
            hint={<LiveStatusHint labKey="platelets" value={parsed.platelets !== null && !Number.isNaN(parsed.platelets) ? parsed.platelets : null} />}
          >
            <NumberInput
              id="cbc-plt"
              step="1"
              min="0"
              value={state.platelets}
              onChange={(e) => setState((s) => ({ ...s, platelets: e.target.value }))}
              unit={LAB_REFERENCE_RANGES.platelets.unit}
              invalid={Boolean(fieldErrors.platelets)}
              placeholder="e.g. 220"
            />
          </Field>

          <Field
            label="ANC (optional)"
            htmlFor="cbc-anc"
            error={fieldErrors.anc}
            hint={<LiveStatusHint labKey="anc" value={parsed.anc !== null && !Number.isNaN(parsed.anc) ? parsed.anc : null} />}
          >
            <NumberInput
              id="cbc-anc"
              step="0.1"
              min="0"
              value={state.anc}
              onChange={(e) => setState((s) => ({ ...s, anc: e.target.value }))}
              unit={LAB_REFERENCE_RANGES.anc.unit}
              invalid={Boolean(fieldErrors.anc)}
              placeholder="e.g. 2.1"
            />
          </Field>
        </FormGrid>

        <Field
          label="Lab source (optional)"
          htmlFor="cbc-source"
          description="Where the blood was drawn — clinic name, lab brand, or 'home draw'."
        >
          <TextInput
            id="cbc-source"
            value={state.labSource}
            onChange={(e) => setState((s) => ({ ...s, labSource: e.target.value }))}
            placeholder="e.g. Quest, Hospital A outpatient lab"
            maxLength={120}
            autoComplete="off"
          />
        </Field>

        <Field label="Notes (optional)" htmlFor="cbc-notes">
          <TextArea
            id="cbc-notes"
            value={state.notes}
            onChange={(e) => setState((s) => ({ ...s, notes: e.target.value }))}
            rows={3}
            maxLength={800}
            placeholder="Anything your care team should know about this draw..."
          />
        </Field>

        <SafetyBanner tone="info" compact>
          {NON_DIAGNOSTIC_DISCLAIMER} Reference ranges shown here are general adult guides — your lab uses its own ranges.
        </SafetyBanner>
      </form>
    </Drawer>
  );
}
