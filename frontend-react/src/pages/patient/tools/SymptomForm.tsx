import { useMemo, useState } from "react";
import { Modal } from "../../../components/ui/Modal";
import {
  Field,
  TextInput,
  TextArea,
  Slider,
  Checkbox,
  FormFooter,
  FormError,
  FormGrid,
  SelectWithCustom,
} from "../../../components/ui/Form";
import {
  SELECT_WITH_CUSTOM_OTHER_VALUE,
  resolveSelectWithCustomValue,
} from "../../../components/ui/selectWithCustom";
import { SafetyBanner } from "../../../components/ui/SafetyBanner";
import { useToolForm } from "../../../hooks/useToolForm";
import { addMySymptom } from "../../../api/client";
import {
  severityBucket,
  NON_DIAGNOSTIC_DISCLAIMER,
  COMMON_SYMPTOMS,
} from "../../../lib/clinical-constants";

interface SymptomFormProps {
  open: boolean;
  onClose: () => void;
  /** Called after a successful save so the parent can refetch + toast. */
  onSaved?: (result: { symptom: string; severity: number; urgent_flag: boolean }) => void;
}

interface FormState {
  /** Canonical selection from the curated dropdown, or
   *  ``SELECT_WITH_CUSTOM_OTHER_VALUE`` when the patient picks Other. */
  symptomChoice: string;
  /** Free-text value used when `symptomChoice === Other`. */
  symptomCustom: string;
  severity: number;
  date: string;       // yyyy-mm-dd
  duration: string;
  notes: string;
  urgentFlag: boolean;
}

const INITIAL_STATE: FormState = {
  symptomChoice: "",
  symptomCustom: "",
  severity: 5,
  date: new Date().toISOString().slice(0, 10),
  duration: "",
  notes: "",
  urgentFlag: false,
};

/**
 * Patient-facing manual symptom-entry form.
 *
 * Behaviour notes
 * ~~~~~~~~~~~~~~~
 * - Severity uses a 0-10 slider with a colour-coded bucket label
 *   (mild / moderate / severe).  Defaults to 5 so the patient can't
 *   accidentally save 0 by hitting Enter on an empty form.
 * - The "urgent red-flag" checkbox is **not auto-routing** anything.  It
 *   only prepends `[urgent flag]` to the saved note so the clinician review
 *   queue picks the entry up.  Routing belongs to the chat agent's safety
 *   layer, not to a patient-set checkbox — the safety promise is that the
 *   system never silently escalates based on form data alone.
 * - Validation is intentionally minimal (presence + numeric range) so the
 *   user is never blocked by a strict format on a tracking entry.  Backend
 *   ``validate_symptom_payload`` is the source of truth for length limits.
 */
export function SymptomForm({ open, onClose, onSaved }: SymptomFormProps) {
  const [state, setState] = useState<FormState>(INITIAL_STATE);

  const resolvedSymptom = resolveSelectWithCustomValue(state.symptomChoice, state.symptomCustom);
  const fieldErrors = useMemo(() => {
    const errors: Partial<Record<"symptom" | "severity" | "date", string>> = {};
    if (!state.symptomChoice) {
      errors.symptom = "Pick a symptom from the list or choose Other.";
    } else if (state.symptomChoice === SELECT_WITH_CUSTOM_OTHER_VALUE && !resolvedSymptom) {
      errors.symptom = "Type the symptom name.";
    } else if (resolvedSymptom.length > 80) {
      errors.symptom = "Keep it under 80 characters.";
    }
    if (!Number.isFinite(state.severity) || state.severity < 0 || state.severity > 10) {
      errors.severity = "Severity must be between 0 and 10.";
    }
    if (!state.date) errors.date = "Pick a date.";
    return errors;
  }, [state.symptomChoice, resolvedSymptom, state.severity, state.date]);

  const isValid = Object.keys(fieldErrors).length === 0;

  const { submitting, submitError, submit, reset } = useToolForm(
    async () => {
      const result = await addMySymptom({
        date: state.date,
        symptom: resolvedSymptom,
        severity: state.severity,
        notes: state.notes.trim() || undefined,
        duration: state.duration.trim() || undefined,
        urgent_flag: state.urgentFlag,
      });
      return result;
    },
    {
      onSuccess: (result) => {
        onSaved?.({
          symptom: resolvedSymptom,
          severity: state.severity,
          urgent_flag: result.urgent_flag,
        });
        // Reset to initial state for next open.
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

  const bucket = severityBucket(state.severity);
  const bucketColor =
    bucket === "severe"   ? "#b91c1c" :
    bucket === "moderate" ? "#92400e" : "#047857";
  const bucketLabel =
    bucket === "severe"   ? "Severe" :
    bucket === "moderate" ? "Moderate" : "Mild";

  return (
    <Modal
      open={open}
      onClose={handleClose}
      title="Log a symptom"
      description="Track how you are feeling so your care team can review trends."
      size="md"
      dismissable={!submitting}
      footer={
        <FormFooter
          onCancel={handleClose}
          submitLabel="Save symptom"
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

        <Field
          label="Symptom"
          htmlFor="symptom-name"
          required
          error={fieldErrors.symptom}
          description="Pick from the most common symptoms below, or choose Other to type your own."
        >
          <SelectWithCustom
            id="symptom-name"
            value={state.symptomChoice}
            customValue={state.symptomCustom}
            options={COMMON_SYMPTOMS}
            onChange={(next) => setState((s) => ({ ...s, symptomChoice: next, symptomCustom: next === SELECT_WITH_CUSTOM_OTHER_VALUE ? s.symptomCustom : "" }))}
            onCustomChange={(next) => setState((s) => ({ ...s, symptomCustom: next }))}
            placeholder="Choose a symptom…"
            customPlaceholder="e.g. tingling in left hand"
            invalid={Boolean(fieldErrors.symptom)}
          />
        </Field>

        <Field
          label="Severity (0–10)"
          htmlFor="symptom-severity"
          required
          error={fieldErrors.severity}
          hint={
            <span style={{ color: bucketColor, fontWeight: 700 }}>
              {bucketLabel}
            </span>
          }
        >
          <Slider
            id="symptom-severity"
            value={state.severity}
            min={0}
            max={10}
            onChange={(v) => setState((s) => ({ ...s, severity: v }))}
          />
        </Field>

        <FormGrid>
          <Field label="Date" htmlFor="symptom-date" required error={fieldErrors.date}>
            <TextInput
              id="symptom-date"
              type="date"
              value={state.date}
              onChange={(e) => setState((s) => ({ ...s, date: e.target.value }))}
              max={new Date().toISOString().slice(0, 10)}
              invalid={Boolean(fieldErrors.date)}
            />
          </Field>
          <Field
            label="Duration (optional)"
            htmlFor="symptom-duration"
            description="How long has this lasted?"
          >
            <TextInput
              id="symptom-duration"
              value={state.duration}
              onChange={(e) => setState((s) => ({ ...s, duration: e.target.value }))}
              placeholder="e.g. since this morning, 2 days"
              maxLength={80}
              autoComplete="off"
            />
          </Field>
        </FormGrid>

        <Field
          label="Notes (optional)"
          htmlFor="symptom-notes"
          description="Anything else your care team should know?"
        >
          <TextArea
            id="symptom-notes"
            value={state.notes}
            onChange={(e) => setState((s) => ({ ...s, notes: e.target.value }))}
            rows={3}
            maxLength={800}
            placeholder="What you were doing, how it feels, anything that makes it better or worse..."
          />
        </Field>

        <Checkbox
          tone="warning"
          checked={state.urgentFlag}
          onChange={(v) => setState((s) => ({ ...s, urgentFlag: v }))}
          label="Mark as urgent for clinician review"
          description="Tags this entry for the review queue. For severe or sudden symptoms (chest pain, trouble breathing, heavy bleeding, fainting), contact your oncology team or local emergency services right away — do not rely on this form."
        />

        <SafetyBanner tone="info" compact>
          {NON_DIAGNOSTIC_DISCLAIMER}
        </SafetyBanner>
      </form>
    </Modal>
  );
}

