import { useMemo, useState } from "react";
import { Modal } from "../../../components/ui/Modal";
import {
  Field,
  TextInput,
  TextArea,
  NumberInput,
  FormFooter,
  FormError,
  FormGrid,
} from "../../../components/ui/Form";
import { SafetyBanner } from "../../../components/ui/SafetyBanner";
import { useToolForm } from "../../../hooks/useToolForm";
import { addMyTreatment } from "../../../api/client";
import { NON_DIAGNOSTIC_DISCLAIMER } from "../../../lib/clinical-constants";

interface TreatmentNoteFormProps {
  open: boolean;
  onClose: () => void;
  onSaved?: (summary: { drug: string }) => void;
}

interface FormState {
  drug: string;
  cycle: string;     // string so empty field doesn't show "0"
  date: string;
  notes: string;
}

const INITIAL_STATE: FormState = {
  drug: "",
  cycle: "",
  date: new Date().toISOString().slice(0, 10),
  notes: "",
};

/**
 * Patient-facing treatment-cycle note.  Cycle number is optional because
 * patients often don't remember the count; the backend defaults to 0 in
 * that case so the existing schema stays valid.  This is a tracking note —
 * no medication-change or scheduling logic lives here.
 */
export function TreatmentNoteForm({ open, onClose, onSaved }: TreatmentNoteFormProps) {
  const [state, setState] = useState<FormState>(INITIAL_STATE);

  const trimmedDrug = state.drug.trim();
  const cycleValue = state.cycle.trim() ? Number(state.cycle) : null;

  const fieldErrors = useMemo(() => {
    const errors: Partial<Record<keyof FormState, string>> = {};
    if (!trimmedDrug) errors.drug = "Add the treatment name.";
    else if (trimmedDrug.length > 120) errors.drug = "Keep under 120 characters.";
    if (!state.date) errors.date = "Pick a date.";
    if (cycleValue !== null && (!Number.isFinite(cycleValue) || cycleValue < 0 || cycleValue > 99)) {
      errors.cycle = "Cycle must be between 0 and 99.";
    }
    return errors;
  }, [trimmedDrug, state.date, cycleValue]);

  const isValid = Object.keys(fieldErrors).length === 0;

  const { submitting, submitError, submit, reset } = useToolForm(
    async () => {
      const result = await addMyTreatment({
        date: state.date,
        drug: trimmedDrug,
        cycle: cycleValue !== null && Number.isFinite(cycleValue) ? Math.trunc(cycleValue) : undefined,
        notes: state.notes.trim() || undefined,
      });
      return result;
    },
    {
      onSuccess: () => {
        onSaved?.({ drug: trimmedDrug });
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
    <Modal
      open={open}
      onClose={handleClose}
      title="Log a treatment cycle"
      description="Note a chemotherapy, infusion, or treatment session so the timeline reflects it."
      size="md"
      dismissable={!submitting}
      footer={
        <FormFooter
          onCancel={handleClose}
          submitLabel="Save treatment note"
          submitting={submitting}
          disabled={!isValid}
          hint="Tracking note — treatment decisions remain with your care team"
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
          label="Treatment / drug"
          htmlFor="tx-drug"
          required
          error={fieldErrors.drug}
          description="Whatever the team called the treatment — e.g. 'Dose-dense AC', 'paclitaxel cycle 3'."
        >
          <TextInput
            id="tx-drug"
            value={state.drug}
            onChange={(e) => setState((s) => ({ ...s, drug: e.target.value }))}
            placeholder="e.g. Paclitaxel"
            invalid={Boolean(fieldErrors.drug)}
            maxLength={120}
            autoComplete="off"
          />
        </Field>

        <FormGrid>
          <Field label="Date" htmlFor="tx-date" required error={fieldErrors.date}>
            <TextInput
              id="tx-date"
              type="date"
              value={state.date}
              onChange={(e) => setState((s) => ({ ...s, date: e.target.value }))}
              max={new Date().toISOString().slice(0, 10)}
              invalid={Boolean(fieldErrors.date)}
            />
          </Field>

          <Field
            label="Cycle (optional)"
            htmlFor="tx-cycle"
            error={fieldErrors.cycle}
            description="Cycle number, if you have it."
          >
            <NumberInput
              id="tx-cycle"
              step="1"
              min="0"
              max="99"
              value={state.cycle}
              onChange={(e) => setState((s) => ({ ...s, cycle: e.target.value }))}
              placeholder="e.g. 3"
              invalid={Boolean(fieldErrors.cycle)}
            />
          </Field>
        </FormGrid>

        <Field label="Notes (optional)" htmlFor="tx-notes">
          <TextArea
            id="tx-notes"
            value={state.notes}
            onChange={(e) => setState((s) => ({ ...s, notes: e.target.value }))}
            rows={3}
            maxLength={800}
            placeholder="How the session went, side effects, anything for the team..."
          />
        </Field>

        <SafetyBanner tone="info" compact>
          {NON_DIAGNOSTIC_DISCLAIMER}
        </SafetyBanner>
      </form>
    </Modal>
  );
}
