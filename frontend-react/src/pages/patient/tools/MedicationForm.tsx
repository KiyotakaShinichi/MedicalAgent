import { useMemo, useState } from "react";
import { Modal } from "../../../components/ui/Modal";
import {
  Field,
  TextInput,
  TextArea,
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
import { addMyMedication } from "../../../api/client";
import { NON_DIAGNOSTIC_DISCLAIMER, COMMON_MEDICATIONS } from "../../../lib/clinical-constants";

interface MedicationFormProps {
  open: boolean;
  onClose: () => void;
  onSaved?: (summary: { medication: string }) => void;
}

interface FormState {
  medicationChoice: string;
  medicationCustom: string;
  dose: string;
  frequency: string;
  date: string;
  sideEffects: string;
  notes: string;
}

const INITIAL_STATE: FormState = {
  medicationChoice: "",
  medicationCustom: "",
  dose: "",
  frequency: "",
  date: new Date().toISOString().slice(0, 10),
  sideEffects: "",
  notes: "",
};

/**
 * Patient-facing medication entry.  We deliberately do NOT auto-classify the
 * medication against any list — patients name what they take, the clinician
 * reviews it.  Safety stays where it belongs: the medication-change refusal
 * lives in the chat agent, not in this tracking form.
 */
export function MedicationForm({ open, onClose, onSaved }: MedicationFormProps) {
  const [state, setState] = useState<FormState>(INITIAL_STATE);

  const resolvedName = resolveSelectWithCustomValue(state.medicationChoice, state.medicationCustom);
  const fieldErrors = useMemo(() => {
    const errors: Partial<Record<"medication" | "date", string>> = {};
    if (!state.medicationChoice) {
      errors.medication = "Pick a medication from the list or choose Other.";
    } else if (state.medicationChoice === SELECT_WITH_CUSTOM_OTHER_VALUE && !resolvedName) {
      errors.medication = "Type the medication name.";
    } else if (resolvedName.length > 120) {
      errors.medication = "Keep under 120 characters.";
    }
    if (!state.date) errors.date = "Pick a date.";
    return errors;
  }, [state.medicationChoice, resolvedName, state.date]);

  const isValid = Object.keys(fieldErrors).length === 0;

  const { submitting, submitError, submit, reset } = useToolForm(
    async () => {
      const result = await addMyMedication({
        medication: resolvedName,
        dose: state.dose.trim() || undefined,
        frequency: state.frequency.trim() || undefined,
        date: state.date,
        side_effects: state.sideEffects.trim() || undefined,
        notes: state.notes.trim() || undefined,
      });
      return result;
    },
    {
      onSuccess: () => {
        onSaved?.({ medication: resolvedName });
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
      title="Log a medication"
      description="Track what you are taking so your care team can review it during follow-ups."
      size="md"
      dismissable={!submitting}
      footer={
        <FormFooter
          onCancel={handleClose}
          submitLabel="Save medication"
          submitting={submitting}
          disabled={!isValid}
          hint="This is a tracking entry, not a prescription"
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
          label="Medication"
          htmlFor="med-name"
          required
          error={fieldErrors.medication}
          description="Pick a common breast-cancer-treatment medication below, or choose Other to type your own. Brand names are noted in parentheses."
        >
          <SelectWithCustom
            id="med-name"
            value={state.medicationChoice}
            customValue={state.medicationCustom}
            options={COMMON_MEDICATIONS}
            onChange={(next) => setState((s) => ({ ...s, medicationChoice: next, medicationCustom: next === SELECT_WITH_CUSTOM_OTHER_VALUE ? s.medicationCustom : "" }))}
            onCustomChange={(next) => setState((s) => ({ ...s, medicationCustom: next }))}
            placeholder="Choose a medication…"
            customPlaceholder="e.g. metformin"
            customMaxLength={120}
            invalid={Boolean(fieldErrors.medication)}
          />
        </Field>

        <FormGrid>
          <Field label="Dose (optional)" htmlFor="med-dose">
            <TextInput
              id="med-dose"
              value={state.dose}
              onChange={(e) => setState((s) => ({ ...s, dose: e.target.value }))}
              placeholder="e.g. 8 mg"
              maxLength={60}
              autoComplete="off"
            />
          </Field>

          <Field label="Frequency (optional)" htmlFor="med-freq">
            <TextInput
              id="med-freq"
              value={state.frequency}
              onChange={(e) => setState((s) => ({ ...s, frequency: e.target.value }))}
              placeholder="e.g. twice a day"
              maxLength={60}
              autoComplete="off"
            />
          </Field>

          <Field label="Date" htmlFor="med-date" required error={fieldErrors.date} fullWidth>
            <TextInput
              id="med-date"
              type="date"
              value={state.date}
              onChange={(e) => setState((s) => ({ ...s, date: e.target.value }))}
              max={new Date().toISOString().slice(0, 10)}
              invalid={Boolean(fieldErrors.date)}
            />
          </Field>
        </FormGrid>

        <Field
          label="Side effects (optional)"
          htmlFor="med-side"
          description="Anything new you have noticed since starting or after a dose."
        >
          <TextArea
            id="med-side"
            value={state.sideEffects}
            onChange={(e) => setState((s) => ({ ...s, sideEffects: e.target.value }))}
            rows={2}
            maxLength={500}
            placeholder="e.g. mild nausea for ~1 hour after each dose"
          />
        </Field>

        <Field label="Notes (optional)" htmlFor="med-notes">
          <TextArea
            id="med-notes"
            value={state.notes}
            onChange={(e) => setState((s) => ({ ...s, notes: e.target.value }))}
            rows={2}
            maxLength={500}
            placeholder="Anything else your care team should know..."
          />
        </Field>

        <SafetyBanner tone="info" compact>
          {NON_DIAGNOSTIC_DISCLAIMER} Dose or schedule changes must be agreed with your care team.
        </SafetyBanner>
      </form>
    </Modal>
  );
}
