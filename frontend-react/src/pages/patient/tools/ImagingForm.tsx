import { useMemo, useState } from "react";
import { Drawer } from "../../../components/ui/Drawer";
import {
  Field,
  TextInput,
  TextArea,
  Select,
  FormFooter,
  FormError,
  FormGrid,
} from "../../../components/ui/Form";
import { SafetyBanner } from "../../../components/ui/SafetyBanner";
import { useToolForm } from "../../../hooks/useToolForm";
import { addMyImagingReport } from "../../../api/client";
import {
  IMAGING_MODALITIES,
  NON_DIAGNOSTIC_DISCLAIMER,
  type ImagingModality,
} from "../../../lib/clinical-constants";

interface ImagingFormProps {
  open: boolean;
  onClose: () => void;
  onSaved?: (summary: { modality: string }) => void;
}

interface FormState {
  date: string;
  modality: ImagingModality;
  bodyArea: string;
  reportType: string;
  findings: string;
  impression: string;
  notes: string;
}

const INITIAL_STATE: FormState = {
  date: new Date().toISOString().slice(0, 10),
  modality: "MRI",
  bodyArea: "Breast",
  reportType: "",
  findings: "",
  impression: "",
  notes: "",
};

/**
 * Patient-facing imaging-report entry form.  Either ``findings`` or
 * ``impression`` must be present — that rule is enforced both client-side
 * (so the Save button stays disabled) and server-side (so a misbehaving
 * client cannot save an empty report row).
 *
 * No file upload yet.  The brief asks for "Upload MRI image / Upload CBC
 * image" — those land in Phase 3.7 as separate tool-tray buttons that go
 * through the existing ``/me/uploads`` endpoint.
 */
export function ImagingForm({ open, onClose, onSaved }: ImagingFormProps) {
  const [state, setState] = useState<FormState>(INITIAL_STATE);

  const findings = state.findings.trim();
  const impression = state.impression.trim();
  const bodyArea = state.bodyArea.trim();

  const fieldErrors = useMemo(() => {
    const errors: Partial<Record<keyof FormState, string>> = {};
    if (!state.date) errors.date = "Pick a date.";
    if (!state.modality) errors.modality = "Pick a modality.";
    if (!findings && !impression) {
      errors.findings = "Either findings or impression is required.";
    }
    if (findings.length > 4000) errors.findings = "Trim to 4000 characters or less.";
    if (impression.length > 4000) errors.impression = "Trim to 4000 characters or less.";
    if (bodyArea && bodyArea.length > 80) errors.bodyArea = "Keep under 80 characters.";
    return errors;
  }, [state.date, state.modality, findings, impression, bodyArea]);

  const isValid = Object.keys(fieldErrors).length === 0;

  const { submitting, submitError, submit, reset } = useToolForm(
    async () => {
      const result = await addMyImagingReport({
        date: state.date,
        modality: state.modality,
        report_type: state.reportType.trim() || undefined,
        body_site: bodyArea || undefined,
        findings: findings || undefined,
        impression: impression || undefined,
        notes: state.notes.trim() || undefined,
      });
      return result;
    },
    {
      onSuccess: (result) => {
        onSaved?.({ modality: result.modality });
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
      title="Save imaging report"
      description="Record an MRI, CT, ultrasound, or mammogram report so your care team can review the text."
      size="lg"
      dismissable={!submitting}
      footer={
        <FormFooter
          onCancel={handleClose}
          submitLabel="Save report"
          submitting={submitting}
          disabled={!isValid}
          hint="Stored as report text — this system does not interpret images"
        />
      }
    >
      <form
        onSubmit={submit}
        noValidate
        style={{ display: "flex", flexDirection: "column", gap: 14 }}
      >
        <FormError message={submitError} />

        <FormGrid>
          <Field label="Modality" htmlFor="img-modality" required error={fieldErrors.modality}>
            <Select
              id="img-modality"
              value={state.modality}
              onChange={(e) =>
                setState((s) => ({ ...s, modality: e.target.value as ImagingModality }))
              }
              invalid={Boolean(fieldErrors.modality)}
            >
              {IMAGING_MODALITIES.map((m) => (
                <option key={m.value} value={m.value}>{m.label}</option>
              ))}
            </Select>
          </Field>

          <Field label="Report date" htmlFor="img-date" required error={fieldErrors.date}>
            <TextInput
              id="img-date"
              type="date"
              value={state.date}
              onChange={(e) => setState((s) => ({ ...s, date: e.target.value }))}
              max={new Date().toISOString().slice(0, 10)}
              invalid={Boolean(fieldErrors.date)}
            />
          </Field>

          <Field
            label="Body area"
            htmlFor="img-body"
            error={fieldErrors.bodyArea}
            description="Where the imaging looked, e.g. 'Breast', 'Chest', 'Abdomen'."
          >
            <TextInput
              id="img-body"
              value={state.bodyArea}
              onChange={(e) => setState((s) => ({ ...s, bodyArea: e.target.value }))}
              placeholder="e.g. Breast"
              maxLength={80}
              autoComplete="off"
            />
          </Field>

          <Field
            label="Report type (optional)"
            htmlFor="img-type"
            description="Free-text label, e.g. 'Diagnostic', 'Follow-up'."
          >
            <TextInput
              id="img-type"
              value={state.reportType}
              onChange={(e) => setState((s) => ({ ...s, reportType: e.target.value }))}
              placeholder="e.g. Follow-up"
              maxLength={80}
              autoComplete="off"
            />
          </Field>
        </FormGrid>

        <Field
          label="Findings"
          htmlFor="img-findings"
          error={fieldErrors.findings}
          description="Paste the findings paragraph from your report. Either findings or impression is required."
        >
          <TextArea
            id="img-findings"
            value={state.findings}
            onChange={(e) => setState((s) => ({ ...s, findings: e.target.value }))}
            rows={4}
            maxLength={4000}
            placeholder="Paste the findings text from the radiology report..."
            invalid={Boolean(fieldErrors.findings)}
          />
        </Field>

        <Field
          label="Impression"
          htmlFor="img-impression"
          error={fieldErrors.impression}
          description="Short clinical summary from the report, if any."
        >
          <TextArea
            id="img-impression"
            value={state.impression}
            onChange={(e) => setState((s) => ({ ...s, impression: e.target.value }))}
            rows={3}
            maxLength={4000}
            placeholder="Paste the impression text..."
            invalid={Boolean(fieldErrors.impression)}
          />
        </Field>

        <Field
          label="Your notes (optional)"
          htmlFor="img-notes"
          description="Anything else for your care team to know about this scan."
        >
          <TextArea
            id="img-notes"
            value={state.notes}
            onChange={(e) => setState((s) => ({ ...s, notes: e.target.value }))}
            rows={2}
            maxLength={800}
          />
        </Field>

        <SafetyBanner tone="info" compact>
          {NON_DIAGNOSTIC_DISCLAIMER} Imaging text is stored as-is; this system does not interpret radiology images.
        </SafetyBanner>
      </form>
    </Drawer>
  );
}
