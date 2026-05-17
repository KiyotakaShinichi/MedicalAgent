import { forwardRef, useId, type ReactNode, type SelectHTMLAttributes, type InputHTMLAttributes, type TextareaHTMLAttributes } from "react";
import { clsx } from "clsx";
import { AlertCircle } from "lucide-react";

/**
 * Small, consistent form primitives.  Each field renders as:
 *   <label> Label <hint? />
 *   <control />
 *   <error?> | <description?>
 *
 * All primitives accept the same shape so tool forms read uniformly.
 *
 * Keep this file pure-presentational.  Validation + submit state belongs in
 * the parent form (see ``useToolForm``).
 */

interface FieldProps {
  label: string;
  htmlFor: string;
  hint?: ReactNode;
  description?: ReactNode;
  error?: string | null;
  required?: boolean;
  children: ReactNode;
  /** Layout helper — span 2 columns of a parent CSS grid. */
  fullWidth?: boolean;
}

export function Field({ label, htmlFor, hint, description, error, required, children, fullWidth }: FieldProps) {
  return (
    <div className={clsx("form-field", fullWidth && "form-field--full")}>
      <div className="form-field-header">
        <label htmlFor={htmlFor} className="form-label">
          {label}
          {required && <span aria-hidden="true" style={{ color: "#b91c1c", marginLeft: 2 }}>*</span>}
        </label>
        {hint && <span className="form-hint">{hint}</span>}
      </div>
      {children}
      {error ? (
        <p className="form-error" role="alert">
          <AlertCircle size={12} aria-hidden="true" />
          {error}
        </p>
      ) : description ? (
        <p className="form-description">{description}</p>
      ) : null}
    </div>
  );
}

interface TextInputProps extends InputHTMLAttributes<HTMLInputElement> {
  invalid?: boolean;
}

export const TextInput = forwardRef<HTMLInputElement, TextInputProps>(function TextInput(
  { invalid, className, ...rest }, ref,
) {
  return (
    <input
      ref={ref}
      className={clsx("form-input", invalid && "form-input--invalid", className)}
      aria-invalid={invalid || undefined}
      {...rest}
    />
  );
});

interface NumberInputProps extends Omit<InputHTMLAttributes<HTMLInputElement>, "type"> {
  invalid?: boolean;
  unit?: string;
}

export const NumberInput = forwardRef<HTMLInputElement, NumberInputProps>(function NumberInput(
  { invalid, unit, className, ...rest }, ref,
) {
  if (!unit) {
    return (
      <input
        ref={ref}
        type="number"
        inputMode="decimal"
        className={clsx("form-input", invalid && "form-input--invalid", className)}
        aria-invalid={invalid || undefined}
        {...rest}
      />
    );
  }
  return (
    <div className={clsx("form-number-wrap", invalid && "form-input--invalid")}>
      <input
        ref={ref}
        type="number"
        inputMode="decimal"
        className="form-number-input"
        aria-invalid={invalid || undefined}
        {...rest}
      />
      <span className="form-number-unit" aria-hidden="true">{unit}</span>
    </div>
  );
});

interface TextAreaProps extends TextareaHTMLAttributes<HTMLTextAreaElement> {
  invalid?: boolean;
}

export const TextArea = forwardRef<HTMLTextAreaElement, TextAreaProps>(function TextArea(
  { invalid, className, rows = 3, ...rest }, ref,
) {
  return (
    <textarea
      ref={ref}
      rows={rows}
      className={clsx("form-input form-textarea", invalid && "form-input--invalid", className)}
      aria-invalid={invalid || undefined}
      {...rest}
    />
  );
});

interface SelectProps extends SelectHTMLAttributes<HTMLSelectElement> {
  invalid?: boolean;
}

export const Select = forwardRef<HTMLSelectElement, SelectProps>(function Select(
  { invalid, className, children, ...rest }, ref,
) {
  return (
    <select
      ref={ref}
      className={clsx("form-input form-select", invalid && "form-input--invalid", className)}
      aria-invalid={invalid || undefined}
      {...rest}
    >
      {children}
    </select>
  );
});

interface CheckboxProps {
  id?: string;
  checked: boolean;
  onChange: (next: boolean) => void;
  label: string;
  description?: ReactNode;
  disabled?: boolean;
  /** Tone signal — "warning" tints the checkbox border for safety flags. */
  tone?: "default" | "warning";
}

export function Checkbox({ id, checked, onChange, label, description, disabled, tone = "default" }: CheckboxProps) {
  const fallbackId = useId();
  const inputId = id ?? fallbackId;
  return (
    <label
      htmlFor={inputId}
      className={clsx("form-checkbox-row", tone === "warning" && "form-checkbox-row--warning")}
    >
      <input
        id={inputId}
        type="checkbox"
        className="form-checkbox"
        checked={checked}
        disabled={disabled}
        onChange={(e) => onChange(e.target.checked)}
      />
      <div style={{ minWidth: 0 }}>
        <span className="form-checkbox-label">{label}</span>
        {description && <p className="form-checkbox-description">{description}</p>}
      </div>
    </label>
  );
}

interface SliderProps {
  id?: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  onChange: (next: number) => void;
  /** Optional formatter — defaults to plain number. */
  format?: (value: number) => string;
}

export function Slider({ id, value, min, max, step = 1, onChange, format }: SliderProps) {
  const fallbackId = useId();
  const inputId = id ?? fallbackId;
  return (
    <div className="form-slider">
      <input
        id={inputId}
        type="range"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(e) => onChange(Number(e.target.value))}
        className="form-slider-input"
        aria-valuemin={min}
        aria-valuemax={max}
        aria-valuenow={value}
      />
      <span className="form-slider-value">{format ? format(value) : value}</span>
    </div>
  );
}

interface FormFooterProps {
  onCancel?: () => void;
  cancelLabel?: string;
  submitLabel: string;
  submitting?: boolean;
  disabled?: boolean;
  /** Tone signal — danger turns the submit button red (used for "discard"
   *  flows, not relevant here yet but cheap to plumb). */
  tone?: "default" | "danger";
  /** Optional inline hint shown left-aligned (e.g. "Auto-saves to your record"). */
  hint?: ReactNode;
}

export function FormFooter({
  onCancel,
  cancelLabel = "Cancel",
  submitLabel,
  submitting,
  disabled,
  tone = "default",
  hint,
}: FormFooterProps) {
  return (
    <>
      {hint && <span className="form-footer-hint">{hint}</span>}
      {onCancel && (
        <button
          type="button"
          className="form-button form-button--ghost"
          onClick={onCancel}
          disabled={submitting}
        >
          {cancelLabel}
        </button>
      )}
      <button
        type="submit"
        className={clsx("form-button", tone === "danger" ? "form-button--danger" : "form-button--primary")}
        disabled={submitting || disabled}
      >
        {submitting ? "Saving…" : submitLabel}
      </button>
    </>
  );
}

interface FormErrorProps {
  message: string | null;
}

export function FormError({ message }: FormErrorProps) {
  if (!message) return null;
  return (
    <div className="form-error-banner" role="alert">
      <AlertCircle size={14} aria-hidden="true" />
      <span>{message}</span>
    </div>
  );
}

interface FormGridProps {
  children: ReactNode;
  /** Default 2 columns at >= 480px; 1 column on small screens. */
  columns?: 1 | 2;
}

export function FormGrid({ children, columns = 2 }: FormGridProps) {
  return <div className={clsx("form-grid", columns === 1 && "form-grid--single")}>{children}</div>;
}


// ─── SelectWithCustom ────────────────────────────────────────────────────────
//
// Curated-dropdown pattern with an "Other (specify)" branch: when the
// patient picks "Other", a text input appears below for free-form entry.
// Used by SymptomForm / MedicationForm so common entries land on a stable
// canonical name (reducing typo drift across the patient record), while
// uncommon entries still flow through cleanly.
//
// The sentinel value, helper, and option type live in `selectWithCustom.ts`
// (sibling module) so Form.tsx only exports components — required by the
// react-refresh/only-export-components rule.

import {
  SELECT_WITH_CUSTOM_OTHER_VALUE,
  type SelectWithCustomOption,
} from "./selectWithCustom";

interface SelectWithCustomProps {
  id?: string;
  /** Current selection — empty string when nothing chosen yet. */
  value: string;
  /** Current free-text value when the "Other" branch is active. */
  customValue: string;
  options: readonly SelectWithCustomOption[];
  /** Fired when the dropdown selection changes (canonical value, or
   *  ``SELECT_WITH_CUSTOM_OTHER_VALUE`` when the user picks "Other"). */
  onChange: (next: string) => void;
  /** Fired when the free-text input changes. */
  onCustomChange: (next: string) => void;
  placeholder?: string;
  customPlaceholder?: string;
  customMaxLength?: number;
  invalid?: boolean;
  /** Label shown for the "Other" option in the dropdown. */
  otherLabel?: string;
}

export function SelectWithCustom({
  id,
  value,
  customValue,
  options,
  onChange,
  onCustomChange,
  placeholder = "Choose one…",
  customPlaceholder = "Type your own",
  customMaxLength = 80,
  invalid,
  otherLabel = "Other (specify)",
}: SelectWithCustomProps) {
  const isOther = value === SELECT_WITH_CUSTOM_OTHER_VALUE;

  // Group options by their `group` field, preserving insertion order.
  const grouped = new Map<string, SelectWithCustomOption[]>();
  for (const opt of options) {
    const key = opt.group ?? "__ungrouped__";
    if (!grouped.has(key)) grouped.set(key, []);
    grouped.get(key)!.push(opt);
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      <select
        id={id}
        className={clsx("form-input form-select", invalid && "form-input--invalid")}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        aria-invalid={invalid || undefined}
      >
        <option value="" disabled>{placeholder}</option>
        {[...grouped.entries()].map(([groupName, opts]) =>
          groupName === "__ungrouped__" ? (
            opts.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)
          ) : (
            <optgroup key={groupName} label={groupName}>
              {opts.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
            </optgroup>
          ),
        )}
        <option value={SELECT_WITH_CUSTOM_OTHER_VALUE}>{otherLabel}</option>
      </select>

      {isOther && (
        <input
          type="text"
          className={clsx("form-input", invalid && "form-input--invalid")}
          value={customValue}
          onChange={(e) => onCustomChange(e.target.value)}
          placeholder={customPlaceholder}
          maxLength={customMaxLength}
          autoComplete="off"
          aria-label={otherLabel}
          aria-invalid={invalid || undefined}
        />
      )}
    </div>
  );
}
