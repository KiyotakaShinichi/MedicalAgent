import { useCallback, useState } from "react";

/**
 * Tiny, reusable submit-lifecycle hook for the tool forms.
 *
 * Why not a full form library?
 * ----------------------------
 * The forms are short (≤ 6 fields) and the validation is mostly per-field
 * presence/numeric-range checks.  Adding react-hook-form or formik for that
 * scope is overkill — and they'd both add bundle weight that hurts the
 * patient-portal initial load we already worked to keep small.
 *
 * What it gives you
 * -----------------
 *   - ``submitting`` boolean for disabling buttons + showing "Saving…"
 *   - ``submitError`` string for the top-of-form red banner
 *   - ``submit`` wrapper that:
 *       * sets submitting=true,
 *       * runs your ``submitFn`` (returns either a Promise or void),
 *       * captures errors into ``submitError`` and KEEPS the form open so
 *         the user can retry without losing their input,
 *       * calls ``onSuccess`` on resolution (parent typically refetches +
 *         closes the modal),
 *       * resets ``submitting`` in a ``finally`` so a thrown error never
 *         leaves the button stuck in "Saving…".
 *   - ``reset`` to clear submit state (used when the modal is closed/opened).
 *
 * Field state is intentionally NOT part of this hook — each form keeps its
 * own useState for fields so we don't fight TypeScript on heterogeneous
 * shapes (CBC has 4 numbers, imaging has 5 strings + a modality enum, etc.).
 */
export function useToolForm<TResult = unknown>(
  submitFn: () => Promise<TResult> | TResult,
  options: {
    onSuccess?: (result: TResult) => void;
  } = {},
) {
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  const submit = useCallback(async (event?: { preventDefault?: () => void }) => {
    event?.preventDefault?.();
    if (submitting) return;
    setSubmitting(true);
    setSubmitError(null);
    try {
      const result = await submitFn();
      options.onSuccess?.(result);
    } catch (error) {
      const friendly =
        error instanceof Error && error.message
          ? error.message
          : "Could not save. Please try again.";
      setSubmitError(friendly);
      if (import.meta.env?.DEV) {
        console.error("[useToolForm] submit failed:", error);
      }
    } finally {
      setSubmitting(false);
    }
  }, [submitFn, submitting, options]);

  const reset = useCallback(() => {
    setSubmitError(null);
    setSubmitting(false);
  }, []);

  return { submitting, submitError, submit, reset };
}
