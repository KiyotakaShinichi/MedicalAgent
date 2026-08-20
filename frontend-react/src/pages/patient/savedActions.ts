import type { SavedAction } from "../../types/api";

/**
 * Saved-action types that mean the patient's clinical record changed.
 *
 * When the chat agent writes to the record, the dashboard must refetch so the
 * labs, symptom log, and timeline reflect it immediately — otherwise the
 * patient is told "saved" while still looking at pre-save data.
 *
 * Both the legacy imperative names (`save_lab`) and the current past-tense
 * names (`saved_labs`) are listed because the backend has emitted both and
 * this list is the only thing standing between a successful write and a stale
 * dashboard. An unrecognised type is treated as non-touching, so if a new
 * write action is added on the backend it must be registered here.
 */
export const REPORT_TOUCHING_ACTION_TYPES: ReadonlySet<string> = new Set([
  "saved_symptom",
  "saved_labs",
  "saved_medication",
  "saved_imaging_report",
  "save_symptom",
  "save_lab",
  "save_medication",
  "save_mri",
  "save_imaging_report",
]);

/** True when any action in the batch requires a patient-report refetch. */
export function touchesPatientReport(actions: readonly SavedAction[]): boolean {
  return actions.some((action) => REPORT_TOUCHING_ACTION_TYPES.has(action.type));
}
