/**
 * Non-component helpers for the `SelectWithCustom` form widget.
 *
 * Lives in its own module so Form.tsx can stay component-only and React's
 * fast-refresh contract (only-export-components) is satisfied.
 */

export const SELECT_WITH_CUSTOM_OTHER_VALUE = "__other__";

export interface SelectWithCustomOption {
  value: string;
  label: string;
  group?: string;
}

/**
 * Resolves the final value to send to the API: if the user picked the
 * "Other" branch, the free-text input wins; otherwise the dropdown's
 * canonical value is used.
 */
export function resolveSelectWithCustomValue(value: string, customValue: string): string {
  if (value === SELECT_WITH_CUSTOM_OTHER_VALUE) {
    return customValue.trim();
  }
  return value.trim();
}
