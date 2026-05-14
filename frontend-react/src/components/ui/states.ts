/**
 * Canonical names for the three "state" panes — re-exported with the
 * shorter, more conventional names other React projects expect.
 *
 * The visuals live in Spinner.tsx; this module is purely organizational so
 * callers can write:
 *
 *   import { EmptyState, LoadingState, ErrorState } from "@ui/states";
 *
 * instead of importing "EmptyPane" / "LoadingPane" / "ErrorPane" from
 * Spinner.tsx, which reads oddly.
 */

export {
  EmptyPane as EmptyState,
  LoadingPane as LoadingState,
  ErrorPane as ErrorState,
  Spinner,
} from "./Spinner";
