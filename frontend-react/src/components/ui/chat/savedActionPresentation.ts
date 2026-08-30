import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  FlaskConical,
  Pill,
  ScanLine,
  ShieldCheck,
  Undo2,
  X,
  type LucideIcon,
} from "lucide-react";

import type { SavedAction } from "../../../types/api";

export type ChipTone = "success" | "warning" | "info";

export interface SavedActionDescriptor {
  label: string;
  Icon: LucideIcon;
  tone: ChipTone;
}

export function savedActionDescriptor(action: SavedAction): SavedActionDescriptor {
  switch (action.type) {
    case "saved_symptom":
    case "save_symptom":
      return { label: "Symptom saved", Icon: Activity, tone: "success" };
    case "saved_labs":
    case "save_lab":
      return { label: "CBC saved", Icon: FlaskConical, tone: "success" };
    case "saved_medication":
    case "save_medication":
      return { label: "Medication saved", Icon: Pill, tone: "success" };
    case "saved_imaging_report":
    case "save_mri": {
      const modality = String((action.data as { modality?: unknown })?.modality ?? "").toLowerCase();
      const label =
        modality.includes("mri") ? "MRI report saved" :
        modality.includes("ct") ? "CT report saved" :
        modality.includes("ultrasound") ? "Ultrasound report saved" :
        "Imaging report saved";
      return { label, Icon: ScanLine, tone: "success" };
    }
    case "possible_metastatic_indicator":
      return { label: "Review flag added", Icon: AlertTriangle, tone: "warning" };
    case "pending_record_confirmation":
      return { label: "Waiting for your confirmation", Icon: ShieldCheck, tone: "info" };
    case "record_write_cancelled":
      return { label: "Save cancelled", Icon: X, tone: "info" };
    case "duplicate_record_prevented":
      return { label: "Duplicate prevented", Icon: ShieldCheck, tone: "warning" };
    case "record_write_undone":
      return { label: "Save undone", Icon: Undo2, tone: "info" };
    default:
      return { label: action.type, Icon: CheckCircle2, tone: "success" };
  }
}

export function describeSavedAction(action: SavedAction): { label: string; tone: ChipTone } {
  const { label, tone } = savedActionDescriptor(action);
  return { label, tone };
}
