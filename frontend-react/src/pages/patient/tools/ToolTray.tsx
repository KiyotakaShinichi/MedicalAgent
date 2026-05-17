import { type ReactNode, useRef } from "react";
import {
  Activity,
  FlaskConical,
  Image as ImageIcon,
  ScanLine,
  Stethoscope,
  Pill,
  Sparkles,
  HelpCircle,
  Upload,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";

export type ToolKey =
  | "symptom"
  | "cbc"
  | "imaging"
  | "medication"
  | "treatment"
  | "upload_cbc"
  | "upload_imaging"
  | "education";

interface ToolDescriptor {
  key: ToolKey;
  icon: LucideIcon;
  label: string;
  /** Tone hint used purely for the icon-tile colour — does not change behaviour. */
  tone: "rose" | "blue" | "amber" | "purple" | "green" | "neutral";
}

const TOOLS: ToolDescriptor[] = [
  { key: "symptom",        icon: Activity,     label: "Log symptom",      tone: "amber"   },
  { key: "cbc",            icon: FlaskConical, label: "Save CBC",         tone: "blue"    },
  { key: "imaging",        icon: ScanLine,     label: "Save imaging",     tone: "rose"    },
  { key: "medication",     icon: Pill,         label: "Log medication",   tone: "purple"  },
  { key: "treatment",      icon: Stethoscope,  label: "Treatment note",   tone: "green"   },
  { key: "upload_cbc",     icon: Upload,       label: "Upload CBC image", tone: "blue"    },
  { key: "upload_imaging", icon: ImageIcon,    label: "Upload MRI image", tone: "rose"    },
  { key: "education",      icon: HelpCircle,   label: "Ask a question",   tone: "neutral" },
];

const TONE_STYLE: Record<ToolDescriptor["tone"], { bg: string; fg: string }> = {
  rose:    { bg: "var(--rose-pale)", fg: "var(--rose-deep)" },
  blue:    { bg: "#dbeafe",          fg: "#1e3a8a" },
  amber:   { bg: "#fef3c7",          fg: "#92400e" },
  purple:  { bg: "#ede9fe",          fg: "#5b21b6" },
  green:   { bg: "#d1fae5",          fg: "#065f46" },
  neutral: { bg: "var(--surface2)",  fg: "var(--text-dim)" },
};

interface ToolTrayProps {
  /** Called when a tool button is clicked.  ``upload_*`` keys come AFTER the
   *  user picked a file — the caller receives the File for processing. */
  onSelect: (key: ToolKey, file?: File) => void;
}

/**
 * Persistent tool tray above the chat composer.
 *
 * Layout: a horizontally scrolling row of icon-chip buttons.  Each chip is
 * an accessible <button>; the two upload chips trigger hidden <input
 * type="file"> pickers and forward the chosen File to ``onSelect``.
 */
export function ToolTray({ onSelect }: ToolTrayProps) {
  const cbcUploadRef = useRef<HTMLInputElement>(null);
  const imagingUploadRef = useRef<HTMLInputElement>(null);

  function handleClick(key: ToolKey) {
    if (key === "upload_cbc")     cbcUploadRef.current?.click();
    else if (key === "upload_imaging") imagingUploadRef.current?.click();
    else                          onSelect(key);
  }

  function handleFile(key: "upload_cbc" | "upload_imaging", e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) onSelect(key, file);
    // Reset so re-uploading the same filename re-triggers onChange.
    e.target.value = "";
  }

  return (
    <div className="patient-tool-tray" role="toolbar" aria-label="Add a health update">
      <span className="patient-tool-tray-label">Add a health update</span>
      <div className="patient-tool-tray-scroller">
        {TOOLS.map(({ key, icon: Icon, label, tone }) => {
          const palette = TONE_STYLE[tone];
          return (
            <button
              key={key}
              type="button"
              className="patient-tool-chip"
              onClick={() => handleClick(key)}
              aria-label={label}
            >
              <span
                className="patient-tool-chip-icon"
                style={{ background: palette.bg, color: palette.fg }}
                aria-hidden="true"
              >
                <Icon size={14} />
              </span>
              <span>{label}</span>
            </button>
          );
        })}
      </div>

      {/* Hidden file pickers — the upload chips trigger these. */}
      <input
        ref={cbcUploadRef}
        type="file"
        accept="image/*,application/pdf"
        style={{ display: "none" }}
        onChange={(e) => handleFile("upload_cbc", e)}
      />
      <input
        ref={imagingUploadRef}
        type="file"
        accept="image/*,application/pdf"
        style={{ display: "none" }}
        onChange={(e) => handleFile("upload_imaging", e)}
      />
    </div>
  );
}

/** Inline icon companion for the toast that fires after a save action. */
export function ToolTraySavedIcon({ tone }: { tone: "rose" | "blue" | "amber" | "purple" | "green" }): ReactNode {
  const palette = TONE_STYLE[tone];
  return (
    <span
      className="inline-flex items-center justify-center"
      style={{
        width: 22, height: 22, borderRadius: 6,
        background: palette.bg, color: palette.fg, flexShrink: 0,
      }}
    >
      <Sparkles size={12} />
    </span>
  );
}
