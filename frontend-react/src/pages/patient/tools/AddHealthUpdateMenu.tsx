import { useCallback, useEffect, useRef, useState } from "react";
import {
  Activity,
  ChevronDown,
  FlaskConical,
  Image as ImageIcon,
  Pill,
  Plus,
  ScanLine,
  Stethoscope,
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
  key: Exclude<ToolKey, "education">;
  icon: LucideIcon;
  label: string;
  description: string;
  tone: "rose" | "blue" | "amber" | "purple" | "green";
}

const TOOLS: ToolDescriptor[] = [
  { key: "symptom",        icon: Activity,     label: "Log symptom",      description: "How you feel right now — severity 0–10.",         tone: "amber"  },
  { key: "cbc",            icon: FlaskConical, label: "Save CBC",         description: "Blood count values from your latest lab.",         tone: "blue"   },
  { key: "imaging",        icon: ScanLine,     label: "Save imaging",     description: "MRI / CT / ultrasound summary text.",              tone: "rose"   },
  { key: "medication",     icon: Pill,         label: "Log medication",   description: "Drug + dose you took today.",                      tone: "purple" },
  { key: "treatment",      icon: Stethoscope,  label: "Treatment note",   description: "Short note about a treatment-related event.",       tone: "green"  },
  { key: "upload_cbc",     icon: Upload,       label: "Upload CBC image", description: "Photo or PDF of a CBC printout.",                   tone: "blue"   },
  { key: "upload_imaging", icon: ImageIcon,    label: "Upload MRI image", description: "Photo or PDF of an imaging report.",                tone: "rose"   },
];

const TONE_STYLE: Record<ToolDescriptor["tone"], { bg: string; fg: string }> = {
  rose:    { bg: "var(--rose-pale)", fg: "var(--rose-deep)" },
  blue:    { bg: "#dbeafe",          fg: "#1e3a8a" },
  amber:   { bg: "#fef3c7",          fg: "#92400e" },
  purple:  { bg: "#ede9fe",          fg: "#5b21b6" },
  green:   { bg: "#d1fae5",          fg: "#065f46" },
};

interface Props {
  onSelect: (key: ToolKey, file?: File) => void;
  /** Optional override for the trigger label.  Defaults to "Add health update". */
  triggerLabel?: string;
}

/**
 * Compact dropdown that replaces the 8-chip tool tray.
 *
 * Accessibility:
 * - The trigger is a real <button> with ``aria-haspopup="menu"``,
 *   ``aria-expanded``, and ``aria-controls`` wired to the menu list.
 * - Arrow Up/Down move focus between items, Home/End jump to ends.
 * - Enter or Space activates the focused item.
 * - Escape closes the menu and returns focus to the trigger.
 * - Outside-click closes the menu.
 *
 * The two upload items trigger hidden <input type="file"> pickers and
 * forward the chosen File to ``onSelect``.  All other items dispatch
 * immediately on click — the parent owns the modal/state.
 */
export function AddHealthUpdateMenu({ onSelect, triggerLabel = "Add health update" }: Props) {
  const [open, setOpen] = useState(false);
  const [focusIndex, setFocusIndex] = useState(0);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const menuRef = useRef<HTMLUListElement>(null);
  const itemRefs = useRef<Array<HTMLButtonElement | null>>([]);
  const cbcUploadRef = useRef<HTMLInputElement>(null);
  const imagingUploadRef = useRef<HTMLInputElement>(null);

  const close = useCallback((restoreFocus = true) => {
    setOpen(false);
    if (restoreFocus) triggerRef.current?.focus();
  }, []);

  // Outside-click handling.  Bound only while the menu is open.
  useEffect(() => {
    if (!open) return;
    function onDocClick(e: MouseEvent) {
      const target = e.target as Node | null;
      if (!target) return;
      if (triggerRef.current?.contains(target)) return;
      if (menuRef.current?.contains(target)) return;
      setOpen(false);
    }
    document.addEventListener("mousedown", onDocClick);
    return () => document.removeEventListener("mousedown", onDocClick);
  }, [open]);

  // Move DOM focus to the focused item whenever it changes (while open).
  useEffect(() => {
    if (!open) return;
    const node = itemRefs.current[focusIndex];
    if (node) node.focus();
  }, [focusIndex, open]);

  function openMenu(initialIndex = 0) {
    setFocusIndex(initialIndex);
    setOpen(true);
  }

  function handleTriggerKey(e: React.KeyboardEvent<HTMLButtonElement>) {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      openMenu(0);
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      openMenu(TOOLS.length - 1);
    } else if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      setOpen((v) => !v);
      if (!open) setFocusIndex(0);
    }
  }

  function handleMenuKey(e: React.KeyboardEvent<HTMLUListElement>) {
    if (e.key === "Escape") {
      e.preventDefault();
      close(true);
    } else if (e.key === "ArrowDown") {
      e.preventDefault();
      setFocusIndex((i) => (i + 1) % TOOLS.length);
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setFocusIndex((i) => (i - 1 + TOOLS.length) % TOOLS.length);
    } else if (e.key === "Home") {
      e.preventDefault();
      setFocusIndex(0);
    } else if (e.key === "End") {
      e.preventDefault();
      setFocusIndex(TOOLS.length - 1);
    } else if (e.key === "Tab") {
      // Tabbing out closes the menu without restoring trigger focus.
      close(false);
    }
  }

  function activate(key: ToolDescriptor["key"]) {
    if (key === "upload_cbc") {
      cbcUploadRef.current?.click();
    } else if (key === "upload_imaging") {
      imagingUploadRef.current?.click();
    } else {
      onSelect(key);
    }
    close(true);
  }

  function handleFile(key: "upload_cbc" | "upload_imaging", e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) onSelect(key, file);
    e.target.value = "";
  }

  return (
    <div className="add-health-update">
      <button
        ref={triggerRef}
        type="button"
        className="add-health-update-trigger"
        aria-haspopup="menu"
        aria-expanded={open}
        aria-controls="add-health-update-menu"
        onClick={() => {
          setFocusIndex(0);
          setOpen((v) => !v);
        }}
        onKeyDown={handleTriggerKey}
      >
        <Plus size={14} aria-hidden="true" />
        <span>{triggerLabel}</span>
        <ChevronDown
          size={14}
          aria-hidden="true"
          style={{ transform: open ? "rotate(180deg)" : "rotate(0deg)", transition: "transform 140ms ease" }}
        />
      </button>

      {open && (
        <ul
          id="add-health-update-menu"
          ref={menuRef}
          className="add-health-update-menu"
          role="menu"
          aria-label="Add a health update"
          onKeyDown={handleMenuKey}
        >
          {TOOLS.map((t, i) => {
            const Icon = t.icon;
            const palette = TONE_STYLE[t.tone];
            return (
              <li key={t.key} role="none">
                <button
                  ref={(el) => { itemRefs.current[i] = el; }}
                  type="button"
                  role="menuitem"
                  className="add-health-update-item"
                  tabIndex={focusIndex === i ? 0 : -1}
                  onClick={() => activate(t.key)}
                  onMouseEnter={() => setFocusIndex(i)}
                >
                  <span
                    className="add-health-update-item-icon"
                    style={{ background: palette.bg, color: palette.fg }}
                    aria-hidden="true"
                  >
                    <Icon size={14} />
                  </span>
                  <span className="add-health-update-item-text">
                    <span className="add-health-update-item-label">{t.label}</span>
                    <span className="add-health-update-item-desc">{t.description}</span>
                  </span>
                </button>
              </li>
            );
          })}
        </ul>
      )}

      {/* Hidden file pickers — the upload items trigger these. */}
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
