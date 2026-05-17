import { useEffect, useRef, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { X } from "lucide-react";

interface ModalProps {
  open: boolean;
  onClose: () => void;
  title: string;
  description?: string;
  /** "sm" ≈ 420px (small forms), "md" ≈ 560px (default), "lg" ≈ 720px (richer forms). */
  size?: "sm" | "md" | "lg";
  /** Disable backdrop-click and Esc dismissal while a form is submitting. */
  dismissable?: boolean;
  children: ReactNode;
  /** Optional footer slot rendered below the body (e.g. cancel + submit). */
  footer?: ReactNode;
}

/**
 * Centered modal dialog rendered via React portal.
 *
 * Behaviour
 * ~~~~~~~~~
 * - Focus is moved to the dialog on open and restored to the trigger on close.
 * - Escape and backdrop click both dismiss, unless ``dismissable`` is false
 *   (the form sets this while a save is in flight so the user can't
 *   accidentally drop a half-completed entry).
 * - Body scroll is locked while open.
 * - All decoration lives in CSS classes (``.modal-*``) so this stays
 *   theme-token-driven and matches the rest of the design system.
 */
export function Modal({
  open,
  onClose,
  title,
  description,
  size = "md",
  dismissable = true,
  children,
  footer,
}: ModalProps) {
  const dialogRef = useRef<HTMLDivElement | null>(null);
  const previousFocusRef = useRef<HTMLElement | null>(null);

  // Lock body scroll while a modal is open.
  useEffect(() => {
    if (!open) return;
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = prev;
    };
  }, [open]);

  // Save + restore focus, and move initial focus into the dialog.
  useEffect(() => {
    if (!open) return;
    previousFocusRef.current = document.activeElement as HTMLElement | null;
    const dialog = dialogRef.current;
    if (dialog) {
      // Defer one tick so child fields are mounted before we look for them.
      const t = window.setTimeout(() => {
        const focusable = dialog.querySelector<HTMLElement>(
          "input, select, textarea, button, [tabindex]:not([tabindex=\"-1\"])",
        );
        (focusable ?? dialog).focus();
      }, 0);
      return () => {
        window.clearTimeout(t);
        previousFocusRef.current?.focus?.();
      };
    }
    return () => {
      previousFocusRef.current?.focus?.();
    };
  }, [open]);

  // Esc to close (when dismissable).
  useEffect(() => {
    if (!open || !dismissable) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.stopPropagation();
        onClose();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, dismissable, onClose]);

  if (!open) return null;

  return createPortal(
    <div
      className="modal-backdrop"
      onMouseDown={(e) => {
        // Only close on backdrop click, never on a click that started inside
        // the dialog and dragged out (e.g. text selection).
        if (dismissable && e.target === e.currentTarget) onClose();
      }}
      role="presentation"
    >
      <div
        ref={dialogRef}
        className={`modal-dialog modal-dialog--${size}`}
        role="dialog"
        aria-modal="true"
        aria-labelledby="modal-title"
        aria-describedby={description ? "modal-description" : undefined}
        tabIndex={-1}
      >
        <header className="modal-header">
          <div style={{ minWidth: 0, flex: 1 }}>
            <h2 id="modal-title" className="modal-title">{title}</h2>
            {description && (
              <p id="modal-description" className="modal-description">{description}</p>
            )}
          </div>
          {dismissable && (
            <button
              type="button"
              className="modal-close"
              onClick={onClose}
              aria-label="Close dialog"
            >
              <X size={16} aria-hidden="true" />
            </button>
          )}
        </header>
        <div className="modal-body">{children}</div>
        {footer && <footer className="modal-footer">{footer}</footer>}
      </div>
    </div>,
    document.body,
  );
}
