import { useEffect, useRef, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { X } from "lucide-react";

interface DrawerProps {
  open: boolean;
  onClose: () => void;
  title: string;
  description?: string;
  /** "sm" ≈ 380px (compact), "md" ≈ 480px (default), "lg" ≈ 640px (richer forms). */
  size?: "sm" | "md" | "lg";
  /** Disable backdrop-click and Esc dismissal while a form is submitting. */
  dismissable?: boolean;
  children: ReactNode;
  footer?: ReactNode;
}

/**
 * Right-side drawer rendered via React portal.
 *
 * Same accessibility contract as ``Modal`` (focus management, body scroll
 * lock, Esc handling, restored focus on close).  The drawer pattern is the
 * preferred surface for **multi-section forms** (e.g. CBC, imaging) where a
 * centered modal would feel too tall on a laptop screen.
 */
export function Drawer({
  open,
  onClose,
  title,
  description,
  size = "md",
  dismissable = true,
  children,
  footer,
}: DrawerProps) {
  const dialogRef = useRef<HTMLDivElement | null>(null);
  const previousFocusRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (!open) return;
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = prev;
    };
  }, [open]);

  useEffect(() => {
    if (!open) return;
    previousFocusRef.current = document.activeElement as HTMLElement | null;
    const dialog = dialogRef.current;
    if (dialog) {
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
      className="drawer-backdrop"
      onMouseDown={(e) => {
        if (dismissable && e.target === e.currentTarget) onClose();
      }}
      role="presentation"
    >
      <div
        ref={dialogRef}
        className={`drawer-panel drawer-panel--${size}`}
        role="dialog"
        aria-modal="true"
        aria-labelledby="drawer-title"
        aria-describedby={description ? "drawer-description" : undefined}
        tabIndex={-1}
      >
        <header className="drawer-header">
          <div style={{ minWidth: 0, flex: 1 }}>
            <h2 id="drawer-title" className="drawer-title">{title}</h2>
            {description && (
              <p id="drawer-description" className="drawer-description">{description}</p>
            )}
          </div>
          {dismissable && (
            <button
              type="button"
              className="drawer-close"
              onClick={onClose}
              aria-label="Close drawer"
            >
              <X size={16} aria-hidden="true" />
            </button>
          )}
        </header>
        <div className="drawer-body">{children}</div>
        {footer && <footer className="drawer-footer">{footer}</footer>}
      </div>
    </div>,
    document.body,
  );
}
