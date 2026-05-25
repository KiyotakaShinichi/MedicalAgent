import { clsx } from "clsx";
import { useState } from "react";
import { ChevronDown } from "lucide-react";
import type { LucideIcon } from "lucide-react";

interface SectionCardProps {
  title: string;
  icon?: LucideIcon;
  meta?: React.ReactNode;
  action?: React.ReactNode;
  footer?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
  bodyClassName?: string;
  padding?: boolean;
  /** When true, the card body collapses behind a chevron toggle. */
  collapsible?: boolean;
  /** Initial open state when ``collapsible`` is true. Defaults to true. */
  defaultOpen?: boolean;
  /** Stable id used to remember the open/closed state across reloads. */
  collapseId?: string;
}

/**
 * Standard section card used across patient/clinician/admin surfaces.
 *
 *   <SectionCard title="Lab values" icon={FlaskConical} meta="12 samples">
 *     ...content...
 *   </SectionCard>
 *
 * Pass ``collapsible`` to show a chevron that hides the body — useful for
 * long lists (symptoms, timeline) so the dashboard isn't a scroll marathon.
 */
export function SectionCard({
  title,
  icon: Icon,
  meta,
  action,
  footer,
  children,
  className,
  bodyClassName,
  padding = true,
  collapsible = false,
  defaultOpen = true,
  collapseId,
}: SectionCardProps) {
  const storageKey = collapsible && collapseId ? `sc-open:${collapseId}` : null;
  const [open, setOpen] = useState<boolean>(() => {
    if (!collapsible) return true;
    if (storageKey && typeof window !== "undefined") {
      const stored = window.localStorage.getItem(storageKey);
      if (stored === "1") return true;
      if (stored === "0") return false;
    }
    return defaultOpen;
  });

  const toggle = () => {
    setOpen((prev) => {
      const next = !prev;
      if (storageKey && typeof window !== "undefined") {
        window.localStorage.setItem(storageKey, next ? "1" : "0");
      }
      return next;
    });
  };

  const headerInteractive = collapsible;
  const bodyId = collapseId ? `${collapseId}-body` : undefined;

  return (
    <section className={clsx("app-card", padding && "p-4", className)}>
      <header
        className={clsx("app-card-header", headerInteractive && "app-card-header--clickable")}
        onClick={headerInteractive ? toggle : undefined}
        role={headerInteractive ? "button" : undefined}
        tabIndex={headerInteractive ? 0 : undefined}
        aria-expanded={collapsible ? open : undefined}
        aria-controls={collapsible ? bodyId : undefined}
        onKeyDown={
          headerInteractive
            ? (e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  toggle();
                }
              }
            : undefined
        }
        style={collapsible && !open ? { paddingBottom: 0, marginBottom: 0, borderBottom: "none" } : undefined}
      >
        <h2 className="app-section-title">
          {Icon && (
            <span className="section-tile" aria-hidden="true">
              <Icon size={14} />
            </span>
          )}
          <span>{title}</span>
        </h2>
        <div className="flex items-center gap-2">
          {meta && <span className="section-meta">{meta}</span>}
          {action}
          {collapsible && (
            <span
              className="section-collapse-chevron"
              aria-hidden="true"
              style={{
                display: "inline-flex",
                alignItems: "center",
                justifyContent: "center",
                width: 24,
                height: 24,
                borderRadius: 6,
                color: "var(--text-faint)",
                transition: "transform 160ms ease",
                transform: open ? "rotate(0deg)" : "rotate(-90deg)",
              }}
            >
              <ChevronDown size={16} />
            </span>
          )}
        </div>
      </header>
      {(!collapsible || open) && (
        <div id={bodyId} className={bodyClassName}>{children}</div>
      )}
      {footer && (!collapsible || open) && (
        <footer
          className="mt-3 pt-3 text-xs"
          style={{
            borderTop: "1px solid var(--border)",
            color: "var(--text-faint)",
          }}
        >
          {footer}
        </footer>
      )}
    </section>
  );
}
