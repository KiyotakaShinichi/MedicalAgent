import { clsx } from "clsx";
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
}

/**
 * Standard section card used across patient/clinician/admin surfaces.
 *
 *   <SectionCard title="Lab values" icon={FlaskConical} meta="12 samples">
 *     ...content...
 *   </SectionCard>
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
}: SectionCardProps) {
  return (
    <section className={clsx("app-card", padding && "p-4", className)}>
      <header className="app-card-header">
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
        </div>
      </header>
      <div className={bodyClassName}>{children}</div>
      {footer && (
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
