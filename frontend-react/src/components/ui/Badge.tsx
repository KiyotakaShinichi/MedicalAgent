import { clsx } from "clsx";
import { type BadgeVariant } from "./badgeUtils";

const variantStyle: Record<BadgeVariant, { bg: string; fg: string; border: string }> = {
  green:  { bg: "var(--green-badge-bg)",  fg: "var(--green-badge-fg)",  border: "#a7f3cf" },
  amber:  { bg: "var(--amber-badge-bg)",  fg: "var(--amber-badge-fg)",  border: "#fcd9a0" },
  red:    { bg: "var(--red-badge-bg)",    fg: "var(--red-badge-fg)",    border: "#fca5a5" },
  blue:   { bg: "var(--blue-badge-bg)",   fg: "var(--blue-badge-fg)",   border: "#bfdbfe" },
  purple: { bg: "var(--purple-badge-bg)", fg: "var(--purple-badge-fg)", border: "#ddd6fe" },
  cyan:   { bg: "var(--cyan-badge-bg)",   fg: "var(--cyan-badge-fg)",   border: "#a5f3fc" },
  muted:  { bg: "var(--muted-badge-bg)",  fg: "var(--muted-badge-fg)",  border: "var(--border)" },
};

interface BadgeProps {
  variant?: BadgeVariant;
  children: React.ReactNode;
  className?: string;
}

export function Badge({ variant = "muted", children, className }: BadgeProps) {
  const style = variantStyle[variant];
  return (
    <span
      className={clsx(
        "inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-semibold border",
        className,
      )}
      style={{
        background: style.bg,
        color: style.fg,
        borderColor: style.border,
      }}
    >
      {children}
    </span>
  );
}
