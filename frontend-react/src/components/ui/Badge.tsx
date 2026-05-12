import { clsx } from "clsx";
import { badgeColorToken, type BadgeVariant } from "./badgeUtils";

const variantClasses: Record<BadgeVariant, string> = {
  green:  "bg-green-950 text-green-400 border-green-800",
  amber:  "bg-amber-950 text-amber-400 border-amber-800",
  red:    "bg-rose-950 text-rose-400 border-rose-800",
  blue:   "bg-blue-950 text-blue-400 border-blue-800",
  purple: "bg-purple-950 text-purple-400 border-purple-800",
  cyan:   "bg-cyan-950 text-cyan-400 border-cyan-800",
  muted:  "bg-slate-800 text-slate-400 border-slate-700",
};

interface BadgeProps {
  variant?: BadgeVariant;
  children: React.ReactNode;
  className?: string;
}

export function Badge({ variant = "muted", children, className }: BadgeProps) {
  return (
    <span
      className={clsx(
        "inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium border",
        variantClasses[variant],
        className
      )}
      style={{
        background: `var(--${badgeColorToken(variant)}-badge-bg, rgba(0,0,0,0.3))`,
      }}
    >
      {children}
    </span>
  );
}
