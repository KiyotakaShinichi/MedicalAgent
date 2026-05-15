import type { LucideIcon } from "lucide-react";
import { clsx } from "clsx";

interface QuickActionChipProps {
  icon?: LucideIcon;
  label: string;
  onClick: () => void;
  disabled?: boolean;
  primary?: boolean;
  className?: string;
}

/**
 * Pill-shaped chip for chat quick actions or hero CTAs.
 */
export function QuickActionChip({
  icon: Icon,
  label,
  onClick,
  disabled = false,
  primary = false,
  className,
}: QuickActionChipProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={clsx(
        "inline-flex items-center gap-1.5 rounded-full border text-[0.78rem] font-medium transition-all",
        "px-3 py-1.5",
        "disabled:opacity-40 disabled:cursor-not-allowed",
        !primary && "hover:-translate-y-px",
        className,
      )}
      style={
        primary
          ? {
              background: "linear-gradient(135deg, var(--rose), var(--rose-strong))",
              color: "#fff",
              borderColor: "transparent",
              boxShadow: "0 2px 8px rgba(236,72,153,0.20)",
            }
          : {
              background: "var(--surface)",
              color: "var(--text)",
              borderColor: "var(--border)",
            }
      }
    >
      {Icon && <Icon size={13} />}
      {label}
    </button>
  );
}
