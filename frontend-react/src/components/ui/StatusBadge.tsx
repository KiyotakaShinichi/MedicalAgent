import { clsx } from "clsx";
import type { LucideIcon } from "lucide-react";

type Tone = "success" | "warning" | "danger" | "info" | "neutral" | "accent";

interface StatusBadgeProps {
  tone?: Tone;
  icon?: LucideIcon;
  children: React.ReactNode;
  className?: string;
  size?: "sm" | "md";
}

const toneStyle: Record<Tone, { bg: string; fg: string; border: string }> = {
  success: { bg: "#ecfdf5", fg: "#047857", border: "#a7f3d0" },
  warning: { bg: "#fffbeb", fg: "#92400e", border: "#fde68a" },
  danger:  { bg: "#fef2f2", fg: "#b91c1c", border: "#fecaca" },
  info:    { bg: "#eff6ff", fg: "#1d4ed8", border: "#bfdbfe" },
  neutral: { bg: "#f9fafb", fg: "#4b5563", border: "#e5e7eb" },
  accent:  { bg: "var(--rose-pale)", fg: "var(--rose-deep)", border: "var(--border-strong)" },
};

export function StatusBadge({
  tone = "neutral",
  icon: Icon,
  children,
  className,
  size = "md",
}: StatusBadgeProps) {
  const style = toneStyle[tone];
  return (
    <span
      className={clsx(
        "inline-flex items-center gap-1.5 rounded-full border font-medium",
        size === "sm" ? "text-[0.68rem] px-2 py-0.5" : "text-[0.74rem] px-2.5 py-1",
        className,
      )}
      style={{
        background: style.bg,
        color: style.fg,
        borderColor: style.border,
      }}
    >
      {Icon && <Icon size={size === "sm" ? 11 : 12} />}
      {children}
    </span>
  );
}
