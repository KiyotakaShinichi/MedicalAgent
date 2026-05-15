import { ShieldCheck, AlertTriangle, Info } from "lucide-react";
import { clsx } from "clsx";

type Tone = "info" | "warning" | "danger" | "success";

interface SafetyBannerProps {
  tone?: Tone;
  title?: string;
  children: React.ReactNode;
  className?: string;
  compact?: boolean;
}

const tonePresets: Record<Tone, { bg: string; border: string; fg: string; icon: typeof Info }> = {
  info:    { bg: "var(--rose-pale)", border: "var(--border-strong)", fg: "var(--rose-deep)", icon: ShieldCheck },
  warning: { bg: "#fffbeb",          border: "#fde68a",              fg: "#92400e",          icon: AlertTriangle },
  danger:  { bg: "#fef2f2",          border: "#fecaca",              fg: "#b91c1c",          icon: AlertTriangle },
  success: { bg: "#ecfdf5",          border: "#a7f3d0",              fg: "#047857",          icon: ShieldCheck },
};

/**
 * Soft inline banner used for safety / disclaimer / context copy.
 * Default tone reuses the brand pink as an "informational" accent.
 */
export function SafetyBanner({
  tone = "info",
  title,
  children,
  className,
  compact = false,
}: SafetyBannerProps) {
  const preset = tonePresets[tone];
  const Icon = preset.icon;
  return (
    <div
      className={clsx(
        "flex gap-2.5 items-start rounded-lg border",
        compact ? "px-3 py-2" : "px-4 py-3",
        className,
      )}
      style={{
        background: preset.bg,
        borderColor: preset.border,
        color: preset.fg,
      }}
      role="note"
    >
      <Icon size={compact ? 14 : 16} className="flex-shrink-0 mt-0.5" aria-hidden="true" />
      <div className="flex-1 min-w-0">
        {title && <strong className={clsx("block", compact ? "text-xs" : "text-sm", "font-semibold")}>{title}</strong>}
        <p className={clsx(compact ? "text-xs" : "text-[0.82rem]", "leading-relaxed")}>{children}</p>
      </div>
    </div>
  );
}
