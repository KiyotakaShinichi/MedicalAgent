import { clsx } from "clsx";
import { Spinner } from "./Spinner";

type Variant = "primary" | "secondary" | "ghost" | "danger";
type Size = "sm" | "md" | "lg";

const variantStyles: Record<Variant, React.CSSProperties> = {
  primary:   { background: "var(--rose)",     color: "#fff",           border: "none" },
  secondary: { background: "var(--surface2)", color: "var(--text)",    border: "1px solid var(--border)" },
  ghost:     { background: "transparent",     color: "var(--text-dim)", border: "1px solid transparent" },
  danger:    { background: "rgba(244,63,94,0.12)", color: "var(--rose)", border: "1px solid rgba(244,63,94,0.3)" },
};

const sizeStyles: Record<Size, string> = {
  sm: "px-2.5 py-1 text-xs gap-1",
  md: "px-3.5 py-1.5 text-sm gap-1.5",
  lg: "px-5 py-2.5 text-sm gap-2",
};

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
  loading?: boolean;
  icon?: React.ReactNode;
}

export function Button({
  variant = "secondary",
  size = "md",
  loading = false,
  icon,
  children,
  className,
  disabled,
  style,
  ...rest
}: ButtonProps) {
  return (
    <button
      disabled={disabled || loading}
      className={clsx(
        "inline-flex items-center justify-center rounded-md font-medium transition-opacity",
        "hover:opacity-80 active:opacity-60 disabled:opacity-40 disabled:cursor-not-allowed",
        sizeStyles[size],
        className
      )}
      style={{ ...variantStyles[variant], ...style }}
      {...rest}
    >
      {loading ? <Spinner size={14} /> : icon}
      {children}
    </button>
  );
}
