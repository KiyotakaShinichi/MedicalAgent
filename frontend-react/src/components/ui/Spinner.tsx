import { AlertCircle, Inbox } from "lucide-react";

export function Spinner({ size = 20 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      className="animate-spin"
      style={{ color: "var(--rose)" }}
      aria-hidden="true"
    >
      <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" strokeOpacity="0.2" />
      <path
        d="M12 2a10 10 0 0 1 10 10"
        stroke="currentColor"
        strokeWidth="3"
        strokeLinecap="round"
      />
    </svg>
  );
}

export function LoadingPane({ label = "Loading..." }: { label?: string }) {
  return (
    <div className="state-pane">
      <Spinner size={30} />
      <span>{label}</span>
    </div>
  );
}

export function ErrorPane({ message }: { message: string }) {
  return (
    <div className="state-pane is-error">
      <AlertCircle size={18} aria-hidden="true" />
      <span>{message}</span>
    </div>
  );
}

export function EmptyPane({ label = "No data available" }: { label?: string }) {
  return (
    <div className="state-pane is-empty">
      <Inbox size={20} aria-hidden="true" />
      <span>{label}</span>
    </div>
  );
}
