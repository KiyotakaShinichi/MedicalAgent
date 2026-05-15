import { AlertCircle, Inbox, RotateCw } from "lucide-react";
import { API_BASE } from "../../api/client";

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

const BACKEND_BASE = (() => {
  try {
    const u = new URL(API_BASE);
    return `${u.hostname}:${u.port || (u.protocol === "https:" ? "443" : "80")}`;
  } catch {
    return API_BASE;
  }
})();

function isNetworkError(message: string): boolean {
  const lower = message.toLowerCase();
  return (
    lower.includes("failed to fetch") ||
    lower.includes("networkerror") ||
    lower.includes("load failed") ||
    lower.includes("err_connection")
  );
}

interface ErrorPaneProps {
  message: string;
  /** Optional retry callback. When provided, renders a "Try again" button. */
  onRetry?: () => void;
}

export function ErrorPane({ message, onRetry }: ErrorPaneProps) {
  const network = isNetworkError(message);
  return (
    <div className="state-pane is-error">
      <AlertCircle size={16} aria-hidden="true" style={{ flexShrink: 0, marginTop: 1 }} />
      <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column", gap: 4 }}>
        <span style={{ fontWeight: 600 }}>
          {network ? `Backend API unavailable at ${BACKEND_BASE}` : "Request failed"}
        </span>
        <span style={{ color: "#9f1239", fontWeight: 400, fontSize: "0.78rem", lineHeight: 1.45 }}>
          {network
            ? "Check that the FastAPI server is running and reachable, then retry."
            : message}
        </span>
        {onRetry && (
          <button
            type="button"
            onClick={onRetry}
            className="inline-flex items-center gap-1.5 rounded-md border self-start"
            style={{
              marginTop: 4,
              padding: "4px 10px",
              fontSize: "0.78rem",
              fontWeight: 600,
              background: "#ffffff",
              borderColor: "#fecaca",
              color: "#b91c1c",
              cursor: "pointer",
            }}
          >
            <RotateCw size={12} />
            Try again
          </button>
        )}
      </div>
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
