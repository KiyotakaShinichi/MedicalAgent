import { createContext, useCallback, useContext, useEffect, useState } from "react";
import { CheckCircle2, AlertTriangle, Info, X } from "lucide-react";
import type { LucideIcon } from "lucide-react";

type ToastTone = "success" | "warning" | "info";

interface Toast {
  id: number;
  tone: ToastTone;
  title: string;
  description?: string;
  durationMs: number;
}

interface ToastContextValue {
  push: (toast: Omit<Toast, "id" | "durationMs"> & { durationMs?: number }) => void;
  dismiss: (id: number) => void;
}

const ToastContext = createContext<ToastContextValue | null>(null);

const TONE_STYLES: Record<ToastTone, { bg: string; border: string; fg: string; Icon: LucideIcon }> = {
  success: { bg: "#ecfdf5", border: "#a7f3d0", fg: "#047857", Icon: CheckCircle2 },
  warning: { bg: "#fffbeb", border: "#fde68a", fg: "#92400e", Icon: AlertTriangle },
  info:    { bg: "#eff6ff", border: "#bfdbfe", fg: "#1d4ed8", Icon: Info },
};

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const dismiss = useCallback((id: number) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const push = useCallback<ToastContextValue["push"]>((toast) => {
    const id = Date.now() + Math.floor(Math.random() * 10000);
    const next: Toast = {
      id,
      tone: toast.tone,
      title: toast.title,
      description: toast.description,
      durationMs: toast.durationMs ?? 3500,
    };
    setToasts((prev) => [...prev, next]);
    window.setTimeout(() => dismiss(id), next.durationMs);
  }, [dismiss]);

  return (
    <ToastContext.Provider value={{ push, dismiss }}>
      {children}
      <div className="toast-stack" role="status" aria-live="polite">
        {toasts.map((t) => (
          <ToastCard key={t.id} toast={t} onDismiss={() => dismiss(t.id)} />
        ))}
      </div>
    </ToastContext.Provider>
  );
}

function ToastCard({ toast, onDismiss }: { toast: Toast; onDismiss: () => void }) {
  const style = TONE_STYLES[toast.tone];
  const Icon = style.Icon;
  const [entering, setEntering] = useState(true);
  useEffect(() => {
    const t = window.setTimeout(() => setEntering(false), 16);
    return () => window.clearTimeout(t);
  }, []);
  return (
    <div
      className="toast-card"
      data-entering={entering ? "true" : "false"}
      style={{ background: style.bg, borderColor: style.border, color: style.fg }}
    >
      <Icon size={16} aria-hidden="true" style={{ flexShrink: 0, marginTop: 1 }} />
      <div style={{ flex: 1, minWidth: 0 }}>
        <p className="toast-title">{toast.title}</p>
        {toast.description && <p className="toast-description">{toast.description}</p>}
      </div>
      <button
        type="button"
        onClick={onDismiss}
        className="toast-dismiss"
        aria-label="Dismiss notification"
      >
        <X size={13} />
      </button>
    </div>
  );
}

// eslint-disable-next-line react-refresh/only-export-components
export function useToast(): ToastContextValue {
  const ctx = useContext(ToastContext);
  if (!ctx) {
    // Outside a provider, return a no-op so components don't crash in isolated tests.
    return {
      push: () => undefined,
      dismiss: () => undefined,
    };
  }
  return ctx;
}
