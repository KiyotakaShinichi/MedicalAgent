import { Component, type ReactNode } from "react";
import { AlertCircle, RefreshCw } from "lucide-react";

interface Props {
  children: ReactNode;
  /** Optional label used in the fallback message, e.g. "Safety & Eval". */
  surface?: string;
}

interface State {
  error: Error | null;
}

/**
 * A render-error boundary that turns a thrown exception into a calm,
 * recoverable surface instead of a blank screen. Wraps any non-trivial
 * sub-tree (e.g. the admin dashboard tab content) so a single bad section
 * does not bring down the whole app.
 *
 * React error boundaries must be class components (hooks cannot
 * implement componentDidCatch).
 */
export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    // Keep console reporting; in a portfolio demo this is enough. In
    // production this is where Sentry / Datadog instrumentation would go.
    // eslint-disable-next-line no-console
    console.error("[ErrorBoundary]", this.props.surface ?? "unknown", error, info);
  }

  handleReset = () => {
    this.setState({ error: null });
  };

  render() {
    if (!this.state.error) {
      return this.props.children;
    }
    const surface = this.props.surface ?? "this section";
    return (
      <div
        role="alert"
        className="m-4 p-5 rounded-xl border flex items-start gap-3"
        style={{
          background: "rgba(244, 63, 94, 0.06)",
          borderColor: "rgba(244, 63, 94, 0.28)",
        }}
      >
        <AlertCircle
          size={18}
          aria-hidden="true"
          style={{ color: "var(--rose)", marginTop: 2, flexShrink: 0 }}
        />
        <div className="flex-1 min-w-0">
          <p className="text-sm font-semibold" style={{ color: "var(--text)" }}>
            Something went wrong rendering {surface}.
          </p>
          <p className="text-xs mt-1" style={{ color: "var(--text-dim)" }}>
            {this.state.error.message || "An unexpected error occurred."}
          </p>
          <button
            type="button"
            onClick={this.handleReset}
            className="mt-3 inline-flex items-center gap-1.5 text-xs font-medium px-3 py-1.5 rounded-md"
            style={{
              background: "var(--rose)",
              color: "#fff",
              border: "1px solid rgba(244, 63, 94, 0.5)",
            }}
          >
            <RefreshCw size={12} aria-hidden="true" /> Try again
          </button>
        </div>
      </div>
    );
  }
}
