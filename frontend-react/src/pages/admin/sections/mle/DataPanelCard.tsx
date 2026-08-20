import type { ReactNode } from "react";
import { RefreshCw } from "lucide-react";
import { Card, CardHeader, SectionTitle } from "../../../../components/ui/Card";
import { Button } from "../../../../components/ui/Button";
import { LoadingPane, ErrorPane, EmptyPane } from "../../../../components/ui/Spinner";

interface DataPanelCardProps {
  title: string;
  /** Small provenance pill, e.g. "Synthetic data" or "Frozen synthetic split". */
  tag?: { label: string; background: string; color: string };
  /** Optional regenerate action rendered in the header. */
  action?: { label: string; onClick: () => void; running: boolean };
  loading: boolean;
  error: string | null;
  /** True when there is no artifact to show even though loading finished. */
  empty: boolean;
  emptyLabel: string;
  errorLabel: string;
  children: ReactNode;
}

/**
 * Card wrapper for one artifact panel, owning the loading / error / empty /
 * ready switch.
 *
 * Every panel in `MleSection` previously repeated a four-branch ternary chain
 * inline. Beyond the duplication, the branches had drifted: some panels
 * checked `status === "error"` and some did not, so a failed fetch on those
 * silently fell through to the empty state and read as "no artifact exists"
 * rather than "the request failed". Centralising the switch makes that
 * impossible.
 *
 * `error` takes precedence over `empty` for exactly that reason.
 */
export function DataPanelCard({
  title,
  tag,
  action,
  loading,
  error,
  empty,
  emptyLabel,
  errorLabel,
  children,
}: DataPanelCardProps) {
  return (
    <Card>
      <CardHeader>
        <SectionTitle>{title}</SectionTitle>
        {tag && (
          <span
            className="text-xs px-2 py-0.5 rounded"
            style={{ background: tag.background, color: tag.color }}
          >
            {tag.label}
          </span>
        )}
        {action && (
          <Button
            variant="secondary"
            size="sm"
            loading={action.running}
            icon={<RefreshCw size={12} aria-hidden="true" />}
            onClick={action.onClick}
            aria-label={`${action.label} — ${title}`}
          >
            {action.label}
          </Button>
        )}
      </CardHeader>

      {loading ? (
        <LoadingPane />
      ) : error ? (
        <ErrorPane message={`${errorLabel}: ${error}`} />
      ) : empty ? (
        <EmptyPane label={emptyLabel} />
      ) : (
        children
      )}
    </Card>
  );
}
