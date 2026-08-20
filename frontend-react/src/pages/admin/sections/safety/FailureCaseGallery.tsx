import { AlertTriangle } from "lucide-react";
import { Badge } from "../../../../components/ui/Badge";
import { EmptyPane } from "../../../../components/ui/Spinner";
import type { SafetyCenter } from "../../../../types/api";

type FailureGallery = SafetyCenter["failure_case_gallery"];

/** The five narrative fields every catalogued failure case carries. */
const FIELDS = [
  { key: "what_happened", label: "What happened", tone: "var(--text)" },
  { key: "why_risky", label: "Why risky", tone: "var(--text-dim)" },
  { key: "system_response", label: "System response", tone: "var(--text-dim)" },
  { key: "mitigation", label: "Mitigation", tone: "var(--text-dim)" },
  { key: "unresolved", label: "Unresolved", tone: "var(--rose)" },
] as const;

/**
 * Catalogue of known failure modes.
 *
 * This is a deliberate anti-marketing surface: it exists to keep unresolved
 * weaknesses visible next to the pass rates, so the "Unresolved" line is
 * always rendered even when empty-ish rather than being conditionally hidden.
 */
export function FailureCaseGallery({ gallery }: { gallery: FailureGallery | undefined }) {
  const cases = gallery?.cases ?? [];
  if (gallery?.status === "not_generated" || cases.length === 0) {
    return <EmptyPane label="No failure cases recorded yet." />;
  }

  return (
    <ul className="grid grid-cols-1 lg:grid-cols-2 gap-3 list-none p-0 m-0">
      {cases.map((c) => (
        <li
          key={c.id}
          className="p-3 rounded-md border"
          style={{ background: "var(--surface2)", borderColor: "var(--border)" }}
        >
          <div className="flex items-start gap-2">
            <AlertTriangle
              size={14}
              aria-hidden="true"
              style={{ color: "var(--amber)", flexShrink: 0, marginTop: 2 }}
            />
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 mb-1">
                <p className="text-xs font-semibold m-0" style={{ color: "var(--text)" }}>
                  {c.id}
                </p>
                <Badge variant="amber">{c.category.replace(/_/g, " ")}</Badge>
              </div>
              {FIELDS.map(({ key, label, tone }) => (
                <p key={key} className="text-xs mb-1 last:mb-0" style={{ color: tone }}>
                  <strong>{label}:</strong> {c[key]}
                </p>
              ))}
            </div>
          </div>
        </li>
      ))}
    </ul>
  );
}
