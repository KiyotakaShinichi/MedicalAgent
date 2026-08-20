import { Pill } from "lucide-react";
import { SectionCard } from "../../components/ui/SectionCard";
import { EmptyPane } from "../../components/ui/Spinner";
import type { MedicationLog } from "../../types/api";

/** Rows rendered before the panel defers to the full patient record. */
const MAX_ROWS = 12;

const COLUMNS = [
  { key: "date", label: "Date", width: 110 },
  { key: "medication", label: "Medication", width: undefined },
  { key: "dose", label: "Dose", width: 90 },
  { key: "frequency", label: "Frequency", width: undefined },
] as const;

const headerStyle = (width?: number) => ({
  color: "var(--text-faint)",
  fontSize: "0.72rem",
  textTransform: "uppercase" as const,
  letterSpacing: "0.06em",
  borderBottom: "1px solid var(--border)",
  width,
});

const cellBorder = { borderBottom: "1px solid var(--border-soft)" };

/**
 * Patient-facing medication log, newest first.
 *
 * Read-only by design: patients log medications through the chat tool tray or
 * their care team, so this panel never offers inline editing. The empty state
 * says who can add entries rather than implying the patient should.
 */
export function MedLogPanel({ meds }: { meds: MedicationLog[] }) {
  // Copy before sorting — mutating the prop array would reorder the caller's
  // report object in place.
  const sorted = [...meds].sort((a, b) => (b.date ?? "").localeCompare(a.date ?? ""));

  return (
    <SectionCard
      title="Medication log"
      icon={Pill}
      meta={sorted.length > 0 ? `${sorted.length} entries` : undefined}
    >
      {sorted.length === 0 ? (
        <EmptyPane label="No medications recorded — your care team can add these from the clinician portal." />
      ) : (
        <div className="overflow-x-auto">
          <table
            className="w-full text-[0.86rem]"
            style={{
              borderCollapse: "separate",
              borderSpacing: 0,
              // Forces horizontal scroll on narrow screens rather than
              // crushing four columns into an unreadable width.
              minWidth: 520,
            }}
          >
            <caption className="sr-only">Recorded medications, most recent first</caption>
            <thead>
              <tr>
                {COLUMNS.map((column) => (
                  <th
                    key={column.key}
                    scope="col"
                    className="text-left font-semibold py-2 px-3"
                    style={headerStyle(column.width)}
                  >
                    {column.label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {sorted.slice(0, MAX_ROWS).map((med, index) => (
                <tr key={`${med.date}-${med.medication}-${index}`}>
                  <th
                    scope="row"
                    className="py-2.5 px-3 tabular-nums text-left font-normal"
                    style={{ color: "var(--text-dim)", ...cellBorder }}
                  >
                    {med.date?.slice(0, 10)}
                  </th>
                  <td className="py-2.5 px-3 font-semibold" style={{ color: "var(--text-strong)", ...cellBorder }}>
                    {med.medication}
                  </td>
                  <td className="py-2.5 px-3 tabular-nums" style={{ color: "var(--text)", ...cellBorder }}>
                    {med.dose}
                  </td>
                  <td className="py-2.5 px-3" style={{ color: "var(--text-dim)", ...cellBorder }}>
                    {med.frequency}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {sorted.length > MAX_ROWS && (
            <p className="text-[0.74rem] mt-2 px-3" style={{ color: "var(--text-faint)" }}>
              + {sorted.length - MAX_ROWS} more in patient record
            </p>
          )}
        </div>
      )}
    </SectionCard>
  );
}
