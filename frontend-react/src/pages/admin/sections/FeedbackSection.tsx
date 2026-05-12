import { useApi } from "../../../hooks/useApi";
import { getAgentFeedback } from "../../../api/client";
import { Card, CardHeader, SectionTitle } from "../../../components/ui/Card";
import { MetricCard } from "../../../components/ui/MetricCard";
import { LoadingPane, ErrorPane, EmptyPane } from "../../../components/ui/Spinner";

function Stars({ rating }: { rating: number | null }) {
  if (rating == null) return <span style={{ color: "var(--text-faint)" }}>-</span>;
  return (
    <span>
      {Array.from({ length: 5 }, (_, i) => (
        <span key={i} style={{ color: i < Math.round(rating) ? "var(--amber)" : "var(--border)" }}>*</span>
      ))}
    </span>
  );
}

export function FeedbackSection() {
  const { data, status, error } = useApi(getAgentFeedback, []);

  return (
    <div className="flex flex-col gap-4">
      {status === "loading" && <LoadingPane />}
      {status === "error"   && <ErrorPane message={error ?? ""} />}
      {status === "success" && data && (
        <>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
            <MetricCard
              label="Responses"
              value={data.summary.count}
            />
            <MetricCard
              label="Avg rating"
              value={data.summary.average_rating != null ? data.summary.average_rating.toFixed(2) : null}
              unit="/5"
              status={data.summary.average_rating != null && data.summary.average_rating >= 4 ? "green" : "amber"}
            />
            <MetricCard
              label="Thumbs-up rate"
              value={data.summary.thumbs_up_rate != null ? `${(data.summary.thumbs_up_rate * 100).toFixed(0)}%` : null}
              status={data.summary.thumbs_up_rate != null && data.summary.thumbs_up_rate >= 0.8 ? "green" : "amber"}
            />
          </div>

          <Card>
            <CardHeader><SectionTitle>Feedback Log</SectionTitle></CardHeader>
            {data.feedback.length === 0 ? (
              <EmptyPane label="No feedback yet" />
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr style={{ borderBottom: "1px solid var(--border)" }}>
                      {["Patient", "Rating", "Thumbs", "Comment", "Date"].map((h) => (
                        <th key={h} className="text-left py-2 pr-3 font-medium" style={{ color: "var(--text-faint)" }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {data.feedback.map((f, i) => (
                      <tr key={i} style={{ borderBottom: "1px solid var(--border)" }} className="last:border-0">
                        <td className="py-2 pr-3 font-medium" style={{ color: "var(--text)" }}>{f.patient_id}</td>
                        <td className="py-2 pr-3"><Stars rating={f.rating} /></td>
                        <td className="py-2 pr-3" style={{ color: f.thumbs_up ? "var(--green)" : "var(--rose)" }}>
                          {f.thumbs_up == null ? "-" : f.thumbs_up ? "Yes" : "No"}
                        </td>
                        <td className="py-2 pr-3 max-w-xs truncate" style={{ color: "var(--text-dim)" }}>{f.feedback_text || "-"}</td>
                        <td className="py-2 tabular-nums" style={{ color: "var(--text-faint)" }}>{f.created_at?.slice(0, 10)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Card>
        </>
      )}
    </div>
  );
}
