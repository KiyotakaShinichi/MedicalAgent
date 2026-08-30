import { Badge } from "../../../../components/ui/Badge";
import type { PublicDataManifest } from "../../../../types/api";

export function PublicDataManifestPanel({ data }: { data: PublicDataManifest }) {
  const visibleNeeds = data.feature_feasibility.slice(0, 6);
  const sourceNames = new Map(data.sources.map((source) => [source.id, source.name]));

  return (
    <div className="flex flex-col gap-3">
      <div className="rounded-md border p-3" style={{ background: "var(--surface2)", borderColor: "var(--border)" }}>
        <p className="text-xs mb-2" style={{ color: "var(--text-dim)" }}>{data.central_data_reality}</p>
        <p className="text-xs" style={{ color: "var(--text-faint)" }}>{data.recommended_strategy}</p>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr style={{ borderBottom: "1px solid var(--border)" }}>
              {["Need", "Status", "Sources", "Project action"].map((heading) => (
                <th key={heading} className="text-left py-2 pr-4 font-medium" style={{ color: "var(--text-faint)" }}>{heading}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {visibleNeeds.map((need) => (
              <tr key={need.need} style={{ borderBottom: "1px solid var(--border)" }} className="last:border-0">
                <td className="py-2 pr-4 font-medium" style={{ color: "var(--text)" }}>{need.need}</td>
                <td className="py-2 pr-4">
                  <Badge variant={
                    need.status === "covered_by_public_data" ? "green" :
                    need.status === "partially_covered" ? "amber" :
                    need.status === "future_extension" ? "blue" :
                    "red"
                  }>
                    {need.status.replace(/_/g, " ")}
                  </Badge>
                </td>
                <td className="py-2 pr-4" style={{ color: "var(--text-dim)" }}>
                  {need.sources.length ? need.sources.map((id) => sourceNames.get(id) || id).join(", ") : "No direct public source"}
                </td>
                <td className="py-2 pr-4" style={{ color: "var(--text-dim)" }}>{need.project_action}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="grid sm:grid-cols-2 gap-2">
        {data.sources.slice(0, 6).map((source) => (
          <div key={source.id} className="rounded-md border p-2" style={{ background: "var(--surface)", borderColor: "var(--border)" }}>
            <div className="flex items-center justify-between gap-2 mb-1">
              <p className="text-xs font-semibold" style={{ color: "var(--text)" }}>{source.name}</p>
              <span className="text-[10px]" style={{ color: "var(--text-faint)" }}>{source.provider}</span>
            </div>
            <p className="text-xs" style={{ color: "var(--text-dim)" }}>{source.use_in_project[0]}</p>
          </div>
        ))}
      </div>

      <p className="text-xs italic" style={{ color: "var(--text-faint)" }}>
        Manifest {data.manifest_hash}. {data.claim_boundary}
      </p>
    </div>
  );
}
