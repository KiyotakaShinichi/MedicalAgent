import { useState } from "react";
import { RefreshCw, Zap, Compass } from "lucide-react";
import { Badge } from "../../../components/ui/Badge";
import { statusVariant } from "../../../components/ui/badgeUtils";
import { Button } from "../../../components/ui/Button";
import { Card, CardHeader, SectionTitle } from "../../../components/ui/Card";
import { ErrorPane, LoadingPane } from "../../../components/ui/Spinner";
import { useApi } from "../../../hooks/useApi";
import {
  getSystemHealth,
  runSystemHealth,
  getAdminFastMode,
  setAdminFastMode,
  probeAdminIntent,
  type IntentProbeResponse,
} from "../../../api/client";


export function SystemHealthSection() {
  const { data, status, error, refetch } = useApi(getSystemHealth, []);

  async function rebuild() {
    await runSystemHealth();
    await refetch();
  }

  if (status === "loading") return <LoadingPane label="Checking system health..." />;
  if (status === "error") return <ErrorPane message={error ?? "Could not load system health"} onRetry={() => void refetch()} />;
  if (!data) return null;

  return (
    <div className="flex flex-col gap-4">
      <Card>
        <CardHeader>
          <SectionTitle>System Health</SectionTitle>
          <Button variant="secondary" size="sm" icon={<RefreshCw size={12} />} onClick={() => void rebuild()}>
            Re-check
          </Button>
        </CardHeader>
        <div className="grid sm:grid-cols-4 gap-3">
          <HealthTile label="Overall" value={data.status} />
          <HealthTile label="Database" value={data.backend.database.status} />
          <HealthTile label="Frontend build" value={data.frontend.production_build_present ? "present" : "missing"} />
          <HealthTile label="Groq" value={data.environment.groq_configured ? "configured" : "not configured"} />
        </div>
        <p className="text-xs mt-3" style={{ color: "var(--text-dim)" }}>{data.claim_boundary}</p>
      </Card>

      <FastModePanel />

      <IntentProbePanel />

      <Card>
        <CardHeader><SectionTitle>Issues & Next Actions</SectionTitle></CardHeader>
        {data.issues.length === 0 ? (
          <p className="text-sm" style={{ color: "var(--green)" }}>No blocking health issues detected.</p>
        ) : (
          <div className="flex flex-col gap-2">
            {data.issues.map((issue, index) => (
              <div key={`${issue.area}-${index}`} className="flex items-start justify-between gap-3 p-2 rounded-md" style={{ background: "var(--surface2)" }}>
                <span className="text-sm">{issue.message}</span>
                <Badge variant={issue.severity === "critical" ? "red" : issue.severity === "warning" ? "amber" : "muted"}>{issue.severity}</Badge>
              </div>
            ))}
          </div>
        )}
        <ul className="mt-3 text-xs" style={{ color: "var(--text-dim)", paddingLeft: 18 }}>
          {data.next_actions.map((action) => <li key={action}>{action}</li>)}
        </ul>
      </Card>

      <div className="grid lg:grid-cols-2 gap-4">
        <Card>
          <CardHeader><SectionTitle>Dependencies</SectionTitle></CardHeader>
          <div className="flex flex-col gap-2">
            {data.dependencies.map((dep) => (
              <div key={dep.package} className="flex items-center justify-between gap-3 text-sm">
                <div>
                  <strong>{dep.package}</strong>
                  <div className="text-xs" style={{ color: "var(--text-faint)" }}>{dep.purpose}</div>
                </div>
                <Badge variant={dep.available ? "green" : "amber"}>{dep.available ? "available" : "missing"}</Badge>
              </div>
            ))}
          </div>
        </Card>

        <Card>
          <CardHeader><SectionTitle>Artifacts</SectionTitle></CardHeader>
          <div className="flex flex-col gap-2">
            {data.artifacts.map((artifact) => (
              <div key={artifact.name} className="flex items-center justify-between gap-3 text-sm">
                <div style={{ minWidth: 0 }}>
                  <strong>{artifact.name.replace(/_/g, " ")}</strong>
                  <div className="text-xs truncate" style={{ color: "var(--text-faint)" }}>{artifact.path}</div>
                </div>
                <div className="flex items-center gap-2">
                  <Badge variant={artifact.exists ? statusVariant(artifact.status) : "amber"}>{artifact.status}</Badge>
                  <Badge variant={artifact.freshness === "fresh" ? "green" : artifact.freshness === "stale" ? "amber" : "muted"}>
                    {artifact.freshness}
                  </Badge>
                </div>
              </div>
            ))}
          </div>
        </Card>
      </div>
    </div>
  );
}

function HealthTile({ label, value }: { label: string; value: string }) {
  return (
    <div className="p-3 rounded-lg border" style={{ background: "var(--surface2)", borderColor: "var(--border)" }}>
      <div className="text-xs mb-1" style={{ color: "var(--text-faint)" }}>{label}</div>
      <Badge variant={statusVariant(value)}>{value.replace(/_/g, " ")}</Badge>
    </div>
  );
}


function FastModePanel() {
  const { data, status, error, refetch } = useApi(getAdminFastMode, []);
  const [busy, setBusy] = useState(false);

  async function apply(next: boolean | null) {
    setBusy(true);
    try {
      await setAdminFastMode(next);
      await refetch();
    } finally {
      setBusy(false);
    }
  }

  if (status === "loading") return <LoadingPane label="Reading FAST_MODE state..." />;
  if (status === "error") return <ErrorPane message={error ?? "Could not load FAST_MODE state"} onRetry={() => void refetch()} />;
  if (!data) return null;

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <Zap size={14} />
          <SectionTitle>LLM FAST_MODE (emergency degradation)</SectionTitle>
          <Badge variant={data.enabled ? "amber" : "green"}>
            {data.enabled ? "ON" : "OFF"}
          </Badge>
          <span className="text-[10px] uppercase" style={{ color: "var(--text-faint)" }}>
            source: {data.source.replace(/_/g, " ")}
          </span>
        </div>
      </CardHeader>
      <p className="text-xs mb-3" style={{ color: "var(--text-dim)" }}>
        When ON, every LLM adjudication on the hot chat path short-circuits to
        "unavailable". The deterministic safety stack (security patterns, route
        boundaries, post-gen validator, claim boundary checker) still enforces
        every refusal contract — what you lose is the LLM second opinion on
        open-ended branches. Use this <strong>only</strong> when the Groq cloud
        provider is degraded / rate-limiting, or for deterministic-only test
        passes.
      </p>
      <div className="flex flex-wrap items-center gap-2">
        <Button
          variant="primary"
          size="sm"
          disabled={busy || data.enabled}
          onClick={() => void apply(true)}
        >
          Force ON
        </Button>
        <Button
          variant="secondary"
          size="sm"
          disabled={busy || (!data.enabled && data.runtime_override === false)}
          onClick={() => void apply(false)}
        >
          Force OFF
        </Button>
        <Button
          variant="ghost"
          size="sm"
          disabled={busy || data.runtime_override === null}
          onClick={() => void apply(null)}
        >
          Clear override (fall back to env var)
        </Button>
      </div>
      <div className="mt-3 text-[11px]" style={{ color: "var(--text-faint)" }}>
        <strong>env_var_value:</strong> {data.env_var_value ?? "(unset)"} ·{" "}
        <strong>runtime_override:</strong>{" "}
        {data.runtime_override === null
          ? "cleared"
          : data.runtime_override
          ? "true"
          : "false"}
      </div>
    </Card>
  );
}


function IntentProbePanel() {
  const [message, setMessage] = useState("");
  const [useLlm, setUseLlm] = useState(true);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<IntentProbeResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function probe() {
    if (!message.trim()) return;
    setBusy(true);
    setError(null);
    try {
      const resp = await probeAdminIntent(message, useLlm);
      setResult(resp);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <Compass size={14} />
          <SectionTitle>Compound-intent live probe</SectionTitle>
        </div>
      </CardHeader>
      <p className="text-xs mb-3" style={{ color: "var(--text-dim)" }}>
        Paste any message (any language) to see how the router would classify
        it. Stateless: does not touch the chat database. Useful for sanity-
        checking multilingual / mixed-language coverage before deploying a
        prompt or vocabulary change.
      </p>
      <div className="flex flex-col gap-2">
        <textarea
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder='e.g. "hi, can you log my symptoms?" or "kamusta, gusto kong i-log ang lagnat ko"'
          rows={2}
          className="w-full text-sm px-2 py-1.5 rounded-md border"
          style={{
            background: "var(--surface)",
            borderColor: "var(--border)",
            color: "var(--text)",
            fontFamily: "var(--font-sans)",
          }}
        />
        <div className="flex flex-wrap items-center gap-2">
          <Button variant="primary" size="sm" disabled={busy || !message.trim()} onClick={() => void probe()}>
            {busy ? "Probing…" : "Probe"}
          </Button>
          <label className="text-xs flex items-center gap-1.5" style={{ color: "var(--text-dim)" }}>
            <input
              type="checkbox"
              checked={useLlm}
              onChange={(e) => setUseLlm(e.target.checked)}
              disabled={busy}
            />
            Use LLM augmenter (70B router tier)
          </label>
        </div>
      </div>

      {error && (
        <p className="mt-3 text-xs" style={{ color: "var(--rose)" }}>
          {error}
        </p>
      )}

      {result && (
        <div className="mt-3 flex flex-col gap-3 text-xs">
          {/* Merged envelope (what the chat layer would actually use). */}
          <ProbeEnvelopeBlock title="Merged (what chat would use)" envelope={result.merged} />
          {/* Deterministic envelope (heuristic-only) for comparison. */}
          <ProbeEnvelopeBlock title="Deterministic (heuristic-only)" envelope={result.deterministic} />
          {/* Raw LLM verdict block. */}
          <div className="p-2 rounded-md" style={{ background: "var(--surface2)", border: "1px solid var(--border)" }}>
            <div className="flex items-center gap-2 mb-1">
              <span style={{ color: "var(--text-faint)" }}>LLM verdict</span>
              <Badge variant={result.llm.available ? "purple" : "muted"}>
                {result.llm.available ? "available" : "unavailable"}
              </Badge>
              {result.llm.language && <Badge variant="cyan">lang: {result.llm.language}</Badge>}
              {result.llm.llm_confidence != null && (
                <Badge variant="muted">conf: {(result.llm.llm_confidence * 100).toFixed(0)}%</Badge>
              )}
            </div>
            {result.llm.provider && (
              <div style={{ color: "var(--text-faint)" }}>
                {result.llm.provider}
                {result.llm.model ? ` / ${result.llm.model}` : ""}
              </div>
            )}
            {result.llm.reason && (
              <div style={{ color: "var(--text-faint)" }}>reason: {result.llm.reason}</div>
            )}
          </div>
        </div>
      )}
    </Card>
  );
}


function ProbeEnvelopeBlock({
  title,
  envelope,
}: {
  title: string;
  envelope: IntentProbeResponse["merged"];
}) {
  return (
    <div className="p-2 rounded-md" style={{ background: "var(--surface2)", border: "1px solid var(--border)" }}>
      <div className="flex flex-wrap items-center gap-2 mb-1.5">
        <span style={{ color: "var(--text-faint)" }}>{title}</span>
        <Badge variant={envelope.is_compound ? "cyan" : "muted"}>
          {envelope.is_compound ? "compound" : "single"}
        </Badge>
        <Badge variant="muted">{envelope.primary_intent.replace(/_/g, " ")}</Badge>
        {envelope.has_tool_request && (
          <Badge variant="cyan">tool: {envelope.tool_request_targets.join(", ") || "n/a"}</Badge>
        )}
      </div>
      <div className="flex flex-col gap-0.5">
        {envelope.segments.length === 0 ? (
          <span style={{ color: "var(--text-faint)" }}>No segments.</span>
        ) : (
          envelope.segments.map((seg, idx) => (
            <div key={idx} className="flex items-baseline gap-2" style={{ color: "var(--text-dim)" }}>
              <span className="font-mono" style={{ color: "var(--text-faint)" }}>{seg.kind}</span>
              <span style={{ color: "var(--text)" }}>{seg.span || "(no span)"}</span>
              {seg.tool_targets?.length > 0 && (
                <span className="text-[10px]" style={{ color: "var(--cyan)" }}>
                  {seg.tool_targets.join(", ")}
                </span>
              )}
            </div>
          ))
        )}
      </div>
      {envelope.suggested_acknowledgment && (
        <div className="mt-1.5 text-[11px]" style={{ color: "var(--text-faint)" }}>
          <span style={{ color: "var(--text-dim)" }}>Suggested ack: </span>
          {envelope.suggested_acknowledgment}
        </div>
      )}
    </div>
  );
}
