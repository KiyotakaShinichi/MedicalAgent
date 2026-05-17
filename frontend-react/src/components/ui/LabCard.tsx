import { useState, useMemo, useRef, useEffect, type ReactNode } from "react";
import { TrendingUp, TrendingDown, Minus, Info } from "lucide-react";
import type { LucideIcon } from "lucide-react";
import {
  LAB_REFERENCE_RANGES,
  classifyLabValue,
  labStatusLabel,
  labStatusTone,
  type LabKey,
  type LabStatus,
} from "../../lib/clinical-constants";

interface LabCardProps {
  labKey: LabKey;
  icon?: LucideIcon;
  value: number | null;
  /**
   * Historical series for the trend line + delta-vs-baseline computation.
   * Pass the FULL time-ordered series; the card slices its own sparkline window.
   * Oldest entry first, latest entry last.
   */
  history?: Array<{ date: string; value: number | null }>;
  /** Date string of the latest reading.  When provided, shown as "Last: …". */
  lastSampleDate?: string | null;
}

const TONE_COLOUR: Record<ReturnType<typeof labStatusTone>, { fg: string; bg: string; border: string; bar: string }> = {
  success: { fg: "#047857", bg: "#ecfdf5", border: "#a7f3d0", bar: "#10b981" },
  warning: { fg: "#92400e", bg: "#fffbeb", border: "#fde68a", bar: "#f59e0b" },
  danger:  { fg: "#b91c1c", bg: "#fef2f2", border: "#fecaca", bar: "#ef4444" },
  neutral: { fg: "var(--text-faint)", bg: "var(--surface2)", border: "var(--border)", bar: "var(--border-strong)" },
};

interface ParsedSeries {
  values: number[];
  baseline: number | null;
  latest: number | null;
  deltaPct: number | null;
}

function parseSeries(history?: LabCardProps["history"]): ParsedSeries {
  const values = (history ?? [])
    .map((p) => (p.value != null && Number.isFinite(p.value) ? p.value : null))
    .filter((v): v is number => v != null);
  if (values.length === 0) return { values, baseline: null, latest: null, deltaPct: null };
  const baseline = values[0];
  const latest = values[values.length - 1];
  const deltaPct =
    values.length >= 2 && baseline !== 0
      ? ((latest - baseline) / Math.abs(baseline)) * 100
      : null;
  return { values, baseline, latest, deltaPct };
}

/**
 * Pure-SVG sparkline.  Renders a 1-pixel-stroked path of recent values with a
 * subtle area fill underneath.  No new chart library — the recharts bundle
 * already lazy-loads only when the main trend chart shows.
 *
 * Drops the points below the in-range floor / above the in-range ceiling
 * as light reference bands so the line's clinical context is visible without
 * a separate y-axis.
 */
function Sparkline({
  values,
  refLow,
  refHigh,
  colour,
  width = 96,
  height = 28,
}: {
  values: number[];
  refLow: number;
  refHigh: number;
  colour: string;
  width?: number;
  height?: number;
}) {
  if (values.length < 2) return null;
  // Use a padded y-range so the line never grazes the edges.
  const padPct = 0.10;
  const series = values.slice(-12); // most recent N points
  const minV = Math.min(...series, refLow);
  const maxV = Math.max(...series, refHigh);
  const span = (maxV - minV) || 1;
  const yPad = span * padPct;
  const yMin = minV - yPad;
  const yMax = maxV + yPad;
  const ySpan = yMax - yMin;

  const xStep = series.length > 1 ? width / (series.length - 1) : width;
  const toY = (v: number) => height - ((v - yMin) / ySpan) * height;
  const points = series.map((v, i) => [i * xStep, toY(v)] as const);
  const path = points.map(([x, y], i) => (i === 0 ? `M${x.toFixed(1)},${y.toFixed(1)}` : `L${x.toFixed(1)},${y.toFixed(1)}`)).join(" ");
  const area = `${path} L${(points[points.length - 1][0]).toFixed(1)},${height} L0,${height} Z`;
  // Reference band as a faint rectangle behind the line.
  const bandTop = toY(refHigh);
  const bandBottom = toY(refLow);
  const bandY = Math.min(bandTop, bandBottom);
  const bandH = Math.max(2, Math.abs(bandBottom - bandTop));

  return (
    <svg
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label="Sparkline of recent values"
      style={{ display: "block" }}
    >
      <rect x="0" y={bandY} width={width} height={bandH} fill={colour} opacity="0.08" />
      <path d={area} fill={colour} opacity="0.12" />
      <path d={path} fill="none" stroke={colour} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
      {/* End-point dot */}
      <circle cx={points[points.length - 1][0]} cy={points[points.length - 1][1]} r="2" fill={colour} />
    </svg>
  );
}

function DeltaBadge({ deltaPct }: { deltaPct: number | null }) {
  if (deltaPct == null) return null;
  if (Math.abs(deltaPct) < 5) {
    return (
      <span className="lab-card-delta lab-card-delta--neutral">
        <Minus size={11} aria-hidden="true" /> Stable
      </span>
    );
  }
  const Icon = deltaPct > 0 ? TrendingUp : TrendingDown;
  // For monitoring, "rising" and "falling" are tonally neutral (the clinical
  // meaning depends on the metric) — colour them grey, not red/green.
  return (
    <span className="lab-card-delta lab-card-delta--neutral">
      <Icon size={11} aria-hidden="true" />
      {deltaPct > 0 ? "+" : ""}{Math.round(deltaPct)}% vs baseline
    </span>
  );
}

/** Patient-safe tooltip body.  Renders as a small popover anchored to an
 *  info icon — no external tooltip library.  Closes on Esc + outside-click. */
function InfoTooltip({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState(false);
  const wrapRef = useRef<HTMLSpanElement | null>(null);

  useEffect(() => {
    if (!open) return;
    const onClick = (e: MouseEvent) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setOpen(false); };
    window.addEventListener("mousedown", onClick);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("mousedown", onClick);
      window.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <span ref={wrapRef} className="lab-card-info-wrap">
      <button
        type="button"
        className="lab-card-info-trigger"
        onClick={() => setOpen((v) => !v)}
        aria-label="About this reference range"
        aria-expanded={open}
      >
        <Info size={11} aria-hidden="true" />
      </button>
      {open && (
        <div className="lab-card-info-popover" role="dialog">
          {children}
        </div>
      )}
    </span>
  );
}

function formatValue(labKey: LabKey, value: number | null): string {
  if (value == null) return "—";
  // Hemoglobin + WBC + ANC carry one decimal place; platelets are usually
  // reported as a whole number.
  if (labKey === "platelets") return Math.round(value).toString();
  return value.toFixed(1);
}

function formatDate(dateString: string | null | undefined): string | null {
  if (!dateString) return null;
  const date = new Date(dateString);
  if (Number.isNaN(date.getTime())) return dateString.slice(0, 10);
  return date.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
}

/**
 * Patient-facing CBC value card.
 *
 * Layout (top to bottom): label + info button | value + unit | status chip +
 * delta + sparkline | reference range + last sample.  Every visible piece of
 * clinical content (label, unit, reference range, status copy, tooltip body)
 * is sourced from ``clinical-constants.ts`` so the dashboard, the CBC entry
 * form, and any future clinician view can never disagree about what a lab
 * value means.
 */
export function LabCard({ labKey, icon: Icon, value, history, lastSampleDate }: LabCardProps) {
  const range = LAB_REFERENCE_RANGES[labKey];
  const parsed = useMemo(() => parseSeries(history), [history]);
  const status: LabStatus = useMemo(() => classifyLabValue(labKey, value), [labKey, value]);
  const tone = labStatusTone(status);
  const palette = TONE_COLOUR[tone];
  const displayValue = formatValue(labKey, value);
  const lastDate = formatDate(lastSampleDate);

  return (
    <div
      className="lab-card"
      style={{
        borderColor: tone === "neutral" ? "var(--border)" : palette.border,
      }}
    >
      <span aria-hidden="true" className="lab-card-bar" style={{ background: palette.bar }} />

      <div className="lab-card-header">
        <span className="lab-card-label">
          {Icon && (
            <span className="lab-card-label-icon" style={{ background: palette.bg, color: palette.fg }}>
              <Icon size={12} aria-hidden="true" />
            </span>
          )}
          {range.label}
        </span>
        <InfoTooltip>
          <p style={{ margin: 0, fontWeight: 600, color: "var(--text-strong)" }}>{range.label} reference</p>
          <p style={{ margin: "4px 0 0", color: "var(--text)" }}>
            {range.refLow}–{range.refHigh} {range.unit}
          </p>
          <p style={{ margin: "6px 0 0", color: "var(--text-dim)", lineHeight: 1.45 }}>
            {range.description}
          </p>
          <p style={{ margin: "6px 0 0", color: "var(--text-faint)", lineHeight: 1.45, fontStyle: "italic" }}>
            {range.disclaimer}
          </p>
        </InfoTooltip>
      </div>

      <div className="lab-card-value-row">
        <span className="lab-card-value">{displayValue}</span>
        <span className="lab-card-unit">{range.unit}</span>
      </div>

      <div className="lab-card-meta-row">
        <span
          className="lab-card-status"
          style={{ background: palette.bg, color: palette.fg, borderColor: palette.border }}
        >
          {labStatusLabel(status)}
        </span>
        <DeltaBadge deltaPct={parsed.deltaPct} />
        <span className="lab-card-spark">
          <Sparkline
            values={parsed.values}
            refLow={range.refLow}
            refHigh={range.refHigh}
            colour={palette.bar}
          />
        </span>
      </div>

      <div className="lab-card-footer">
        <span>Ref {range.refLow}–{range.refHigh} {range.unit}</span>
        {lastDate && <span>· Last sample {lastDate}</span>}
      </div>
    </div>
  );
}
