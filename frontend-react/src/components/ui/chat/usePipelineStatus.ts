import { useEffect, useState } from "react";

const PIPELINE_STAGES = [
  { label: "Checking safety gate...", delay: 0 },
  { label: "Routing intent...", delay: 300 },
  { label: "Retrieving context...", delay: 700 },
  { label: "Generating response...", delay: 1500 },
];

export function usePipelineStatus(active: boolean): string {
  const [timing, setTiming] = useState<{ startedAt: number; now: number } | null>(null);

  useEffect(() => {
    let cancelled = false;
    if (!active) {
      window.setTimeout(() => { if (!cancelled) setTiming(null); }, 0);
      return () => { cancelled = true; };
    }
    const startedAt = Date.now();
    window.setTimeout(() => { if (!cancelled) setTiming({ startedAt, now: startedAt }); }, 0);
    const interval = window.setInterval(() => {
      setTiming((current) => current ? { ...current, now: Date.now() } : current);
    }, 150);
    return () => { cancelled = true; window.clearInterval(interval); };
  }, [active]);

  if (!active || !timing) return PIPELINE_STAGES[0].label;
  const elapsedMs = Math.max(0, timing.now - timing.startedAt);
  const idx = PIPELINE_STAGES.findLastIndex((step) => elapsedMs >= step.delay);
  return PIPELINE_STAGES[Math.max(0, idx)].label;
}
