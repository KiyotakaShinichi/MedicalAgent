import { useEffect, useState } from "react";

interface Props {
  timestamp: number | null;
  /** Prefix shown before the relative phrase, e.g. "Last refreshed" or "Updated". */
  prefix?: string;
  className?: string;
}

function format(diffMs: number): string {
  const sec = Math.max(0, Math.floor(diffMs / 1000));
  if (sec < 5)    return "just now";
  if (sec < 60)   return `${sec}s ago`;
  const min = Math.floor(sec / 60);
  if (min < 60)   return `${min} min${min === 1 ? "" : "s"} ago`;
  const hr = Math.floor(min / 60);
  if (hr < 24)    return `${hr} hr${hr === 1 ? "" : "s"} ago`;
  const d = Math.floor(hr / 24);
  return `${d} day${d === 1 ? "" : "s"} ago`;
}

/**
 * Tiny self-refreshing "X seconds ago" badge. Re-renders every 10 seconds
 * for sub-minute precision without thrashing the page.
 */
export function RelativeTime({ timestamp, prefix = "Updated", className }: Props) {
  const [now, setNow] = useState<number | null>(null);

  useEffect(() => {
    if (timestamp == null) return;
    const initial = window.setTimeout(() => setNow(Date.now()), 0);
    const interval = window.setInterval(() => setNow(Date.now()), 10000);
    return () => {
      window.clearTimeout(initial);
      window.clearInterval(interval);
    };
  }, [timestamp]);

  if (timestamp == null) return null;
  const label = now == null ? "just now" : format(now - timestamp);
  return (
    <span className={className} style={{ color: "var(--text-faint)", fontSize: "0.72rem", fontWeight: 500 }}>
      {prefix} {label}
    </span>
  );
}
