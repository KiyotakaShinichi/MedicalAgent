import { clsx } from "clsx";
import { Card } from "./Card";

interface SkeletonBlockProps {
  width?: string | number;
  height?: string | number;
  className?: string;
  style?: React.CSSProperties;
}

/**
 * Single shimmering rectangle. Use as a building block — for full-card and
 * full-page silhouettes use SkeletonCard / SkeletonGrid.
 */
export function SkeletonBlock({ width, height, className, style }: SkeletonBlockProps) {
  return (
    <span
      className={clsx("skeleton-block", className)}
      aria-hidden="true"
      style={{
        width: typeof width === "number" ? `${width}px` : width ?? "100%",
        height: typeof height === "number" ? `${height}px` : height ?? "12px",
        ...style,
      }}
    />
  );
}

/**
 * A card-shaped skeleton — title + 3 lines. Drop into a grid cell to
 * preview the eventual content's silhouette while data loads.
 */
export function SkeletonCard({ rows = 3, className }: { rows?: number; className?: string }) {
  return (
    <Card className={clsx("skeleton-card", className)} padding={false}>
      <SkeletonBlock width="40%" height={14} />
      {Array.from({ length: rows }).map((_, i) => (
        <SkeletonBlock
          key={i}
          width={i === rows - 1 ? "70%" : "100%"}
          height={10}
        />
      ))}
    </Card>
  );
}

/**
 * Full-dashboard skeleton — a hero row followed by a 12-column grid of
 * card-shaped skeletons. Matches the layout that follows it so the page
 * does not visually jump when data arrives.
 */
export function SkeletonDashboard({
  label = "Loading dashboard...",
}: {
  label?: string;
}) {
  return (
    <div className="dashboard-page" role="status" aria-label={label}>
      <div
        className="patient-hero"
        style={{ background: "rgba(23, 28, 39, 0.7)" }}
      >
        <div style={{ display: "grid", gap: 10 }}>
          <SkeletonBlock width="30%" height={10} />
          <SkeletonBlock width="55%" height={22} />
          <SkeletonBlock width="40%" height={12} />
        </div>
        <div style={{ display: "grid", gap: 10 }}>
          <SkeletonBlock width="100%" height={48} />
          <SkeletonBlock width="100%" height={48} />
        </div>
      </div>
      <div className="dashboard-content">
        <div className="skeleton-grid">
          <SkeletonCard className="skeleton-grid-full" rows={4} />
          <SkeletonCard />
          <SkeletonCard />
          <SkeletonCard />
          <SkeletonCard />
        </div>
      </div>
    </div>
  );
}
