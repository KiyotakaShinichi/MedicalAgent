export function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

export function readPath(record: Record<string, unknown> | null, path: string[]): unknown {
  let current: unknown = record;
  for (const key of path) {
    if (!current || typeof current !== "object" || Array.isArray(current)) return null;
    current = (current as Record<string, unknown>)[key];
  }
  return current;
}

export function readString(record: Record<string, unknown> | null, path: string[]): string | null {
  const value = readPath(record, path);
  return typeof value === "string" ? value : null;
}

export function formatMaybePercent(value: unknown): string {
  return typeof value === "number" ? `${(value * 100).toFixed(1)}%` : "-";
}

export function formatMaybeNumber(value: unknown): string {
  return typeof value === "number" ? value.toFixed(3) : "-";
}

export function formatMaybeMs(value: unknown): string {
  return typeof value === "number" ? `${value.toFixed(0)}ms` : "-";
}
