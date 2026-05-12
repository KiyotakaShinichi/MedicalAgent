export type BadgeVariant = "green" | "amber" | "red" | "blue" | "purple" | "muted" | "cyan";

export function badgeColorToken(variant: BadgeVariant) {
  const map: Record<BadgeVariant, string> = {
    green: "green",
    amber: "amber",
    red: "rose",
    blue: "blue",
    purple: "purple",
    cyan: "cyan",
    muted: "slate",
  };
  return map[variant];
}

export function statusVariant(status: string): BadgeVariant {
  const s = (status || "").toLowerCase();
  if (
    s.includes("strong") ||
    s.includes("pass") ||
    s.includes("normal") ||
    s.includes("stable") ||
    s.includes("approv") ||
    s.includes("low_risk")
  ) return "green";
  if (
    s.includes("acceptable") ||
    s.includes("watch") ||
    s.includes("warn") ||
    s.includes("amber") ||
    s.includes("review")
  ) return "amber";
  if (
    s.includes("fail") ||
    s.includes("error") ||
    s.includes("unsafe") ||
    s.includes("high_risk") ||
    s.includes("urgent") ||
    s.includes("reject")
  ) return "red";
  if (s.includes("blue") || s.includes("info")) return "blue";
  if (s.includes("purple")) return "purple";
  return "muted";
}
