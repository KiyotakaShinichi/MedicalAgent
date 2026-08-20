import type { CSSProperties, ReactNode } from "react";
import { safeExternalUrl } from "../../lib/safeUrl";

interface ExternalLinkProps {
  /** Backend-supplied URL. Untrusted — sanitised before it becomes an href. */
  href: string | null | undefined;
  children: ReactNode;
  className?: string;
  style?: CSSProperties;
}

/**
 * Anchor for URLs that came from the backend or an ingested knowledge base.
 *
 * Two things this guarantees that a bare `<a href={x}>` does not:
 *
 *   1. The scheme is allow-listed, so a `javascript:` or `data:text/html` URL
 *      in source metadata renders as inert text instead of executing on click.
 *   2. `rel="noopener noreferrer"` is always present alongside `target=_blank`,
 *      so the opened page cannot reach back through `window.opener`.
 */
export function ExternalLink({ href, children, className, style }: ExternalLinkProps) {
  const safeHref = safeExternalUrl(href);

  if (!safeHref) {
    // Degrade to text rather than dropping the label — the source name is
    // still meaningful evidence even when its link is unusable.
    return (
      <span className={className} style={style} title="Link unavailable or blocked">
        {children}
      </span>
    );
  }

  return (
    <a
      href={safeHref}
      target="_blank"
      rel="noopener noreferrer"
      className={className}
      style={style}
    >
      {children}
    </a>
  );
}
