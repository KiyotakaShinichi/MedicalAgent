/**
 * URL sanitisation for backend-supplied links.
 *
 * The admin evidence panels render `href` values that originate from ingested
 * knowledge-base source metadata. That metadata is not authored by us, so an
 * href is a trust boundary: React escapes text but does NOT stop
 * `javascript:` or `data:text/html` URLs from executing on click.
 *
 * `safeExternalUrl` returns the URL only when it parses and uses a scheme that
 * cannot execute script, and `null` otherwise so callers can degrade to plain
 * text rather than rendering a live link.
 */

const ALLOWED_PROTOCOLS = new Set(["http:", "https:", "mailto:"]);

/**
 * True when the string contains a C0 control character or DEL.
 *
 * Browsers strip newline, carriage return, and tab while parsing a URL, which
 * is how "java\nscript:alert(1)" historically slipped past naive scheme
 * filters. Any URL carrying one is rejected before parsing rather than after.
 * Tested by code point rather than by regex so that no control character has
 * to appear literally in this source file.
 */
function hasControlCharacter(value: string): boolean {
  for (let i = 0; i < value.length; i += 1) {
    const code = value.charCodeAt(i);
    if (code <= 0x1f || code === 0x7f) return true;
  }
  return false;
}

export function safeExternalUrl(raw: string | null | undefined): string | null {
  if (!raw) return null;
  const trimmed = raw.trim();
  if (!trimmed) return null;
  if (hasControlCharacter(trimmed)) return null;

  let parsed: URL;
  try {
    // Relative URLs resolve against the current origin, which is safe and
    // keeps backend-relative artifact paths working.
    parsed = new URL(trimmed, window.location.origin);
  } catch {
    return null;
  }

  if (!ALLOWED_PROTOCOLS.has(parsed.protocol)) return null;
  return parsed.href;
}

/** True when the URL points somewhere other than the app's own origin. */
export function isExternalUrl(href: string): boolean {
  try {
    return new URL(href, window.location.origin).origin !== window.location.origin;
  } catch {
    return false;
  }
}
