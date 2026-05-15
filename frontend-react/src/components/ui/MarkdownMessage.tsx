import { Fragment, type ReactNode } from "react";

/**
 * Tiny safe markdown renderer for chat assistant messages.
 *
 * Supports:
 *   **bold**      → <strong>
 *   *italic*      → <em>
 *   `code`        → <code>
 *   numbered list (1. / 2. ...)
 *   bullet list   (- / *)
 *   paragraphs separated by blank lines
 *
 * Does NOT use dangerouslySetInnerHTML — every node is constructed as a React
 * element so user-supplied content cannot inject HTML.
 */

interface Props {
  text: string;
  className?: string;
}

// ── Inline parsing ────────────────────────────────────────────────────────────

type InlineToken =
  | { kind: "text"; value: string }
  | { kind: "bold"; value: string }
  | { kind: "italic"; value: string }
  | { kind: "code"; value: string };

/**
 * Parse a single line of inline markdown into tokens. We use a left-to-right
 * scanner instead of regex-replace so we never have to inject HTML.
 */
function parseInline(line: string): InlineToken[] {
  const tokens: InlineToken[] = [];
  let i = 0;
  let buffer = "";
  const flush = () => {
    if (buffer.length) {
      tokens.push({ kind: "text", value: buffer });
      buffer = "";
    }
  };
  while (i < line.length) {
    const ch = line[i];
    const next = line[i + 1];

    // **bold**
    if (ch === "*" && next === "*") {
      const close = line.indexOf("**", i + 2);
      if (close !== -1 && close > i + 2) {
        flush();
        tokens.push({ kind: "bold", value: line.slice(i + 2, close) });
        i = close + 2;
        continue;
      }
    }

    // *italic* (single asterisk) — only when not part of **
    if (ch === "*" && next !== "*") {
      const close = line.indexOf("*", i + 1);
      // Don't treat as italic if the closing * is part of a **
      const isBoldNeighbor = close !== -1 && line[close + 1] === "*";
      if (close !== -1 && close > i + 1 && !isBoldNeighbor) {
        const inside = line.slice(i + 1, close);
        // Italics shouldn't span huge ranges — keep it sensible (no newlines inside)
        if (!inside.includes("\n") && inside.length < 200) {
          flush();
          tokens.push({ kind: "italic", value: inside });
          i = close + 1;
          continue;
        }
      }
    }

    // `code`
    if (ch === "`") {
      const close = line.indexOf("`", i + 1);
      if (close !== -1 && close > i + 1) {
        flush();
        tokens.push({ kind: "code", value: line.slice(i + 1, close) });
        i = close + 1;
        continue;
      }
    }

    buffer += ch;
    i++;
  }
  flush();
  return tokens;
}

function renderInline(tokens: InlineToken[]): ReactNode[] {
  return tokens.map((t, i) => {
    switch (t.kind) {
      case "bold":   return <strong key={i}>{t.value}</strong>;
      case "italic": return <em key={i}>{t.value}</em>;
      case "code":
        return (
          <code
            key={i}
            style={{
              fontFamily: "var(--mono)",
              fontSize: "0.86em",
              padding: "1px 5px",
              borderRadius: 4,
              background: "rgba(0,0,0,0.05)",
            }}
          >
            {t.value}
          </code>
        );
      default:       return <Fragment key={i}>{t.value}</Fragment>;
    }
  });
}

// ── Block parsing ─────────────────────────────────────────────────────────────

type Block =
  | { kind: "p"; lines: string[] }
  | { kind: "ul"; items: string[] }
  | { kind: "ol"; items: string[] };

function parseBlocks(source: string): Block[] {
  const blocks: Block[] = [];
  const lines = source.replace(/\r\n/g, "\n").split("\n");
  let i = 0;
  while (i < lines.length) {
    const raw = lines[i];
    const line = raw.trim();

    if (!line) {
      i++;
      continue;
    }

    // Bullet list
    if (/^[-*]\s+/.test(line)) {
      const items: string[] = [];
      while (i < lines.length && /^[-*]\s+/.test(lines[i].trim())) {
        items.push(lines[i].trim().replace(/^[-*]\s+/, ""));
        i++;
      }
      blocks.push({ kind: "ul", items });
      continue;
    }

    // Numbered list
    if (/^\d+\.\s+/.test(line)) {
      const items: string[] = [];
      while (i < lines.length && /^\d+\.\s+/.test(lines[i].trim())) {
        items.push(lines[i].trim().replace(/^\d+\.\s+/, ""));
        i++;
      }
      blocks.push({ kind: "ol", items });
      continue;
    }

    // Paragraph — consume non-blank, non-list lines
    const paraLines: string[] = [];
    while (
      i < lines.length &&
      lines[i].trim() !== "" &&
      !/^[-*]\s+/.test(lines[i].trim()) &&
      !/^\d+\.\s+/.test(lines[i].trim())
    ) {
      paraLines.push(lines[i].trim());
      i++;
    }
    blocks.push({ kind: "p", lines: paraLines });
  }
  return blocks;
}

// ── Component ─────────────────────────────────────────────────────────────────

export function MarkdownMessage({ text, className }: Props) {
  const blocks = parseBlocks(text);
  return (
    <div className={className}>
      {blocks.map((block, bi) => {
        if (block.kind === "ul") {
          return (
            <ul
              key={bi}
              style={{
                listStyle: "disc",
                paddingLeft: "1.25rem",
                margin: "4px 0",
                display: "flex",
                flexDirection: "column",
                gap: 2,
              }}
            >
              {block.items.map((it, ii) => (
                <li key={ii} style={{ lineHeight: 1.55 }}>{renderInline(parseInline(it))}</li>
              ))}
            </ul>
          );
        }
        if (block.kind === "ol") {
          return (
            <ol
              key={bi}
              style={{
                listStyle: "decimal",
                paddingLeft: "1.5rem",
                margin: "4px 0",
                display: "flex",
                flexDirection: "column",
                gap: 2,
              }}
            >
              {block.items.map((it, ii) => (
                <li key={ii} style={{ lineHeight: 1.55 }}>{renderInline(parseInline(it))}</li>
              ))}
            </ol>
          );
        }
        // paragraph — line-break sensitive but markdown-style line joins
        return (
          <p key={bi} style={{ margin: bi === 0 ? "0" : "8px 0 0", lineHeight: 1.6 }}>
            {block.lines.map((l, li) => (
              <Fragment key={li}>
                {li > 0 && <br />}
                {renderInline(parseInline(l))}
              </Fragment>
            ))}
          </p>
        );
      })}
    </div>
  );
}
