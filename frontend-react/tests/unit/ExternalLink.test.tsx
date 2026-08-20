import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { ExternalLink } from "../../src/components/ui/ExternalLink";

describe("ExternalLink", () => {
  it("renders a link with noopener noreferrer for an allowed scheme", () => {
    render(<ExternalLink href="https://example.org/paper">Source paper</ExternalLink>);

    const link = screen.getByRole("link", { name: "Source paper" });
    expect(link).toHaveAttribute("href", "https://example.org/paper");
    expect(link).toHaveAttribute("target", "_blank");
    // noopener stops the opened page reaching back via window.opener.
    expect(link.getAttribute("rel")).toContain("noopener");
    expect(link.getAttribute("rel")).toContain("noreferrer");
  });

  it("degrades a javascript: URL to inert text but keeps the label", () => {
    render(<ExternalLink href="javascript:alert(document.cookie)">Malicious source</ExternalLink>);

    expect(screen.queryByRole("link")).not.toBeInTheDocument();
    expect(screen.getByText("Malicious source")).toBeInTheDocument();
  });

  it("degrades a data: URL to inert text", () => {
    render(<ExternalLink href="data:text/html;base64,PHNjcmlwdD4=">Data source</ExternalLink>);
    expect(screen.queryByRole("link")).not.toBeInTheDocument();
  });

  it("renders the label as text when the URL is missing", () => {
    render(<ExternalLink href={null}>Unlinked source</ExternalLink>);
    expect(screen.queryByRole("link")).not.toBeInTheDocument();
    expect(screen.getByText("Unlinked source")).toBeInTheDocument();
  });
});
