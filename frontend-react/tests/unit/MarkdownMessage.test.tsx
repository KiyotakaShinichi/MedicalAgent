import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { MarkdownMessage } from "../../src/components/ui/MarkdownMessage";


describe("MarkdownMessage list rendering", () => {
  it("removes orphaned numbering from a single generated item", () => {
    const { container } = render(
      <MarkdownMessage text="1. I can help explain records in this portal." />,
    );

    expect(container.querySelector("ol")).toBeNull();
    expect(screen.getByText("I can help explain records in this portal.")).toBeInTheDocument();
  });

  it("keeps a real multi-item ordered list", () => {
    const { container } = render(
      <MarkdownMessage text={"1. Check the record date.\n2. Prepare a care-team question."} />,
    );

    expect(container.querySelector("ol")).not.toBeNull();
    expect(container.querySelectorAll("li")).toHaveLength(2);
  });
});
