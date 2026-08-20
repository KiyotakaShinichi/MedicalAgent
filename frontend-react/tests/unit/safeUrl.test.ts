import { describe, it, expect } from "vitest";
import { safeExternalUrl, isExternalUrl } from "../../src/lib/safeUrl";

describe("safeExternalUrl", () => {
  it("allows http and https", () => {
    expect(safeExternalUrl("https://pubmed.ncbi.nlm.nih.gov/123")).toBe(
      "https://pubmed.ncbi.nlm.nih.gov/123",
    );
    expect(safeExternalUrl("http://example.org/doc")).toBe("http://example.org/doc");
  });

  it("allows mailto for contact links", () => {
    expect(safeExternalUrl("mailto:research@example.org")).toBe("mailto:research@example.org");
  });

  it("blocks javascript: URLs", () => {
    expect(safeExternalUrl("javascript:alert(1)")).toBeNull();
    expect(safeExternalUrl("JaVaScRiPt:alert(1)")).toBeNull();
  });

  it("blocks javascript: obfuscated with embedded control characters", () => {
    // Browsers strip these while parsing, so the scheme check alone is not enough.
    expect(safeExternalUrl("java\nscript:alert(1)")).toBeNull();
    expect(safeExternalUrl("java\tscript:alert(1)")).toBeNull();
    expect(safeExternalUrl("java\rscript:alert(1)")).toBeNull();
  });

  it("blocks data: URLs that could carry markup", () => {
    expect(safeExternalUrl("data:text/html;base64,PHNjcmlwdD4=")).toBeNull();
  });

  it("blocks other executable or unexpected schemes", () => {
    expect(safeExternalUrl("vbscript:msgbox(1)")).toBeNull();
    expect(safeExternalUrl("file:///etc/passwd")).toBeNull();
  });

  it("returns null for empty and absent input", () => {
    expect(safeExternalUrl(null)).toBeNull();
    expect(safeExternalUrl(undefined)).toBeNull();
    expect(safeExternalUrl("")).toBeNull();
    expect(safeExternalUrl("   ")).toBeNull();
  });

  it("resolves relative paths against the current origin", () => {
    const resolved = safeExternalUrl("/Data/evals/report.json");
    expect(resolved).toBe(`${window.location.origin}/Data/evals/report.json`);
  });
});

describe("isExternalUrl", () => {
  it("distinguishes third-party hosts from the app origin", () => {
    expect(isExternalUrl("https://example.org/x")).toBe(true);
    expect(isExternalUrl("/local/path")).toBe(false);
  });
});
