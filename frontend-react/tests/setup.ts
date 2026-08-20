import "@testing-library/jest-dom/vitest";
import { afterEach, vi } from "vitest";
import { cleanup } from "@testing-library/react";

/**
 * jsdom implements no layout engine, so a handful of browser APIs the app
 * legitimately uses are simply absent. These are environment shims, not
 * behaviour stubs — each one stands in for something jsdom cannot compute, and
 * none of them changes what the components under test do.
 */

// Used for responsive layout and reduced-motion checks. Reports "no match" so
// tests exercise the default (desktop, motion-allowed) branch.
if (!window.matchMedia) {
  window.matchMedia = ((query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: () => {},
    removeListener: () => {},
    addEventListener: () => {},
    removeEventListener: () => {},
    dispatchEvent: () => false,
  })) as typeof window.matchMedia;
}

// Scrolling is a no-op without layout. The dashboard calls these for deep-link
// anchoring; they must exist but have nothing to do here.
if (!window.scrollTo) {
  window.scrollTo = (() => {}) as typeof window.scrollTo;
}
if (!Element.prototype.scrollTo) {
  Element.prototype.scrollTo = (() => {}) as typeof Element.prototype.scrollTo;
}
if (!Element.prototype.scrollIntoView) {
  Element.prototype.scrollIntoView = (() => {}) as typeof Element.prototype.scrollIntoView;
}

// Recharts and other size-aware components observe their container. jsdom
// reports every element as zero-sized, so the callback is never meaningful.
if (!globalThis.ResizeObserver) {
  globalThis.ResizeObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
  } as unknown as typeof ResizeObserver;
}

afterEach(() => {
  cleanup();
  vi.clearAllTimers();
});
