import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import type { ReactNode } from "react";

vi.mock("../../src/api/client", () => ({
  API_BASE: "http://test.local",
  getMyReportCore: vi.fn(),
  getMyReportEnrichment: vi.fn(),
  getMyChatHistory: vi.fn(),
  sendMyChat: vi.fn(),
  sendMyChatStream: vi.fn(),
  undoMyConfirmedRecordWrite: vi.fn(),
  uploadFile: vi.fn(),
  logout: vi.fn(),
}));

// Recharts needs real layout; jsdom reports zero-size containers and the chart
// renders nothing useful. The lab *panel* behaviour is what matters here.
vi.mock("../../src/components/charts/LabTrendsChart", () => ({
  LabTrendsChart: () => <div data-testid="lab-trends-chart" />,
}));

import * as api from "../../src/api/client";
import PatientDashboard from "../../src/pages/patient/PatientDashboard";
import { AuthProvider } from "../../src/context/AuthContext";
import { ToastProvider } from "../../src/components/ui/Toast";
import { resetTelemetrySinks } from "../../src/lib/telemetry";

const mocked = vi.mocked(api);

/** Minimal core report: every collection empty, so empty states are exercised. */
const CORE_REPORT = {
  patient_id: "P001",
  patient_name: "Ana Reyes",
  labs: [],
  symptoms: [],
  timeline: [],
  medication_logs: [],
  chat_history: [],
  ai_summary: null,
  evidence_aware_prediction: null,
  hybrid_prediction: null,
  genetic_counseling_readiness: null,
};

function renderDashboard(initialPath = "/patient") {
  return render(
    <MemoryRouter initialEntries={[initialPath]}>
      <ToastProvider>
        <AuthProvider>
          <PatientDashboard />
        </AuthProvider>
      </ToastProvider>
    </MemoryRouter>,
  );
}

/** A promise that never settles — holds a request in flight. */
const pending = <T,>(): Promise<T> => new Promise<T>(() => {});

describe("PatientDashboard", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    resetTelemetrySinks();
    vi.spyOn(console, "warn").mockImplementation(() => {});
    vi.spyOn(console, "error").mockImplementation(() => {});
    sessionStorage.setItem("patientPortalAccessToken", "test-token");
    sessionStorage.setItem("currentPatientId", "P001");
    mocked.getMyChatHistory.mockResolvedValue({ messages: [] } as never);
    // Default: enrichment never completes, so tests that do not care about it
    // simply see the core report.
    mocked.getMyReportEnrichment.mockReturnValue(pending());
  });

  afterEach(() => {
    sessionStorage.clear();
    resetTelemetrySinks();
    vi.restoreAllMocks();
  });

  describe("core report loading", () => {
    it("shows a loading state while the report is in flight", () => {
      mocked.getMyReportCore.mockReturnValue(pending());
      renderDashboard();
      // The skeleton exposes its label as the accessible name of a live region
      // rather than as visible text.
      expect(screen.getByRole("status", { name: /Loading your records/i })).toBeInTheDocument();
    });

    it("shows an error state with a retry affordance when the report fails", async () => {
      mocked.getMyReportCore.mockRejectedValue(new Error("records unavailable"));
      renderDashboard();

      expect(await screen.findByText("records unavailable")).toBeInTheDocument();
      expect(screen.getByRole("button", { name: /try again/i })).toBeInTheDocument();
    });

    it("refetches the report when retry is pressed", async () => {
      const user = userEvent.setup();
      mocked.getMyReportCore.mockRejectedValueOnce(new Error("records unavailable"));
      mocked.getMyReportCore.mockResolvedValue(CORE_REPORT as never);
      renderDashboard();

      await screen.findByRole("button", { name: /try again/i });
      await user.click(screen.getByRole("button", { name: /try again/i }));

      await waitFor(() => expect(mocked.getMyReportCore).toHaveBeenCalledTimes(2));
    });

    it("renders the patient's records once loaded", async () => {
      mocked.getMyReportCore.mockResolvedValue(CORE_REPORT as never);
      renderDashboard();

      expect(await screen.findByRole("button", { name: "Support chat" })).toBeInTheDocument();
      expect(screen.getAllByText("Ana Reyes").length).toBeGreaterThan(0);
    });
  });

  describe("clinical safety presentation", () => {
    it("always shows the clinical boundary banner", async () => {
      mocked.getMyReportCore.mockResolvedValue(CORE_REPORT as never);
      const { container } = renderDashboard();

      await screen.findByRole("button", { name: "Support chat" });
      expect(container.textContent).toMatch(/not.*(diagnos|replace)/i);
    });

    it("carries a proof-of-concept footnote stating it does not diagnose", async () => {
      mocked.getMyReportCore.mockResolvedValue(CORE_REPORT as never);
      renderDashboard();

      await screen.findByRole("button", { name: "Support chat" });
      // The boundary is stated in more than one place by design — the topbar
      // pill and the page footnote — so assert presence, not uniqueness.
      expect(
        screen.getAllByText(/does not diagnose, recommend treatment, or\s+replace clinician judgement/i).length,
      ).toBeGreaterThan(0);
    });

    it("does not present model output when enrichment has not completed", async () => {
      // The safety-critical case: core records render immediately, but the
      // model-derived panels must not show anything until the job completes.
      mocked.getMyReportCore.mockResolvedValue(CORE_REPORT as never);
      mocked.getMyReportEnrichment.mockResolvedValue({
        report_enrichment: { status: "pending", retry_after_ms: 10_000 },
      } as never);

      renderDashboard();
      await screen.findByRole("button", { name: "Support chat" });

      expect(
        await screen.findByText(/Loading synthetic engineering details separately/i),
      ).toBeInTheDocument();
    });

    it("states plainly when enrichment fails instead of implying no findings", async () => {
      mocked.getMyReportCore.mockResolvedValue(CORE_REPORT as never);
      mocked.getMyReportEnrichment.mockResolvedValue({
        report_enrichment: { status: "failed" },
      } as never);

      renderDashboard();
      await screen.findByRole("button", { name: "Support chat" });

      // Bounded retries elapse before the error surfaces.
      expect(
        await screen.findByText(
          /synthetic engineering details could not be loaded/i,
          {},
          { timeout: 10_000 },
        ),
      ).toBeInTheDocument();
      // Core records stay available — a failed enrichment is not a failed page.
      expect(screen.getAllByText("Ana Reyes").length).toBeGreaterThan(0);
    }, 15_000);
  });

  describe("empty states", () => {
    it("tells the patient who can add medications when the log is empty", async () => {
      mocked.getMyReportCore.mockResolvedValue(CORE_REPORT as never);
      renderDashboard();

      expect(
        await screen.findByText(/No medications recorded — your care team can add these/i),
      ).toBeInTheDocument();
    });
  });

  describe("malformed API responses", () => {
    it("renders when the report omits every optional collection", async () => {
      // The report endpoint has returned partial payloads; missing arrays must
      // not blank the portal.
      mocked.getMyReportCore.mockResolvedValue({
        patient_id: "P001",
        patient_name: "Ana Reyes",
      } as never);

      renderDashboard();

      expect(await screen.findByRole("button", { name: "Support chat" })).toBeInTheDocument();
      expect(screen.getByText(/No medications recorded/i)).toBeInTheDocument();
    });

    it("falls back to the report's chat history when the history endpoint fails", async () => {
      mocked.getMyReportCore.mockResolvedValue({
        ...CORE_REPORT,
        chat_history: [{ role: "assistant", message: "Recorded your symptom.", timestamp: "2026-01-01T00:00:00Z" }],
      } as never);
      mocked.getMyChatHistory.mockRejectedValue(new Error("history unavailable"));

      renderDashboard();

      expect(await screen.findByText(/Recorded your symptom/)).toBeInTheDocument();
    });

    it("survives a chat history payload with no messages array", async () => {
      mocked.getMyReportCore.mockResolvedValue(CORE_REPORT as never);
      mocked.getMyChatHistory.mockResolvedValue({} as never);

      renderDashboard();
      expect(await screen.findByRole("button", { name: "Support chat" })).toBeInTheDocument();
    });
  });

  describe("navigation", () => {
    it("renders the support chat surface on the chat route", async () => {
      mocked.getMyReportCore.mockResolvedValue(CORE_REPORT as never);
      renderDashboard("/patient/chat");

      // The composer is the defining element of the chat tab.
      expect(await screen.findByRole("textbox")).toBeInTheDocument();
    });

    it("exposes both tabs as controls from the overview", async () => {
      mocked.getMyReportCore.mockResolvedValue(CORE_REPORT as never);
      renderDashboard();

      expect(await screen.findByRole("button", { name: "Overview" })).toBeInTheDocument();
      expect(screen.getByRole("button", { name: "Support chat" })).toBeInTheDocument();
    });
  });
});

describe("ToastProvider integration", () => {
  it("mounts the dashboard inside providers without throwing", () => {
    mocked.getMyReportCore.mockReturnValue(pending());
    const wrapper = ({ children }: { children: ReactNode }) => (
      <MemoryRouter>
        <ToastProvider>
          <AuthProvider>{children}</AuthProvider>
        </ToastProvider>
      </MemoryRouter>
    );
    expect(() => render(<PatientDashboard />, { wrapper })).not.toThrow();
  });
});
