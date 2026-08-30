import { beforeEach, describe, expect, it, vi } from "vitest";

const transport = vi.hoisted(() => ({
  get: vi.fn(),
  post: vi.fn(),
  del: vi.fn(),
  request: vi.fn(),
  getToken: vi.fn(),
  responseError: vi.fn(),
  BASE: "http://test.local",
}));

vi.mock("../../src/api/client.transport", () => transport);

import {
  cancelWorkspaceJob,
  createWorkspaceJob,
  getWorkspaceOverview,
  login,
} from "../../src/api/client.platform";
import {
  addMyLab,
  addMySymptom,
  getMyChatHistory,
  undoMyConfirmedRecordWrite,
} from "../../src/api/client.patient";
import {
  acknowledgeHighRiskConversationAlert,
  getPatientReport,
  submitSummaryReview,
} from "../../src/api/client.clinician";
import {
  getAgentTraceLogs,
  getClinicianPatientPredictionTraces,
  getPredictionTraces,
  probeAdminIntent,
  runCbioportalBiomarkerSchemaMapping,
  setAdminFastMode,
} from "../../src/api/client.admin";

describe("domain API contracts", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    transport.get.mockResolvedValue({});
    transport.post.mockResolvedValue({});
    transport.del.mockResolvedValue({});
    transport.request.mockResolvedValue({});
  });

  it("serializes demo login credentials through the shared transport", async () => {
    await login("P001", "patient-demo");
    expect(transport.post).toHaveBeenCalledWith("/auth/demo-credential-login", {
      username: "P001",
      password: "patient-demo",
    });
  });

  it("preserves organization and idempotency headers for workspace jobs", async () => {
    await createWorkspaceJob("org one", "project-a", { job_type: "rag_eval" }, "idem-7");
    expect(transport.request).toHaveBeenCalledWith(
      "POST",
      "/platform/organizations/org one/projects/project-a/jobs",
      { job_type: "rag_eval" },
      { "X-NLCare-Organization-ID": "org one", "Idempotency-Key": "idem-7" },
    );

    await getWorkspaceOverview("org one");
    expect(transport.request).toHaveBeenLastCalledWith(
      "GET",
      "/platform/organizations/org one/overview",
      undefined,
      { "X-NLCare-Organization-ID": "org one" },
    );

    await cancelWorkspaceJob("org one", "job-2");
    expect(transport.request).toHaveBeenLastCalledWith(
      "DELETE",
      "/platform/organizations/org one/jobs/job-2",
      undefined,
      { "X-NLCare-Organization-ID": "org one" },
    );
  });

  it("uses only patient-scoped /me routes for patient record operations", async () => {
    const symptom = { date: "2026-08-30", symptom: "nausea", severity: 4 };
    const lab = { date: "2026-08-30", wbc: 4.2, hemoglobin: 11.1, platelets: 180 };

    await getMyChatHistory();
    await addMySymptom(symptom);
    await addMyLab(lab);
    await undoMyConfirmedRecordWrite(17);

    expect(transport.get).toHaveBeenCalledWith("/me/chat");
    expect(transport.post).toHaveBeenCalledWith("/me/symptoms", symptom);
    expect(transport.post).toHaveBeenCalledWith("/me/labs", lab);
    expect(transport.del).toHaveBeenCalledWith("/me/record-write-actions/17");
    for (const call of [
      ...transport.get.mock.calls,
      ...transport.post.mock.calls,
      ...transport.del.mock.calls,
    ]) {
      expect(String(call[0])).not.toMatch(/\/patients?\//);
    }
  });

  it("keeps clinician patient identifiers in clinician-owned routes", async () => {
    const review = { decision: "approve", clinician_notes: "Reviewed." };
    await getPatientReport("P002");
    await submitSummaryReview("P002", review);

    expect(transport.get).toHaveBeenCalledWith("/patient-report/P002");
    expect(transport.post).toHaveBeenCalledWith("/patients/P002/summary-review", review);
  });

  it("uses the documented acknowledgement default without inventing clinical content", async () => {
    await acknowledgeHighRiskConversationAlert(8);
    expect(transport.post).toHaveBeenCalledWith(
      "/clinician/high-risk-conversation-alerts/8/acknowledge",
      { note: "Acknowledged from the clinician review dashboard." },
    );
  });

  it("serializes admin booleans and query limits exactly", async () => {
    await setAdminFastMode(null);
    await probeAdminIntent("Can you explain this?", false);
    await runCbioportalBiomarkerSchemaMapping(false);
    await getAgentTraceLogs(7);

    expect(transport.post).toHaveBeenNthCalledWith(1, "/admin/fast-mode", { enabled: null });
    expect(transport.post).toHaveBeenNthCalledWith(2, "/admin/intent-classifier-probe", {
      message: "Can you explain this?",
      use_llm: false,
    });
    expect(transport.post).toHaveBeenNthCalledWith(
      3,
      "/admin/cbioportal-biomarker-schema-mapping?live_fetch=false",
    );
    expect(transport.get).toHaveBeenCalledWith("/admin/agent-trace-logs?limit=7");
  });

  it("encodes patient IDs and optional prediction-trace filters", async () => {
    await getClinicianPatientPredictionTraces("P 001/unsafe", {
      limit: 12,
      abstained_only: true,
    });
    expect(transport.get).toHaveBeenCalledWith(
      "/clinician/patients/P%20001%2Funsafe/prediction-traces?limit=12&abstained_only=true",
    );

    await getPredictionTraces({
      limit: 20,
      patient_id: "P001",
      decision: "review required",
      abstained_only: true,
    });
    expect(transport.get).toHaveBeenLastCalledWith(
      "/admin/prediction-traces?limit=20&patient_id=P001&decision=review+required&abstained_only=true",
    );
  });
});
