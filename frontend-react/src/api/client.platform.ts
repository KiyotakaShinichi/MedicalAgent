import type * as Api from "../types/api";
import { BASE, del, get, getToken, post, request, responseError } from "./client.transport";

export const getPlatformSession = () =>
  get<Api.SaaSPlatformSession>("/platform/session");

export const createPlatformOrganization = (payload: { name: string; slug?: string }) =>
  post<import("../types/api").SaaSOrganization>("/platform/organizations", payload);

export const getWorkspaceOverview = (organizationId: string) =>
  request<Api.SaaSWorkspaceOverview>(
    "GET",
    `/platform/organizations/${organizationId}/overview`,
    undefined,
    { "X-NLCare-Organization-ID": organizationId },
  );

export const createWorkspaceProject = (
  organizationId: string,
  payload: { name: string; slug?: string; description?: string },
) => request<Api.SaaSProject>(
  "POST",
  `/platform/organizations/${organizationId}/projects`,
  payload,
  { "X-NLCare-Organization-ID": organizationId },
);

export const createWorkspaceJob = (
  organizationId: string,
  projectId: string,
  payload: { job_type: string; environment_id?: string; payload?: Record<string, unknown> },
  idempotencyKey: string,
) => request<{ job: Api.SaaSPlatformJob; idempotent_reuse: boolean }>(
  "POST",
  `/platform/organizations/${organizationId}/projects/${projectId}/jobs`,
  payload,
  {
    "X-NLCare-Organization-ID": organizationId,
    "Idempotency-Key": idempotencyKey,
  },
);

export const cancelWorkspaceJob = (organizationId: string, jobId: string) =>
  request<{ job: Api.SaaSPlatformJob }>(
    "DELETE",
    `/platform/organizations/${organizationId}/jobs/${jobId}`,
    undefined,
    { "X-NLCare-Organization-ID": organizationId },
  );

// Auth
export const login = (username: string, password: string) =>
  post<Api.LoginResponse>("/auth/demo-credential-login", { username, password });

export const getDemoPatients = () =>
  get<{ patients: Api.DemoPatient[] }>("/auth/demo-patients");

export const whoami = () =>
  get<{ role: string; patient_id: string | null }>("/auth/whoami");

export const logout = () =>
  del<{ revoked: boolean; message: string }>("/auth/session");

export async function getAuthenticatedObjectUrl(path: string): Promise<string> {
  const token = getToken();
  const res = await fetch(`${BASE}${path}`, {
    headers: {
      "X-NLCare-Data-Class": "synthetic",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    cache: "no-store",
  });
  if (!res.ok) {
    throw await responseError(res);
  }
  return URL.createObjectURL(await res.blob());
}

// Patient
