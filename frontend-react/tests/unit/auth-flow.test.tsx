import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes, useLocation } from "react-router-dom";

const loginMock = vi.hoisted(() => vi.fn());
vi.mock("../../src/api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../src/api/client")>();
  return { ...actual, login: loginMock };
});

import { AuthProvider } from "../../src/context/AuthContext";
import { RouteGuard } from "../../src/components/layout/RouteGuard";
import { useAuth } from "../../src/hooks/useAuth";
import LoginPage from "../../src/pages/Login";

function SessionProbe() {
  const auth = useAuth();
  return (
    <div>
      <output>{`${auth.role ?? "none"}:${auth.patientId ?? "none"}:${auth.token ?? "none"}`}</output>
      <button onClick={() => auth.setSession("new-token", "patient", "P002")}>set</button>
      <button onClick={auth.clearSession}>clear</button>
    </div>
  );
}

function LocationProbe() {
  return <output>{useLocation().pathname}</output>;
}

function renderGuard(role: "patient" | "clinician" | "admin") {
  return render(
    <MemoryRouter initialEntries={["/protected"]}>
      <AuthProvider>
        <Routes>
          <Route path="/login" element={<><span>login route</span><LocationProbe /></>} />
          <Route path="/patient" element={<><span>patient route</span><LocationProbe /></>} />
          <Route path="/clinician" element={<><span>clinician route</span><LocationProbe /></>} />
          <Route path="/admin" element={<><span>admin route</span><LocationProbe /></>} />
          <Route path="/protected" element={<RouteGuard role={role}><span>protected content</span></RouteGuard>} />
        </Routes>
      </AuthProvider>
    </MemoryRouter>,
  );
}

describe("authentication and route boundaries", () => {
  beforeEach(() => {
    sessionStorage.clear();
    loginMock.mockReset();
  });

  afterEach(() => sessionStorage.clear());

  it("restores an existing patient session and clears every role token on logout", async () => {
    sessionStorage.setItem("patientPortalAccessToken", "saved-token");
    sessionStorage.setItem("adminAccessToken", "stale-admin-token");
    sessionStorage.setItem("currentPatientId", "P001");
    const user = userEvent.setup();
    render(<AuthProvider><SessionProbe /></AuthProvider>);

    expect(screen.getByText("patient:P001:saved-token")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "clear" }));
    expect(screen.getByText("none:none:none")).toBeInTheDocument();
    expect(sessionStorage.length).toBe(0);
  });

  it("replaces stale role credentials when a new session is established", async () => {
    sessionStorage.setItem("adminAccessToken", "old-admin");
    const user = userEvent.setup();
    render(<AuthProvider><SessionProbe /></AuthProvider>);

    await user.click(screen.getByRole("button", { name: "set" }));
    expect(sessionStorage.getItem("adminAccessToken")).toBeNull();
    expect(sessionStorage.getItem("patientPortalAccessToken")).toBe("new-token");
    expect(sessionStorage.getItem("currentPatientId")).toBe("P002");
  });

  it("redirects a missing session to login", () => {
    renderGuard("patient");
    expect(screen.getByText("login route")).toBeInTheDocument();
    expect(screen.queryByText("protected content")).not.toBeInTheDocument();
  });

  it("redirects a mismatched role to its own workspace", () => {
    sessionStorage.setItem("clinicianAccessToken", "clinician-token");
    renderGuard("admin");
    expect(screen.getByText("clinician route")).toBeInTheDocument();
    expect(screen.queryByText("protected content")).not.toBeInTheDocument();
  });

  it("renders protected content only for the required role", () => {
    sessionStorage.setItem("adminAccessToken", "admin-token");
    renderGuard("admin");
    expect(screen.getByText("protected content")).toBeInTheDocument();
  });

  it("fills demo credentials without submitting them", async () => {
    const user = userEvent.setup();
    render(<MemoryRouter><AuthProvider><LoginPage /></AuthProvider></MemoryRouter>);

    await user.click(screen.getByRole("button", { name: /Patient P002/i }));
    expect(screen.getByLabelText("Username")).toHaveValue("P002");
    expect(screen.getByLabelText("Password")).toHaveValue("patient-demo");
    expect(loginMock).not.toHaveBeenCalled();
  });

  it("stores a successful session and routes by the backend role", async () => {
    loginMock.mockResolvedValue({ access_token: "token-1", role: "clinician", patient_id: null });
    const user = userEvent.setup();
    render(
      <MemoryRouter initialEntries={["/login"]}>
        <AuthProvider>
          <Routes>
            <Route path="/login" element={<LoginPage />} />
            <Route path="/clinician" element={<LocationProbe />} />
          </Routes>
        </AuthProvider>
      </MemoryRouter>,
    );

    await user.type(screen.getByLabelText("Username"), " clinician ");
    await user.type(screen.getByLabelText("Password"), "clinician-demo");
    await user.click(screen.getByRole("button", { name: /sign in/i }));

    expect(await screen.findByText("/clinician")).toBeInTheDocument();
    expect(loginMock).toHaveBeenCalledWith("clinician", "clinician-demo");
    expect(sessionStorage.getItem("clinicianAccessToken")).toBe("token-1");
  });

  it("shows a bounded login error without exposing an internal exception", async () => {
    loginMock.mockRejectedValue(new Error("POST https://internal-api.local/auth failed with stack secret"));
    const user = userEvent.setup();
    render(<MemoryRouter><AuthProvider><LoginPage /></AuthProvider></MemoryRouter>);

    await user.type(screen.getByLabelText("Username"), "P001");
    await user.type(screen.getByLabelText("Password"), "wrong");
    await user.click(screen.getByRole("button", { name: /sign in/i }));

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(/could not sign in/i);
    expect(alert).not.toHaveTextContent(/internal-api|stack secret/i);
    await waitFor(() => expect(screen.getByRole("button", { name: /sign in/i })).toBeEnabled());
  });
});
