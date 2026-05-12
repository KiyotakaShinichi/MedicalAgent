import { createContext } from "react";
import type { Role } from "../types/api";

export interface AuthState {
  token: string | null;
  role: Role | null;
  patientId: string | null;
}

export interface AuthContextValue extends AuthState {
  setSession: (token: string, role: Role, patientId: string | null) => void;
  clearSession: () => void;
}

export const TOKEN_KEYS: Record<Role, string> = {
  patient: "patientPortalAccessToken",
  clinician: "clinicianAccessToken",
  admin: "adminAccessToken",
};

export function loadSession(): AuthState {
  for (const [role, key] of Object.entries(TOKEN_KEYS)) {
    const token = localStorage.getItem(key);
    if (token) {
      return {
        token,
        role: role as Role,
        patientId: localStorage.getItem("currentPatientId"),
      };
    }
  }
  return { token: null, role: null, patientId: null };
}

export const AuthContext = createContext<AuthContextValue | null>(null);
