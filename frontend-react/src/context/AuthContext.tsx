import React, { useState, useCallback } from "react";
import type { Role } from "../types/api";
import { AuthContext, TOKEN_KEYS, loadSession } from "./authContextCore";
import type { AuthState } from "./authContextCore";

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [state, setState] = useState<AuthState>(loadSession);

  const setSession = useCallback(
    (token: string, role: Role, patientId: string | null) => {
      Object.values(TOKEN_KEYS).forEach((k) => localStorage.removeItem(k));
      localStorage.setItem(TOKEN_KEYS[role], token);
      if (patientId) localStorage.setItem("currentPatientId", patientId);
      setState({ token, role, patientId });
    },
    []
  );

  const clearSession = useCallback(() => {
    Object.values(TOKEN_KEYS).forEach((k) => localStorage.removeItem(k));
    localStorage.removeItem("currentPatientId");
    setState({ token: null, role: null, patientId: null });
  }, []);

  return (
    <AuthContext.Provider value={{ ...state, setSession, clearSession }}>
      {children}
    </AuthContext.Provider>
  );
}
