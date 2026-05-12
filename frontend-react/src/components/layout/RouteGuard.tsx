import { Navigate } from "react-router-dom";
import { useAuth } from "../../hooks/useAuth";
import type { Role } from "../../types/api";

interface RouteGuardProps {
  role: Role;
  children: React.ReactNode;
}

export function RouteGuard({ role, children }: RouteGuardProps) {
  const { token, role: sessionRole } = useAuth();
  if (!token) return <Navigate to="/login" replace />;
  if (sessionRole !== role) {
    const dest = sessionRole === "patient" ? "/patient" : sessionRole === "clinician" ? "/clinician" : "/admin";
    return <Navigate to={dest} replace />;
  }
  return <>{children}</>;
}
