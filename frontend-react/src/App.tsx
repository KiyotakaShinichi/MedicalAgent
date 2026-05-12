import { lazy, Suspense } from "react";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { AuthProvider } from "./context/AuthContext";
import { RouteGuard } from "./components/layout/RouteGuard";
import { LoadingPane } from "./components/ui/Spinner";
import LoginPage from "./pages/Login";

const PatientDashboard = lazy(() => import("./pages/patient/PatientDashboard"));
const ClinicianDashboard = lazy(() => import("./pages/clinician/ClinicianDashboard"));
const AdminDashboard = lazy(() => import("./pages/admin/AdminDashboard"));

function RouteLoader() {
  return (
    <div className="route-loader">
      <LoadingPane label="Loading workspace..." />
    </div>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Suspense fallback={<RouteLoader />}>
          <Routes>
            <Route path="/login" element={<LoginPage />} />

            <Route
              path="/patient/*"
              element={
                <RouteGuard role="patient">
                  <PatientDashboard />
                </RouteGuard>
              }
            />

            <Route
              path="/clinician/*"
              element={
                <RouteGuard role="clinician">
                  <ClinicianDashboard />
                </RouteGuard>
              }
            />

            <Route
              path="/admin/*"
              element={
                <RouteGuard role="admin">
                  <AdminDashboard />
                </RouteGuard>
              }
            />

            <Route path="*" element={<Navigate to="/login" replace />} />
          </Routes>
        </Suspense>
      </BrowserRouter>
    </AuthProvider>
  );
}
