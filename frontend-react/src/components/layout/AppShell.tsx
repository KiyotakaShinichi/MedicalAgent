import { useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { clsx } from "clsx";
import { Activity, ChevronLeft, LogOut, Menu, ShieldCheck } from "lucide-react";
import { useAuth } from "../../hooks/useAuth";
import type { LucideIcon } from "lucide-react";

interface NavItem {
  to: string;
  label: string;
  icon: LucideIcon;
}

interface AppShellProps {
  children: React.ReactNode;
  navItems: NavItem[];
  title: string;
  subtitle?: string;
}

export function AppShell({ children, navItems, title, subtitle }: AppShellProps) {
  const { clearSession, role, patientId } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const [sidebarOpen, setSidebarOpen] = useState(true);

  function handleLogout() {
    clearSession();
    navigate("/login");
  }

  const roleLabel =
    role === "patient" ? `Patient ${patientId ?? ""}` :
    role === "clinician" ? "Clinician" :
    role === "admin" ? "Admin / MLE" : "Workspace";
  const activePath = [...navItems]
    .sort((a, b) => b.to.length - a.to.length)
    .find(({ to }) => location.pathname === to || location.pathname.startsWith(`${to}/`))
    ?.to;

  return (
    <div className="app-shell">
      <aside className={clsx("app-sidebar", sidebarOpen ? "is-open" : "is-collapsed")}>
        <div className="app-sidebar-brand">
          <span className="app-sidebar-logo">
            <Activity size={22} aria-hidden="true" />
          </span>
          {sidebarOpen && (
            <div className="app-sidebar-brand-text">
              <strong>OncoTrack</strong>
              <span>Breast monitoring</span>
            </div>
          )}
          <button
            type="button"
            onClick={() => setSidebarOpen((open) => !open)}
            className="app-sidebar-toggle"
            aria-label={sidebarOpen ? "Collapse sidebar" : "Expand sidebar"}
          >
            {sidebarOpen ? <ChevronLeft size={18} /> : <Menu size={18} />}
          </button>
        </div>

        <div className="app-sidebar-profile">
          <span className="app-sidebar-avatar">{roleLabel.slice(0, 1)}</span>
          {sidebarOpen && (
            <div>
              <strong>{roleLabel}</strong>
              <span>Demo secure session</span>
            </div>
          )}
        </div>

        <nav className="app-sidebar-nav" aria-label="Primary navigation">
          {navItems.map(({ to, label, icon: Icon }) => {
            const active = activePath === to;
            return (
              <Link
                key={to}
                to={to}
                className={clsx("app-nav-link", active && "is-active")}
                title={!sidebarOpen ? label : undefined}
              >
                <Icon size={18} aria-hidden="true" />
                {sidebarOpen && <span>{label}</span>}
              </Link>
            );
          })}
        </nav>

        <div className="app-sidebar-footer">
          {sidebarOpen && (
            <p>
              Monitoring support only. Not for diagnosis or treatment decisions.
            </p>
          )}
          <button type="button" onClick={handleLogout} className="app-logout-button">
            <LogOut size={17} aria-hidden="true" />
            {sidebarOpen && <span>Sign out</span>}
          </button>
        </div>
      </aside>

      <div className="app-frame">
        <header className="app-topbar">
          <div>
            <p className="app-topbar-kicker">{roleLabel}</p>
            <h1>{title}</h1>
            {subtitle && <span>{subtitle}</span>}
          </div>
          <div className="app-topbar-actions">
            <span className="app-safety-pill">
              <ShieldCheck size={15} aria-hidden="true" />
              PoC - not for clinical use
            </span>
          </div>
        </header>

        <main className="app-main">
          {children}
        </main>
      </div>
    </div>
  );
}
