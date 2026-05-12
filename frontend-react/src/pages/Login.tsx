import { useState, type FormEvent } from "react";
import { useNavigate } from "react-router-dom";
import {
  Activity,
  ArrowRight,
  BarChart3,
  Database,
  Lock,
  ShieldCheck,
  Stethoscope,
  User,
  Users,
} from "lucide-react";
import { login } from "../api/client";
import { useAuth } from "../hooks/useAuth";
import { Button } from "../components/ui/Button";
import type { Role } from "../types/api";

const DEMOS = [
  {
    label: "Patient P001",
    username: "P001",
    password: "patient-demo",
    hint: "Own records, support chat, timeline",
    icon: User,
  },
  {
    label: "Patient P002",
    username: "P002",
    password: "patient-demo",
    hint: "Second scoped patient account",
    icon: User,
  },
  {
    label: "Clinician",
    username: "clinician",
    password: "clinician-demo",
    hint: "Review queue and approvals",
    icon: Stethoscope,
  },
  {
    label: "Admin / MLE",
    username: "admin",
    password: "admin-demo",
    hint: "Model, RAG, safety analytics",
    icon: BarChart3,
  },
];

const platformSignals = [
  { label: "Guardrailed RAG", value: "33/33 eval pass" },
  { label: "Safety regression", value: "1.0 attack block rate" },
  { label: "Human oversight", value: "approve/edit/reject" },
];

export default function LoginPage() {
  const { setSession } = useAuth();
  const navigate = useNavigate();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      const res = await login(username.trim(), password);
      setSession(res.access_token, res.role as Role, res.patient_id);
      if (res.role === "patient") navigate("/patient");
      else if (res.role === "clinician") navigate("/clinician");
      else navigate("/admin");
    } catch (err: unknown) {
      setError((err as Error).message || "Login failed");
    } finally {
      setLoading(false);
    }
  }

  function fillDemo(usernameValue: string, passwordValue: string) {
    setUsername(usernameValue);
    setPassword(passwordValue);
    setError(null);
  }

  return (
    <main className="login-page">
      <section className="login-shell" aria-label="MedicalAgent sign in">
        <div className="login-hero">
          <div>
            <div className="login-brand">
              <span className="login-brand-mark">
                <Activity size={24} aria-hidden="true" />
              </span>
              <div>
                <p className="login-brand-kicker">MedicalAgent</p>
                <h1>OncoTrack</h1>
              </div>
            </div>

            <div className="login-copy">
              <p className="login-eyebrow">Safety-first oncology monitoring POC</p>
              <h2>One role-aware gateway for patients, clinicians, and MLE review.</h2>
              <p>
                Explore a breast cancer monitoring workflow with patient-scoped records,
                clinician review, guardrailed RAG, model evaluation, and audit-ready
                safety signals. The system supports review and education only; it does
                not diagnose or recommend treatment.
              </p>
            </div>
          </div>

          <div className="login-signal-grid">
            {platformSignals.map((item) => (
              <div className="login-signal" key={item.label}>
                <span>{item.label}</span>
                <strong>{item.value}</strong>
              </div>
            ))}
          </div>

          <div className="login-guardrail">
            <ShieldCheck size={18} aria-hidden="true" />
            <span>
              Deterministic safety checks run before LLM reasoning. Clinicians remain
              the decision-makers.
            </span>
          </div>
        </div>

        <div className="login-panel">
          <div className="login-panel-header">
            <div>
              <p className="login-eyebrow">Secure demo access</p>
              <h2>Sign in</h2>
            </div>
            <span className="login-lock">
              <Lock size={18} aria-hidden="true" />
            </span>
          </div>

          <form onSubmit={handleSubmit} className="login-form">
            <label className="login-field" htmlFor="username">
              <span>Username</span>
              <div className="login-input-wrap">
                <User size={17} aria-hidden="true" />
                <input
                  id="username"
                  type="text"
                  autoComplete="username"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  required
                  placeholder="P001, clinician, or admin"
                />
              </div>
            </label>

            <label className="login-field" htmlFor="password">
              <span>Password</span>
              <div className="login-input-wrap">
                <Lock size={17} aria-hidden="true" />
                <input
                  id="password"
                  type="password"
                  autoComplete="current-password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                  placeholder="Enter demo password"
                />
              </div>
            </label>

            {error && <p className="login-error">{error}</p>}

            <Button
              type="submit"
              variant="primary"
              size="lg"
              loading={loading}
              className="login-submit"
              icon={!loading ? <ArrowRight size={17} aria-hidden="true" /> : undefined}
            >
              Sign in to workspace
            </Button>
          </form>

          <div className="login-demo-section">
            <div className="login-demo-heading">
              <span>Demo credentials</span>
              <small>Click one to fill the form</small>
            </div>
            <div className="login-demo-grid">
              {DEMOS.map((demo) => {
                const Icon = demo.icon;
                const selected = username === demo.username && password === demo.password;
                return (
                  <button
                    type="button"
                    key={demo.label}
                    onClick={() => fillDemo(demo.username, demo.password)}
                    className={`login-demo-card${selected ? " selected" : ""}`}
                  >
                    <span className="login-demo-icon">
                      <Icon size={17} aria-hidden="true" />
                    </span>
                    <span>
                      <strong>{demo.label}</strong>
                      <small>{demo.hint}</small>
                    </span>
                  </button>
                );
              })}
            </div>
          </div>

          <div className="login-note">
            <Database size={16} aria-hidden="true" />
            <span>
              Demo data may be synthetic. This app is not a diagnostic tool and is not
              validated for clinical deployment.
            </span>
          </div>
        </div>
      </section>

      <footer className="login-footer">
        <Users size={14} aria-hidden="true" />
        Role is inferred from credentials. Patients are routed to their own records;
        clinicians and admins are routed to their review surfaces.
      </footer>
    </main>
  );
}
