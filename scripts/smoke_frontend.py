from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


CHECKS = {
    "frontend/login.html": [
        "OncoTrack",
        "Demo Access",
        "Sign in to your workspace",
        "demo-credential-login",
        "patientPortalAccessToken",
        "adminAccessToken",
    ],
    "frontend/admin.html": [
        "OncoTrack",
        "RAG Evaluation",
        "Cost & Latency",
        "Security Guardrails",
        "Agent Feedback",
        "Agent Regression Suite",
        "runAgentRegression",
        "MLE Release Gates",
        "Release Readiness",
        "runMleReadiness",
    ],
    "frontend/patient.html": [
        "Oncology Monitor",
        "My Oncology Journey",
        "Enter to send",
        "Message the support agent",
        "Hybrid MLE",
        "msg-citations",
        "submitAgentFeedback",
    ],
    "frontend/index.html": [
        "OncoTrack",
        "Clinician",
        "Review Queue",
        "logoutClinician",
        "btn-ghost",
    ],
}


def main():
    missing = []
    for relative_path, expected_values in CHECKS.items():
        path = ROOT / relative_path
        if not path.exists():
            missing.append(f"{relative_path}: file missing")
            continue
        content = path.read_text(encoding="utf-8")
        for value in expected_values:
            if value not in content:
                missing.append(f"{relative_path}: missing {value!r}")
    if missing:
        for item in missing:
            print(item)
        raise SystemExit(1)
    print("Frontend smoke checks passed.")


if __name__ == "__main__":
    main()
