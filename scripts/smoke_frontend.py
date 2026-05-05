from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


CHECKS = {
    "frontend/admin.html": [
        "RAG Evaluation & Guardrails",
        "API Cost & Latency",
        "Security Guardrails",
        "Agent Feedback",
    ],
    "frontend/patient.html": [
        "My Oncology Journey",
        "Rate answer",
        "msg-citations",
        "submitAgentFeedback",
    ],
    "frontend/index.html": [
        "Clinician",
        "Review Queue",
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
