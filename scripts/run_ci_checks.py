import subprocess
import sys


COMMANDS = [
    [sys.executable, "-m", "unittest", "tests.test_breast_monitoring"],
    [sys.executable, "scripts/evaluate_agent_rag.py"],
    [sys.executable, "scripts/run_summary_quality_eval.py"],
    [sys.executable, "scripts/run_mle_checks.py"],
]


def main():
    for command in COMMANDS:
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
