import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.services.llm_judge_eval import DEFAULT_OUTPUT_PATH, DEFAULT_RAG_EVAL_PATH, run_llm_judge_eval  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Run optional LLM-as-judge evaluation on RAG gold outputs.")
    parser.add_argument("--rag-eval-path", default=DEFAULT_RAG_EVAL_PATH)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--max-cases", type=int, default=30)
    args = parser.parse_args()
    report = run_llm_judge_eval(
        rag_eval_path=args.rag_eval_path,
        output_path=args.output_path,
        max_cases=args.max_cases,
    )
    print(json.dumps({
        "output_path": args.output_path,
        "status": report.get("status"),
        "provider": report.get("provider"),
        "summary": report.get("summary"),
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
