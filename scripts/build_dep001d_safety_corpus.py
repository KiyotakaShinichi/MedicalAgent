from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.dep001d_safety_corpus import build_dep001d_corpora


if __name__ == "__main__":
    result = build_dep001d_corpora()
    print(f"DEP-001D development cases: {result['development_case_n']}")
    print(f"DEP-001D output-actionability cases: {result['output_actionability_case_n']}")
