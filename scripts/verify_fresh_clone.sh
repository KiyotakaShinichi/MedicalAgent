#!/usr/bin/env sh
# Canonical end-to-end verification for a clean NLCare checkout.
# Dependency and safety-encoder provisioning may use the network. Every test
# command runs afterwards with NLCARE_TEST_OFFLINE=true and pytest blocks all
# non-loopback DNS/socket traffic.

set -eu

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p Data/test_tmp

command -v uv >/dev/null 2>&1 || {
  echo "uv is required (install uv==0.8.24)" >&2
  exit 1
}
command -v npm >/dev/null 2>&1 || {
  echo "npm is required" >&2
  exit 1
}

echo "==> 1/6 Locked dependency installation"
uv sync --frozen
(cd frontend-react && npm ci)

echo "==> 2/6 Rebuild local runtime assets from declared sources"
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 uv run python scripts/provision_semantic_safety_encoders.py
uv run python scripts/provision_semantic_safety_encoders.py --check-only --verify-runtimes
uv run python scripts/provision_derived_artifacts.py
uv run python scripts/provision_derived_artifacts.py --check-only

export NLCARE_TEST_OFFLINE=true
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LLM_ADJUDICATION_ENABLED=false
export RAG_FORCE_SPARSE=true
export RAG_ENABLE_CROSS_ENCODER=false
unset GROQ_API_KEY OPENAI_API_KEY AZURE_OPENAI_API_KEY GOOGLE_API_KEY GEMINI_API_KEY
unset PINECONE_API_KEY AZURE_SEARCH_API_KEY N8N_API_KEY N8N_WEBHOOK_URL

echo "==> 3/6 Backend static and repository contracts"
uv lock --check
uv run python scripts/check_dependency_contract.py
uv run python scripts/check_env_documentation.py
uv run ruff check backend scripts tests
uv run mypy
uv run python scripts/check_file_size.py backend --max-loc 500 --baseline tests/contracts/backend_authored_loc_baseline.json
uv run python scripts/check_file_size.py frontend-react/src --extensions .ts,.tsx,.css --max-loc 500 --baseline tests/contracts/frontend_authored_loc_baseline.json

echo "==> 4/6 Backend suite (hermetic, branch coverage floor 60%)"
uv run python scripts/check_fresh_clone_offline.py \
  --full-suite \
  --pytest-command "python -m pytest tests -q --cov=backend --cov-branch --cov-fail-under=60 --cov-report=term-missing:skip-covered" \
  --json-output Data/test_tmp/fresh_clone_full_suite.json

echo "==> 5/6 Frontend lint, typecheck, coverage, and build"
cd frontend-react
npm run lint
npm run typecheck
npm run test:coverage -- --reporter=json --outputFile=../Data/test_tmp/frontend_vitest_results.json
npm run build
cd "$ROOT"

echo "==> 6/6 Build ephemeral verification summary"
uv run python scripts/build_fresh_clone_summary.py \
  --backend-report Data/test_tmp/fresh_clone_full_suite.json \
  --frontend-report Data/test_tmp/frontend_vitest_results.json \
  --coverage-report frontend-react/coverage/coverage-summary.json \
  --output Data/test_tmp/fresh_clone_summary.json

echo "FRESH CLONE OK"
