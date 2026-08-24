#!/usr/bin/env sh
#
# End-to-end verification that a fresh clone of this repository works.
#
#   sh scripts/verify_fresh_clone.sh
#
# Run it on a clean checkout with no .venv, no node_modules, and no .env. It
# bootstraps both stacks, runs the backend suite, builds the frontend, and
# prints FRESH CLONE OK only if every step succeeded.
#
# Why the backend suite runs through the verifier
# -----------------------------------------------
# `scripts/check_fresh_clone_offline.py --full-suite` invokes exactly the
# pytest command printed below and adds the accounting a bare pytest run cannot
# do: it strips every live provider credential, refuses to inherit
# NLCARE_ALLOW_TEST_NETWORK, and fails if any test skipped because a network or
# credential was unavailable. A skip is invisible in a green summary line, and
# that is how a suite quietly stops testing what it claims to.
#
# The suite therefore runs once, here, and CI runs this script rather than
# repeating it.

set -eu

ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$ROOT"

echo "==> 1/4 Bootstrapping backend and frontend"
python scripts/bootstrap.py --with-frontend

# The pytest command is written out in full rather than held in a variable, so
# the command that runs is the command you read - here and in any tool that
# scans this file.
echo "==> 2/4 Backend suite (hermetic: no network, no credentials)"
python scripts/check_fresh_clone_offline.py \
  --full-suite \
  --pytest-command "python -m pytest tests -q --cov=backend --cov-branch --cov-fail-under=60 --cov-report=term-missing:skip-covered" \
  --json-output Data/test_tmp/fresh_clone_full_suite.json

echo "==> 3/4 Frontend production build"
cd frontend-react
npm ci
npm run build
cd "$ROOT"

echo "==> 4/4 Done"
echo "FRESH CLONE OK"
