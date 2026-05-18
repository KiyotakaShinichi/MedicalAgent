.PHONY: backend-tests frontend-vitest frontend-playwright-smoke frontend-lint-build release-gate ship

backend-tests:
	RAG_FORCE_SPARSE=true python -m pytest tests/test_breast_monitoring.py -q

frontend-vitest:
	cd frontend-react && npm run test

frontend-playwright-smoke:
	cd frontend-react && npm run test:e2e -- tests/e2e/smoke.spec.ts

frontend-lint-build:
	cd frontend-react && npm run lint && npm run build

release-gate:
	python scripts/run_release_gate.py

ship: backend-tests frontend-vitest frontend-playwright-smoke frontend-lint-build release-gate

ship-py:
	python scripts/ship.py
