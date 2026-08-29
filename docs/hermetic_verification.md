# Hermetic Verification Contract

NLCare separates **core engineering verification** from **release evidence**.
The distinction prevents a genuine DEP-001 safety finding from being hidden or
misreported as a broken linter, type checker, or unit-test environment.

## Canonical command

From a clean checkout with Python 3.11, Node 20, and `uv==0.8.24` available:

```sh
sh scripts/verify_fresh_clone.sh
```

The command performs a frozen Python sync and `npm ci`, provisions the declared
local safety/runtime assets, then enables `NLCARE_TEST_OFFLINE=true`. It runs:

- dependency and environment contracts;
- backend and frontend file-size ratchets;
- Ruff and mypy;
- the complete backend suite with branch coverage and a 60% floor;
- frontend lint, typecheck, unit coverage, and production build;
- frontend coverage floors of 35% statements, 62% branches, 31% functions,
  and 35% lines.

It writes ephemeral run evidence to `Data/test_tmp/fresh_clone_summary.json`.
CI uploads that file; it is not a tracked evaluation artifact.

## Offline mode

`NLCARE_TEST_OFFLINE=true` is the canonical test-mode signal. The pytest
bootstrap also sets Hugging Face and Transformers offline flags, removes live
provider credentials, and blocks non-loopback DNS and socket connections.
Loopback traffic remains available for TestClient and local process tests.

Tests that genuinely contact an external system must use
`@pytest.mark.requires_network`. They are excluded from the default command and
must be invoked explicitly:

```sh
pytest -m requires_network
```

The current default suite has no required live-provider test. Groq, adjudication,
RAG generation, managed-vector, and automation tests use existing injected
functions/transports or deterministic local fallbacks. Production provider code
is not replaced by test fakes.

## Provisioning boundary

Dependency installation and initial semantic-safety encoder provisioning may use
the network before verification. Test execution itself must not download models
or contact Groq, OpenAI, Gemini, Pinecone, Azure Search, n8n, external databases,
or external HTTP services. Derived RAG and lakehouse artifacts are rebuilt
offline from tracked inputs.

Bicep installation, Playwright browser provisioning, release-evidence generation,
and the release gate remain in `.github/workflows/ship.yml`. They do not gate the
core CI workflow. Ship may remain red on the preserved DEP-001 behavioral
evidence; core CI must not bypass or rewrite that evidence.

## Scope

Passing this contract proves clean-checkout engineering reproducibility. It does
not prove clinical validity, deployment approval, real-world safety, or a green
release gate.
