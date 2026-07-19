# Dependency reproducibility

`requirements-lock.txt` pins the direct Python dependencies resolved for the
July 2026 engineering environment. `python scripts/run_dependency_lock_audit.py`
checks direct lock coverage and reports packages missing or drifting in the
current interpreter.

This snapshot is intentionally described as a direct-dependency lock. It does
not fully pin the transitive dependency graph, perform vulnerability scanning,
prove secure deployment, establish compliance, or make NLCare healthcare
production ready. A deployment pipeline should additionally build from a
reviewed container digest, produce an SBOM, scan transitive packages and the
container, sign the image, and verify it in a separate environment.
