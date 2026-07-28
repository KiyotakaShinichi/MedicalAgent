# Dependency reproducibility

`requirements-lock.txt` pins direct Python dependencies. The generated
`requirements-lock-py314-win.txt` captures every installed distribution for the
known-good CPython 3.14 Windows engineering environment, together with its
interpreter/platform fingerprint. Refresh and audit it from that environment:

```powershell
.\.venv\Scripts\python.exe scripts\run_dependency_lock_audit.py --refresh-transitive-lock
```

The audit reports direct-lock drift, transitive graph drift, missing or extra
distributions, the lock SHA-256, and an environment-fingerprint mismatch.

This transitive lock is intentionally platform-scoped. The lock audit itself
does not reproduce the Python 3.11 Linux CI environment or prove secure
deployment. Run `python scripts/run_dependency_security_scan.py` separately to
audit the Windows Python lock and frontend lockfile; advisory findings remain
release warnings until upgraded, mitigated, or explicitly accepted with scope.
A deployment pipeline still needs a separately generated Linux lock, reviewed
container digest, SBOM, transitive/container scans, image signing, and
verification in an isolated environment. None of these controls establishes
compliance or makes NLCare healthcare production ready.
