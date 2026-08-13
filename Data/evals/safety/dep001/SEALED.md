# Sealed DEP-001 evidence

`final_holdout_safety_bank.jsonl` and `final_holdout_manifest.json` are sealed
historical evidence. DEP-001A development, training, calibration, threshold
selection, and feature engineering must not read or import their case content.

Only a one-way integrity process may verify the frozen SHA-256 and exact-hash
non-overlap. It must emit counts only and must never expose holdout text,
nearest examples, tokens, labels, or per-case failures to development code.

The old holdout must not be rerun for optimization. A new eligible external
human no-read holdout is required after DEP-001A reaches internal readiness.
