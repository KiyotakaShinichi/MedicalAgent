from backend.services.mle_readiness_checks.core import check


def lineage_leakage_holdout_checks(lineage, leakage_audit, locked_holdout):
    lineage = lineage or {}
    leakage_audit = leakage_audit or {}
    locked_holdout = locked_holdout or {}
    return [
        check(
            name="dataset_lineage_manifest_present",
            category="lineage",
            status="passed" if lineage.get("dataset_hash") else "unideal",
            value={
                "dataset_hash": lineage.get("dataset_hash"),
                "schema_version": lineage.get("schema_version"),
            },
            threshold="dataset hash and schema version recorded",
            meaning="Dataset hashes, generation seeds, schema signatures, and feature lineage make runs reproducible.",
            hard_gate=False,
            remediation="Run python scripts/generate_mle_maturity_artifacts.py.",
        ),
        check(
            name="temporal_leakage_audit_passed",
            category="lineage",
            status="passed" if leakage_audit.get("status") == "passed" else "failed",
            value=leakage_audit.get("status") or "missing",
            threshold="passed",
            meaning="Checks that final outcomes and future-only fields are not model inputs.",
            hard_gate=True,
            remediation="Inspect Data/complete_synthetic_training/leakage_audit/temporal_leakage_audit.json.",
        ),
        check(
            name="locked_holdout_manifest_present",
            category="lineage",
            status="passed" if locked_holdout.get("locked_holdout_patients") else "unideal",
            value={
                "locked_holdout_patients": locked_holdout.get("locked_holdout_patients"),
                "seed": locked_holdout.get("seed"),
                "dataset_hash": locked_holdout.get("dataset_hash"),
            },
            threshold="frozen patient-level holdout split recorded",
            meaning="A frozen holdout prevents tuning directly against every synthetic test artifact.",
            hard_gate=False,
            remediation="Run python scripts/generate_mle_maturity_artifacts.py.",
        ),
    ]


def realism_checks(realism_report: dict) -> list[dict]:
    if not realism_report or realism_report.get("status") == "unavailable":
        return [
            check(
                name="synthetic_realism_report_present",
                category="realism",
                status="unideal",
                value="missing",
                threshold="realism report generated",
                meaning="A realism audit compares synthetic distributions to basic clinical thresholds and external baselines.",
                hard_gate=False,
                remediation="Run python scripts/run_synthetic_realism_report.py to generate a report.",
            )
        ]

    sim_to_real = realism_report.get("sim_to_real_comparison") or {}
    sim_status = sim_to_real.get("status") or realism_report.get("status")
    return [
        check(
            name="synthetic_realism_report_present",
            category="realism",
            status="passed",
            value={
                "training_patients": realism_report.get("training_patients"),
                "training_rows": realism_report.get("training_rows"),
            },
            threshold="report present",
            meaning="Synthetic realism should be audited before treating a PoC as MLE-ready.",
            hard_gate=False,
            remediation="Regenerate training data and rerun the realism audit.",
        ),
        check(
            name="sim_to_real_gap_review",
            category="realism",
            status=sim_status or "unavailable",
            value={
                "status": sim_status,
                "comparisons": sim_to_real.get("comparisons"),
            },
            threshold="KS/JS divergence within acceptable bounds",
            meaning="Sim-to-real checks flag distribution gaps that may limit external validity.",
            hard_gate=False,
            remediation="Tune the synthetic generator to align age, subtype, and baseline size distributions.",
        ),
    ]
