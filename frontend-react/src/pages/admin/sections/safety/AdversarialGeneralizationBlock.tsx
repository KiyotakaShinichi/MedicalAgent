import { MetricCard } from "../../../../components/ui/MetricCard";
import { EmptyPane } from "../../../../components/ui/Spinner";
import { EvalIntegrityFooter } from "../../../../components/ui/EvalIntegrityFooter";
import {
  coerceIntegrityStatus,
  fmtRate,
  readArray,
  readNumber,
  readRecord,
  readString,
} from "./safetyFormat";

/**
 * Adversarial generalization (v2) held-out evaluation.
 *
 * The artifact shape is owned by an eval script rather than the API schema, so
 * it arrives as an untyped record. Every field is read through the guarded
 * `read*` helpers — a shape change upstream degrades individual metrics to "—"
 * instead of crashing the admin dashboard.
 *
 * The "Not solved: yes" tile is intentional and hard-coded. This benchmark has
 * known open failures, and the surface states that unconditionally rather than
 * letting a good pass rate imply the problem is closed.
 */
export function AdversarialGeneralizationBlock({ artifact }: { artifact?: Record<string, unknown> }) {
  if (!artifact || artifact.status === "not_generated") {
    return (
      <EmptyPane label="No adversarial generalization v2 artifact yet - run scripts/run_adversarial_generalization_v2_eval.py." />
    );
  }

  const metrics = readRecord(artifact, "metrics");
  const heldoutV2 = readRecord(artifact, "heldout_v2");
  const failures = readArray(heldoutV2, "failures");

  const heldoutV1Rate = readNumber(metrics, "heldout_v1_pass_rate");
  const heldoutV2Rate = readNumber(metrics, "heldout_v2_pass_rate");
  const leakageRate = readNumber(metrics, "unsafe_leakage_rate");
  const safeNegativeRate = readNumber(metrics, "safe_negative_control_pass_rate");
  const totalN = readNumber(heldoutV2, "total_n");

  return (
    <div className="flex flex-col gap-3">
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <MetricCard label="Original bank" value={fmtRate(readNumber(metrics, "original_bank_pass_rate"))} status="muted" />
        <MetricCard label="Heldout v1" value={fmtRate(heldoutV1Rate)} status={heldoutV1Rate != null && heldoutV1Rate >= 0.9 ? "green" : "amber"} />
        <MetricCard label="Heldout v2" value={fmtRate(heldoutV2Rate)} status={heldoutV2Rate != null && heldoutV2Rate >= 0.8 ? "green" : "amber"} />
        <MetricCard label="Unsafe leakage" value={fmtRate(leakageRate)} status={leakageRate === 0 ? "green" : "amber"} />
        <MetricCard label="Paraphrase" value={fmtRate(readNumber(metrics, "paraphrase_pass_rate"))} status="muted" />
        <MetricCard label="Safe negatives" value={fmtRate(safeNegativeRate)} status={safeNegativeRate === 1 ? "green" : "amber"} />
        <MetricCard label="Heldout v2 N" value={String(totalN ?? "—")} status="muted" />
        <MetricCard label="Not solved" value="yes" status="amber" />
      </div>

      {failures.length > 0 && (
        <p className="text-xs" style={{ color: "var(--text-faint)" }}>
          Remaining weak cases: {failures.length}. Review the artifact for category-level gaps before claiming robustness.
        </p>
      )}

      <EvalIntegrityFooter
        totalN={totalN ?? null}
        passCount={readNumber(heldoutV2, "pass_count") ?? null}
        failCount={readNumber(heldoutV2, "fail_count") ?? failures.length}
        skippedCount={readNumber(heldoutV2, "skipped_count") ?? 0}
        authorship="internal"
        clinicalValidation={false}
        wasUsedForTuning={false}
        status={coerceIntegrityStatus(readString(artifact, "status"))}
        artifactPath={readString(artifact, "artifact_path") ?? "Data/evals/safety/latest_adversarial_generalization_v2.json"}
        caveat="Internal heldout v2 — frozen synthetic adversarial bank. Engineering signal only; clinical generalization is not implied."
      />
    </div>
  );
}
