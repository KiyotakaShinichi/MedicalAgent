# Student-Constraint Elevation Plan

This plan lists controllable engineering upgrades under student-accessible constraints. It is not clinical validation and does not replace clinician review or real-world outcome evidence.

## Highest Leverage Steps

1. **External distribution alignment from exported cBioPortal rows**
   - Why: Moves beyond dataset-name mapping into actual public-row distribution checks.
   - Proof artifact: `Data/evals/models/latest_external_distribution_alignment.json`

2. **Common-feature model transfer stress test**
   - Why: Train on one source and evaluate feature behavior on another without pretending labels match.
   - Proof artifact: `Data/evals/models/latest_common_feature_transfer_stress.json`

3. **Synthetic realism candidate generator tuned against public distributions**
   - Why: Creates a separate candidate dataset with documented improvements and regressions.
   - Proof artifact: `Data/evals/models/latest_public_distribution_realism_candidate.json`

4. **Human-review simulation packet with blinded rubric**
   - Why: Prepares for eventual nurse/clinician review while letting non-clinicians audit clarity and safety boundaries now.
   - Proof artifact: `future docs/reviewer_packet_blinded/`

5. **Model behavior cards per head**
   - Why: Separates classification, regression, toxicity, genetics, and tumor-marker behavior so reviewers do not confuse scope.
   - Proof artifact: `future docs/model_behavior_cards/`

6. **RAG answer provenance snapshots**
   - Why: Exports trace replay into compact reviewer-facing before/after examples.
   - Proof artifact: `future Data/evals/rag/latest_trace_replay_gallery.json`

## Do Not Do Yet

- Do not claim real-world response prediction.
- Do not promote toxicity target v2 beyond review-priority experiment.
- Do not use TCGA/METABRIC survival labels as if they were pCR or NLCare response-score labels.
- Do not train patient-facing treatment recommendations.
