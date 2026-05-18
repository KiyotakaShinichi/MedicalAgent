# Genetic Mutation Predictor Boundary

Genetic information can be included in OncoTrack only as structured record context and clinician/genetic-counselor review support.

## How Genetic Mutations Are Detected

Inherited or germline variants are usually detected from a blood, saliva, or cheek-swab sample. Tumor or somatic variants are usually detected from tumor tissue obtained through biopsy/surgery, liquid biopsy, or molecular pathology testing. Ordinary MRI, CT, ultrasound, or mammogram images do not directly diagnose a BRCA1/BRCA2/PALB2/TP53/PTEN/CHEK2/ATM mutation.

## How OncoTrack Should Use Them

Allowed:

- organize genetic-test records,
- track whether the test was germline, somatic, tumor sequencing, or unknown,
- store gene, variant text, classification, sample type, report date, and review status,
- route VUS, pathogenic/likely pathogenic, or unclear results for clinician/genetic-counselor review,
- use synthetic genetic-context features in offline ML ablation only.

Blocked:

- diagnosing inherited cancer risk,
- treating a VUS as positive,
- predicting a relative's cancer risk,
- recommending treatment changes from a variant,
- inferring mutation status from imaging.

## Predictor Weighting Policy

For response monitoring, direct longitudinal signals should carry more weight than genetic context:

1. Imaging trend and clinician-reviewed imaging report summaries.
2. CBC/lab trajectory, symptoms, treatment cycle timing, dose delays, and interventions.
3. Biomarker/pathology context such as ER, PR, HER2, and Ki-67.
4. Genetic-test record context and family-history readiness.
5. Tumor-marker trends as review context only, not standalone proof.

This is an engineering policy, not clinical advice. It should be reviewed by an oncology clinician/genetic counselor before any real-world use.
