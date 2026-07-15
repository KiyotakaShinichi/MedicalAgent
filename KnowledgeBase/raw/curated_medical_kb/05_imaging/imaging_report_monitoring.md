# Imaging Report Monitoring: MRI, CT, Ultrasound, and Response Language

## Purpose

This document explains how imaging report text may be tracked in NLCare. The system does not read scans like a radiologist and does not diagnose treatment response or metastasis. It can store and summarize report wording for clinician review.

## Breast MRI

Breast MRI reports may describe lesion size, enhancement patterns, response compared with baseline, and lymph node findings. Baseline MRI, interim MRI, and end-of-course MRI can be used to monitor changes over time. MRI findings require radiologist and clinician interpretation.

## Ultrasound

Breast ultrasound reports may describe a mass, cystic or solid features, location, size, margins, lymph nodes, and BI-RADS assessment when reported. Ultrasound text can contribute to the timeline, but it is not a standalone diagnosis in NLCare.

## CT and PET/CT

CT or PET/CT reports may mention lesions, nodules, lymph nodes, effusion, ascites, liver findings, bone lesions, or other sites. NLCare can flag words that may need clinician review. It must not confirm metastasis.

## Response assessment wording

Reports may use terms such as complete response, partial response, stable disease, progression, interval decrease, interval increase, or residual disease. These terms should be presented as report wording, not as the assistant's conclusion.

## Pathology confirmation

Imaging response is not the same as pathologic complete response. pCR requires pathology assessment after surgery in the appropriate clinical context.

## Safe wording examples

- "The report text says interval decrease; a clinician should interpret what that means for your care."
- "I can save the CT report wording and flag it for review."
- "This system cannot confirm metastasis from report text."

## Sources

- NCI Breast Cancer Treatment PDQ: https://www.cancer.gov/types/breast/hp/breast-treatment-pdq
- American College of Radiology BI-RADS overview: https://www.acr.org/Clinical-Resources/Reporting-and-Data-Systems/Bi-Rads
- The Cancer Imaging Archive QIN-Breast collection: https://www.cancerimagingarchive.net/collection/qin-breast/

Last reviewed: 2026-05-15
