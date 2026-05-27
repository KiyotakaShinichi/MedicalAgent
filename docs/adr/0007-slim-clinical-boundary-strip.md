# 0007 — Slim clinical-boundary strip · full disclaimer always in DOM

**Status**: accepted

## Context

The clinical-boundary banner is the single most safety-load-bearing
piece of UI in the patient surface. The user-facing wording is:

> Engineering prototype only. Not clinically validated. Not for real
> patient care. No clinician approval. Synthetic-only ML signals.
> Outputs must not be used for diagnosis, treatment, prognosis,
> genetic-risk interpretation, tumor-marker interpretation, or
> medication decisions.

The original implementation was a 100px-tall red block that the user
described as "a giant visual block". Reducing the visual weight is OK;
weakening the wording is not.

## Decision

A two-state strip:

1. **Collapsed** (default): single line, red border, eyebrow
   `Clinical boundary`, short headline `Engineering prototype only.
   Not clinically validated. Synthetic-only ML signals.`, chevron.
2. **Expanded**: full disclaimer text rendered below the strip.

Implementation rules:

- The full disclaimer string is **always present in the DOM** (using
  `hidden={!open}`, not conditional render), so screen readers and
  page-scrape tooling see it on every render.
- The headline is a *shorter* phrasing, NEVER a different claim.
- The chevron is keyboard-operable + has `aria-expanded`/`aria-controls`.

## Consequences

- ✅ Visual register matches the rest of the KPI-dashboard UI.
- ✅ Full disclaimer is one click or one screen-reader pass away.
- ✅ A reviewer scanning HTML never sees a "missing disclaimer" page.
- ⚠ The strip is now small enough that a designer might try to remove
  it. The wording is duplicated in `ClinicalBoundaryBanner.tsx` AND
  the PoC footnote at the dashboard bottom. Two-source redundancy is
  intentional.

## Reversal cost

Low — switch to the old full banner. Don't make the wording shorter
than the current "Engineering prototype only. Not clinically validated.
Synthetic-only ML signals." headline.
