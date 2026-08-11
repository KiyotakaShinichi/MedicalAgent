# Architecture Decision Records (ADRs)

This folder captures the decisions that **actually moved the
architecture** — the ones a future contributor needs to understand
before changing the surrounding code, and the ones that would be
expensive to reverse silently.

Format: each ADR is one short markdown file. Status is one of
`accepted`, `superseded`, or `deprecated`. New ADRs use the next
sequential number — never reuse a number even after deprecation.

## Index

| # | Title | Status |
|---:|---|---|
| 0001 | [agent_rag.py god-module split into 15 focused modules](0001-agent-rag-split.md) | accepted |
| 0002 | [Tier-aware LLM adjudication (70B router · 120B answer)](0002-tier-aware-adjudication.md) | accepted |
| 0003 | [FAST_MODE escape hatch + admin runtime toggle](0003-fast-mode-escape-hatch.md) | accepted |
| 0004 | [Per-turn trace stores decisions, never chain-of-thought](0004-no-chain-of-thought-in-trace.md) | accepted |
| 0005 | [Adversarial bank held-out variant set with anti-contamination test](0005-adversarial-holdout-variants.md) | accepted |
| 0006 | [Release-gate informational tier + warn-on-regression floors](0006-release-gate-informational-tier.md) | accepted |
| 0007 | [Slim clinical-boundary strip · full text always in DOM](0007-slim-clinical-boundary-strip.md) | accepted |
| 0008 | [Composer "+" attachment popover replaces 8-chip tray](0008-composer-attachment-popover.md) | accepted |
| 0009 | [Source-alias normalisation for the frozen retrieval goldset](0009-source-alias-normalization.md) | accepted |
| 0010 | [Separate SaaS control plane from the synthetic patient demo](0010-saas-control-plane-boundary.md) | accepted |

## When to write a new ADR

- A decision is **structural** (it affects how multiple modules talk).
- A decision is **load-bearing** (other decisions assume it).
- A decision **trades one risk for another** (not just "we picked X
  for convenience").
- A decision **enforces a boundary** that future contributors might
  weaken without realising.

Skip ADRs for: refactors that don't change interfaces, dependency
bumps, formatting, comment edits.
