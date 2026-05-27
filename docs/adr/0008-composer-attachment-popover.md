# 0008 — Composer "+" attachment popover replaces 8-chip tool tray

**Status**: accepted

## Context

The patient chat originally had an always-visible tool tray of 8 chips
(Log symptom · Save CBC · Save imaging · Log medication · Treatment
note · Upload CBC image · Upload MRI image · Ask a question). The
user described it as "too cluttered and unintuitive" and asked for a
ChatGPT-style attachment trigger.

The tray was also a discoverability mismatch: the patient's most
common ask is "let me type a question", not "open one of 8 forms".

## Decision

Move the tools into a single `+` button on the LEFT of the chat
composer. Clicking opens a popover with two labelled groups:

- **Record**: Log symptom · Save CBC · Save imaging · Log medication ·
  Treatment note (5 items)
- **Upload**: Upload CBC image · Upload MRI image (2 items)

"Ask a question" is no longer a tool — it's the composer textarea.

Implementation rules:

- The `+` is a real `<button>` with `aria-haspopup="menu"`,
  `aria-expanded`, `aria-controls`.
- ArrowUp/Down/Home/End/Enter/Space/Escape are all wired.
- Outside-click and Tab-out close; Escape returns focus to the trigger.
- The popover floats **above** the trigger (`bottom: calc(100% + 8px)`)
  because the composer lives at the bottom of the workspace.
- On mobile the popover width is `min(320px, calc(100vw - 32px))` so
  it cannot overflow horizontally.
- `ChatPanel.composerLeading?: ReactNode` is a generic slot — the
  clinician chat can reuse it later without the patient-only tools.

## Consequences

- ✅ The chat workspace lost ~80px of vertical real estate.
- ✅ All seven form/upload flows still work (hidden file inputs back
  the two upload items; the parent owns the modal state).
- ⚠ Discoverability of the seven tools dropped by one click. The
  popover description text on each item is the mitigation.

## Reversal cost

Low. Re-introduce the `ToolTray` from git history. Note that the
clinician chat does not currently use any leading element, so reverting
only affects the patient surface.
