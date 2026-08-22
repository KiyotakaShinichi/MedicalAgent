"""NLCare backend package.

Intentionally empty of logic. The file exists so `backend` is a regular
package rather than an implicit namespace package: without it, tooling that
resolves modules by file path sees `backend/services/` as both `services` and
`backend.services`. That ambiguity is why `mypy` aborted with "Source file
found twice under different module names" as soon as any type-checked module
imported another `backend.*` module, and why every file in [tool.mypy] `files`
had to avoid such imports.
"""
