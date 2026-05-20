"""Pytest bootstrap for local console-script invocations.

Some Windows environments invoke ``pytest`` through a console script that
does not put the repository root on ``sys.path``.  The project release gate
documents ``pytest tests/test_breast_monitoring.py -q`` as the smoke command,
so keep that exact entrypoint reliable.
"""
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
