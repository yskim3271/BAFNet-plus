"""Pytest configuration for the BAFNet+ test suite.

Adds the project root (the directory containing ``src/``) to ``sys.path`` so
tests can ``import src...`` regardless of the working directory pytest is
invoked from.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
