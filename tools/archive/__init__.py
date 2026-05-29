"""Archived compatibility scripts.

The original scripts used sibling imports from ``tools/``. Keep that directory
on ``sys.path`` so archived modules remain importable when loaded as a package.
"""

import sys
from pathlib import Path

_TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))
