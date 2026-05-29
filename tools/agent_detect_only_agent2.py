#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Compatibility wrapper for the archived detect-only agent CLI."""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.archive.agent_detect_only_agent2 import *  # noqa: F401,F403


if __name__ == "__main__":
    from tools.archive.agent_detect_only_agent2 import main

    raise SystemExit(main())
