#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Compatibility wrapper for the archived spherical-camera smoke script."""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.archive.test_spherical_camera import *  # noqa: F401,F403


if __name__ == "__main__":
    from tools.archive.test_spherical_camera import main

    raise SystemExit(main())
