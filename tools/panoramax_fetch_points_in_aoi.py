#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Compatibility wrapper for the archived Panoramax AOI-point fetcher."""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.archive.panoramax_fetch_points_in_aoi import *  # noqa: F401,F403


if __name__ == "__main__":
    from tools.archive.panoramax_fetch_points_in_aoi import main

    raise SystemExit(main())
