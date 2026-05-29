#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Agent pipeline entry points.

The current CLI scripts still own the full fetch/order/download orchestration.
New reusable pipeline code should be added here as those stages are split out.
"""

from __future__ import annotations

from .refine_policy import plan_refine_view


def plan_detection_refine(*args, **kwargs):
    """Thin pipeline helper used as the first migration point for refine planning."""
    return plan_refine_view(*args, **kwargs)


# TODO: Move detect_from_panos_stage from tools/basestation_agent_complete.py
# here after the fetch/order/download helper functions are also package-local.

