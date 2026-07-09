#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Configuration objects for the base-station exploration agent."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AgentConfig:
    det_w: int = 1280
    det_h: int = 1280
    fov_front: float = 105.0
    fov_side: float = 90.0
    crop_strategy: str = "ui_like"
    crop_supersample: float = 1.25
    crop_interpolation: str = "cubic"
    level_method: str = "none"
    level_horizon: bool = True
    level_min_confidence: float = 0.25
    level_preview_fov: float = 90.0
    level_preview_w: int = 768
    level_preview_h: int = 768

    zoom_min_fov: float = 50.0
    high_conf: float = 0.60
    low_conf: float = 0.20

    small_area_frac: float = 0.02
    large_area_frac: float = 0.08
    edge_center_margin: float = 0.20
    zoom_safe_factor: float = 0.90
    bbox_margin_deg: float = 3.0
    recenter_pitch: bool = True
    refine_zoom_ratio_small: float = 0.55
    refine_zoom_ratio_medium: float = 0.75

    max_refine: int = 2
    yaw_side_deg: float = 90.0
