#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Refine-search planning for the base-station detection agent."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

from .spherical_camera import compute_next_view_from_bbox, wrap_yaw_deg


TARGET_BBOX_AREA_RATIO = 1.0 / 12.0
TARGET_BBOX_AREA_MIN = 1.0 / 16.0
TARGET_BBOX_AREA_MAX = 1.0 / 8.0
DEFAULT_MAX_ZOOM_RATIO = 0.6


def _bbox_xyxy(det: Any) -> Tuple[float, float, float, float]:
    raw = det.get("xyxy") if isinstance(det, dict) else det
    if raw is None or len(raw) != 4:
        raise ValueError(f"det must be xyxy or dict with xyxy: {det}")
    x1, y1, x2, y2 = [float(x) for x in raw]
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def bbox_area_ratio(det: Any, image_w: int, image_h: int) -> float:
    """Return bbox area divided by image area."""
    x1, y1, x2, y2 = _bbox_xyxy(det)
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    image_area = max(1.0, float(image_w) * float(image_h))
    return float(area / image_area)


def classify_detection(best_det: Optional[dict], high_conf: float, low_conf: Optional[float] = None) -> str:
    """Classify detection state for refine control.

    ``low_conf`` is retained for call-site compatibility. Confidence below
    ``high_conf`` still returns ``"refine"``; existing callers may decide to
    reject very low confidence detections before calling the refine planner.
    """
    del low_conf
    if not best_det:
        return "no_detection"
    if float(best_det.get("conf", 0.0)) >= float(high_conf):
        return "confirmed"
    return "refine"


def compute_target_fov_by_bbox_area(
    current_fov: float,
    area_ratio: float,
    target_area_ratio: float = TARGET_BBOX_AREA_RATIO,
    max_zoom_ratio: float = DEFAULT_MAX_ZOOM_RATIO,
    min_fov: float = 20.0,
) -> float:
    """Compute a target FOV that moves bbox area toward the target ratio."""
    current = float(current_fov)
    area = max(0.0, float(area_ratio))
    target = max(1e-9, float(target_area_ratio))
    zoom_floor = current * max(0.0, min(1.0, float(max_zoom_ratio)))

    if area <= 0.0:
        target_fov = current
    else:
        target_fov = current * math.sqrt(area / target)
    target_fov = max(target_fov, zoom_floor, float(min_fov))
    return float(min(179.0, max(1.0, target_fov)))


def _refine_action(yaw_delta: float, pitch_delta: float, fov_delta: float) -> str:
    recenter = abs(float(yaw_delta)) >= 0.5 or abs(float(pitch_delta)) >= 0.5
    zoom_in = float(fov_delta) < -0.5
    zoom_out = float(fov_delta) > 0.5
    if recenter and zoom_in:
        return "recenter_and_zoom"
    if recenter and zoom_out:
        return "recenter_and_widen"
    if recenter:
        return "recenter_only"
    if zoom_in:
        return "zoom_only"
    if zoom_out:
        return "widen_only"
    return "keep"


def plan_refine_view(
    det: Any,
    yaw: float,
    pitch: float,
    roll: float,
    current_fov: float,
    image_w: int,
    image_h: int,
    min_fov: float,
    margin_deg: float,
    R_level: Any = None,
    recenter_pitch: bool = True,
    target_bbox_area_ratio: float = TARGET_BBOX_AREA_RATIO,
    target_bbox_area_min: float = TARGET_BBOX_AREA_MIN,
    target_bbox_area_max: float = TARGET_BBOX_AREA_MAX,
    max_zoom_ratio: float = DEFAULT_MAX_ZOOM_RATIO,
) -> Tuple[float, float, float, str, Dict[str, Any]]:
    """Plan the next refine crop using spherical rays and bbox-area FOV control."""
    area_ratio = bbox_area_ratio(det, image_w, image_h)
    yaw_next, pitch_next, _, spherical_debug = compute_next_view_from_bbox(
        bbox=det,
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        fov_x=current_fov,
        out_w=image_w,
        out_h=image_h,
        zoom_ratio=max_zoom_ratio,
        min_fov=min_fov,
        margin_deg=margin_deg,
        R_level=R_level,
    )
    if not recenter_pitch:
        pitch_next = float(pitch)

    safe_fov = float(spherical_debug["safe_fov"])
    target_fov = compute_target_fov_by_bbox_area(
        current_fov=current_fov,
        area_ratio=area_ratio,
        target_area_ratio=target_bbox_area_ratio,
        max_zoom_ratio=max_zoom_ratio,
        min_fov=min_fov,
    )
    next_fov = max(safe_fov, target_fov, float(min_fov))
    next_fov = float(min(179.0, max(1.0, next_fov)))

    yaw_delta = wrap_yaw_deg(float(yaw_next) - float(yaw))
    pitch_delta = float(pitch_next) - float(pitch)
    fov_delta = next_fov - float(current_fov)
    refine_action = _refine_action(yaw_delta, pitch_delta, fov_delta)

    debug_info: Dict[str, Any] = dict(spherical_debug)
    debug_info.update(
        {
            "current_yaw": float(yaw),
            "current_pitch": float(pitch),
            "current_fov": float(current_fov),
            "previous_yaw": float(yaw),
            "previous_pitch": float(pitch),
            "previous_fov": float(current_fov),
            "bbox_area_ratio": float(area_ratio),
            "target_bbox_area_ratio": float(target_bbox_area_ratio),
            "target_bbox_area_min": float(target_bbox_area_min),
            "target_bbox_area_max": float(target_bbox_area_max),
            "safe_fov": float(safe_fov),
            "target_fov": float(target_fov),
            "next_fov": float(next_fov),
            "final_fov": float(next_fov),
            "max_corner_angle": float(spherical_debug["max_corner_angle"]),
            "yaw_next": float(yaw_next),
            "pitch_next": float(pitch_next),
            "target_yaw": float(yaw_next),
            "target_pitch": float(pitch_next),
            "next_yaw": float(yaw_next),
            "next_pitch": float(pitch_next),
            "yaw_delta": float(yaw_delta),
            "pitch_delta": float(pitch_delta),
            "fov_delta": float(fov_delta),
            "max_zoom_ratio": float(max_zoom_ratio),
            "zoom_fov": float(target_fov),
            "refine_action": refine_action,
            "recenter_pitch": bool(recenter_pitch),
        }
    )
    return float(yaw_next), float(pitch_next), next_fov, refine_action, debug_info

