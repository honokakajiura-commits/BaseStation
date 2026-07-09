#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Crop naming and rendering helpers for agent views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .spherical_camera import equirect_to_perspective


@dataclass(frozen=True)
class CropViewSpec:
    view: str
    yaw_off_deg: float
    fov_deg: float


def fmt_deg_tag(x: float, ndigits: int = 0) -> str:
    sign = "p" if x >= 0 else "m"
    ax = abs(float(x))
    if ndigits <= 0:
        return f"{sign}{int(round(ax))}"
    scale = 10 ** ndigits
    v = int(round(ax * scale))
    whole = v // scale
    frac = v % scale
    return f"{sign}{whole}p{frac}"


def build_action_tag(step: int, last_yaw_delta: float, last_zoom: bool) -> str:
    if step == 0:
        return "init"
    parts = []
    if abs(last_yaw_delta) > 1e-6:
        parts.append(f"yaw_{fmt_deg_tag(last_yaw_delta)}")
    if last_zoom:
        parts.append("zoom")
    if not parts:
        parts.append("keep")
    return "_".join(parts)


def build_crop_name(
    idx: int,
    fid: str,
    view: str,
    step: int,
    yaw: float,
    fov: float,
    last_yaw_delta: float,
    last_zoom: bool,
) -> str:
    act = build_action_tag(step, last_yaw_delta, last_zoom)
    return (
        f"{idx:05d}__{fid}__{view}"
        f"__r{step}"
        f"__yaw{fmt_deg_tag(yaw)}"
        f"__fov{fmt_deg_tag(fov)}"
        f"__act{act}.jpg"
    )


def get_front_left_right_views(fov_front: float, fov_side: float, yaw_side_deg: float = 90.0) -> List[Tuple[str, float, float]]:
    return [
        ("front", 0.0, float(fov_front)),
        ("left", -float(yaw_side_deg), float(fov_side)),
        ("right", float(yaw_side_deg), float(fov_side)),
    ]


def get_cv_interpolation(name: str) -> int:
    key = str(name or "").strip().lower()
    if key == "nearest":
        return cv2.INTER_NEAREST
    if key == "linear":
        return cv2.INTER_LINEAR
    if key == "cubic":
        return cv2.INTER_CUBIC
    if key == "lanczos":
        return cv2.INTER_LANCZOS4
    raise ValueError(f"unsupported interpolation: {name}")


def render_detection_crop(
    pano_bgr: np.ndarray,
    yaw_deg: float,
    pitch_deg: float,
    fov_deg: float,
    out_w: int,
    out_h: int,
    crop_strategy: str,
    supersample: float,
    interpolation: str,
    R_level: Optional[np.ndarray] = None,
    roll_deg: float = 0.0,
    level_meta: Optional[dict] = None,
) -> Tuple[np.ndarray, dict]:
    strategy = str(crop_strategy or "").strip().lower()
    if strategy not in {"legacy", "ui_like"}:
        raise ValueError(f"unsupported crop_strategy: {crop_strategy}")

    remap_interp = cv2.INTER_LINEAR if strategy == "legacy" else get_cv_interpolation(interpolation)
    ss = max(1.0, float(supersample))
    render_w = int(round(int(out_w) * ss))
    render_h = int(round(int(out_h) * ss))

    crop = equirect_to_perspective(
        pano_bgr,
        yaw=float(yaw_deg),
        pitch=float(pitch_deg),
        roll=float(roll_deg),
        fov_x=float(fov_deg),
        out_w=int(render_w),
        out_h=int(render_h),
        R_level=R_level,
        interpolation=remap_interp,
    )

    if render_w != int(out_w) or render_h != int(out_h):
        crop = cv2.resize(crop, (int(out_w), int(out_h)), interpolation=cv2.INTER_AREA)

    remap_name = str(interpolation if strategy != "legacy" else "linear")
    meta = {
        "strategy": strategy,
        "supersample": float(ss),
        "remap_interpolation": remap_name,
        "interpolation": remap_name,
        "render_size": [int(render_w), int(render_h)],
        "output_size": [int(out_w), int(out_h)],
        "fov_deg": float(fov_deg),
        "yaw_deg": float(yaw_deg),
        "pitch_deg": float(pitch_deg),
        "roll_deg": float(roll_deg),
        "leveling_enabled": bool(R_level is not None),
        "level_roll_deg": float((level_meta or {}).get("roll_deg", 0.0) or 0.0),
        "level_confidence": float((level_meta or {}).get("confidence", 0.0) or 0.0),
        "level_sample_count": int((level_meta or {}).get("sample_count", 0) or 0),
        "level_used_sample_count": int((level_meta or {}).get("used_sample_count", 0) or 0),
        "R_level_applied": bool(R_level is not None),
    }
    return crop, meta
