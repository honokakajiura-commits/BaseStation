#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Crop naming and rendering helpers for agent views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from tools.make_yolo_crops_from_panoramax import render_panoramax_crop


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
) -> Tuple[np.ndarray, dict]:
    crop, meta = render_panoramax_crop(
        pano_bgr=pano_bgr,
        yaw_deg=yaw_deg,
        pitch_deg=pitch_deg,
        fov_deg=fov_deg,
        out_w=out_w,
        out_h=out_h,
        crop_strategy=crop_strategy,
        supersample=supersample,
        interpolation=interpolation,
    )
    if "remap_interpolation" in meta and "interpolation" not in meta:
        meta["interpolation"] = meta["remap_interpolation"]
    return crop, meta

