#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Minimal geolocation helpers for future bbox/ray to map projection work."""

from __future__ import annotations

from typing import Any, Dict

from .spherical_camera import bbox_center_to_world_ray, ray_to_yaw_pitch


def ray_to_bearing_elevation(ray: Any) -> Dict[str, float]:
    yaw_deg, pitch_deg = ray_to_yaw_pitch(ray)
    return {"bearing_deg": float(yaw_deg), "elevation_deg": float(pitch_deg)}


def bbox_to_bearing_elevation(
    bbox: Any,
    yaw: float,
    pitch: float,
    roll: float,
    fov_x: float,
    out_w: int,
    out_h: int,
) -> Dict[str, float]:
    ray = bbox_center_to_world_ray(
        bbox=bbox,
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        fov_x=fov_x,
        out_w=out_w,
        out_h=out_h,
    )
    return ray_to_bearing_elevation(ray)

