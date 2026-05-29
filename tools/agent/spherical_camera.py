#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Spherical camera geometry for Panoramax equirectangular images.

Coordinate system
-----------------
- Public angles are degrees.
- Camera/world axes are x:right, y:up, z:forward.
- Positive yaw turns the optical axis to the right: +z -> +x.
- Positive pitch looks upward: +z -> +y.
- Positive roll rotates camera/image x toward y around the optical axis.
- Image pixels still use OpenCV coordinates: u grows right, v grows down.

The view rotation is camera-to-world and is applied to column vectors as:

    R_view = R_yaw(yaw) @ R_pitch_up(pitch) @ R_roll(roll)

For row-vector numpy arrays this module uses ``ray @ R_view.T``.

``R_level`` is optional and maps level-corrected world rays to the source
panorama rays for sampling. The returned bbox/world rays stay in the same
level-corrected coordinate system as yaw/pitch, so they can be fed back into
``ray_to_yaw_pitch`` for the next view.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


EPS = 1e-12


def wrap_yaw_deg(yaw_deg: float) -> float:
    return (float(yaw_deg) + 180.0) % 360.0 - 180.0


def clamp_pitch_deg(pitch_deg: float, margin_deg: float = 1.0) -> float:
    return max(-90.0 + float(margin_deg), min(90.0 - float(margin_deg), float(pitch_deg)))


def _validate_fov(fov_x: float) -> None:
    if not (0.0 < float(fov_x) < 180.0):
        raise ValueError(f"fov_x must be in (0, 180) degrees: {fov_x}")


def normalize_ray(ray: np.ndarray) -> np.ndarray:
    arr = np.asarray(ray, dtype=np.float64)
    norm = np.linalg.norm(arr, axis=-1, keepdims=True)
    return arr / np.maximum(norm, EPS)


def pixel_to_camera_ray(u: Any, v: Any, W: int, H: int, fov_x: float) -> np.ndarray:
    """Convert perspective-image pixel coordinates to camera-space rays.

    ``fov_x`` is horizontal FOV in degrees. The focal length is shared by x and
    y, so the vertical FOV follows from ``W/H``. The optical center is
    ``(W/2, H/2)`` to match the existing crop renderer.
    """
    _validate_fov(fov_x)
    if int(W) <= 0 or int(H) <= 0:
        raise ValueError(f"W and H must be positive: W={W}, H={H}")

    u_arr = np.asarray(u, dtype=np.float64)
    v_arr = np.asarray(v, dtype=np.float64)
    fx = (float(W) / 2.0) / math.tan(math.radians(float(fov_x)) / 2.0)
    cx = float(W) / 2.0
    cy = float(H) / 2.0

    x = (u_arr - cx) / fx
    y = -(v_arr - cy) / fx
    z = np.ones_like(x, dtype=np.float64)
    return normalize_ray(np.stack([x, y, z], axis=-1))


def yaw_pitch_to_ray(yaw: Any, pitch: Any) -> np.ndarray:
    """Build a world ray from yaw/pitch in degrees."""
    yaw_rad = np.radians(np.asarray(yaw, dtype=np.float64))
    pitch_rad = np.radians(np.asarray(pitch, dtype=np.float64))
    x = np.cos(pitch_rad) * np.sin(yaw_rad)
    y = np.sin(pitch_rad)
    z = np.cos(pitch_rad) * np.cos(yaw_rad)
    return normalize_ray(np.stack([x, y, z], axis=-1))


def ray_to_yaw_pitch(ray: Any) -> Tuple[Any, Any]:
    """Convert a 3D ray to yaw/pitch in degrees."""
    arr = normalize_ray(np.asarray(ray, dtype=np.float64))
    x = arr[..., 0]
    y = arr[..., 1]
    z = arr[..., 2]
    yaw = np.degrees(np.arctan2(x, z))
    pitch = np.degrees(np.arctan2(y, np.sqrt(x * x + z * z)))
    if np.ndim(yaw) == 0:
        return wrap_yaw_deg(float(yaw)), float(pitch)
    return ((yaw + 180.0) % 360.0 - 180.0), pitch


def _rotation_yaw_deg(yaw_deg: float) -> np.ndarray:
    rad = math.radians(float(yaw_deg))
    c = math.cos(rad)
    s = math.sin(rad)
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float64,
    )


def _rotation_pitch_up_deg(pitch_deg: float) -> np.ndarray:
    rad = math.radians(float(pitch_deg))
    c = math.cos(rad)
    s = math.sin(rad)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, s],
            [0.0, -s, c],
        ],
        dtype=np.float64,
    )


def _rotation_roll_deg(roll_deg: float) -> np.ndarray:
    rad = math.radians(float(roll_deg))
    c = math.cos(rad)
    s = math.sin(rad)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def make_rotation(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Return camera-to-world rotation for yaw/pitch/roll in degrees.

    Rotation order for camera rays is roll first around camera +z, then pitch
    around camera +x with positive pitch upward, then yaw around world +y.
    With roll=0, ``make_rotation(yaw, pitch, 0) @ [0, 0, 1]`` equals
    ``yaw_pitch_to_ray(yaw, pitch)``.
    """
    return _rotation_yaw_deg(yaw) @ _rotation_pitch_up_deg(pitch) @ _rotation_roll_deg(roll)


def apply_rotation_to_rays(rays: np.ndarray, R: np.ndarray) -> np.ndarray:
    return normalize_ray(np.asarray(rays, dtype=np.float64) @ np.asarray(R, dtype=np.float64).T)


def _ray_to_equirect_uv(ray: np.ndarray, pano_w: int, pano_h: int) -> Tuple[np.ndarray, np.ndarray]:
    dirs = normalize_ray(ray)
    lon = np.arctan2(dirs[..., 0], dirs[..., 2])
    lat = np.arctan2(dirs[..., 1], np.sqrt(dirs[..., 0] ** 2 + dirs[..., 2] ** 2))
    u = (lon / (2.0 * math.pi) + 0.5) * float(pano_w)
    v = (0.5 - lat / math.pi) * float(pano_h)
    u = np.mod(u, float(pano_w))
    v = np.clip(v, 0.0, float(pano_h - 1))
    return u.astype(np.float32), v.astype(np.float32)


def equirect_to_perspective(
    pano: np.ndarray,
    yaw: float,
    pitch: float,
    roll: float,
    fov_x: float,
    out_w: int,
    out_h: int,
    R_level: Optional[np.ndarray] = None,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Render a rectilinear perspective crop from an equirectangular panorama."""
    if pano is None or pano.ndim < 2:
        raise ValueError("pano must be an image array")
    _validate_fov(fov_x)

    pano_h, pano_w = pano.shape[:2]
    xs = np.arange(int(out_w), dtype=np.float64)
    ys = np.arange(int(out_h), dtype=np.float64)
    xv, yv = np.meshgrid(xs, ys)

    camera_rays = pixel_to_camera_ray(xv, yv, int(out_w), int(out_h), float(fov_x))
    R_view = make_rotation(float(yaw), float(pitch), float(roll))
    world_rays = apply_rotation_to_rays(camera_rays, R_view)

    sample_rays = world_rays
    if R_level is not None:
        sample_rays = apply_rotation_to_rays(world_rays, np.asarray(R_level, dtype=np.float64))

    map_u, map_v = _ray_to_equirect_uv(sample_rays, pano_w=pano_w, pano_h=pano_h)
    return cv2.remap(
        pano,
        map_u,
        map_v,
        interpolation=interpolation,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _bbox_xyxy(bbox: Any) -> Tuple[float, float, float, float]:
    raw = bbox.get("xyxy") if isinstance(bbox, dict) else bbox
    if raw is None or len(raw) != 4:
        raise ValueError(f"bbox must be xyxy or dict with xyxy: {bbox}")
    x1, y1, x2, y2 = [float(x) for x in raw]
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def _camera_pixel_to_world_ray(
    u: float,
    v: float,
    yaw: float,
    pitch: float,
    roll: float,
    fov_x: float,
    out_w: int,
    out_h: int,
) -> np.ndarray:
    camera_ray = pixel_to_camera_ray(float(u), float(v), int(out_w), int(out_h), float(fov_x))
    return apply_rotation_to_rays(camera_ray, make_rotation(float(yaw), float(pitch), float(roll)))


def bbox_center_to_world_ray(
    bbox: Any,
    yaw: float,
    pitch: float,
    roll: float,
    fov_x: float,
    out_w: int,
    out_h: int,
    R_level: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Back-project a YOLO bbox center to a world ray.

    ``R_level`` is accepted for API symmetry with rendering. The returned ray is
    not rotated by ``R_level`` because next-view yaw/pitch are also defined in
    the level-corrected world coordinate system.
    """
    del R_level
    x1, y1, x2, y2 = _bbox_xyxy(bbox)
    return _camera_pixel_to_world_ray(
        (x1 + x2) / 2.0,
        (y1 + y2) / 2.0,
        yaw,
        pitch,
        roll,
        fov_x,
        out_w,
        out_h,
    )


def _angle_between_rays_deg(a: np.ndarray, b: np.ndarray) -> float:
    aa = normalize_ray(np.asarray(a, dtype=np.float64))
    bb = normalize_ray(np.asarray(b, dtype=np.float64))
    dot = float(np.clip(np.sum(aa * bb, axis=-1), -1.0, 1.0))
    return math.degrees(math.acos(dot))


def _bbox_corner_world_rays(
    bbox: Any,
    yaw: float,
    pitch: float,
    roll: float,
    fov_x: float,
    out_w: int,
    out_h: int,
) -> List[np.ndarray]:
    x1, y1, x2, y2 = _bbox_xyxy(bbox)
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    return [
        _camera_pixel_to_world_ray(u, v, yaw, pitch, roll, fov_x, out_w, out_h)
        for u, v in corners
    ]


def compute_next_view_from_bbox(
    bbox: Any,
    yaw: float,
    pitch: float,
    roll: float,
    fov_x: float,
    out_w: int,
    out_h: int,
    zoom_ratio: float,
    min_fov: float,
    margin_deg: float,
    R_level: Optional[np.ndarray] = None,
) -> Tuple[float, float, float, Dict[str, Any]]:
    """Compute the next refine view from a bbox using spherical geometry."""
    target_ray = bbox_center_to_world_ray(
        bbox,
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        fov_x=fov_x,
        out_w=out_w,
        out_h=out_h,
        R_level=R_level,
    )
    target_yaw, target_pitch = ray_to_yaw_pitch(target_ray)
    target_pitch = clamp_pitch_deg(float(target_pitch), margin_deg=0.25)

    corner_rays = _bbox_corner_world_rays(
        bbox,
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        fov_x=fov_x,
        out_w=out_w,
        out_h=out_h,
    )
    corner_angles = [_angle_between_rays_deg(corner_ray, target_ray) for corner_ray in corner_rays]
    max_corner_angle = max(corner_angles) if corner_angles else 0.0
    safe_fov = 2.0 * max_corner_angle + float(margin_deg)
    zoom_fov = float(fov_x) * float(zoom_ratio)
    final_fov = max(float(zoom_fov), float(safe_fov), float(min_fov))
    final_fov = min(179.0, max(1.0, final_fov))

    yaw_delta = wrap_yaw_deg(float(target_yaw) - float(yaw))
    pitch_delta = float(target_pitch) - float(pitch)
    fov_delta = float(final_fov) - float(fov_x)
    if abs(yaw_delta) < 0.5 and abs(pitch_delta) < 0.5 and abs(fov_delta) < 0.5:
        refine_action = "keep"
    elif final_fov < float(fov_x) - 0.5 and (abs(yaw_delta) >= 0.5 or abs(pitch_delta) >= 0.5):
        refine_action = "recenter_and_zoom"
    elif final_fov < float(fov_x) - 0.5:
        refine_action = "zoom_only"
    elif abs(yaw_delta) >= 0.5 or abs(pitch_delta) >= 0.5:
        refine_action = "recenter_only"
    else:
        refine_action = "adjust_fov"

    x1, y1, x2, y2 = _bbox_xyxy(bbox)
    debug_info: Dict[str, Any] = {
        "previous_yaw": float(yaw),
        "previous_pitch": float(pitch),
        "previous_fov": float(fov_x),
        "bbox_center": [float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)],
        "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
        "target_yaw": float(target_yaw),
        "target_pitch": float(target_pitch),
        "next_yaw": float(target_yaw),
        "next_pitch": float(target_pitch),
        "next_fov": float(final_fov),
        "max_corner_angle": float(max_corner_angle),
        "max_corner_angle_deg": float(max_corner_angle),
        "corner_angles_deg": [float(a) for a in corner_angles],
        "safe_fov": float(safe_fov),
        "safe_fov_deg": float(safe_fov),
        "zoom_fov": float(zoom_fov),
        "zoom_fov_deg": float(zoom_fov),
        "final_fov": float(final_fov),
        "final_fov_deg": float(final_fov),
        "zoom_ratio": float(zoom_ratio),
        "min_fov": float(min_fov),
        "margin_deg": float(margin_deg),
        "yaw_delta": float(yaw_delta),
        "pitch_delta": float(pitch_delta),
        "fov_delta": float(fov_delta),
        "refine_action": refine_action,
        "R_level_applied_for_sampling": bool(R_level is not None),
    }
    return float(target_yaw), float(target_pitch), float(final_fov), debug_info
