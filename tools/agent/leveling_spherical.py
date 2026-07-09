#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Spherical RANSAC leveling helpers for the agent."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .crop import render_detection_crop
from .spherical_camera import apply_rotation_to_rays, make_rotation, normalize_ray, pixel_to_camera_ray


WORLD_UP = np.array([0.0, 1.0, 0.0], dtype=np.float64)


@dataclass
class LevelEstimate:
    method: str
    applied: bool
    R_level: Optional[np.ndarray]
    reject_reason: Optional[str]
    v_up: Optional[list[float]]
    angle_to_world_up_deg: Optional[float]
    total_line_count: int
    inlier_count: int
    inlier_ratio: float
    mean_residual_deg: Optional[float]
    median_residual_deg: Optional[float]
    residual_thresh_deg: float
    ransac_iters: int
    debug: dict


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_image(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), img):
        raise ValueError(f"failed to write image: {path}")


def _wrap_yaw_deg(yaw: float) -> float:
    return (float(yaw) + 180.0) % 360.0 - 180.0


def _rotation_from_to(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(normalize_ray(a), dtype=np.float64)
    b = np.asarray(normalize_ray(b), dtype=np.float64)
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if c > 1.0 - 1e-10:
        return np.eye(3, dtype=np.float64)
    if c < -1.0 + 1e-10:
        helper = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(a, helper))) > 0.9:
            helper = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        axis = np.asarray(normalize_ray(np.cross(a, helper)), dtype=np.float64)
        angle = math.pi
    else:
        axis = np.asarray(np.cross(a, b), dtype=np.float64)
        s = float(np.linalg.norm(axis))
        axis = axis / max(s, 1e-12)
        angle = math.acos(c)

    x, y, z = [float(v) for v in axis]
    K = np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)


def _line_orientation_deg(x1: float, y1: float, x2: float, y2: float) -> float:
    angle = math.degrees(math.atan2(float(y2) - float(y1), float(x2) - float(x1)))
    while angle < -90.0:
        angle += 180.0
    while angle >= 90.0:
        angle -= 180.0
    return float(angle)


def _detect_hough_segments(preview_bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    if preview_bgr is None or preview_bgr.ndim < 2:
        return np.zeros((0, 4), dtype=np.float64), {"reason": "invalid_image", "raw_line_count": 0}

    h, w = preview_bgr.shape[:2]
    min_dim = max(1, min(int(w), int(h)))
    gray = cv2.cvtColor(preview_bgr, cv2.COLOR_BGR2GRAY) if preview_bgr.ndim == 3 else preview_bgr
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3, L2gradient=True)

    min_line_len = max(40, int(round(min_dim * 0.08)))
    max_line_gap = max(8, int(round(min_dim * 0.025)))
    threshold = max(45, int(round(min_dim * 0.08)))
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=threshold,
        minLineLength=min_line_len,
        maxLineGap=max_line_gap,
    )
    meta = {
        "raw_line_count": 0 if lines is None else int(len(lines)),
        "min_line_length": int(min_line_len),
        "max_line_gap": int(max_line_gap),
        "hough_threshold": int(threshold),
    }
    if lines is None:
        meta["reason"] = "no_hough_lines"
        return np.zeros((0, 4), dtype=np.float64), meta
    return lines[:, 0, :].astype(np.float64), meta


def _segment_to_great_circle(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    yaw_sample: float,
    pitch_detect: float,
    preview_w: int,
    preview_h: int,
    preview_fov: float,
) -> Optional[np.ndarray]:
    r1_cam = pixel_to_camera_ray(float(x1), float(y1), int(preview_w), int(preview_h), float(preview_fov))
    r2_cam = pixel_to_camera_ray(float(x2), float(y2), int(preview_w), int(preview_h), float(preview_fov))
    R_view = make_rotation(float(yaw_sample), float(pitch_detect), 0.0)
    r1_world = apply_rotation_to_rays(r1_cam, R_view)
    r2_world = apply_rotation_to_rays(r2_cam, R_view)
    n = np.cross(np.asarray(r1_world, dtype=np.float64), np.asarray(r2_world, dtype=np.float64))
    norm = float(np.linalg.norm(n))
    if norm <= 1e-9:
        return None
    return np.asarray(normalize_ray(n), dtype=np.float64)


def _candidate_from_normals(n1: np.ndarray, n2: np.ndarray) -> Optional[np.ndarray]:
    v = np.cross(np.asarray(n1, dtype=np.float64), np.asarray(n2, dtype=np.float64))
    norm = float(np.linalg.norm(v))
    if norm <= 1e-9:
        return None
    v = np.asarray(normalize_ray(v), dtype=np.float64)
    if float(v[1]) < 0.0:
        v = -v
    return v


def _clamp_unit(x: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=np.float64), -1.0, 1.0)


def _angular_residuals_rad(normals: np.ndarray, v: np.ndarray) -> np.ndarray:
    dots = normals @ np.asarray(v, dtype=np.float64)
    return np.abs(np.arcsin(_clamp_unit(dots)))


def _weighted_median(values: Sequence[float], weights: Sequence[float]) -> float:
    vals = np.asarray(values, dtype=np.float64)
    wts = np.asarray(weights, dtype=np.float64)
    mask = np.isfinite(vals) & np.isfinite(wts) & (wts > 0.0)
    vals = vals[mask]
    wts = wts[mask]
    if vals.size == 0:
        return 0.0
    order = np.argsort(vals)
    vals = vals[order]
    wts = wts[order]
    total = float(np.sum(wts))
    if total <= 0.0:
        return float(np.median(vals))
    idx = int(np.searchsorted(np.cumsum(wts), total * 0.5, side="left"))
    return float(vals[min(idx, vals.size - 1)])


def _refine_up_vector(normals: np.ndarray, weights: np.ndarray, keep: np.ndarray) -> Optional[np.ndarray]:
    if int(np.count_nonzero(keep)) < 2:
        return None
    N = normals[keep]
    W = weights[keep]
    A = N * np.sqrt(np.maximum(W, 1e-9))[:, None]
    try:
        _, _, vh = np.linalg.svd(A, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    v = np.asarray(vh[-1, :], dtype=np.float64)
    v = np.asarray(normalize_ray(v), dtype=np.float64)
    if float(v[1]) < 0.0:
        v = -v
    return v


def _draw_preview_lines(preview_bgr: np.ndarray, lines: Sequence[Dict[str, Any]]) -> np.ndarray:
    out = preview_bgr.copy()
    for record in lines:
        color = (150, 150, 150)
        thickness = 1
        if record.get("inlier", False):
            color = (0, 140, 255)
            thickness = 2
        elif record.get("candidate", False):
            color = (0, 210, 255)
            thickness = 2
        x1, y1, x2, y2 = [int(round(v)) for v in record["xyxy"]]
        cv2.line(out, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    return out


def _save_debug(
    debug_dir: Path,
    previews: Sequence[Dict[str, Any]],
    preview_images: Sequence[np.ndarray],
    preview_lines: Sequence[Sequence[Dict[str, Any]]],
    estimate: LevelEstimate,
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    for preview_meta, preview_img, lines in zip(previews, preview_images, preview_lines):
        stem = str(preview_meta["preview_path"]).split("/")[-1].replace(".jpg", "")
        _write_image(debug_dir / f"{stem}_lines.jpg", _draw_preview_lines(preview_img, lines))
        (debug_dir / f"{stem}_lines.json").write_text(
            json.dumps(_json_safe(lines), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    meta = {
        "method": estimate.method,
        "applied": bool(estimate.applied),
        "reject_reason": estimate.reject_reason,
        "v_up": estimate.v_up,
        "angle_to_world_up_deg": estimate.angle_to_world_up_deg,
        "total_line_count": int(estimate.total_line_count),
        "inlier_count": int(estimate.inlier_count),
        "inlier_ratio": float(estimate.inlier_ratio),
        "mean_residual_deg": estimate.mean_residual_deg,
        "median_residual_deg": estimate.median_residual_deg,
        "residual_thresh_deg": float(estimate.residual_thresh_deg),
        "ransac_iters": int(estimate.ransac_iters),
        "R_level": None if estimate.R_level is None else estimate.R_level.tolist(),
        "debug": _json_safe(estimate.debug),
    }
    (debug_dir / "spherical_ransac_level_meta.json").write_text(
        json.dumps(_json_safe(meta), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def estimate_spherical_ransac_level(
    pano_bgr: np.ndarray,
    yaw_center: float,
    preview_yaw_offsets: Sequence[float] = (-135, -90, -45, 0, 45, 90, 135),
    preview_pitch: float = 0.0,
    preview_fov: float = 90.0,
    preview_w: int = 1024,
    preview_h: int = 768,
    ransac_iters: int = 1000,
    residual_thresh_deg: float = 3.0,
    min_inliers: int = 8,
    min_total_lines: int = 20,
    max_apply_deg: float = 5.0,
    debug: bool = False,
    debug_dir: Optional[Path] = None,
) -> LevelEstimate:
    method = "spherical_ransac"
    if pano_bgr is None or pano_bgr.ndim < 2:
        return LevelEstimate(
            method=method,
            applied=False,
            R_level=None,
            reject_reason="invalid_image",
            v_up=None,
            angle_to_world_up_deg=None,
            total_line_count=0,
            inlier_count=0,
            inlier_ratio=0.0,
            mean_residual_deg=None,
            median_residual_deg=None,
            residual_thresh_deg=float(residual_thresh_deg),
            ransac_iters=int(ransac_iters),
            debug={"reason": "invalid_image"},
        )

    preview_images: List[np.ndarray] = []
    preview_meta: List[Dict[str, Any]] = []
    preview_lines: List[List[Dict[str, Any]]] = []
    records: List[Dict[str, Any]] = []
    preview_yaw_offsets = tuple(float(x) for x in preview_yaw_offsets)

    for yaw_off in preview_yaw_offsets:
        yaw_sample = _wrap_yaw_deg(float(yaw_center) + float(yaw_off))
        preview, crop_meta = render_detection_crop(
            pano_bgr=pano_bgr,
            yaw_deg=float(yaw_sample),
            pitch_deg=float(preview_pitch),
            fov_deg=float(preview_fov),
            out_w=int(preview_w),
            out_h=int(preview_h),
            crop_strategy="ui_like",
            supersample=1.0,
            interpolation="linear",
            R_level=None,
            roll_deg=0.0,
            level_meta=None,
        )
        raw_lines, detect_meta = _detect_hough_segments(preview)
        local_lines: List[Dict[str, Any]] = []
        for raw_line in raw_lines:
            x1, y1, x2, y2 = [float(v) for v in raw_line]
            length = float(math.hypot(x2 - x1, y2 - y1))
            if length < float(detect_meta.get("min_line_length", 0) or 0):
                continue
            normal = _segment_to_great_circle(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                yaw_sample=yaw_sample,
                pitch_detect=float(preview_pitch),
                preview_w=int(preview_w),
                preview_h=int(preview_h),
                preview_fov=float(preview_fov),
            )
            if normal is None:
                continue
            record = {
                "yaw_sample": float(yaw_sample),
                "preview_path": f"preview_yaw_{yaw_sample:.6f}.jpg",
                "xyxy": [x1, y1, x2, y2],
                "length": float(length),
                "angle_deg": float(_line_orientation_deg(x1, y1, x2, y2)),
                "great_circle_normal": normal.tolist(),
                "weight": float(max(1.0, math.sqrt(length))),
                "candidate": True,
                "inlier": False,
                "residual_deg": None,
            }
            records.append(record)
            local_lines.append(record)
        preview_images.append(preview)
        preview_meta.append(
            {
                "preview_path": f"preview_yaw_{yaw_sample:.6f}.jpg",
                "yaw_sample": float(yaw_sample),
                "yaw_offset": float(yaw_off),
                "crop_meta": _json_safe(crop_meta),
                "detect_meta": _json_safe(detect_meta),
            }
        )
        preview_lines.append(local_lines)

    total_line_count = int(len(records))
    debug_info: Dict[str, Any] = {
        "yaw_center": float(yaw_center),
        "preview_pitch": float(preview_pitch),
        "preview_fov": float(preview_fov),
        "preview_w": int(preview_w),
        "preview_h": int(preview_h),
        "preview_yaw_offsets": list(preview_yaw_offsets),
        "preview_count": len(preview_meta),
        "total_line_count": total_line_count,
    }

    if total_line_count < int(min_total_lines):
        estimate = LevelEstimate(
            method=method,
            applied=False,
            R_level=None,
            reject_reason="not_enough_total_lines",
            v_up=None,
            angle_to_world_up_deg=None,
            total_line_count=total_line_count,
            inlier_count=0,
            inlier_ratio=0.0,
            mean_residual_deg=None,
            median_residual_deg=None,
            residual_thresh_deg=float(residual_thresh_deg),
            ransac_iters=int(ransac_iters),
            debug=debug_info,
        )
        if debug and debug_dir is not None:
            _save_debug(debug_dir, preview_meta, preview_images, preview_lines, estimate)
        return estimate

    normals = np.asarray([record["great_circle_normal"] for record in records], dtype=np.float64)
    weights = np.asarray([max(1.0, float(record["weight"])) for record in records], dtype=np.float64)
    threshold_rad = math.radians(float(residual_thresh_deg))
    rng = random.Random(0)
    best: Optional[Dict[str, Any]] = None
    weight_list = weights.tolist()

    for _ in range(max(1, int(ransac_iters))):
        i, j = rng.choices(range(len(records)), weights=weight_list, k=2)
        if i == j:
            continue
        v_candidate = _candidate_from_normals(normals[i], normals[j])
        if v_candidate is None or not np.all(np.isfinite(v_candidate)):
            continue
        residuals = _angular_residuals_rad(normals, v_candidate)
        keep = residuals <= threshold_rad
        inlier_count = int(np.count_nonzero(keep))
        if inlier_count < 2:
            continue
        inlier_weight = float(np.sum(weights[keep]))
        median_residual = _weighted_median(residuals[keep].tolist(), weights[keep].tolist())
        score = (inlier_count, inlier_weight, -float(median_residual))
        if best is None or score > best["score"]:
            best = {
                "score": score,
                "candidate": v_candidate,
                "keep": keep,
                "median_residual_rad": float(median_residual),
            }

    if best is None:
        estimate = LevelEstimate(
            method=method,
            applied=False,
            R_level=None,
            reject_reason="ransac_failed",
            v_up=None,
            angle_to_world_up_deg=None,
            total_line_count=total_line_count,
            inlier_count=0,
            inlier_ratio=0.0,
            mean_residual_deg=None,
            median_residual_deg=None,
            residual_thresh_deg=float(residual_thresh_deg),
            ransac_iters=int(ransac_iters),
            debug=debug_info,
        )
        if debug and debug_dir is not None:
            _save_debug(debug_dir, preview_meta, preview_images, preview_lines, estimate)
        return estimate

    keep = np.asarray(best["keep"], dtype=bool)
    refined = _refine_up_vector(normals, weights, keep)
    if refined is None or not np.all(np.isfinite(refined)):
        estimate = LevelEstimate(
            method=method,
            applied=False,
            R_level=None,
            reject_reason="svd_failed",
            v_up=None,
            angle_to_world_up_deg=None,
            total_line_count=total_line_count,
            inlier_count=int(np.count_nonzero(keep)),
            inlier_ratio=float(np.count_nonzero(keep)) / float(total_line_count),
            mean_residual_deg=None,
            median_residual_deg=None,
            residual_thresh_deg=float(residual_thresh_deg),
            ransac_iters=int(ransac_iters),
            debug=debug_info,
        )
        if debug and debug_dir is not None:
            _save_debug(debug_dir, preview_meta, preview_images, preview_lines, estimate)
        return estimate

    residuals = _angular_residuals_rad(normals, refined)
    keep = residuals <= threshold_rad
    second = _refine_up_vector(normals, weights, keep)
    if second is not None and np.all(np.isfinite(second)):
        refined = second
        residuals = _angular_residuals_rad(normals, refined)
        keep = residuals <= threshold_rad

    inlier_count = int(np.count_nonzero(keep))
    if inlier_count > 0:
        mean_residual_deg = math.degrees(float(np.average(residuals[keep], weights=weights[keep])))
        median_residual_deg = math.degrees(_weighted_median(residuals[keep].tolist(), weights[keep].tolist()))
    else:
        mean_residual_deg = None
        median_residual_deg = None

    angle_to_world_up_deg = math.degrees(
        math.acos(float(np.clip(np.dot(WORLD_UP, refined), -1.0, 1.0)))
    )
    inlier_ratio = float(inlier_count) / float(total_line_count) if total_line_count > 0 else 0.0
    v_up_list = refined.tolist()
    R_level = _rotation_from_to(WORLD_UP, refined)

    applied = bool(
        total_line_count >= int(min_total_lines)
        and inlier_count >= int(min_inliers)
        and angle_to_world_up_deg <= float(max_apply_deg)
        and np.all(np.isfinite(refined))
        and np.all(np.isfinite(R_level))
    )
    reject_reason = None if applied else (
        "angle_exceeds_max_apply_deg"
        if angle_to_world_up_deg > float(max_apply_deg)
        else "not_enough_inliers"
        if inlier_count < int(min_inliers)
        else "low_confidence"
    )

    for record, residual, is_inlier in zip(records, residuals.tolist(), keep.tolist()):
        record["residual_deg"] = math.degrees(float(residual))
        record["inlier"] = bool(is_inlier)

    debug_info.update(
        {
            "inlier_count": int(inlier_count),
            "inlier_ratio": float(inlier_ratio),
            "mean_residual_deg": mean_residual_deg,
            "median_residual_deg": median_residual_deg,
            "angle_to_world_up_deg": float(angle_to_world_up_deg),
            "reject_reason": reject_reason,
            "applied": bool(applied),
        }
    )
    estimate = LevelEstimate(
        method=method,
        applied=bool(applied),
        R_level=None if not applied else R_level,
        reject_reason=reject_reason,
        v_up=v_up_list,
        angle_to_world_up_deg=float(angle_to_world_up_deg),
        total_line_count=total_line_count,
        inlier_count=int(inlier_count),
        inlier_ratio=float(inlier_ratio),
        mean_residual_deg=None if mean_residual_deg is None else float(mean_residual_deg),
        median_residual_deg=None if median_residual_deg is None else float(median_residual_deg),
        residual_thresh_deg=float(residual_thresh_deg),
        ransac_iters=int(ransac_iters),
        debug=debug_info,
    )
    if debug and debug_dir is not None:
        _save_debug(debug_dir, preview_meta, preview_images, preview_lines, estimate)
    return estimate
