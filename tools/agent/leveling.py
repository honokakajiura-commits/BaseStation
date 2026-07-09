#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Panorama horizon leveling helpers."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .spherical_camera import equirect_to_perspective, make_rotation


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


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
    cutoff = total * 0.5
    idx = int(np.searchsorted(np.cumsum(wts), cutoff, side="left"))
    idx = min(idx, vals.size - 1)
    return float(vals[idx])


def _line_orientation_deg(x1: float, y1: float, x2: float, y2: float) -> float:
    angle = math.degrees(math.atan2(float(y2) - float(y1), float(x2) - float(x1)))
    while angle < -90.0:
        angle += 180.0
    while angle >= 90.0:
        angle -= 180.0
    return float(angle)


def _roll_from_line_angle(angle_deg: float) -> Optional[Tuple[float, str]]:
    """Return visual image roll from a near-horizontal or near-vertical line."""
    angle = float(angle_deg)
    if -30.0 <= angle <= 30.0:
        return angle, "horizontal"
    if 60.0 <= abs(angle) <= 90.0:
        if angle < 0.0:
            return angle + 90.0, "vertical"
        return angle - 90.0, "vertical"
    return None


def _base_failure(method: str = "hough_lines", **extra: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "roll_deg": 0.0,
        "confidence": 0.0,
        "line_count": 0,
        "method": method,
    }
    out.update(extra)
    return out


def estimate_roll_from_crop_lines(img_bgr: np.ndarray) -> Dict[str, Any]:
    """Estimate visual roll in degrees from line segments in a perspective crop.

    Positive roll means near-horizontal lines slope downward to the right in
    OpenCV image coordinates. ``make_level_rotation`` converts this visual roll
    into the opposite source-ray correction used by ``equirect_to_perspective``.
    """
    if img_bgr is None or img_bgr.ndim < 2:
        return _base_failure(reason="invalid_image")

    h, w = img_bgr.shape[:2]
    min_dim = max(1, min(int(w), int(h)))
    if min_dim < 32:
        return _base_failure(reason="image_too_small")

    if img_bgr.ndim == 2:
        gray = img_bgr
    else:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3, L2gradient=True)

    min_line_len = max(32, int(round(min_dim * 0.08)))
    max_line_gap = max(8, int(round(min_dim * 0.02)))
    threshold = max(40, int(round(min_dim * 0.08)))
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=threshold,
        minLineLength=min_line_len,
        maxLineGap=max_line_gap,
    )
    raw_line_count = 0 if lines is None else int(len(lines))
    if lines is None:
        return _base_failure(raw_line_count=0, reason="no_hough_lines")

    candidates: List[Tuple[float, float, str]] = []
    horizontal_count = 0
    vertical_count = 0
    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = [float(v) for v in line]
        length = math.hypot(x2 - x1, y2 - y1)
        if length < float(min_line_len):
            continue
        angle = _line_orientation_deg(x1, y1, x2, y2)
        roll_info = _roll_from_line_angle(angle)
        if roll_info is None:
            continue
        roll, kind = roll_info
        candidates.append((float(roll), float(length), kind))
        if kind == "horizontal":
            horizontal_count += 1
        else:
            vertical_count += 1

    if not candidates:
        return _base_failure(
            raw_line_count=raw_line_count,
            horizontal_count=0,
            vertical_count=0,
            reason="no_level_candidates",
        )

    values = np.asarray([c[0] for c in candidates], dtype=np.float64)
    weights = np.asarray([c[1] for c in candidates], dtype=np.float64)
    initial = _weighted_median(values.tolist(), weights.tolist())
    residuals = np.abs(values - initial)
    mad = _weighted_median(residuals.tolist(), weights.tolist())
    gate = max(4.0, min(15.0, 3.0 * max(float(mad), 1.0)))
    keep = residuals <= gate
    if not bool(np.any(keep)):
        keep = np.ones_like(values, dtype=bool)

    kept_values = values[keep]
    kept_weights = weights[keep]
    roll_deg = _weighted_median(kept_values.tolist(), kept_weights.tolist())
    final_residuals = np.abs(kept_values - roll_deg)
    dispersion = _weighted_median(final_residuals.tolist(), kept_weights.tolist())

    kept_weight = float(np.sum(kept_weights))
    total_weight = float(np.sum(weights))
    consensus_score = kept_weight / total_weight if total_weight > 0.0 else 0.0
    line_score = min(1.0, float(kept_values.size) / 8.0)
    length_score = min(1.0, kept_weight / (float(min_dim) * 3.0))
    dispersion_score = _clamp01(1.0 - float(dispersion) / 10.0)
    confidence = consensus_score * dispersion_score * (0.70 * line_score + 0.30 * length_score)
    if kept_values.size < 2:
        confidence *= 0.25
    elif kept_values.size < 3:
        confidence *= 0.60

    return {
        "roll_deg": float(roll_deg),
        "confidence": _clamp01(confidence),
        "line_count": int(kept_values.size),
        "method": "hough_lines",
        "raw_line_count": raw_line_count,
        "candidate_count": int(len(candidates)),
        "horizontal_count": int(horizontal_count),
        "vertical_count": int(vertical_count),
        "dispersion_deg": float(dispersion),
    }


def estimate_pano_level_correction(
    pano_bgr: np.ndarray,
    yaw_samples: Iterable[float] = (-120, -60, 0, 60, 120),
    pitch_deg: float = 0.0,
    preview_fov: float = 90.0,
    preview_w: int = 768,
    preview_h: int = 768,
    interpolation: int = cv2.INTER_LINEAR,
) -> Dict[str, Any]:
    """Estimate a single visual roll correction for an equirectangular panorama."""
    samples: List[Dict[str, Any]] = []
    yaw_values = [float(yaw) for yaw in yaw_samples]
    if pano_bgr is None or pano_bgr.ndim < 2:
        return {
            "enabled": False,
            "roll_deg": 0.0,
            "confidence": 0.0,
            "sample_count": len(yaw_values),
            "used_sample_count": 0,
            "samples": samples,
            "method": "hough_lines",
            "reason": "invalid_image",
        }

    for yaw in yaw_values:
        sample: Dict[str, Any] = {"yaw_deg": float(yaw), "used_for_leveling": False}
        try:
            preview = equirect_to_perspective(
                pano_bgr,
                yaw=float(yaw),
                pitch=float(pitch_deg),
                roll=0.0,
                fov_x=float(preview_fov),
                out_w=int(preview_w),
                out_h=int(preview_h),
                R_level=None,
                interpolation=interpolation,
            )
            estimate = estimate_roll_from_crop_lines(preview)
            sample.update(estimate)
        except Exception as exc:  # keep the pipeline running on bad images
            sample.update(_base_failure(reason="sample_failed", error=str(exc)))
        samples.append(sample)

    usable_indices: List[int] = []
    usable_rolls: List[float] = []
    usable_weights: List[float] = []
    for idx, sample in enumerate(samples):
        confidence = float(sample.get("confidence", 0.0) or 0.0)
        line_count = int(sample.get("line_count", 0) or 0)
        roll = float(sample.get("roll_deg", 0.0) or 0.0)
        if confidence < 0.15 or line_count < 2 or not math.isfinite(roll):
            continue
        dispersion = max(0.0, float(sample.get("dispersion_deg", 0.0) or 0.0))
        usable_indices.append(idx)
        usable_rolls.append(roll)
        usable_weights.append(math.sqrt(max(1.0, float(line_count))) * max(0.05, confidence) / (1.0 + dispersion))

    if not usable_rolls:
        return {
            "enabled": False,
            "roll_deg": 0.0,
            "confidence": 0.0,
            "sample_count": len(samples),
            "used_sample_count": 0,
            "samples": samples,
            "method": "hough_lines",
            "reason": "no_confident_samples",
        }

    values = np.asarray(usable_rolls, dtype=np.float64)
    weights = np.asarray(usable_weights, dtype=np.float64)
    initial = _weighted_median(values.tolist(), weights.tolist())
    residuals = np.abs(values - initial)
    mad = _weighted_median(residuals.tolist(), weights.tolist())
    gate = max(3.0, min(12.0, 3.0 * max(float(mad), 0.75)))
    keep = residuals <= gate
    if not bool(np.any(keep)):
        keep = np.ones_like(values, dtype=bool)

    kept_rolls = values[keep]
    kept_weights = weights[keep]
    kept_indices = [idx for idx, is_kept in zip(usable_indices, keep.tolist()) if is_kept]
    roll_deg = _weighted_median(kept_rolls.tolist(), kept_weights.tolist())
    final_residuals = np.abs(kept_rolls - roll_deg)
    dispersion = _weighted_median(final_residuals.tolist(), kept_weights.tolist())

    for idx in kept_indices:
        samples[idx]["used_for_leveling"] = True

    kept_sample_count = int(kept_rolls.size)
    avg_conf = float(
        np.average(
            [float(samples[idx].get("confidence", 0.0) or 0.0) for idx in kept_indices],
            weights=kept_weights,
        )
    )
    count_score = min(1.0, kept_sample_count / 3.0)
    if kept_sample_count == 1:
        only = samples[kept_indices[0]]
        if float(only.get("confidence", 0.0) or 0.0) >= 0.65 and int(only.get("line_count", 0) or 0) >= 6:
            count_score = 0.45
        else:
            count_score = 0.0
    dispersion_score = _clamp01(1.0 - float(dispersion) / 8.0)
    confidence = _clamp01(avg_conf * count_score * dispersion_score)
    enabled = bool(confidence > 0.0 and count_score > 0.0)

    if not enabled:
        return {
            "enabled": False,
            "roll_deg": 0.0,
            "confidence": 0.0,
            "sample_count": len(samples),
            "used_sample_count": kept_sample_count,
            "samples": samples,
            "method": "hough_lines",
            "reason": "insufficient_consensus",
        }

    return {
        "enabled": True,
        "roll_deg": float(roll_deg),
        "confidence": confidence,
        "sample_count": len(samples),
        "used_sample_count": kept_sample_count,
        "samples": samples,
        "method": "hough_lines",
        "dispersion_deg": float(dispersion),
    }


def make_level_rotation(roll_deg: float) -> np.ndarray:
    """Build the source-ray leveling rotation for a visual crop roll estimate.

    A positive visual roll slopes downward to the right in image coordinates.
    ``R_level`` maps level-corrected world rays to source panorama rays, so the
    source-frame rotation uses the opposite sign.
    """
    return make_rotation(0.0, 0.0, -float(roll_deg))
