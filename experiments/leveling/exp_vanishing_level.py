#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# EXPERIMENTAL: vanishing-point based crop leveling for left/right views.
# Safe to delete: remove experiments/leveling/exp_vanishing_level.py.
# This script does not affect the main agent pipeline.

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.agent.crop import get_front_left_right_views, render_detection_crop
from tools.agent.leveling import estimate_pano_level_correction, make_level_rotation
from tools.agent.spherical_camera import wrap_yaw_deg


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


@dataclass
class LineRecord:
    x1: float
    y1: float
    x2: float
    y2: float
    length: float
    angle_deg: float
    line: np.ndarray
    candidate: bool
    inlier: bool = False
    residual: Optional[float] = None


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


def _draw_label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    pad = 10
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.55, min(1.0, out.shape[1] / 1200.0))
    thickness = max(1, int(round(scale * 2.0)))
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(out, (0, 0), (tw + pad * 2, th + baseline + pad * 2), (0, 0, 0), -1)
    cv2.putText(out, text, (pad, pad + th), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return out


def _resize_to_height(img: np.ndarray, height: int) -> np.ndarray:
    if img.shape[0] == height:
        return img
    width = max(1, int(round(img.shape[1] * (float(height) / float(img.shape[0])))))
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(value)))


def _clamp01(value: float) -> float:
    return _clamp(float(value), 0.0, 1.0)


def _line_orientation_deg(x1: float, y1: float, x2: float, y2: float) -> float:
    angle = math.degrees(math.atan2(float(y2) - float(y1), float(x2) - float(x1)))
    while angle < -90.0:
        angle += 180.0
    while angle >= 90.0:
        angle -= 180.0
    return float(angle)


def _segment_to_normalized_line(x1: float, y1: float, x2: float, y2: float) -> Optional[np.ndarray]:
    p1 = np.array([float(x1), float(y1), 1.0], dtype=np.float64)
    p2 = np.array([float(x2), float(y2), 1.0], dtype=np.float64)
    line = np.cross(p1, p2)
    norm = float(math.hypot(float(line[0]), float(line[1])))
    if norm <= 1e-9:
        return None
    return line / norm


def _line_intersection(line_a: np.ndarray, line_b: np.ndarray) -> Optional[Tuple[float, float]]:
    p = np.cross(line_a, line_b)
    if abs(float(p[2])) <= 1e-9:
        return None
    x = float(p[0] / p[2])
    y = float(p[1] / p[2])
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return x, y


def _detect_hough_line_records(
    img_bgr: np.ndarray,
    min_length_frac: float,
    vertical_exclude_deg: float,
    horizontal_exclude_deg: float,
) -> Tuple[List[LineRecord], Dict[str, Any]]:
    if img_bgr is None or img_bgr.ndim < 2:
        return [], {"reason": "invalid_image", "raw_line_count": 0, "candidate_count": 0}

    h, w = img_bgr.shape[:2]
    min_dim = max(1, min(int(w), int(h)))
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3, L2gradient=True)

    min_line_len = max(40, int(round(min_dim * float(min_length_frac))))
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
    if lines is None:
        return [], {
            "reason": "no_hough_lines",
            "raw_line_count": 0,
            "candidate_count": 0,
            "min_line_length": int(min_line_len),
            "hough_threshold": int(threshold),
        }

    records: List[LineRecord] = []
    vertical_exclude = abs(float(vertical_exclude_deg))
    horizontal_exclude = abs(float(horizontal_exclude_deg))
    for raw_line in lines[:, 0, :]:
        x1, y1, x2, y2 = [float(v) for v in raw_line]
        length = float(math.hypot(x2 - x1, y2 - y1))
        angle = _line_orientation_deg(x1, y1, x2, y2)
        line = _segment_to_normalized_line(x1, y1, x2, y2)
        if line is None:
            continue
        abs_angle = abs(float(angle))
        candidate = bool(
            length >= float(min_line_len)
            and abs_angle <= (90.0 - vertical_exclude)
            and abs_angle >= horizontal_exclude
        )
        records.append(
            LineRecord(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                length=length,
                angle_deg=angle,
                line=line,
                candidate=candidate,
            )
        )

    return records, {
        "raw_line_count": int(len(lines)),
        "candidate_count": int(sum(1 for record in records if record.candidate)),
        "min_line_length": int(min_line_len),
        "max_line_gap": int(max_line_gap),
        "hough_threshold": int(threshold),
    }


def _view_expected_vp_sign(view: str) -> int:
    if str(view).lower() == "left":
        return 1
    if str(view).lower() == "right":
        return -1
    return 0


def _vp_side_score(vp_x: float, width: int, expected_sign: int) -> float:
    if expected_sign == 0:
        return 1.0
    signed_dx = float(expected_sign) * (float(vp_x) - float(width) / 2.0)
    if signed_dx <= 0.0:
        return 0.0
    if signed_dx >= float(width) * 0.5:
        return 1.0
    return max(0.25, signed_dx / (float(width) * 0.5))


def _weighted_least_squares_vp(records: Sequence[LineRecord]) -> Optional[Tuple[float, float]]:
    if len(records) < 2:
        return None
    lines = np.asarray([record.line for record in records], dtype=np.float64)
    weights = np.asarray([max(1.0, record.length) for record in records], dtype=np.float64)
    A = lines[:, :2]
    b = -lines[:, 2]
    sw = np.sqrt(weights / max(1.0, float(np.mean(weights))))
    Aw = A * sw[:, None]
    bw = b * sw
    try:
        solution, _, rank, _ = np.linalg.lstsq(Aw, bw, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if int(rank) < 2:
        return None
    x, y = float(solution[0]), float(solution[1])
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return x, y


def _robust_weighted_median(values: Sequence[float], weights: Sequence[float]) -> float:
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


def _estimate_vanishing_point(
    records: Sequence[LineRecord],
    width: int,
    height: int,
    expected_sign: int,
    iterations: int,
    residual_threshold_px: float,
    rng_seed: int,
) -> Dict[str, Any]:
    candidates = [record for record in records if record.candidate]
    if len(candidates) < 2:
        return {
            "vp": None,
            "inliers": [],
            "residual": None,
            "confidence": 0.0,
            "reason": "not_enough_candidates",
        }

    rng = random.Random(int(rng_seed))
    weights = [max(1.0, record.length) for record in candidates]
    total_weight = float(sum(weights))
    residual_threshold = max(4.0, float(residual_threshold_px))
    best: Optional[Dict[str, Any]] = None

    index_pairs: List[Tuple[int, int]] = []
    if len(candidates) <= 80:
        for i in range(len(candidates) - 1):
            for j in range(i + 1, len(candidates)):
                index_pairs.append((i, j))
        rng.shuffle(index_pairs)
        index_pairs = index_pairs[: max(1, int(iterations))]
    else:
        for _ in range(max(1, int(iterations))):
            i, j = rng.choices(range(len(candidates)), weights=weights, k=2)
            if i == j:
                j = (j + 1) % len(candidates)
            index_pairs.append((min(i, j), max(i, j)))

    max_abs_coord = float(max(width, height)) * 30.0
    for i, j in index_pairs:
        vp = _line_intersection(candidates[i].line, candidates[j].line)
        if vp is None:
            continue
        vp_x, vp_y = vp
        if abs(vp_x) > max_abs_coord or abs(vp_y) > max_abs_coord:
            continue
        side_score = _vp_side_score(vp_x, width, expected_sign)
        if expected_sign != 0 and side_score <= 0.0:
            continue

        residuals = np.asarray([abs(float(record.line[0]) * vp_x + float(record.line[1]) * vp_y + float(record.line[2])) for record in candidates])
        keep = residuals <= residual_threshold
        inlier_count = int(np.count_nonzero(keep))
        if inlier_count < 2:
            continue
        inlier_weight = float(np.sum(np.asarray(weights, dtype=np.float64)[keep]))
        kept_residuals = residuals[keep]
        kept_weights = np.asarray(weights, dtype=np.float64)[keep]
        residual = _robust_weighted_median(kept_residuals.tolist(), kept_weights.tolist())
        score = (
            inlier_count,
            inlier_weight,
            side_score,
            -float(residual),
        )
        if best is None or score > best["score"]:
            best = {
                "score": score,
                "vp": (float(vp_x), float(vp_y)),
                "inlier_mask": keep,
                "residual": float(residual),
                "side_score": float(side_score),
            }

    if best is None:
        return {
            "vp": None,
            "inliers": [],
            "residual": None,
            "confidence": 0.0,
            "reason": "no_ransac_consensus",
        }

    initial_keep = np.asarray(best["inlier_mask"], dtype=bool)
    inliers = [record for record, keep in zip(candidates, initial_keep.tolist()) if keep]
    refined_vp = _weighted_least_squares_vp(inliers) or best["vp"]
    vp_x, vp_y = float(refined_vp[0]), float(refined_vp[1])

    residuals = np.asarray([abs(float(record.line[0]) * vp_x + float(record.line[1]) * vp_y + float(record.line[2])) for record in candidates])
    keep = residuals <= residual_threshold
    inliers = [record for record, is_inlier in zip(candidates, keep.tolist()) if is_inlier]
    if len(inliers) >= 2:
        refined_vp = _weighted_least_squares_vp(inliers) or refined_vp
        vp_x, vp_y = float(refined_vp[0]), float(refined_vp[1])
        residuals = np.asarray([abs(float(record.line[0]) * vp_x + float(record.line[1]) * vp_y + float(record.line[2])) for record in candidates])
        keep = residuals <= residual_threshold
        inliers = [record for record, is_inlier in zip(candidates, keep.tolist()) if is_inlier]

    kept_residuals = residuals[keep]
    kept_weights = np.asarray(weights, dtype=np.float64)[keep]
    residual = _robust_weighted_median(kept_residuals.tolist(), kept_weights.tolist()) if len(inliers) else float("inf")
    inlier_weight = float(np.sum(kept_weights)) if len(inliers) else 0.0
    consensus_score = inlier_weight / total_weight if total_weight > 0.0 else 0.0
    line_score = min(1.0, float(len(inliers)) / 8.0)
    length_score = min(1.0, inlier_weight / (float(min(width, height)) * 3.0))
    residual_score = _clamp01(1.0 - float(residual) / max(1.0, residual_threshold * 2.0))
    side_score = _vp_side_score(vp_x, width, expected_sign)
    confidence = side_score * residual_score * (0.45 * consensus_score + 0.35 * line_score + 0.20 * length_score)
    if len(inliers) < 3:
        confidence *= 0.25
    elif len(inliers) < 5:
        confidence *= 0.70

    for record in records:
        record.inlier = False
        record.residual = None
    inlier_ids = {id(record) for record in inliers}
    candidate_residual_by_id = {id(record): float(res) for record, res in zip(candidates, residuals.tolist())}
    for record in records:
        if id(record) in candidate_residual_by_id:
            record.residual = candidate_residual_by_id[id(record)]
        if id(record) in inlier_ids:
            record.inlier = True

    return {
        "vp": (float(vp_x), float(vp_y)),
        "inliers": inliers,
        "residual": float(residual),
        "confidence": _clamp01(confidence),
        "reason": None,
    }


def _horizon_y(width: int, height: int, fov_deg: float, pitch_deg: float) -> float:
    f = (float(width) / 2.0) / math.tan(math.radians(float(fov_deg)) / 2.0)
    return float(height) / 2.0 + f * math.tan(math.radians(float(pitch_deg)))


def _estimate_roll_from_vp(
    vp: Optional[Tuple[float, float]],
    horizon_y: float,
    width: int,
) -> Tuple[float, str]:
    if vp is None:
        return 0.0, "no_vp"
    vp_x, vp_y = float(vp[0]), float(vp[1])
    dx = vp_x - float(width) / 2.0
    if abs(dx) < max(1.0, float(width) * 0.08):
        return 0.0, "vp_too_close_to_center_x"
    roll_rad = (float(horizon_y) - vp_y) / dx
    roll_deg = math.degrees(roll_rad)
    if not math.isfinite(roll_deg):
        return 0.0, "nonfinite_roll"
    return float(roll_deg), "ok"


def _clip_point_to_image(x: float, y: float, width: int, height: int) -> Tuple[int, int]:
    return (
        int(round(_clamp(float(x), 0.0, float(width - 1)))),
        int(round(_clamp(float(y), 0.0, float(height - 1)))),
    )


def _ray_to_box_endpoint(
    start_x: float,
    start_y: float,
    target_x: float,
    target_y: float,
    width: int,
    height: int,
) -> Optional[Tuple[int, int]]:
    dx = float(target_x) - float(start_x)
    dy = float(target_y) - float(start_y)
    if abs(dx) <= 1e-9 and abs(dy) <= 1e-9:
        return None
    ts: List[float] = []
    if dx > 1e-9:
        ts.append((float(width - 1) - float(start_x)) / dx)
    elif dx < -1e-9:
        ts.append((0.0 - float(start_x)) / dx)
    if dy > 1e-9:
        ts.append((float(height - 1) - float(start_y)) / dy)
    elif dy < -1e-9:
        ts.append((0.0 - float(start_y)) / dy)
    valid = [t for t in ts if t > 0.0 and math.isfinite(t)]
    if not valid:
        return None
    t = min(valid)
    return _clip_point_to_image(float(start_x) + dx * t, float(start_y) + dy * t, width, height)


def _draw_hough_lines(img_bgr: np.ndarray, records: Sequence[LineRecord]) -> np.ndarray:
    out = img_bgr.copy()
    for record in records:
        color = (120, 120, 120)
        thickness = 1
        if record.candidate:
            color = (0, 210, 255)
            thickness = 2
        if record.inlier:
            color = (0, 140, 255)
            thickness = 3
        cv2.line(
            out,
            (int(round(record.x1)), int(round(record.y1))),
            (int(round(record.x2)), int(round(record.y2))),
            color,
            thickness,
            cv2.LINE_AA,
        )
    return out


def _draw_vp_debug(
    img_bgr: np.ndarray,
    records: Sequence[LineRecord],
    vp: Optional[Tuple[float, float]],
    horizon_y: float,
    dy: Optional[float],
    estimated_roll_deg: float,
    applied_roll_deg: float,
    confidence: float,
    view: str,
) -> np.ndarray:
    out = _draw_hough_lines(img_bgr, records)
    h, w = out.shape[:2]

    if vp is not None:
        vp_x, vp_y = float(vp[0]), float(vp[1])
        for record in records:
            if not record.inlier:
                continue
            mx = (record.x1 + record.x2) / 2.0
            my = (record.y1 + record.y2) / 2.0
            end = _ray_to_box_endpoint(mx, my, vp_x, vp_y, w, h)
            if end is not None:
                cv2.line(out, (int(round(mx)), int(round(my))), end, (255, 255, 0), 1, cv2.LINE_AA)

        vp_point = _clip_point_to_image(vp_x, vp_y, w, h)
        vp_inside = 0.0 <= vp_x <= float(w - 1) and 0.0 <= vp_y <= float(h - 1)
        cv2.circle(out, vp_point, 12, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(out, vp_point, 16, (255, 255, 255), 2, cv2.LINE_AA)
        label = "vp" if vp_inside else "vp off"
        cv2.putText(out, label, (vp_point[0] + 14, max(22, vp_point[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, label, (vp_point[0] + 14, max(22, vp_point[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

    horizon_inside = 0.0 <= float(horizon_y) <= float(h - 1)
    hy = int(round(_clamp(float(horizon_y), 0.0, float(h - 1))))
    cv2.line(out, (0, hy), (w - 1, hy), (255, 0, 255), 2, cv2.LINE_AA)
    horizon_label = f"horizon_y={horizon_y:.1f}" if horizon_inside else f"horizon_y={horizon_y:.1f} off"
    cv2.putText(out, horizon_label, (12, max(24, min(h - 12, hy - 10))), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, horizon_label, (12, max(24, min(h - 12, hy - 10))), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 1, cv2.LINE_AA)

    text_lines = [
        f"EXPERIMENTAL vp leveling: {view}",
        f"dy={0.0 if dy is None else dy:.1f}px",
        f"estimated_roll={estimated_roll_deg:.3f} deg",
        f"applied_roll={applied_roll_deg:.3f} deg",
        f"confidence={confidence:.3f}",
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.65
    thickness = 1
    line_h = 24
    box_w = 430
    box_h = 16 + line_h * len(text_lines)
    cv2.rectangle(out, (0, h - box_h), (box_w, h), (0, 0, 0), -1)
    for idx, text in enumerate(text_lines):
        y = h - box_h + 26 + idx * line_h
        cv2.putText(out, text, (12, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return out


def _make_comparison(
    crop_no_level: np.ndarray,
    crop_level: np.ndarray,
    crop_vp_level: np.ndarray,
    vp_debug: np.ndarray,
) -> np.ndarray:
    target_h = min(620, max(320, crop_no_level.shape[0] // 2))
    panels = [
        _draw_label(_resize_to_height(crop_no_level, target_h), "crop_no_level"),
        _draw_label(_resize_to_height(crop_level, target_h), "crop_level_hough_angle"),
        _draw_label(_resize_to_height(crop_vp_level, target_h), "crop_vp_level"),
        _draw_label(_resize_to_height(vp_debug, target_h), "vp_debug"),
    ]
    return np.hstack(panels)


def _line_records_for_json(records: Sequence[LineRecord], max_records: int = 300) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    ordered = sorted(records, key=lambda record: (not record.inlier, not record.candidate, -record.length))
    for record in ordered[: max(0, int(max_records))]:
        out.append(
            {
                "xyxy": [int(round(record.x1)), int(round(record.y1)), int(round(record.x2)), int(round(record.y2))],
                "length": float(record.length),
                "angle_deg": float(record.angle_deg),
                "candidate": bool(record.candidate),
                "inlier": bool(record.inlier),
                "residual": None if record.residual is None else float(record.residual),
            }
        )
    return out


def _read_yaw_map(path: Optional[Path]) -> Dict[str, float]:
    if path is None or not path.exists():
        return {}
    out: Dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        fid = str(row.get("fid") or "")
        if not fid:
            continue
        try:
            out[fid] = float(row.get("yaw_center", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
    return out


def _discover_panos(pano_dir: Path) -> List[Path]:
    if not pano_dir.exists():
        raise ValueError(f"pano_dir does not exist: {pano_dir}")
    return sorted(path for path in pano_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def _build_hough_level(pano: np.ndarray, min_confidence: float) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    level_meta = dict(
        estimate_pano_level_correction(
            pano,
            yaw_samples=(-120.0, -60.0, 0.0, 60.0, 120.0),
            pitch_deg=0.0,
            preview_fov=90.0,
            preview_w=768,
            preview_h=768,
        )
    )
    confidence = float(level_meta.get("confidence", 0.0) or 0.0)
    applied = bool(level_meta.get("enabled", False)) and confidence >= float(min_confidence)
    level_meta["applied"] = bool(applied)
    level_meta["min_confidence"] = float(min_confidence)
    R_level = make_level_rotation(float(level_meta.get("roll_deg", 0.0) or 0.0)) if applied else None
    return R_level, level_meta


def _process_view(
    pano: np.ndarray,
    pano_path: Path,
    out_dir: Path,
    view: str,
    yaw: float,
    pitch: float,
    fov: float,
    width: int,
    height: int,
    crop_strategy: str,
    supersample: float,
    interpolation: str,
    R_hough_level: Optional[np.ndarray],
    hough_level_meta: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    crop_no_level, crop_no_level_meta = render_detection_crop(
        pano_bgr=pano,
        yaw_deg=float(yaw),
        pitch_deg=float(pitch),
        fov_deg=float(fov),
        out_w=int(width),
        out_h=int(height),
        crop_strategy=crop_strategy,
        supersample=float(supersample),
        interpolation=interpolation,
        R_level=None,
        roll_deg=0.0,
        level_meta=hough_level_meta,
    )
    crop_level, crop_level_meta = render_detection_crop(
        pano_bgr=pano,
        yaw_deg=float(yaw),
        pitch_deg=float(pitch),
        fov_deg=float(fov),
        out_w=int(width),
        out_h=int(height),
        crop_strategy=crop_strategy,
        supersample=float(supersample),
        interpolation=interpolation,
        R_level=R_hough_level,
        roll_deg=0.0,
        level_meta=hough_level_meta,
    )

    records, detect_meta = _detect_hough_line_records(
        crop_no_level,
        min_length_frac=float(args.min_length_frac),
        vertical_exclude_deg=float(args.vertical_exclude_deg),
        horizontal_exclude_deg=float(args.horizontal_exclude_deg),
    )
    expected_sign = _view_expected_vp_sign(view)
    residual_threshold = max(4.0, float(args.residual_threshold_px))
    vp_estimate = _estimate_vanishing_point(
        records,
        width=int(width),
        height=int(height),
        expected_sign=expected_sign,
        iterations=int(args.ransac_iterations),
        residual_threshold_px=residual_threshold,
        rng_seed=int(args.seed) + int(abs(round(float(yaw) * 10.0))),
    )
    vp = vp_estimate.get("vp")
    horizon_y = _horizon_y(width=int(width), height=int(height), fov_deg=float(fov), pitch_deg=float(pitch))
    vp_y: Optional[float] = None if vp is None else float(vp[1])
    dy: Optional[float] = None if vp_y is None else float(vp_y - horizon_y)
    estimated_roll_deg, roll_reason = _estimate_roll_from_vp(
        vp,
        horizon_y=horizon_y,
        width=int(width),
    )
    confidence = float(vp_estimate.get("confidence", 0.0) or 0.0)
    inlier_count = int(len(vp_estimate.get("inliers", []) or []))
    residual = vp_estimate.get("residual")
    apply_vp_roll = bool(
        roll_reason == "ok"
        and confidence >= float(args.vp_min_confidence)
        and inlier_count >= int(args.min_inliers)
        and residual is not None
        and float(residual) <= float(args.max_apply_residual_px)
    )
    applied_roll_deg = (
        _clamp(float(estimated_roll_deg), -abs(float(args.max_roll_deg)), abs(float(args.max_roll_deg)))
        if apply_vp_roll
        else 0.0
    )

    crop_vp_level, crop_vp_level_meta = render_detection_crop(
        pano_bgr=pano,
        yaw_deg=float(yaw),
        pitch_deg=float(pitch),
        fov_deg=float(fov),
        out_w=int(width),
        out_h=int(height),
        crop_strategy=crop_strategy,
        supersample=float(supersample),
        interpolation=interpolation,
        R_level=None,
        roll_deg=applied_roll_deg,
        level_meta=hough_level_meta,
    )

    vp_lines = _draw_hough_lines(crop_no_level, records)
    vp_debug = _draw_vp_debug(
        crop_no_level,
        records,
        None if vp is None else (float(vp[0]), float(vp[1])),
        horizon_y=horizon_y,
        dy=dy,
        estimated_roll_deg=float(estimated_roll_deg),
        applied_roll_deg=applied_roll_deg,
        confidence=confidence,
        view=view,
    )
    comparison = _make_comparison(crop_no_level, crop_level, crop_vp_level, vp_debug)

    _write_image(out_dir / "crop_no_level.jpg", crop_no_level)
    _write_image(out_dir / "crop_level.jpg", crop_level)
    _write_image(out_dir / "crop_vp_level.jpg", crop_vp_level)
    _write_image(out_dir / "vp_lines.jpg", vp_lines)
    _write_image(out_dir / "vp_debug.jpg", vp_debug)
    _write_image(out_dir / "comparison.jpg", comparison)

    meta = {
        "experiment": "EXPERIMENTAL vanishing point crop leveling",
        "pano": str(pano_path),
        "view": str(view),
        "yaw": float(yaw),
        "pitch": float(pitch),
        "fov": float(fov),
        "width": int(width),
        "height": int(height),
        "vp_x": None if vp is None else float(vp[0]),
        "vp_y": None if vp is None else float(vp[1]),
        "horizon_y": float(horizon_y),
        "dy": dy,
        "estimated_roll_deg": float(estimated_roll_deg),
        "applied_roll_deg": float(applied_roll_deg),
        "confidence": float(confidence),
        "line_count": int(detect_meta.get("candidate_count", 0) or 0),
        "inlier_count": int(inlier_count),
        "residual": None if residual is None else float(residual),
        "raw_line_count": int(detect_meta.get("raw_line_count", 0) or 0),
        "candidate_count": int(detect_meta.get("candidate_count", 0) or 0),
        "expected_vp_side": "right" if expected_sign > 0 else "left" if expected_sign < 0 else "none",
        "roll_reason": str(roll_reason),
        "reason": vp_estimate.get("reason"),
        "applied": bool(apply_vp_roll),
        "vp_min_confidence": float(args.vp_min_confidence),
        "min_inliers": int(args.min_inliers),
        "max_roll_deg": float(args.max_roll_deg),
        "residual_threshold_px": float(residual_threshold),
        "max_apply_residual_px": float(args.max_apply_residual_px),
        "hough_level_meta": _json_safe(hough_level_meta),
        "crop_no_level_meta": _json_safe(crop_no_level_meta),
        "crop_level_meta": _json_safe(crop_level_meta),
        "crop_vp_level_meta": _json_safe(crop_vp_level_meta),
        "detect_meta": _json_safe(detect_meta),
        "outputs": {
            "crop_no_level": str(out_dir / "crop_no_level.jpg"),
            "crop_level": str(out_dir / "crop_level.jpg"),
            "crop_vp_level": str(out_dir / "crop_vp_level.jpg"),
            "vp_lines": str(out_dir / "vp_lines.jpg"),
            "vp_debug": str(out_dir / "vp_debug.jpg"),
            "comparison": str(out_dir / "comparison.jpg"),
            "vp_level_meta": str(out_dir / "vp_level_meta.json"),
        },
        "lines": _line_records_for_json(records),
    }
    (out_dir / "vp_level_meta.json").write_text(json.dumps(_json_safe(meta), ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def _view_specs_for_args(args: argparse.Namespace) -> Dict[str, Tuple[float, float]]:
    specs = get_front_left_right_views(float(args.front_fov), float(args.side_fov), yaw_side_deg=90.0)
    return {view: (float(yaw_off), float(fov)) for view, yaw_off, fov in specs}


def _process_pano(
    pano_path: Path,
    out_dir: Path,
    yaw_center: float,
    views: Sequence[str],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    pano = cv2.imread(str(pano_path))
    if pano is None:
        raise ValueError(f"failed to read panorama: {pano_path}")

    R_hough_level, hough_level_meta = _build_hough_level(pano, float(args.level_min_confidence))
    view_specs = _view_specs_for_args(args)
    outputs: List[Dict[str, Any]] = []
    for view in views:
        key = str(view).strip().lower()
        if key not in view_specs:
            raise ValueError(f"unsupported view: {view}")
        yaw_off, fov = view_specs[key]
        yaw = wrap_yaw_deg(float(yaw_center) + yaw_off)
        view_out_dir = out_dir / key
        outputs.append(
            _process_view(
                pano=pano,
                pano_path=pano_path,
                out_dir=view_out_dir,
                view=key,
                yaw=float(yaw),
                pitch=float(args.pitch),
                fov=float(fov),
                width=int(args.width),
                height=int(args.height),
                crop_strategy=str(args.crop_strategy),
                supersample=float(args.supersample),
                interpolation=str(args.interpolation),
                R_hough_level=R_hough_level,
                hough_level_meta=hough_level_meta,
                args=args,
            )
        )
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="EXPERIMENTAL: test vanishing-point based crop leveling for left/right Panoramax crops.",
    )
    parser.add_argument("--pano", type=Path, help="Single panorama image.")
    parser.add_argument("--pano_dir", type=Path, help="Directory of panorama images for batch/random runs.")
    parser.add_argument("--yaw_map", type=Path, help="Optional yaw_map.jsonl with fid/yaw_center entries.")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--yaw_center", type=float, default=0.0, help="Yaw center for --pano mode or fallback for batch.")
    parser.add_argument("--views", default="left,right", help="Comma-separated views. Default: left,right.")
    parser.add_argument("--random", type=int, default=0, help="Randomly sample this many panos from --pano_dir. 0 means all.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pitch", type=float, default=40.0)
    parser.add_argument("--front_fov", type=float, default=105.0)
    parser.add_argument("--side_fov", type=float, default=90.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=1280)
    parser.add_argument("--crop_strategy", default="ui_like")
    parser.add_argument("--supersample", type=float, default=1.25)
    parser.add_argument("--interpolation", default="cubic")
    parser.add_argument("--level_min_confidence", type=float, default=0.25)

    parser.add_argument("--min_length_frac", type=float, default=0.08)
    parser.add_argument("--vertical_exclude_deg", type=float, default=10.0)
    parser.add_argument("--horizontal_exclude_deg", type=float, default=3.0)
    parser.add_argument("--ransac_iterations", type=int, default=1200)
    parser.add_argument("--residual_threshold_px", type=float, default=12.0)
    parser.add_argument("--max_apply_residual_px", type=float, default=14.0)
    parser.add_argument("--vp_min_confidence", type=float, default=0.25)
    parser.add_argument("--min_inliers", type=int, default=4)
    parser.add_argument("--max_roll_deg", type=float, default=2.5)
    parser.add_argument("--print_full_index", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    views = [view.strip().lower() for view in str(args.views).split(",") if view.strip()]
    if not views:
        raise ValueError("--views must contain at least one view")

    all_meta: List[Dict[str, Any]] = []
    if args.pano is not None:
        pano_path = Path(args.pano)
        pano_out_dir = Path(args.out_dir)
        all_meta.extend(_process_pano(pano_path, pano_out_dir, float(args.yaw_center), views, args))
    else:
        if args.pano_dir is None:
            raise ValueError("either --pano or --pano_dir is required")
        yaw_map_path = args.yaw_map
        if yaw_map_path is None:
            candidate = Path(args.pano_dir).parent / "yaw_map.jsonl"
            yaw_map_path = candidate if candidate.exists() else None
        yaw_map = _read_yaw_map(yaw_map_path)
        panos = _discover_panos(Path(args.pano_dir))
        if int(args.random) > 0 and len(panos) > int(args.random):
            rng = random.Random(int(args.seed))
            panos = sorted(rng.sample(panos, int(args.random)))

        for idx, pano_path in enumerate(panos, start=1):
            fid = pano_path.stem
            yaw_center = float(yaw_map.get(fid, float(args.yaw_center)))
            pano_out_dir = Path(args.out_dir) / f"{idx:02d}_{fid}"
            all_meta.extend(_process_pano(pano_path, pano_out_dir, yaw_center, views, args))

    index = {
        "experiment": "EXPERIMENTAL vanishing point crop leveling",
        "count": len(all_meta),
        "views": views,
        "outputs": _json_safe([{k: v for k, v in row.items() if k != "lines"} for row in all_meta]),
    }
    index_path = Path(args.out_dir) / "vp_level_index.json"
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    if bool(args.print_full_index):
        print(json.dumps(_json_safe(index), ensure_ascii=False, indent=2))
    else:
        summary = {
            "experiment": index["experiment"],
            "count": index["count"],
            "views": views,
            "index": str(index_path),
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
