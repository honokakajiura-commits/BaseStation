#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# EXPERIMENTAL: spherical upright adjustment validation
# This script is inspired by "Upright Adjustment of 360 Spherical Panoramas".
# It is only for testing panorama-level R_level estimation.
# Safe to delete: remove experiments/leveling/ directory.
# It does not affect the main agent pipeline.

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

from tools.agent.crop import render_detection_crop
from tools.agent.spherical_camera import (
    pixel_to_camera_ray,
    apply_rotation_to_rays,
    make_rotation,
    normalize_ray,
    ray_to_yaw_pitch,
)


YAW_OFFSETS = (-135.0, -90.0, -45.0, 0.0, 45.0, 90.0, 135.0)
WORLD_UP = np.array([0.0, 1.0, 0.0], dtype=np.float64)
INLIER_THRESHOLD_DEG = 2.5


@dataclass
class GreatCircleLine:
    preview_index: int
    yaw_sample: float
    x1: float
    y1: float
    x2: float
    y2: float
    length: float
    angle_deg: float
    normal: np.ndarray
    weight: float
    residual_deg: Optional[float] = None
    inlier: bool = False


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


def _yaw_tag(yaw: float) -> str:
    value = _wrap_yaw_deg(float(yaw))
    sign = "p" if value >= 0.0 else "m"
    abs_value = abs(value)
    if abs(abs_value - round(abs_value)) < 1e-6:
        return f"{sign}{int(round(abs_value)):03d}"
    return f"{sign}{abs_value:.1f}".replace(".", "p")


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


def _make_comparison(no_level: np.ndarray, level: np.ndarray, level_inv: np.ndarray) -> np.ndarray:
    target_h = min(620, max(320, no_level.shape[0] // 2))
    panels = [
        _draw_label(_resize_to_height(no_level, target_h), "crop_no_level"),
        _draw_label(_resize_to_height(level, target_h), "crop_spherical_level"),
        _draw_label(_resize_to_height(level_inv, target_h), "crop_spherical_level_inv"),
    ]
    return np.hstack(panels)


def _line_orientation_deg(x1: float, y1: float, x2: float, y2: float) -> float:
    angle = math.degrees(math.atan2(float(y2) - float(y1), float(x2) - float(x1)))
    while angle < -90.0:
        angle += 180.0
    while angle >= 90.0:
        angle -= 180.0
    return float(angle)


def _detect_hough_segments(preview_bgr: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    if preview_bgr is None or preview_bgr.ndim < 2:
        return np.zeros((0, 4), dtype=np.float64), None, {"reason": "invalid_image", "raw_line_count": 0}

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
        return np.zeros((0, 4), dtype=np.float64), edges, meta
    return lines[:, 0, :].astype(np.float64), edges, meta


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


def _collect_great_circles(
    pano_bgr: np.ndarray,
    out_dir: Path,
    args: argparse.Namespace,
) -> Tuple[List[GreatCircleLine], List[Dict[str, Any]]]:
    debug_dir = out_dir / "preview_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    records: List[GreatCircleLine] = []
    previews: List[Dict[str, Any]] = []

    for idx, yaw_off in enumerate(YAW_OFFSETS, start=1):
        yaw_sample = _wrap_yaw_deg(float(args.yaw_center) + float(yaw_off))
        preview, preview_meta = render_detection_crop(
            pano_bgr=pano_bgr,
            yaw_deg=yaw_sample,
            pitch_deg=float(args.pitch_detect),
            fov_deg=float(args.preview_fov),
            out_w=int(args.preview_width),
            out_h=int(args.preview_height),
            crop_strategy="ui_like",
            supersample=1.0,
            interpolation="linear",
            R_level=None,
            roll_deg=0.0,
            level_meta=None,
        )
        tag = _yaw_tag(yaw_sample)
        preview_path = debug_dir / f"preview_yaw_{tag}.jpg"
        _write_image(preview_path, preview)

        raw_lines, _, detect_meta = _detect_hough_segments(preview)
        local_count = 0
        for raw_line in raw_lines:
            x1, y1, x2, y2 = [float(v) for v in raw_line]
            length = float(math.hypot(x2 - x1, y2 - y1))
            min_len = float(detect_meta.get("min_line_length", 0) or 0)
            if length < min_len:
                continue
            normal = _segment_to_great_circle(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                yaw_sample=yaw_sample,
                pitch_detect=float(args.pitch_detect),
                preview_w=int(args.preview_width),
                preview_h=int(args.preview_height),
                preview_fov=float(args.preview_fov),
            )
            if normal is None:
                continue
            records.append(
                GreatCircleLine(
                    preview_index=idx,
                    yaw_sample=float(yaw_sample),
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    length=length,
                    angle_deg=_line_orientation_deg(x1, y1, x2, y2),
                    normal=normal,
                    weight=max(1.0, length),
                )
            )
            local_count += 1

        previews.append(
            {
                "preview_index": idx,
                "yaw_sample": float(yaw_sample),
                "yaw_offset": float(yaw_off),
                "preview_path": str(preview_path),
                "lines_path": str(debug_dir / f"preview_yaw_{tag}_lines.jpg"),
                "lines_json_path": str(debug_dir / f"preview_yaw_{tag}_lines.json"),
                "preview_meta": _json_safe(preview_meta),
                "detect_meta": _json_safe(detect_meta),
                "line_count": int(local_count),
            }
        )

    return records, previews


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


def _candidate_from_normals(n1: np.ndarray, n2: np.ndarray) -> Optional[np.ndarray]:
    v = np.cross(np.asarray(n1, dtype=np.float64), np.asarray(n2, dtype=np.float64))
    norm = float(np.linalg.norm(v))
    if norm <= 1e-9:
        return None
    v = np.asarray(normalize_ray(v), dtype=np.float64)
    if float(v[1]) < 0.0:
        v = -v
    return v


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


def _estimate_up_vector(records: List[GreatCircleLine], args: argparse.Namespace) -> Dict[str, Any]:
    if len(records) < 2:
        return {
            "ok": False,
            "v_up": None,
            "reject_reason": "not_enough_great_circles",
            "inlier_count": 0,
            "mean_residual_deg": None,
            "median_residual_deg": None,
        }

    normals = np.asarray([record.normal for record in records], dtype=np.float64)
    weights = np.asarray([max(1.0, record.weight) for record in records], dtype=np.float64)
    threshold_rad = math.radians(INLIER_THRESHOLD_DEG)
    rng = random.Random(int(args.seed))
    best: Optional[Dict[str, Any]] = None
    mean_weight = max(1.0, float(np.mean(weights)))

    for _ in range(max(1, int(args.ransac_iters))):
        i, j = rng.choices(range(len(records)), weights=weights.tolist(), k=2)
        if i == j:
            continue
        v = _candidate_from_normals(normals[i], normals[j])
        if v is None:
            continue
        residuals = _angular_residuals_rad(normals, v)
        keep = residuals <= threshold_rad
        inlier_count = int(np.count_nonzero(keep))
        if inlier_count < 2:
            continue
        inlier_weight = float(np.sum(weights[keep]))
        med_residual = _weighted_median(residuals[keep].tolist(), weights[keep].tolist())
        up_prior = max(0.0, float(v[1])) ** 4
        # This simplified experiment does not classify Manhattan/Atlanta line
        # families. Without a weak +Y prior, dense horizontal structures such as
        # wires can dominate and produce a horizontal vanishing direction.
        score = up_prior * (float(inlier_count) + inlier_weight / mean_weight) - float(med_residual)
        if best is None or score > float(best["score"]):
            best = {
                "score": float(score),
                "up_prior": float(up_prior),
                "v": v,
                "keep": keep,
                "median_residual_rad": float(med_residual),
            }

    if best is None:
        return {
            "ok": False,
            "v_up": None,
            "reject_reason": "no_ransac_consensus",
            "inlier_count": 0,
            "mean_residual_deg": None,
            "median_residual_deg": None,
        }

    refined = _refine_up_vector(normals, weights, np.asarray(best["keep"], dtype=bool))
    if refined is None:
        refined = np.asarray(best["v"], dtype=np.float64)

    residuals = _angular_residuals_rad(normals, refined)
    keep = residuals <= threshold_rad
    second_refined = _refine_up_vector(normals, weights, keep)
    if second_refined is not None:
        refined = second_refined
        residuals = _angular_residuals_rad(normals, refined)
        keep = residuals <= threshold_rad

    for record, residual, is_inlier in zip(records, residuals.tolist(), keep.tolist()):
        record.residual_deg = math.degrees(float(residual))
        record.inlier = bool(is_inlier)

    inlier_count = int(np.count_nonzero(keep))
    if inlier_count == 0:
        mean_residual = None
        median_residual = None
    else:
        mean_residual = math.degrees(float(np.average(residuals[keep], weights=weights[keep])))
        median_residual = math.degrees(_weighted_median(residuals[keep].tolist(), weights[keep].tolist()))

    reject_reason = None
    ok = True
    if inlier_count < int(args.min_inliers):
        ok = False
        reject_reason = "not_enough_inliers"

    return {
        "ok": ok,
        "v_up": refined,
        "reject_reason": reject_reason,
        "inlier_count": inlier_count,
        "mean_residual_deg": mean_residual,
        "median_residual_deg": median_residual,
        "inlier_threshold_deg": INLIER_THRESHOLD_DEG,
        "initial_v_up": np.asarray(best["v"], dtype=np.float64),
        "initial_up_prior": float(best["up_prior"]),
        "initial_median_residual_deg": math.degrees(float(best["median_residual_rad"])),
    }


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


def _line_to_json(record: GreatCircleLine) -> Dict[str, Any]:
    return {
        "preview_index": int(record.preview_index),
        "yaw_sample": float(record.yaw_sample),
        "xyxy": [float(record.x1), float(record.y1), float(record.x2), float(record.y2)],
        "length": float(record.length),
        "angle_deg": float(record.angle_deg),
        "great_circle_normal": _json_safe(record.normal),
        "weight": float(record.weight),
        "residual_deg": record.residual_deg,
        "inlier": bool(record.inlier),
    }


def _draw_preview_lines(preview_bgr: np.ndarray, records: Sequence[GreatCircleLine]) -> np.ndarray:
    out = preview_bgr.copy()
    for record in records:
        color = (150, 150, 150)
        thickness = 1
        if record.inlier:
            color = (0, 140, 255)
            thickness = 2
        cv2.line(
            out,
            (int(round(record.x1)), int(round(record.y1))),
            (int(round(record.x2)), int(round(record.y2))),
            color,
            thickness,
            cv2.LINE_AA,
        )
    return out


def _save_debug_outputs(
    pano_bgr: np.ndarray,
    out_dir: Path,
    records: List[GreatCircleLine],
    previews: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    debug_dir = out_dir / "preview_debug"
    records_by_idx: Dict[int, List[GreatCircleLine]] = {}
    for record in records:
        records_by_idx.setdefault(int(record.preview_index), []).append(record)

    for preview_meta in previews:
        idx = int(preview_meta["preview_index"])
        preview = cv2.imread(str(preview_meta["preview_path"]))
        if preview is None:
            preview, _ = render_detection_crop(
                pano_bgr=pano_bgr,
                yaw_deg=float(preview_meta["yaw_sample"]),
                pitch_deg=float(args.pitch_detect),
                fov_deg=float(args.preview_fov),
                out_w=int(args.preview_width),
                out_h=int(args.preview_height),
                crop_strategy="ui_like",
                supersample=1.0,
                interpolation="linear",
                R_level=None,
                roll_deg=0.0,
                level_meta=None,
            )
        local_records = records_by_idx.get(idx, [])
        lines_img = _draw_preview_lines(preview, local_records)
        _write_image(Path(preview_meta["lines_path"]), lines_img)
        Path(preview_meta["lines_json_path"]).write_text(
            json.dumps(_json_safe([_line_to_json(record) for record in local_records]), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    with (out_dir / "great_circles.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(_json_safe(_line_to_json(record)), ensure_ascii=False) + "\n")


def _render_view_outputs(
    pano_bgr: np.ndarray,
    out_dir: Path,
    view: str,
    yaw: float,
    fov: float,
    args: argparse.Namespace,
    R_level: Optional[np.ndarray],
    R_level_inverse: Optional[np.ndarray],
) -> Dict[str, Any]:
    view_dir = out_dir / view
    view_dir.mkdir(parents=True, exist_ok=True)
    common = {
        "pano_bgr": pano_bgr,
        "pitch_deg": float(args.crop_pitch),
        "out_w": int(args.crop_width),
        "out_h": int(args.crop_height),
        "crop_strategy": "ui_like",
        "supersample": 1.0,
        "interpolation": "cubic",
        "roll_deg": 0.0,
        "level_meta": None,
    }
    no_level, meta_no = render_detection_crop(
        yaw_deg=float(yaw),
        fov_deg=float(fov),
        R_level=None,
        **common,
    )
    level, meta_level = render_detection_crop(
        yaw_deg=float(yaw),
        fov_deg=float(fov),
        R_level=R_level,
        **common,
    )
    level_inv, meta_inv = render_detection_crop(
        yaw_deg=float(yaw),
        fov_deg=float(fov),
        R_level=R_level_inverse,
        **common,
    )
    comparison = _make_comparison(no_level, level, level_inv)

    _write_image(view_dir / "crop_no_level.jpg", no_level)
    _write_image(view_dir / "crop_spherical_level.jpg", level)
    _write_image(view_dir / "crop_spherical_level_inv.jpg", level_inv)
    _write_image(out_dir / f"comparison_{view}.jpg", comparison)
    return {
        "view": view,
        "yaw": float(yaw),
        "pitch": float(args.crop_pitch),
        "fov": float(fov),
        "out_dir": str(view_dir),
        "crop_no_level": str(view_dir / "crop_no_level.jpg"),
        "crop_spherical_level": str(view_dir / "crop_spherical_level.jpg"),
        "crop_spherical_level_inv": str(view_dir / "crop_spherical_level_inv.jpg"),
        "comparison": str(out_dir / f"comparison_{view}.jpg"),
        "crop_no_level_meta": _json_safe(meta_no),
        "crop_spherical_level_meta": _json_safe(meta_level),
        "crop_spherical_level_inv_meta": _json_safe(meta_inv),
    }


def _run(args: argparse.Namespace) -> Dict[str, Any]:
    pano_path = Path(args.pano)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pano = cv2.imread(str(pano_path))
    if pano is None:
        raise ValueError(f"failed to read panorama image: {pano_path}")

    records, previews = _collect_great_circles(pano, out_dir, args)
    up_result = _estimate_up_vector(records, args)
    v_up = up_result.get("v_up")

    R_level: Optional[np.ndarray] = None
    R_level_inverse: Optional[np.ndarray] = None
    angle_to_world_up_deg: Optional[float] = None
    v_up_yaw_pitch: Optional[Tuple[float, float]] = None
    reject_reason = up_result.get("reject_reason")
    applied = bool(up_result.get("ok", False))

    if v_up is not None:
        v_up_arr = np.asarray(v_up, dtype=np.float64)
        angle_to_world_up_deg = math.degrees(math.acos(float(np.clip(np.dot(WORLD_UP, v_up_arr), -1.0, 1.0))))
        v_up_yaw_pitch = ray_to_yaw_pitch(v_up_arr)
        R_level = _rotation_from_to(WORLD_UP, v_up_arr)
        R_level_inverse = R_level.T
        if angle_to_world_up_deg > float(args.max_apply_deg):
            applied = False
            reject_reason = "angle_exceeds_max_apply_deg"
    else:
        applied = False
        if reject_reason is None:
            reject_reason = "no_up_vector"

    # These comparison crops intentionally render both candidate directions even
    # when ``applied`` is false, so the experiment can reveal which convention is correct.
    R_level_for_output = R_level if R_level is not None else None
    R_level_inverse_for_output = R_level_inverse if R_level_inverse is not None else None
    view_outputs = []
    for view, yaw, fov in [
        ("front", float(args.yaw_center), float(args.front_fov)),
        ("left", _wrap_yaw_deg(float(args.yaw_center) - 90.0), float(args.side_fov)),
        ("right", _wrap_yaw_deg(float(args.yaw_center) + 90.0), float(args.side_fov)),
    ]:
        view_outputs.append(
            _render_view_outputs(
                pano,
                out_dir,
                view=view,
                yaw=float(yaw),
                fov=float(fov),
                args=args,
                R_level=R_level_for_output,
                R_level_inverse=R_level_inverse_for_output,
            )
        )

    _save_debug_outputs(pano, out_dir, records, previews, args)

    meta = {
        "experiment": "EXPERIMENTAL spherical upright adjustment validation",
        "method_note": (
            "Simplified great-circle/up-vector estimator inspired by Jung et al.; "
            "this is not a full Atlanta-world optimization."
        ),
        "pano": str(pano_path),
        "yaw_center": float(args.yaw_center),
        "pitch_detect": float(args.pitch_detect),
        "preview_fov": float(args.preview_fov),
        "preview_width": int(args.preview_width),
        "preview_height": int(args.preview_height),
        "crop_pitch": float(args.crop_pitch),
        "v_up": None if v_up is None else _json_safe(np.asarray(v_up, dtype=np.float64)),
        "v_up_yaw_pitch": None if v_up_yaw_pitch is None else [float(v_up_yaw_pitch[0]), float(v_up_yaw_pitch[1])],
        "angle_to_world_up_deg": angle_to_world_up_deg,
        "applied": bool(applied),
        "reject_reason": reject_reason,
        "renders_candidate_crops_even_when_rejected": True,
        "max_apply_deg": float(args.max_apply_deg),
        "ransac_iters": int(args.ransac_iters),
        "min_inliers": int(args.min_inliers),
        "inlier_threshold_deg": INLIER_THRESHOLD_DEG,
        "inlier_count": int(up_result.get("inlier_count", 0) or 0),
        "total_line_count": int(len(records)),
        "mean_residual_deg": up_result.get("mean_residual_deg"),
        "median_residual_deg": up_result.get("median_residual_deg"),
        "initial_v_up": _json_safe(up_result.get("initial_v_up")),
        "initial_up_prior": up_result.get("initial_up_prior"),
        "initial_median_residual_deg": up_result.get("initial_median_residual_deg"),
        "R_level": None if R_level is None else _json_safe(R_level),
        "R_level_inverse": None if R_level_inverse is None else _json_safe(R_level_inverse),
        "preview_debug": _json_safe(previews),
        "view_outputs": _json_safe(view_outputs),
        "outputs": {
            "preview_debug_dir": str(out_dir / "preview_debug"),
            "great_circles": str(out_dir / "great_circles.jsonl"),
            "upright_meta": str(out_dir / "upright_meta.json"),
            "comparison_front": str(out_dir / "comparison_front.jpg"),
            "comparison_left": str(out_dir / "comparison_left.jpg"),
            "comparison_right": str(out_dir / "comparison_right.jpg"),
        },
    }
    (out_dir / "upright_meta.json").write_text(
        json.dumps(_json_safe(meta), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return meta


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="EXPERIMENTAL spherical great-circle upright adjustment validation.",
    )
    parser.add_argument("--pano", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--yaw_center", type=float, default=0.0)
    parser.add_argument("--pitch_detect", type=float, default=0.0)
    parser.add_argument("--preview_fov", type=float, default=90.0)
    parser.add_argument("--preview_width", type=int, default=1024)
    parser.add_argument("--preview_height", type=int, default=768)
    parser.add_argument("--crop_width", type=int, default=1280)
    parser.add_argument("--crop_height", type=int, default=1280)
    parser.add_argument("--crop_pitch", type=float, default=40.0)
    parser.add_argument("--front_fov", type=float, default=105.0)
    parser.add_argument("--side_fov", type=float, default=90.0)
    parser.add_argument("--max_apply_deg", type=float, default=5.0)
    parser.add_argument("--ransac_iters", type=int, default=1000)
    parser.add_argument("--min_inliers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    meta = _run(args)
    summary = {
        "pano": meta["pano"],
        "out_dir": str(args.out_dir),
        "v_up": meta["v_up"],
        "angle_to_world_up_deg": meta["angle_to_world_up_deg"],
        "applied": meta["applied"],
        "reject_reason": meta["reject_reason"],
        "inlier_count": meta["inlier_count"],
        "total_line_count": meta["total_line_count"],
        "median_residual_deg": meta["median_residual_deg"],
        "upright_meta": meta["outputs"]["upright_meta"],
    }
    print(json.dumps(_json_safe(summary), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
