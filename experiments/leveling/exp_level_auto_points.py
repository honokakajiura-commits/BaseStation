# EXPERIMENTAL: horizontal leveling validation
# This script is only for testing crop/panorama point mapping.
# Safe to delete: remove experiments/leveling/ directory.
# It does not affect the main agent pipeline.

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.agent.crop import render_detection_crop
from tools.agent.leveling import estimate_pano_level_correction, make_level_rotation
from tools.agent.spherical_camera import (
    apply_rotation_to_rays,
    make_rotation,
    pixel_to_camera_ray,
    ray_to_yaw_pitch,
)


LEVEL_YAW_SAMPLES = (-120.0, -60.0, 0.0, 60.0, 120.0)
LEVEL_PREVIEW_PITCH = 0.0
LEVEL_PREVIEW_FOV = 90.0
LEVEL_PREVIEW_W = 768
LEVEL_PREVIEW_H = 768


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _ray_to_equirect_uv(ray: np.ndarray, pano_w: int, pano_h: int) -> Tuple[float, float]:
    arr = np.asarray(ray, dtype=np.float64)
    norm = np.linalg.norm(arr, axis=-1, keepdims=True)
    dirs = arr / np.maximum(norm, 1e-12)
    lon = np.arctan2(dirs[..., 0], dirs[..., 2])
    lat = np.arctan2(dirs[..., 1], np.sqrt(dirs[..., 0] ** 2 + dirs[..., 2] ** 2))
    u = (lon / (2.0 * math.pi) + 0.5) * float(pano_w)
    v = (0.5 - lat / math.pi) * float(pano_h)
    return float(np.mod(u, float(pano_w))), float(np.clip(v, 0.0, float(pano_h - 1)))


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
    idx = int(np.searchsorted(np.cumsum(wts), total * 0.5, side="left"))
    return float(vals[min(idx, vals.size - 1)])


def _line_orientation_deg(x1: float, y1: float, x2: float, y2: float) -> float:
    angle = math.degrees(math.atan2(float(y2) - float(y1), float(x2) - float(x1)))
    while angle < -90.0:
        angle += 180.0
    while angle >= 90.0:
        angle -= 180.0
    return float(angle)


def _roll_from_line_angle(angle_deg: float) -> Optional[Tuple[float, str]]:
    angle = float(angle_deg)
    if -30.0 <= angle <= 30.0:
        return angle, "horizontal"
    if 60.0 <= abs(angle) <= 90.0:
        return (angle + 90.0 if angle < 0.0 else angle - 90.0), "vertical"
    return None


def _yaw_tag(yaw: float) -> str:
    value = float(yaw)
    sign = "p" if value >= 0.0 else "m"
    abs_value = abs(value)
    if abs(abs_value - round(abs_value)) < 1e-6:
        return f"{sign}{int(round(abs_value)):03d}"
    return f"{sign}{abs_value:.1f}".replace(".", "p")


def _analyze_level_debug_lines(preview_bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    if preview_bgr is None or preview_bgr.ndim < 2:
        out = np.zeros((LEVEL_PREVIEW_H, LEVEL_PREVIEW_W, 3), dtype=np.uint8)
        return out, {
            "estimated_roll_deg": 0.0,
            "confidence": 0.0,
            "line_count": 0,
            "horizontal_line_count": 0,
            "vertical_line_count": 0,
            "reason": "invalid_image",
        }

    out = preview_bgr.copy()
    h, w = preview_bgr.shape[:2]
    min_dim = max(1, min(int(w), int(h)))
    gray = cv2.cvtColor(preview_bgr, cv2.COLOR_BGR2GRAY) if preview_bgr.ndim == 3 else preview_bgr
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
    if lines is None:
        return out, {
            "estimated_roll_deg": 0.0,
            "confidence": 0.0,
            "line_count": 0,
            "horizontal_line_count": 0,
            "vertical_line_count": 0,
            "candidate_line_count": 0,
            "used_line_count": 0,
            "reason": "no_hough_lines",
        }

    records: List[Dict[str, Any]] = []
    candidates: List[Tuple[float, float, str]] = []
    horizontal_count = 0
    vertical_count = 0
    for raw_line in lines[:, 0, :]:
        x1, y1, x2, y2 = [float(v) for v in raw_line]
        length = math.hypot(x2 - x1, y2 - y1)
        angle = _line_orientation_deg(x1, y1, x2, y2)
        roll_info = _roll_from_line_angle(angle) if length >= float(min_line_len) else None
        kind = "other"
        roll_deg: Optional[float] = None
        color = (160, 160, 160)
        if roll_info is not None:
            roll_deg, kind = roll_info
            candidates.append((float(roll_deg), float(length), kind))
            if kind == "horizontal":
                horizontal_count += 1
                color = (0, 0, 255)
            else:
                vertical_count += 1
                color = (255, 0, 0)

        cv2.line(
            out,
            (int(round(x1)), int(round(y1))),
            (int(round(x2)), int(round(y2))),
            color,
            2,
            cv2.LINE_AA,
        )
        records.append(
            {
                "xyxy": [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))],
                "length": float(length),
                "angle_deg": float(angle),
                "kind": kind,
                "roll_deg": None if roll_deg is None else float(roll_deg),
            }
        )

    meta: Dict[str, Any] = {
        "estimated_roll_deg": 0.0,
        "confidence": 0.0,
        "line_count": int(len(lines)),
        "horizontal_line_count": int(horizontal_count),
        "vertical_line_count": int(vertical_count),
        "candidate_line_count": int(len(candidates)),
        "used_line_count": 0,
        "lines": records,
    }
    if not candidates:
        meta["reason"] = "no_level_candidates"
        return out, meta

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

    meta.update(
        {
            "estimated_roll_deg": float(roll_deg),
            "confidence": _clamp01(confidence),
            "used_line_count": int(kept_values.size),
            "dispersion_deg": float(dispersion),
        }
    )
    return out, meta


def _save_level_debug(pano_bgr: np.ndarray, out_dir: Path, level_meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    debug_dir = out_dir / "level_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    sample_by_yaw: Dict[float, Dict[str, Any]] = {}
    for sample in level_meta.get("samples", []) or []:
        if isinstance(sample, dict) and "yaw_deg" in sample:
            sample_by_yaw[round(float(sample["yaw_deg"]), 6)] = sample

    outputs: List[Dict[str, Any]] = []
    for yaw in LEVEL_YAW_SAMPLES:
        preview, _ = render_detection_crop(
            pano_bgr=pano_bgr,
            yaw_deg=float(yaw),
            pitch_deg=LEVEL_PREVIEW_PITCH,
            fov_deg=LEVEL_PREVIEW_FOV,
            out_w=LEVEL_PREVIEW_W,
            out_h=LEVEL_PREVIEW_H,
            crop_strategy="ui_like",
            supersample=1.0,
            interpolation="linear",
            R_level=None,
            roll_deg=0.0,
            level_meta=level_meta,
        )
        lines_img, meta = _analyze_level_debug_lines(preview)
        tag = _yaw_tag(float(yaw))
        preview_path = debug_dir / f"preview_yaw_{tag}.jpg"
        lines_path = debug_dir / f"preview_yaw_{tag}_lines.jpg"
        meta_path = debug_dir / f"preview_yaw_{tag}_meta.json"
        _write_image(preview_path, preview)
        _write_image(lines_path, lines_img)

        meta = {
            "yaw": float(yaw),
            "estimated_roll_deg": float(meta.get("estimated_roll_deg", 0.0) or 0.0),
            "confidence": float(meta.get("confidence", 0.0) or 0.0),
            "line_count": int(meta.get("line_count", 0) or 0),
            "horizontal_line_count": int(meta.get("horizontal_line_count", 0) or 0),
            "vertical_line_count": int(meta.get("vertical_line_count", 0) or 0),
            "candidate_line_count": int(meta.get("candidate_line_count", 0) or 0),
            "used_line_count": int(meta.get("used_line_count", 0) or 0),
            "dispersion_deg": float(meta.get("dispersion_deg", 0.0) or 0.0),
            "reason": meta.get("reason"),
            "preview": str(preview_path),
            "lines": str(lines_path),
            "meta": str(meta_path),
            "level_sample": _json_safe(sample_by_yaw.get(round(float(yaw), 6), {})),
            "line_segments": _json_safe(meta.get("lines", [])),
        }
        meta_path.write_text(json.dumps(_json_safe(meta), ensure_ascii=False, indent=2), encoding="utf-8")
        outputs.append({k: v for k, v in meta.items() if k != "line_segments"})

    (debug_dir / "level_debug_index.json").write_text(
        json.dumps(_json_safe(outputs), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return outputs


def _fallback_grid_points(width: int, height: int, max_points: int) -> List[Tuple[float, float]]:
    count = max(1, int(max_points))
    side = int(math.ceil(math.sqrt(count)))
    xs = np.linspace(float(width) * 0.35, float(width) * 0.65, side)
    ys = np.linspace(float(height) * 0.35, float(height) * 0.65, side)
    points: List[Tuple[float, float]] = []
    for y in ys:
        for x in xs:
            points.append((float(x), float(y)))
            if len(points) >= count:
                return points
    return points


def _detect_crop_points(
    crop_bgr: np.ndarray,
    max_points: int,
    quality: float,
    min_distance: float,
) -> Tuple[List[Tuple[float, float]], str]:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if crop_bgr.ndim == 3 else crop_bgr
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max(1, int(max_points)),
        qualityLevel=max(1e-6, float(quality)),
        minDistance=max(1.0, float(min_distance)),
        blockSize=7,
    )
    if corners is None or len(corners) == 0:
        h, w = gray.shape[:2]
        return _fallback_grid_points(w, h, max_points), "fallback_grid"

    points = [(float(pt[0][0]), float(pt[0][1])) for pt in corners]
    points.sort(key=lambda p: (p[1], p[0]))
    return points[: max(1, int(max_points))], "goodFeaturesToTrack"


def _project_crop_point_to_pano(
    x: float,
    y: float,
    width: int,
    height: int,
    fov: float,
    yaw: float,
    pitch: float,
    pano_w: int,
    pano_h: int,
    R_level: Optional[np.ndarray],
) -> Dict[str, float]:
    camera_ray = pixel_to_camera_ray(float(x), float(y), int(width), int(height), float(fov))
    world_ray = apply_rotation_to_rays(camera_ray, make_rotation(float(yaw), float(pitch), 0.0))
    source_ray = apply_rotation_to_rays(world_ray, R_level) if R_level is not None else world_ray
    pano_u, pano_v = _ray_to_equirect_uv(source_ray, pano_w=pano_w, pano_h=pano_h)
    local_yaw, local_pitch = ray_to_yaw_pitch(source_ray)
    return {
        "crop_x": float(x),
        "crop_y": float(y),
        "pano_u": float(pano_u),
        "pano_v": float(pano_v),
        "local_yaw": float(local_yaw),
        "local_pitch": float(local_pitch),
    }


def _draw_crop_points(crop_bgr: np.ndarray, points: Sequence[Dict[str, float]]) -> np.ndarray:
    out = crop_bgr.copy()
    for point in points:
        idx = int(point["index"])
        x = int(round(point["crop_x"]))
        y = int(round(point["crop_y"]))
        cv2.circle(out, (x, y), 8, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(out, (x, y), 11, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, str(idx), (x + 12, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, str(idx), (x + 12, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _draw_pano_points(pano_bgr: np.ndarray, points: Sequence[Dict[str, float]]) -> np.ndarray:
    out = pano_bgr.copy()
    radius = max(5, int(round(min(out.shape[:2]) / 180.0)))
    font_scale = max(0.5, min(1.1, out.shape[1] / 2500.0))
    thickness = max(1, int(round(font_scale * 2.0)))
    for point in points:
        idx = int(point["index"])
        x = int(round(point["pano_u"])) % out.shape[1]
        y = int(round(point["pano_v"]))
        y = max(0, min(out.shape[0] - 1, y))
        cv2.circle(out, (x, y), radius, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(out, (x, y), radius + 3, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(
            out,
            str(idx),
            (min(out.shape[1] - 1, x + radius + 4), max(15, y - radius - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness + 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            str(idx),
            (min(out.shape[1] - 1, x + radius + 4), max(15, y - radius - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
    return out


def _unwrap_us(us: Sequence[float], pano_w: int) -> List[float]:
    plain_span = max(us) - min(us)
    shifted = [u + float(pano_w) if u < float(pano_w) / 2.0 else u for u in us]
    shifted_span = max(shifted) - min(shifted)
    return shifted if shifted_span < plain_span else [float(u) for u in us]


def _pano_zoom(pano_annotated: np.ndarray, points: Sequence[Dict[str, float]]) -> np.ndarray:
    pano_h, pano_w = pano_annotated.shape[:2]
    if not points:
        return pano_annotated.copy()

    us = [float(point["pano_u"]) for point in points]
    vs = [float(point["pano_v"]) for point in points]
    us_unwrapped = _unwrap_us(us, pano_w)
    margin = max(80, int(round(min(pano_w, pano_h) * 0.04)))
    zoom_w = int(round(max(240.0, min(float(pano_w), max(us_unwrapped) - min(us_unwrapped) + margin * 2.0))))
    zoom_h = int(round(max(240.0, min(float(pano_h), max(vs) - min(vs) + margin * 2.0))))
    center_u = (min(us_unwrapped) + max(us_unwrapped)) / 2.0
    center_v = (min(vs) + max(vs)) / 2.0

    x0 = int(round(center_u - zoom_w / 2.0))
    y0 = int(round(center_v - zoom_h / 2.0))
    y0 = max(0, min(pano_h - zoom_h, y0))
    xs = np.mod(np.arange(x0, x0 + zoom_w), pano_w)
    return pano_annotated[y0 : y0 + zoom_h, :][:, xs].copy()


def _resize_to_height(img: np.ndarray, height: int) -> np.ndarray:
    if img.shape[0] == height:
        return img
    width = max(1, int(round(img.shape[1] * (float(height) / float(img.shape[0])))))
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


def _make_comparison(
    crop_no_level: np.ndarray,
    crop_level: np.ndarray,
    crop_points: np.ndarray,
    pano_zoom: np.ndarray,
) -> np.ndarray:
    target_h = min(640, max(320, crop_no_level.shape[0] // 2))
    panels = [
        _draw_label(_resize_to_height(crop_no_level, target_h), "crop_no_level"),
        _draw_label(_resize_to_height(crop_level, target_h), "crop_level"),
        _draw_label(_resize_to_height(crop_points, target_h), "crop_level_points"),
        _draw_label(_resize_to_height(pano_zoom, target_h), "pano_zoom_points"),
    ]
    return np.hstack(panels)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pano", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--pitch", type=float, default=40.0)
    parser.add_argument("--fov", type=float, default=105.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=1280)
    parser.add_argument("--max_points", type=int, default=20)
    parser.add_argument("--quality", type=float, default=0.01)
    parser.add_argument("--min_distance", type=float, default=30.0)
    parser.add_argument("--level_min_confidence", type=float, default=0.25)
    parser.add_argument("--no_level_horizon", action="store_true")
    parser.add_argument("--save_level_debug", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    pano_path = Path(args.pano)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pano = cv2.imread(str(pano_path))
    if pano is None:
        raise ValueError(f"failed to read panorama: {pano_path}")
    pano_h, pano_w = pano.shape[:2]

    if args.no_level_horizon:
        level_meta: Dict[str, Any] = {
            "enabled": False,
            "roll_deg": 0.0,
            "confidence": 0.0,
            "sample_count": 0,
            "used_sample_count": 0,
            "samples": [],
            "method": "disabled",
            "reason": "no_level_horizon",
            "applied": False,
            "min_confidence": float(args.level_min_confidence),
        }
        R_level = None
    else:
        level_meta = dict(
            estimate_pano_level_correction(
                pano,
                yaw_samples=LEVEL_YAW_SAMPLES,
                pitch_deg=LEVEL_PREVIEW_PITCH,
                preview_fov=LEVEL_PREVIEW_FOV,
                preview_w=LEVEL_PREVIEW_W,
                preview_h=LEVEL_PREVIEW_H,
            )
        )
        level_confidence = float(level_meta.get("confidence", 0.0) or 0.0)
        applied = bool(level_meta.get("enabled", False)) and level_confidence >= float(args.level_min_confidence)
        level_meta["applied"] = bool(applied)
        level_meta["min_confidence"] = float(args.level_min_confidence)
        R_level = make_level_rotation(float(level_meta.get("roll_deg", 0.0) or 0.0)) if applied else None

    level_debug_outputs: List[Dict[str, Any]] = []
    if args.save_level_debug:
        level_debug_outputs = _save_level_debug(pano, out_dir, level_meta)

    crop_no_level, meta_no_level = render_detection_crop(
        pano_bgr=pano,
        yaw_deg=args.yaw,
        pitch_deg=args.pitch,
        fov_deg=args.fov,
        out_w=args.width,
        out_h=args.height,
        crop_strategy="ui_like",
        supersample=1.0,
        interpolation="cubic",
        R_level=None,
        roll_deg=0.0,
        level_meta=level_meta,
    )
    crop_level, meta_level = render_detection_crop(
        pano_bgr=pano,
        yaw_deg=args.yaw,
        pitch_deg=args.pitch,
        fov_deg=args.fov,
        out_w=args.width,
        out_h=args.height,
        crop_strategy="ui_like",
        supersample=1.0,
        interpolation="cubic",
        R_level=R_level,
        roll_deg=0.0,
        level_meta=level_meta,
    )

    crop_points_xy, point_source = _detect_crop_points(
        crop_level,
        max_points=args.max_points,
        quality=args.quality,
        min_distance=args.min_distance,
    )
    points: List[Dict[str, float]] = []
    for idx, (x, y) in enumerate(crop_points_xy, start=1):
        projected = _project_crop_point_to_pano(
            x=x,
            y=y,
            width=args.width,
            height=args.height,
            fov=args.fov,
            yaw=args.yaw,
            pitch=args.pitch,
            pano_w=pano_w,
            pano_h=pano_h,
            R_level=R_level,
        )
        projected["index"] = int(idx)
        points.append(projected)

    crop_points_img = _draw_crop_points(crop_level, points)
    pano_points_img = _draw_pano_points(pano, points)
    pano_zoom_img = _pano_zoom(pano_points_img, points)
    comparison_img = _make_comparison(crop_no_level, crop_level, crop_points_img, pano_zoom_img)

    _write_image(out_dir / "crop_no_level.jpg", crop_no_level)
    _write_image(out_dir / "crop_level.jpg", crop_level)
    _write_image(out_dir / "crop_level_points.jpg", crop_points_img)
    _write_image(out_dir / "pano_projected_points.jpg", pano_points_img)
    _write_image(out_dir / "pano_zoom_points.jpg", pano_zoom_img)
    _write_image(out_dir / "comparison.jpg", comparison_img)

    mapping = {
        "pano": str(pano_path),
        "yaw": float(args.yaw),
        "pitch": float(args.pitch),
        "fov": float(args.fov),
        "width": int(args.width),
        "height": int(args.height),
        "level_meta": _json_safe(level_meta),
        "R_level_applied": bool(R_level is not None),
        "point_source": point_source,
        "crop_no_level_meta": _json_safe(meta_no_level),
        "crop_level_meta": _json_safe(meta_level),
        "outputs": {
            "crop_no_level": str(out_dir / "crop_no_level.jpg"),
            "crop_level": str(out_dir / "crop_level.jpg"),
            "crop_level_points": str(out_dir / "crop_level_points.jpg"),
            "pano_projected_points": str(out_dir / "pano_projected_points.jpg"),
            "pano_zoom_points": str(out_dir / "pano_zoom_points.jpg"),
            "comparison": str(out_dir / "comparison.jpg"),
        },
        "level_debug_outputs": _json_safe(level_debug_outputs),
        "points": _json_safe(points),
    }
    if args.save_level_debug:
        mapping["outputs"]["level_debug_dir"] = str(out_dir / "level_debug")
    (out_dir / "auto_points_mapping.json").write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(mapping, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
