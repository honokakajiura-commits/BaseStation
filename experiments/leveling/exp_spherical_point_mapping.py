# EXPERIMENTAL: spherical level point mapping validation
# This script validates point correspondence between level crop and source equirectangular panorama.
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
from tools.agent.spherical_camera import (
    apply_rotation_to_rays,
    make_rotation,
    normalize_ray,
    pixel_to_camera_ray,
)

from experiments.leveling.exp_spherical_upright_level import (
    _estimate_up_vector,
    _collect_great_circles,
    _json_safe,
    _rotation_from_to,
)


def _write_image(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), img):
        raise ValueError(f"failed to write image: {path}")


def _wrap_yaw_deg(yaw: float) -> float:
    return (float(yaw) + 180.0) % 360.0 - 180.0


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        v = float(value)
        if not math.isfinite(v):
            return None
        return v
    except (TypeError, ValueError):
        return None


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


def _make_point_comparison(original: np.ndarray, roundtrip: np.ndarray, pano: np.ndarray) -> np.ndarray:
    target_h = min(620, max(320, original.shape[0] // 2))
    return np.hstack(
        [
            _draw_label(_resize_to_height(original, target_h), "crop_points_original"),
            _draw_label(_resize_to_height(roundtrip, target_h), "crop_points_roundtrip"),
            _draw_label(_resize_to_height(pano, target_h), "pano_projected_points"),
        ]
    )


def _make_zoom(pano_img: np.ndarray, points: Sequence[Dict[str, Any]]) -> np.ndarray:
    if not points:
        return pano_img.copy()
    h, w = pano_img.shape[:2]
    us = [float(p["pano_u"]) for p in points if p.get("pano_u") is not None]
    vs = [float(p["pano_v"]) for p in points if p.get("pano_v") is not None]
    if not us or not vs:
        return pano_img.copy()
    span_u = max(us) - min(us)
    span_v = max(vs) - min(vs)
    if span_u > w * 0.7:
        return pano_img.copy()
    margin = max(80, int(round(min(w, h) * 0.05)))
    x0 = int(round(min(us) - margin))
    x1 = int(round(max(us) + margin))
    y0 = int(round(min(vs) - margin))
    y1 = int(round(max(vs) + margin))
    if span_v > h * 0.6:
        return pano_img.copy()
    x0 = max(0, min(w - 1, x0))
    x1 = max(x0 + 1, min(w, x1))
    y0 = max(0, min(h - 1, y0))
    y1 = max(y0 + 1, min(h, y1))
    return pano_img[y0:y1, x0:x1].copy()


def _draw_points(
    img: np.ndarray,
    points: Sequence[Dict[str, Any]],
    use_roundtrip: bool = False,
    highlight_error: bool = False,
) -> np.ndarray:
    out = img.copy()
    radius = max(5, int(round(min(out.shape[:2]) / 180.0)))
    font_scale = max(0.45, min(0.9, out.shape[1] / 2600.0))
    thickness = max(1, int(round(font_scale * 2.0)))
    for point in points:
        idx = int(point["id"])
        key_x = "roundtrip_x" if use_roundtrip else "crop_x"
        key_y = "roundtrip_y" if use_roundtrip else "crop_y"
        x = int(round(float(point[key_x])))
        y = int(round(float(point[key_y])))
        if out.shape[0] > 0 and out.shape[1] > 0:
            x = max(0, min(out.shape[1] - 1, x))
            y = max(0, min(out.shape[0] - 1, y))
        err = float(point.get("roundtrip_error_px") or 0.0)
        color = (0, 140, 255) if highlight_error and err > 5.0 else (0, 0, 255)
        cv2.circle(out, (x, y), radius, color, -1, cv2.LINE_AA)
        cv2.circle(out, (x, y), radius + 3, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, str(idx), (x + radius + 4, max(15, y - radius - 4)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(out, str(idx), (x + radius + 4, max(15, y - radius - 4)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return out


def _generate_points(
    crop_bgr: np.ndarray,
    max_points: int,
    use_grid: bool,
    seed: int,
) -> Tuple[List[Tuple[float, float]], str]:
    h, w = crop_bgr.shape[:2]
    if use_grid:
        cols = 5
        rows = 4
        xs = np.linspace(float(w) * 0.15, float(w) * 0.85, cols)
        ys = np.linspace(float(h) * 0.15, float(h) * 0.85, rows)
        pts = [(float(x), float(y)) for y in ys for x in xs][: max(1, int(max_points))]
        return pts, "grid"
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if crop_bgr.ndim == 3 else crop_bgr
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max(1, int(max_points)),
        qualityLevel=0.01,
        minDistance=max(8.0, min(h, w) * 0.03),
        blockSize=7,
        useHarrisDetector=False,
    )
    if corners is None or len(corners) == 0:
        cols = 5
        rows = 4
        xs = np.linspace(float(w) * 0.15, float(w) * 0.85, cols)
        ys = np.linspace(float(h) * 0.15, float(h) * 0.85, rows)
        pts = [(float(x), float(y)) for y in ys for x in xs][: max(1, int(max_points))]
        return pts, "grid_fallback"
    pts = [(float(pt[0][0]), float(pt[0][1])) for pt in corners]
    pts.sort(key=lambda p: (p[1], p[0]))
    if len(pts) < min(8, int(max_points)):
        cols = 5
        rows = 4
        xs = np.linspace(float(w) * 0.15, float(w) * 0.85, cols)
        ys = np.linspace(float(h) * 0.15, float(h) * 0.85, rows)
        grid = [(float(x), float(y)) for y in ys for x in xs]
        pts.extend(grid)
        seen = set()
        uniq = []
        for x, y in pts:
            key = (round(x, 1), round(y, 1))
            if key in seen:
                continue
            seen.add(key)
            uniq.append((x, y))
        pts = uniq
    return pts[: max(1, int(max_points))], "goodFeaturesToTrack"


def _project_crop_to_pano(
    x: float,
    y: float,
    width: int,
    height: int,
    yaw: float,
    pitch: float,
    fov: float,
    R_level: np.ndarray,
    pano_w: int,
    pano_h: int,
) -> Dict[str, float]:
    camera_ray = pixel_to_camera_ray(float(x), float(y), int(width), int(height), float(fov))
    world_ray = apply_rotation_to_rays(camera_ray, make_rotation(float(yaw), float(pitch), 0.0))
    source_ray = apply_rotation_to_rays(world_ray, R_level)
    source_ray = np.asarray(normalize_ray(source_ray), dtype=np.float64)
    lon = math.atan2(float(source_ray[0]), float(source_ray[2]))
    lat = math.atan2(float(source_ray[1]), math.sqrt(float(source_ray[0]) ** 2 + float(source_ray[2]) ** 2))
    u = (lon / (2.0 * math.pi) + 0.5) * float(pano_w)
    v = (0.5 - lat / math.pi) * float(pano_h)
    u = float(np.mod(u, float(pano_w)))
    v = float(np.clip(v, 0.0, float(pano_h - 1)))
    return {"pano_u": u, "pano_v": v}


def _project_pano_to_crop(
    u: float,
    v: float,
    pano_w: int,
    pano_h: int,
    width: int,
    height: int,
    yaw: float,
    pitch: float,
    fov: float,
    R_level: np.ndarray,
) -> Tuple[float, float]:
    lon = (float(u) / float(pano_w) - 0.5) * 2.0 * math.pi
    lat = (0.5 - float(v) / float(pano_h)) * math.pi
    source_ray = np.array([math.cos(lat) * math.sin(lon), math.sin(lat), math.cos(lat) * math.cos(lon)], dtype=np.float64)
    level_world_ray = apply_rotation_to_rays(source_ray, R_level.T)
    view_inv = make_rotation(float(yaw), float(pitch), 0.0).T
    camera_ray = apply_rotation_to_rays(level_world_ray, view_inv)
    if float(camera_ray[2]) <= 1e-12:
        return float("nan"), float("nan")
    fx = (float(width) / 2.0) / math.tan(math.radians(float(fov)) / 2.0)
    cx = float(width) / 2.0
    cy = float(height) / 2.0
    x = float(camera_ray[0]) / float(camera_ray[2]) * fx + cx
    y = cy - float(camera_ray[1]) / float(camera_ray[2]) * fx
    return float(x), float(y)


def _run(args: argparse.Namespace) -> Dict[str, Any]:
    pano_path = Path(args.pano)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pano = cv2.imread(str(pano_path))
    if pano is None:
        raise ValueError(f"failed to read panorama image: {pano_path}")
    pano_h, pano_w = pano.shape[:2]

    # Reuse the upright estimator from the spherical leveling experiment.
    records, previews = _collect_great_circles(pano, out_dir, argparse.Namespace(
        yaw_center=float(args.yaw_center or 0.0),
        pitch_detect=0.0,
        preview_fov=90.0,
        preview_width=1024,
        preview_height=768,
        seed=int(args.seed),
        ransac_iters=1000,
        min_inliers=8,
        max_apply_deg=float(args.max_apply_deg),
    ))
    up_result = _estimate_up_vector(records, argparse.Namespace(
        ransac_iters=1000,
        min_inliers=8,
        seed=int(args.seed),
    ))
    v_up = up_result.get("v_up")
    if v_up is None:
        raise ValueError(f"failed to estimate v_up: {up_result.get('reject_reason')}")
    v_up_arr = np.asarray(v_up, dtype=np.float64)
    angle_to_world_up_deg = math.degrees(math.acos(float(np.clip(np.dot(np.array([0.0, 1.0, 0.0], dtype=np.float64), v_up_arr), -1.0, 1.0))))
    R_level = _rotation_from_to(np.array([0.0, 1.0, 0.0], dtype=np.float64), v_up_arr)
    R_level_inverse = R_level.T
    R_level_applied = bool(angle_to_world_up_deg <= float(args.max_apply_deg))

    crop_no_level, meta_no = render_detection_crop(
        pano_bgr=pano,
        yaw_deg=float(args.yaw),
        pitch_deg=float(args.pitch),
        fov_deg=float(args.fov),
        out_w=int(args.width),
        out_h=int(args.height),
        crop_strategy="ui_like",
        supersample=1.0,
        interpolation="cubic",
        R_level=None,
        roll_deg=0.0,
        level_meta=None,
    )
    crop_spherical_level, meta_level = render_detection_crop(
        pano_bgr=pano,
        yaw_deg=float(args.yaw),
        pitch_deg=float(args.pitch),
        fov_deg=float(args.fov),
        out_w=int(args.width),
        out_h=int(args.height),
        crop_strategy="ui_like",
        supersample=1.0,
        interpolation="cubic",
        R_level=R_level,
        roll_deg=0.0,
        level_meta=None,
    )

    if args.yaw_center is None:
        args.yaw_center = float(args.yaw)

    points_xy, point_source = _generate_points(crop_spherical_level, int(args.max_points), bool(args.use_grid), int(args.seed))
    points: List[Dict[str, Any]] = []
    for idx, (x, y) in enumerate(points_xy, start=1):
        proj = _project_crop_to_pano(
            x=x,
            y=y,
            width=int(args.width),
            height=int(args.height),
            yaw=float(args.yaw),
            pitch=float(args.pitch),
            fov=float(args.fov),
            R_level=R_level,
            pano_w=pano_w,
            pano_h=pano_h,
        )
        roundtrip_x, roundtrip_y = _project_pano_to_crop(
            u=proj["pano_u"],
            v=proj["pano_v"],
            pano_w=pano_w,
            pano_h=pano_h,
            width=int(args.width),
            height=int(args.height),
            yaw=float(args.yaw),
            pitch=float(args.pitch),
            fov=float(args.fov),
            R_level=R_level,
        )
        error = math.hypot(float(roundtrip_x) - float(x), float(roundtrip_y) - float(y))
        points.append(
            {
                "id": int(idx),
                "crop_x": float(x),
                "crop_y": float(y),
                "pano_u": float(proj["pano_u"]),
                "pano_v": float(proj["pano_v"]),
                "roundtrip_x": float(roundtrip_x),
                "roundtrip_y": float(roundtrip_y),
                "roundtrip_error_px": float(error),
            }
        )

    error_values = [float(point["roundtrip_error_px"]) for point in points]
    error_mean = float(np.mean(error_values)) if error_values else None
    error_median = float(np.median(error_values)) if error_values else None
    error_max = float(np.max(error_values)) if error_values else None

    pano_overlay = pano.copy()
    for point in points:
        x = int(round(point["pano_u"])) % pano_overlay.shape[1]
        y = max(0, min(pano_overlay.shape[0] - 1, int(round(point["pano_v"]))))
        err = float(point["roundtrip_error_px"])
        color = (0, 0, 255) if err <= 5.0 else (0, 165, 255)
        cv2.circle(pano_overlay, (x, y), 7, color, -1, cv2.LINE_AA)
        cv2.circle(pano_overlay, (x, y), 10, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(pano_overlay, str(int(point["id"])), (x + 12, max(16, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(pano_overlay, str(int(point["id"])), (x + 12, max(16, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    roundtrip_crop = crop_spherical_level.copy()
    for point in points:
        x1 = int(round(point["crop_x"]))
        y1 = int(round(point["crop_y"]))
        x2 = int(round(point["roundtrip_x"]))
        y2 = int(round(point["roundtrip_y"]))
        err = float(point["roundtrip_error_px"])
        color = (0, 0, 255) if err <= 5.0 else (0, 165, 255)
        cv2.circle(roundtrip_crop, (x1, y1), 7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(roundtrip_crop, (x1, y1), 5, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(roundtrip_crop, (x2, y2), 7, color, -1, cv2.LINE_AA)
        cv2.circle(roundtrip_crop, (x2, y2), 10, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(roundtrip_crop, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        cv2.putText(roundtrip_crop, str(int(point["id"])), (x1 + 10, max(16, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(roundtrip_crop, str(int(point["id"])), (x1 + 10, max(16, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(roundtrip_crop, str(int(point["id"])), (x2 + 10, max(16, y2 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(roundtrip_crop, str(int(point["id"])), (x2 + 10, max(16, y2 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    original_crop = crop_spherical_level.copy()
    original_crop = _draw_points(original_crop, points, use_roundtrip=False, highlight_error=True)
    pano_zoom = _make_zoom(pano_overlay, points)
    comparison = _make_point_comparison(original_crop, roundtrip_crop, pano_zoom)

    _write_image(out_dir / "crop_points_original.jpg", original_crop)
    _write_image(out_dir / "crop_points_roundtrip.jpg", roundtrip_crop)
    _write_image(out_dir / "crop_no_level.jpg", crop_no_level)
    _write_image(out_dir / "crop_spherical_level.jpg", crop_spherical_level)
    _write_image(out_dir / "pano_projected_points.jpg", pano_overlay)
    _write_image(out_dir / "pano_zoom_points.jpg", pano_zoom)
    _write_image(out_dir / "comparison.jpg", comparison)

    point_mapping_meta = {
        "pano": str(pano_path),
        "yaw": float(args.yaw),
        "pitch": float(args.pitch),
        "fov": float(args.fov),
        "width": int(args.width),
        "height": int(args.height),
        "R_level_applied": bool(R_level_applied),
        "R_level": None if R_level is None else _json_safe(R_level),
        "v_up": _json_safe(v_up_arr),
        "angle_to_world_up_deg": float(angle_to_world_up_deg),
        "inlier_count": int(up_result.get("inlier_count", 0) or 0),
        "total_line_count": int(len(records)),
        "mean_residual_deg": up_result.get("mean_residual_deg"),
        "median_residual_deg": up_result.get("median_residual_deg"),
        "points": _json_safe(points),
        "point_source": point_source,
        "error_mean_px": error_mean,
        "error_median_px": error_median,
        "error_max_px": error_max,
        "crop_no_level_meta": _json_safe(meta_no),
        "crop_spherical_level_meta": _json_safe(meta_level),
        "R_level_inverse": None if R_level_inverse is None else _json_safe(R_level_inverse),
        "preview_debug": _json_safe(previews),
        "outputs": {
            "crop_no_level": str(out_dir / "crop_no_level.jpg"),
            "crop_spherical_level": str(out_dir / "crop_spherical_level.jpg"),
            "crop_points_original": str(out_dir / "crop_points_original.jpg"),
            "crop_points_roundtrip": str(out_dir / "crop_points_roundtrip.jpg"),
            "pano_projected_points": str(out_dir / "pano_projected_points.jpg"),
            "pano_zoom_points": str(out_dir / "pano_zoom_points.jpg"),
            "comparison": str(out_dir / "comparison.jpg"),
        },
        "status": None,
    }

    if error_mean is not None and error_max is not None and error_mean <= 2.0 and error_max <= 5.0:
        status = "PASS"
    elif error_mean is not None and error_max is not None and error_mean <= 5.0 and error_max <= 10.0:
        status = "WARN"
    else:
        status = "FAIL"
    point_mapping_meta["status"] = status

    (out_dir / "point_mapping_meta.json").write_text(json.dumps(_json_safe(point_mapping_meta), ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "point_count": len(points),
        "error_mean_px": error_mean,
        "error_median_px": error_median,
        "error_max_px": error_max,
        "status": status,
        "point_mapping_meta": str(out_dir / "point_mapping_meta.json"),
    }
    print(json.dumps(_json_safe(summary), ensure_ascii=False, indent=2))
    return point_mapping_meta


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EXPERIMENTAL spherical level point mapping validation.")
    parser.add_argument("--pano", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--yaw", type=float, required=True)
    parser.add_argument("--pitch", type=float, default=40.0)
    parser.add_argument("--fov", type=float, default=90.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=1280)
    parser.add_argument("--yaw_center", type=float, default=None)
    parser.add_argument("--max_points", type=int, default=20)
    parser.add_argument("--use_grid", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_apply_deg", type=float, default=5.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    _run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
