#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generate a synthetic equirectangular panorama and spherical-camera crops.

Usage:
python tools/test_spherical_camera.py --out_dir runs/spherical_camera_synthetic
"""

import argparse
import importlib.util
import json
import math
from pathlib import Path

import cv2
import numpy as np

try:
    from spherical_camera import (
        apply_rotation_to_rays,
        compute_next_view_from_bbox,
        equirect_to_perspective,
        make_rotation,
        pixel_to_camera_ray,
        wrap_yaw_deg,
        yaw_pitch_to_ray,
    )
except ModuleNotFoundError:
    _SC_PATH = Path(__file__).resolve().with_name("spherical_camera.py")
    _SC_SPEC = importlib.util.spec_from_file_location("spherical_camera", _SC_PATH)
    if _SC_SPEC is None or _SC_SPEC.loader is None:
        raise
    _SC_MOD = importlib.util.module_from_spec(_SC_SPEC)
    _SC_SPEC.loader.exec_module(_SC_MOD)
    apply_rotation_to_rays = _SC_MOD.apply_rotation_to_rays
    compute_next_view_from_bbox = _SC_MOD.compute_next_view_from_bbox
    equirect_to_perspective = _SC_MOD.equirect_to_perspective
    make_rotation = _SC_MOD.make_rotation
    pixel_to_camera_ray = _SC_MOD.pixel_to_camera_ray
    wrap_yaw_deg = _SC_MOD.wrap_yaw_deg
    yaw_pitch_to_ray = _SC_MOD.yaw_pitch_to_ray


def equirect_uv(yaw_deg: float, pitch_deg: float, width: int, height: int) -> tuple[int, int]:
    u = int(round((float(yaw_deg) / 360.0 + 0.5) * width)) % width
    v = int(round((0.5 - float(pitch_deg) / 180.0) * height))
    return u, max(0, min(height - 1, v))


def draw_marker(img: np.ndarray, yaw_deg: float, pitch_deg: float, color: tuple[int, int, int], label: str) -> None:
    h, w = img.shape[:2]
    u, v = equirect_uv(yaw_deg, pitch_deg, w, h)
    cv2.circle(img, (u, v), 20, color, -1, lineType=cv2.LINE_AA)
    cv2.putText(img, label, (min(w - 110, u + 8), max(22, v - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)


def make_synthetic_equirect(width: int = 2048, height: int = 1024) -> np.ndarray:
    img = np.full((height, width, 3), 245, dtype=np.uint8)
    for pitch in range(-90, 91):
        v = int(round((0.5 - pitch / 180.0) * height))
        if 0 <= v < height:
            shade = 225 - int(abs(pitch) * 0.8)
            img[v, :, :] = np.clip(shade, 120, 235)

    for yaw in [0, 90, 180, -90]:
        u, _ = equirect_uv(yaw, 0, width, height)
        cv2.line(img, (u, 0), (u, height - 1), (0, 0, 0), 4)
        if yaw == 180:
            cv2.line(img, (width - 1, 0), (width - 1, height - 1), (0, 0, 0), 4)
        cv2.putText(img, f"yaw {yaw}", (max(4, min(width - 130, u + 8)), 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    for pitch in [0, 30, -30, 60, -60]:
        _, v = equirect_uv(0, pitch, width, height)
        cv2.line(img, (0, v), (width - 1, v), (40, 40, 40), 3)
        cv2.putText(img, f"pitch {pitch}", (12, max(24, v - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (40, 40, 40), 2)

    draw_marker(img, 0, 0, (0, 220, 0), "front marker")
    draw_marker(img, 90, 0, (220, 0, 0), "right marker")
    draw_marker(img, 0, 30, (0, 0, 220), "up marker")
    draw_marker(img, 25, 10, (220, 0, 220), "refine target")
    return img


def project_world_ray(ray: np.ndarray, yaw: float, pitch: float, fov_x: float, out_w: int, out_h: int) -> tuple[float, float]:
    R = make_rotation(yaw, pitch, 0.0)
    cam = R.T @ np.asarray(ray, dtype=np.float64)
    fx = (out_w / 2.0) / math.tan(math.radians(fov_x) / 2.0)
    u = (cam[0] / cam[2]) * fx + out_w / 2.0
    v = out_h / 2.0 - (cam[1] / cam[2]) * fx
    return float(u), float(v)


def draw_status(img: np.ndarray, lines: list[str]) -> np.ndarray:
    out = img.copy()
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (out.shape[1], 96), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.45, out, 0.55, 0)
    y = 28
    for line in lines:
        cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
        y += 30
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="runs/spherical_camera_synthetic")
    parser.add_argument("--out_w", type=int, default=512)
    parser.add_argument("--out_h", type=int, default=512)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pano = make_synthetic_equirect()
    cv2.imwrite(str(out_dir / "synthetic_equirect.png"), pano)

    crops = {
        "crop_yaw0_pitch0.png": (0.0, 0.0, 80.0),
        "crop_yaw90_pitch0.png": (90.0, 0.0, 80.0),
        "crop_yaw0_pitch30.png": (0.0, 30.0, 80.0),
    }
    for name, (yaw, pitch, fov) in crops.items():
        crop = equirect_to_perspective(pano, yaw=yaw, pitch=pitch, roll=0.0, fov_x=fov, out_w=args.out_w, out_h=args.out_h)
        cv2.imwrite(str(out_dir / name), crop)

    cur_yaw = 0.0
    cur_pitch = 0.0
    cur_fov = 105.0
    target_ray = yaw_pitch_to_ray(25.0, 10.0)
    bbox_cx, bbox_cy = project_world_ray(target_ray, cur_yaw, cur_pitch, cur_fov, args.out_w, args.out_h)
    bbox = [bbox_cx - 44.0, bbox_cy - 30.0, bbox_cx + 44.0, bbox_cy + 30.0]

    before = equirect_to_perspective(pano, yaw=cur_yaw, pitch=cur_pitch, roll=0.0, fov_x=cur_fov, out_w=args.out_w, out_h=args.out_h)
    cv2.rectangle(before, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 255), 2)

    next_yaw, next_pitch, next_fov, debug = compute_next_view_from_bbox(
        bbox=bbox,
        yaw=cur_yaw,
        pitch=cur_pitch,
        roll=0.0,
        fov_x=cur_fov,
        out_w=args.out_w,
        out_h=args.out_h,
        zoom_ratio=0.45,
        min_fov=20.0,
        margin_deg=4.0,
    )
    after = equirect_to_perspective(pano, yaw=next_yaw, pitch=next_pitch, roll=0.0, fov_x=next_fov, out_w=args.out_w, out_h=args.out_h)
    cv2.drawMarker(after, (args.out_w // 2, args.out_h // 2), (0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=28, thickness=2)

    before_status = draw_status(before, [f"before yaw={cur_yaw:.1f} pitch={cur_pitch:.1f} fov={cur_fov:.1f}", f"bbox center=({bbox_cx:.1f},{bbox_cy:.1f})"])
    after_status = draw_status(after, [f"after yaw={next_yaw:.1f} pitch={next_pitch:.1f} fov={next_fov:.1f}", f"action={debug['refine_action']}"])
    cv2.imwrite(str(out_dir / "refine_before.png"), before_status)
    cv2.imwrite(str(out_dir / "refine_after.png"), after_status)
    cv2.imwrite(str(out_dir / "refine_compare.png"), np.concatenate([before_status, after_status], axis=1))

    corner_pixels = []
    R_cur = make_rotation(cur_yaw, cur_pitch, 0.0)
    for u, v in [(bbox[0], bbox[1]), (bbox[2], bbox[1]), (bbox[2], bbox[3]), (bbox[0], bbox[3])]:
        corner_ray = apply_rotation_to_rays(pixel_to_camera_ray(u, v, args.out_w, args.out_h, cur_fov), R_cur)
        corner_pixels.append(project_world_ray(corner_ray, next_yaw, next_pitch, next_fov, args.out_w, args.out_h))

    report = {
        "output_dir": str(out_dir),
        "previous": {"yaw": cur_yaw, "pitch": cur_pitch, "fov": cur_fov},
        "bbox": bbox,
        "next": {"yaw": next_yaw, "pitch": next_pitch, "fov": next_fov},
        "yaw_error_to_target": wrap_yaw_deg(next_yaw - 25.0),
        "pitch_error_to_target": next_pitch - 10.0,
        "corner_pixels_after": corner_pixels,
        "debug_info": debug,
    }
    (out_dir / "debug.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
