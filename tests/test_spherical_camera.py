import importlib.util
import math
from pathlib import Path

import cv2
import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "agent" / "spherical_camera.py"
spec = importlib.util.spec_from_file_location("tools.agent.spherical_camera", MODULE_PATH)
sc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sc)


def equirect_uv(yaw_deg, pitch_deg, width, height):
    u = int(round((float(yaw_deg) / 360.0 + 0.5) * width)) % width
    v = int(round((0.5 - float(pitch_deg) / 180.0) * height))
    return u, max(0, min(height - 1, v))


def draw_marker(img, yaw_deg, pitch_deg, color, label):
    h, w = img.shape[:2]
    u, v = equirect_uv(yaw_deg, pitch_deg, w, h)
    cv2.circle(img, (u, v), 18, color, -1, lineType=cv2.LINE_AA)
    cv2.putText(img, label, (min(w - 90, u + 8), max(18, v - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def make_synthetic_equirect(width=1024, height=512):
    img = np.full((height, width, 3), 245, dtype=np.uint8)
    for pitch in range(-90, 91):
        v = int(round((0.5 - pitch / 180.0) * height))
        if 0 <= v < height:
            shade = 225 - int(abs(pitch) * 0.8)
            img[v, :, :] = np.clip(shade, 120, 235)

    for yaw in [0, 90, 180, -90]:
        u, _ = equirect_uv(yaw, 0, width, height)
        cv2.line(img, (u, 0), (u, height - 1), (0, 0, 0), 3)
        if yaw == 180:
            cv2.line(img, (width - 1, 0), (width - 1, height - 1), (0, 0, 0), 3)
        cv2.putText(img, f"Y{yaw}", (max(2, min(width - 80, u + 5)), 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)

    for pitch in [0, 30, -30, 60, -60]:
        _, v = equirect_uv(0, pitch, width, height)
        cv2.line(img, (0, v), (width - 1, v), (40, 40, 40), 2)
        cv2.putText(img, f"P{pitch}", (8, max(20, v - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 2)

    draw_marker(img, 0, 0, (0, 220, 0), "front")
    draw_marker(img, 90, 0, (220, 0, 0), "right")
    draw_marker(img, 0, 30, (0, 0, 220), "up")
    draw_marker(img, 25, 10, (220, 0, 220), "target")
    return img


def center_mean(img, radius=5):
    h, w = img.shape[:2]
    patch = img[h // 2 - radius : h // 2 + radius + 1, w // 2 - radius : w // 2 + radius + 1]
    return patch.mean(axis=(0, 1))


def assert_channel_dominates(color, channel):
    assert color[channel] > 120
    others = [i for i in range(3) if i != channel]
    assert color[channel] > max(float(color[i]) for i in others) + 40


def project_world_ray(ray, yaw, pitch, fov_x, out_w, out_h):
    R = sc.make_rotation(yaw, pitch, 0.0)
    cam = R.T @ np.asarray(ray, dtype=np.float64)
    assert cam[2] > 0.0
    fx = (out_w / 2.0) / math.tan(math.radians(fov_x) / 2.0)
    u = (cam[0] / cam[2]) * fx + out_w / 2.0
    v = out_h / 2.0 - (cam[1] / cam[2]) * fx
    return float(u), float(v)


def bbox_corner_world_rays(bbox, yaw, pitch, fov_x, out_w, out_h):
    x1, y1, x2, y2 = bbox
    R = sc.make_rotation(yaw, pitch, 0.0)
    rays = []
    for u, v in [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]:
        rays.append(sc.apply_rotation_to_rays(sc.pixel_to_camera_ray(u, v, out_w, out_h, fov_x), R))
    return rays


def test_ray_conventions_and_rotation_roundtrip():
    np.testing.assert_allclose(sc.pixel_to_camera_ray(640, 640, 1280, 1280, 105.0), [0.0, 0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(sc.yaw_pitch_to_ray(90.0, 0.0), [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(sc.yaw_pitch_to_ray(0.0, 30.0), [0.0, 0.5, math.sqrt(3) / 2.0], atol=1e-12)

    for yaw, pitch in [(0.0, 0.0), (90.0, 0.0), (-45.0, 20.0), (135.0, -15.0)]:
        center = sc.make_rotation(yaw, pitch, 0.0) @ np.array([0.0, 0.0, 1.0])
        np.testing.assert_allclose(center, sc.yaw_pitch_to_ray(yaw, pitch), atol=1e-12)
        yy, pp = sc.ray_to_yaw_pitch(center)
        assert abs(sc.wrap_yaw_deg(yy - yaw)) < 1e-9
        assert abs(pp - pitch) < 1e-9


def test_synthetic_perspective_center_directions():
    pano = make_synthetic_equirect()
    crop0 = sc.equirect_to_perspective(pano, yaw=0.0, pitch=0.0, roll=0.0, fov_x=80.0, out_w=512, out_h=512)
    assert_channel_dominates(center_mean(crop0), 1)

    crop90 = sc.equirect_to_perspective(pano, yaw=90.0, pitch=0.0, roll=0.0, fov_x=80.0, out_w=512, out_h=512)
    assert_channel_dominates(center_mean(crop90), 0)

    crop_up = sc.equirect_to_perspective(pano, yaw=0.0, pitch=30.0, roll=0.0, fov_x=80.0, out_w=512, out_h=512)
    assert_channel_dominates(center_mean(crop_up), 2)


def test_compute_next_view_recenters_bbox_and_keeps_corners_visible():
    pano = make_synthetic_equirect()
    out_w = out_h = 512
    cur_yaw = 0.0
    cur_pitch = 0.0
    cur_fov = 105.0
    target_ray = sc.yaw_pitch_to_ray(25.0, 10.0)
    u, v = project_world_ray(target_ray, cur_yaw, cur_pitch, cur_fov, out_w, out_h)
    bbox = [u - 40.0, v - 28.0, u + 40.0, v + 28.0]

    next_yaw, next_pitch, next_fov, debug = sc.compute_next_view_from_bbox(
        bbox=bbox,
        yaw=cur_yaw,
        pitch=cur_pitch,
        roll=0.0,
        fov_x=cur_fov,
        out_w=out_w,
        out_h=out_h,
        zoom_ratio=0.45,
        min_fov=20.0,
        margin_deg=4.0,
    )

    assert abs(sc.wrap_yaw_deg(next_yaw - 25.0)) < 0.5
    assert abs(next_pitch - 10.0) < 0.5
    assert next_fov < cur_fov
    assert next_fov >= debug["safe_fov"]

    after = sc.equirect_to_perspective(pano, yaw=next_yaw, pitch=next_pitch, roll=0.0, fov_x=next_fov, out_w=out_w, out_h=out_h)
    center = center_mean(after)
    assert center[0] > 100 and center[2] > 100

    for corner_ray in bbox_corner_world_rays(bbox, cur_yaw, cur_pitch, cur_fov, out_w, out_h):
        cu, cv = project_world_ray(corner_ray, next_yaw, next_pitch, next_fov, out_w, out_h)
        assert 0.0 <= cu <= out_w
        assert 0.0 <= cv <= out_h
