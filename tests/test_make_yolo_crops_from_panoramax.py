import importlib.util
from pathlib import Path

import cv2
import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "make_yolo_crops_from_panoramax.py"
spec = importlib.util.spec_from_file_location("make_yolo_crops_from_panoramax", MODULE_PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def make_blank(width=1600, height=800):
    return np.full((height, width, 3), 255, dtype=np.uint8)


def draw_line(img, x1, y1, x2, y2, thickness=3):
    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), thickness)
    return img


def make_horizontal_scene(angle_deg=0.0, n=6):
    img = make_blank()
    cx = img.shape[1] // 2
    cy = img.shape[0] // 2
    ang = np.deg2rad(angle_deg)
    dx = int(np.cos(ang) * 450)
    dy = int(np.sin(ang) * 450)
    ys = np.linspace(120, 680, n).astype(int)
    for y in ys:
        draw_line(img, cx - dx, y - dy, cx + dx, y + dy, 3)
    return img


def make_vertical_dominant_scene(horizontal_angle_deg=2.0, n_horizontal=7, n_vertical=12):
    img = make_horizontal_scene(angle_deg=horizontal_angle_deg, n=n_horizontal)
    xs = np.linspace(150, 1450, n_vertical).astype(int)
    for x in xs:
        draw_line(img, x, 100, x, 700, 3)
    return img


def test_choose_best_image_href_prefers_hd_over_sd_and_thumb():
    assets = {
        "hd": {"href": "https://example.com/images/full.jpg", "type": "image/jpeg", "roles": ["data"]},
        "sd": {"href": "https://example.com/derivates/full/sd.jpg", "type": "image/jpeg", "roles": ["visual"]},
        "thumb": {"href": "https://example.com/derivates/full/thumb.jpg", "type": "image/jpeg", "roles": ["thumbnail"]},
    }
    chosen = mod.choose_best_image_href_from_assets_dict(assets)
    assert chosen == assets["hd"]["href"]



def test_choose_best_image_href_uses_asset_key_fallback_without_roles():
    assets = {
        "hd": {"href": "https://example.com/images/full.jpg", "type": "image/jpeg"},
        "sd": {"href": "https://example.com/derivates/full/sd.jpg", "type": "image/jpeg"},
        "thumb": {"href": "https://example.com/derivates/full/thumb.jpg", "type": "image/jpeg"},
    }
    chosen = mod.choose_best_image_href_from_assets_dict(assets)
    assert chosen == assets["hd"]["href"]



def test_estimate_roll_fallback_few_horizontal():
    img = make_horizontal_scene(angle_deg=2.0, n=2)
    roll_deg, reason, meta = mod.estimate_roll_deg_from_crop(img)
    assert roll_deg == 0.0
    assert reason == "fallback_few_horizontal"
    assert meta["n_horizontal"] < 5



def test_estimate_roll_fallback_not_horizontal_dominant():
    img = make_vertical_dominant_scene(horizontal_angle_deg=2.0, n_horizontal=7, n_vertical=12)
    roll_deg, reason, meta = mod.estimate_roll_deg_from_crop(img)
    assert roll_deg == 0.0
    assert reason == "fallback_not_horizontal_dominant"
    assert meta["n_horizontal"] <= meta["n_vertical"]



def test_estimate_roll_fallback_small_angle():
    img = make_horizontal_scene(angle_deg=1.0, n=8)
    roll_deg, reason, meta = mod.estimate_roll_deg_from_crop(img)
    assert roll_deg == 0.0
    assert reason == "fallback_small_angle"
    assert abs(meta["angle_deg"]) < 1.5



def test_estimate_roll_fallback_large_angle():
    img = make_horizontal_scene(angle_deg=6.0, n=8)
    roll_deg, reason, meta = mod.estimate_roll_deg_from_crop(img)
    assert roll_deg == 0.0
    assert reason == "fallback_large_angle"
    assert abs(meta["angle_deg"]) > 5.0



def test_estimate_roll_applies_only_when_safe():
    img = make_horizontal_scene(angle_deg=3.0, n=8)
    roll_deg, reason, meta = mod.estimate_roll_deg_from_crop(img)
    assert reason == "applied"
    assert 1.5 <= abs(roll_deg) <= 5.0
    assert meta["n_horizontal"] >= 5
    assert meta["n_horizontal"] > meta["n_vertical"]
