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
    img = make_blank()
    draw_line(img, 700, 400, 900, 404, 3)
    roll_deg, reason, meta = mod.estimate_roll_deg_from_crop(img)
    assert roll_deg == 0.0
    assert reason == "fallback_few_horizontal"
    assert meta["n_horizontal"] < 5



def test_estimate_roll_fallback_not_horizontal_dominant():
    img = make_vertical_dominant_scene(horizontal_angle_deg=2.0, n_horizontal=7, n_vertical=20)
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


def test_estimate_pano_roll_applies_consensus_from_multiple_views():
    orig = mod.estimate_roll_deg_from_crop
    results = iter([
        (-4.0, "applied", {"angle_deg": -4.0, "horizontal_weight": 120.0, "n_horizontal": 10, "n_vertical": 1}),
        (-3.8, "applied", {"angle_deg": -3.8, "horizontal_weight": 110.0, "n_horizontal": 9, "n_vertical": 1}),
        (-4.2, "applied", {"angle_deg": -4.2, "horizontal_weight": 105.0, "n_horizontal": 8, "n_vertical": 1}),
        (0.0, "fallback_few_horizontal", {"angle_deg": 0.0, "horizontal_weight": 0.0, "n_horizontal": 0, "n_vertical": 0}),
    ])

    def fake_estimate(*args, **kwargs):
        return next(results)

    mod.estimate_roll_deg_from_crop = fake_estimate
    try:
        pano = make_blank(width=4096, height=2048)
        roll_deg, reason, meta = mod.estimate_pano_roll_deg(
            pano,
            yaw_center_deg=0.0,
            pitch_deg=-30.0,
            out_w=1600,
            out_h=800,
            fov_front=115.0,
            fov_side=115.0,
            fov_back=115.0,
        )
    finally:
        mod.estimate_roll_deg_from_crop = orig

    assert reason == "applied"
    assert abs(roll_deg + 4.0) < 1e-6
    assert meta["n_candidates"] == 3


def test_estimate_pano_roll_rejects_inconsistent_signs():
    orig = mod.estimate_roll_deg_from_crop
    results = iter([
        (-4.0, "applied", {"angle_deg": -4.0, "horizontal_weight": 100.0, "n_horizontal": 8, "n_vertical": 1}),
        (4.2, "applied", {"angle_deg": 4.2, "horizontal_weight": 95.0, "n_horizontal": 8, "n_vertical": 1}),
        (-0.5, "fallback_small_angle", {"angle_deg": -0.5, "horizontal_weight": 90.0, "n_horizontal": 8, "n_vertical": 1}),
        (0.0, "fallback_few_horizontal", {"angle_deg": 0.0, "horizontal_weight": 0.0, "n_horizontal": 0, "n_vertical": 0}),
    ])

    def fake_estimate(*args, **kwargs):
        return next(results)

    mod.estimate_roll_deg_from_crop = fake_estimate
    try:
        pano = make_blank(width=4096, height=2048)
        roll_deg, reason, meta = mod.estimate_pano_roll_deg(
            pano,
            yaw_center_deg=0.0,
            pitch_deg=-30.0,
            out_w=1600,
            out_h=800,
            fov_front=115.0,
            fov_side=115.0,
            fov_back=115.0,
        )
    finally:
        mod.estimate_roll_deg_from_crop = orig

    assert roll_deg == 0.0
    assert reason == "fallback_insufficient_candidates"
    assert meta["n_candidates"] == 2


def test_estimate_pano_roll_applies_weak_single_view_when_support_is_sparse():
    orig = mod.estimate_roll_deg_from_crop
    results = iter([
        (0.0, "fallback_few_horizontal", {"angle_deg": 0.0, "horizontal_weight": 0.0, "vertical_weight": 0.0, "n_horizontal": 0, "n_vertical": 0}),
        (0.0, "fallback_small_angle", {"angle_deg": 0.6, "horizontal_weight": 1800.0, "vertical_weight": 200.0, "n_horizontal": 12, "n_vertical": 1}),
        (-3.0, "applied", {"angle_deg": -3.0, "horizontal_weight": 4200.0, "vertical_weight": 300.0, "n_horizontal": 18, "n_vertical": 1}),
        (0.0, "fallback_few_horizontal", {"angle_deg": 0.0, "horizontal_weight": 0.0, "vertical_weight": 0.0, "n_horizontal": 0, "n_vertical": 0}),
    ])

    def fake_estimate(*args, **kwargs):
        return next(results)

    mod.estimate_roll_deg_from_crop = fake_estimate
    try:
        pano = make_blank(width=4096, height=2048)
        roll_deg, reason, meta = mod.estimate_pano_roll_deg(
            pano,
            yaw_center_deg=0.0,
            pitch_deg=-30.0,
            out_w=1600,
            out_h=800,
            fov_front=115.0,
            fov_side=115.0,
            fov_back=115.0,
        )
    finally:
        mod.estimate_roll_deg_from_crop = orig

    assert reason == "applied_weak_single_view"
    assert -2.5 <= roll_deg <= -1.5
    assert meta["n_candidates"] == 1


def test_estimate_pano_roll_trims_same_sign_outlier_before_consensus():
    orig = mod.estimate_roll_deg_from_crop
    results = iter([
        (-4.2, "applied", {"angle_deg": -4.2, "horizontal_weight": 6000.0, "vertical_weight": 300.0, "n_horizontal": 20, "n_vertical": 1}),
        (-3.8, "applied", {"angle_deg": -3.8, "horizontal_weight": 5200.0, "vertical_weight": 250.0, "n_horizontal": 18, "n_vertical": 1}),
        (-7.7, "applied", {"angle_deg": -7.7, "horizontal_weight": 1800.0, "vertical_weight": 150.0, "n_horizontal": 9, "n_vertical": 1}),
        (0.0, "fallback_small_angle", {"angle_deg": 0.4, "horizontal_weight": 3000.0, "vertical_weight": 100.0, "n_horizontal": 14, "n_vertical": 0}),
    ])

    def fake_estimate(*args, **kwargs):
        return next(results)

    mod.estimate_roll_deg_from_crop = fake_estimate
    try:
        pano = make_blank(width=4096, height=2048)
        roll_deg, reason, meta = mod.estimate_pano_roll_deg(
            pano,
            yaw_center_deg=0.0,
            pitch_deg=-30.0,
            out_w=1600,
            out_h=800,
            fov_front=115.0,
            fov_side=115.0,
            fov_back=115.0,
        )
    finally:
        mod.estimate_roll_deg_from_crop = orig

    assert reason == "applied"
    assert abs(roll_deg + 4.2) < 0.5
    assert meta["consensus_abs_dev_deg"] < 2.4


def make_grid_pano(width=2048, height=1024):
    img = make_blank(width=width, height=height)
    for x in np.linspace(0, width - 1, 16).astype(int):
        cv2.line(img, (x, 0), (x, height - 1), (0, 0, 0), 2)
    for y in np.linspace(0, height - 1, 8).astype(int):
        cv2.line(img, (0, y), (width - 1, y), (0, 0, 0), 2)
    return img


def test_extract_line_segments_finds_structure():
    img = make_horizontal_scene(angle_deg=0.0, n=8)
    segs = mod.extract_line_segments(img)
    assert len(segs) >= 6
    assert all(seg['length'] > 0 for seg in segs)


def test_apply_global_upright_identity_returns_same_object():
    pano = make_grid_pano()
    out = mod.apply_global_upright_to_equirectangular(pano, roll_deg=0.0, pitch_deg=0.0)
    assert out is pano


def test_apply_global_upright_preserves_shape_for_nonzero_rotation():
    pano = make_grid_pano()
    out = mod.apply_global_upright_to_equirectangular(pano, roll_deg=2.0, pitch_deg=-1.0)
    assert out.shape == pano.shape


def test_score_upright_candidate_reports_view_stats():
    pano = make_grid_pano()
    score = mod.score_upright_candidate(
        pano,
        yaw_center_deg=0.0,
        cand_roll_deg=0.0,
        cand_pitch_deg=0.0,
        sample_out_w=320,
        sample_out_h=192,
    )
    assert score['n_valid_views'] >= 1
    assert score['n_lines'] > 0
    assert len(score['views']) == 8


def test_estimate_global_upright_roll_pitch_falls_back_on_blank_pano():
    pano = make_blank(width=2048, height=1024)
    roll_deg, pitch_deg, reason, meta = mod.estimate_global_upright_roll_pitch(
        pano,
        yaw_center_deg=0.0,
    )
    assert roll_deg == 0.0
    assert pitch_deg == 0.0
    assert reason in {'fallback_insufficient_structure', 'fallback_low_gain', 'fallback_small_adjustment'}
    assert meta['n_lines'] == 0



def test_estimate_global_upright_roll_pitch_prefers_baseline_when_gain_is_small():
    orig = mod.score_upright_candidate

    def fake_score(*args, **kwargs):
        roll_deg = float(kwargs['cand_roll_deg'])
        pitch_deg = float(kwargs['cand_pitch_deg'])
        score = 1000.0
        if abs(roll_deg - 1.0) < 1e-6 and abs(pitch_deg) < 1e-6:
            score = 1050.0
        return {
            'roll_deg': roll_deg,
            'pitch_deg': pitch_deg,
            'score': score,
            'raw_score': score,
            'horizontal_score': 500.0,
            'vertical_score': 500.0,
            'balance': 1.0,
            'side_score': 500.0,
            'front_back_score': 500.0,
            'side_balance': 1.0,
            'effective_views': 8,
            'view_coverage': 1.0,
            'dominant_view_share': 0.2,
            'magnitude_penalty': 1.0,
            'n_valid_views': 8,
            'n_lines': 120,
            'views': [],
        }

    mod.score_upright_candidate = fake_score
    try:
        pano = make_blank(width=2048, height=1024)
        roll_deg, pitch_deg, reason, meta = mod.estimate_global_upright_roll_pitch(pano, yaw_center_deg=0.0)
    finally:
        mod.score_upright_candidate = orig

    assert roll_deg == 0.0
    assert pitch_deg == 0.0
    assert reason in {'fallback_low_gain', 'fallback_low_gain_ratio'}
    assert meta['baseline_score'] == 1000.0
    assert meta['best_score'] == 1050.0
    assert meta['score_gain'] == 50.0


def test_estimate_global_upright_roll_pitch_rejects_large_adjustment():
    orig = mod.score_upright_candidate

    def fake_score(*args, **kwargs):
        roll_deg = float(kwargs['cand_roll_deg'])
        pitch_deg = float(kwargs['cand_pitch_deg'])
        score = 1000.0
        if abs(roll_deg - 5.0) < 1e-6 and abs(pitch_deg - 3.0) < 1e-6:
            score = 2000.0
        return {
            'roll_deg': roll_deg,
            'pitch_deg': pitch_deg,
            'score': score,
            'raw_score': score,
            'horizontal_score': 1000.0,
            'vertical_score': 1000.0,
            'balance': 1.0,
            'side_score': 1000.0,
            'front_back_score': 1000.0,
            'side_balance': 1.0,
            'effective_views': 8,
            'view_coverage': 1.0,
            'dominant_view_share': 0.2,
            'magnitude_penalty': 0.2,
            'n_valid_views': 8,
            'n_lines': 120,
            'views': [],
        }

    mod.score_upright_candidate = fake_score
    try:
        pano = make_blank(width=2048, height=1024)
        roll_deg, pitch_deg, reason, meta = mod.estimate_global_upright_roll_pitch(pano, yaw_center_deg=0.0)
    finally:
        mod.score_upright_candidate = orig

    assert roll_deg == 0.0
    assert pitch_deg == 0.0
    assert reason == 'fallback_large_adjustment'
    assert meta['best_before_fallback']['roll_deg'] == 5.0
    assert meta['best_before_fallback']['pitch_deg'] == 3.0
