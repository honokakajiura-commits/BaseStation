from tools.agent.refine_policy import (
    bbox_area_ratio,
    classify_detection,
    compute_target_fov_by_bbox_area,
    plan_refine_view,
)


def test_compute_target_fov_zooms_in_for_small_bbox():
    fov = compute_target_fov_by_bbox_area(
        current_fov=100.0,
        area_ratio=1.0 / 100.0,
        target_area_ratio=1.0 / 12.0,
        max_zoom_ratio=0.6,
        min_fov=20.0,
    )
    assert fov < 100.0
    assert fov == 60.0


def test_compute_target_fov_stays_near_current_at_target_area():
    fov = compute_target_fov_by_bbox_area(
        current_fov=100.0,
        area_ratio=1.0 / 12.0,
        target_area_ratio=1.0 / 12.0,
        max_zoom_ratio=0.6,
        min_fov=20.0,
    )
    assert abs(fov - 100.0) < 1e-9


def test_compute_target_fov_respects_min_fov():
    fov = compute_target_fov_by_bbox_area(
        current_fov=30.0,
        area_ratio=1.0 / 10_000.0,
        target_area_ratio=1.0 / 12.0,
        max_zoom_ratio=0.1,
        min_fov=20.0,
    )
    assert fov == 20.0


def test_compute_target_fov_respects_max_zoom_ratio_floor():
    fov = compute_target_fov_by_bbox_area(
        current_fov=100.0,
        area_ratio=1.0 / 10_000.0,
        target_area_ratio=1.0 / 12.0,
        max_zoom_ratio=0.6,
        min_fov=20.0,
    )
    assert fov == 60.0


def test_plan_refine_view_prefers_safe_fov_when_larger_than_target():
    det = {"xyxy": [0.0, 250.0, 512.0, 262.0], "conf": 0.4}
    next_yaw, next_pitch, next_fov, action, debug = plan_refine_view(
        det=det,
        yaw=0.0,
        pitch=0.0,
        roll=0.0,
        current_fov=105.0,
        image_w=512,
        image_h=512,
        min_fov=20.0,
        margin_deg=4.0,
        max_zoom_ratio=0.6,
    )
    assert debug["safe_fov"] > debug["target_fov"]
    assert next_fov == debug["safe_fov"]
    assert next_fov > 100.0
    assert abs(next_pitch) < 1e-6
    assert action in {"widen_only", "recenter_and_widen"}


def test_bbox_area_ratio_and_classify_detection():
    det = {"xyxy": [10.0, 10.0, 30.0, 50.0], "conf": 0.7}
    assert bbox_area_ratio(det, 100, 100) == 0.08
    assert classify_detection(None, high_conf=0.6) == "no_detection"
    assert classify_detection(det, high_conf=0.6, low_conf=0.2) == "confirmed"
    assert classify_detection({"conf": 0.3, "xyxy": [0, 0, 1, 1]}, high_conf=0.6, low_conf=0.2) == "refine"

