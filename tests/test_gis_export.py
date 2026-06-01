import math
from pathlib import Path

from tools.agent.geolocation import (
    classify_confidence,
    infer_refine_status,
    make_observation_point_record,
    safe_id,
)
from tools.agent.gis_export import prepare_annotated_attachments


def _distance_m(lon1, lat1, lon2, lat2):
    radius_m = 6371008.8
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return radius_m * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def test_conf_class_thresholds():
    assert classify_confidence(0.60) == "high"
    assert classify_confidence(0.599) == "medium"
    assert classify_confidence(0.30) == "medium"
    assert classify_confidence(0.299) == "low"


def test_refine_status_from_step_and_crop_path():
    assert infer_refine_status({"s": 0, "view": "front"}) == "initial"
    assert infer_refine_status({"s": 1, "view": "front"}) == "refined"
    assert infer_refine_status({"step": "retry", "view": "front"}) == "refined"
    assert infer_refine_status({"crop_path": "crops/00001__fid__front__r2__yawp0.jpg"}) == "refined"
    assert infer_refine_status({"crop_path": "crops/refine/fid.jpg"}) == "refined"
    assert infer_refine_status({"view": "left"}) == "initial"
    assert infer_refine_status({"view": "diagonal"}) == "unknown"


def test_safe_ray_id_is_filename_safe():
    assert safe_id("fid:abc/view left step#1") == "fid_abc_view_left_step_1"
    assert safe_id("../") == "ray"


def test_observation_point_offsets_from_camera_by_distance():
    ray_record = {
        "ray_id": "r1",
        "camera_lon": 139.0,
        "camera_lat": 35.0,
        "end_lon": 139.001,
        "end_lat": 35.0,
        "geo_azimuth": 90.0,
    }
    point = make_observation_point_record(ray_record, offset_m=5.0)

    assert point["lon"] > ray_record["camera_lon"]
    assert abs(point["lat"] - ray_record["camera_lat"]) < 0.0001
    assert abs(_distance_m(139.0, 35.0, point["lon"], point["lat"]) - 5.0) < 0.01


def test_annotated_attachments_include_only_existing_images_and_windows_paths(tmp_path):
    run_dir = tmp_path / "run"
    annotated_dir = run_dir / "annotated"
    annotated_dir.mkdir(parents=True)
    source = annotated_dir / "crop.jpg"
    source.write_bytes(b"jpg")

    copy_dir = tmp_path / "arcgis_annotated"
    records = [
        {"ray_id": "fid_front_0_1", "annotated_path": str(source)},
        {"ray_id": "missing", "annotated_path": str(run_dir / "annotated" / "missing.jpg")},
    ]

    rows, windows_rows = prepare_annotated_attachments(
        records=records,
        run_dir=run_dir,
        arcgis_annotated_dir=copy_dir,
        arcgis_windows_annotated_dir=r"C:\Users\kajiura\Desktop\arcGIS_data\detection_annotated",
    )

    assert rows == [
        {
            "ray_id": "fid_front_0_1",
            "image_type": "annotated",
            "image_path": str(copy_dir / "fid_front_0_1_annotated.jpg"),
        }
    ]
    assert Path(rows[0]["image_path"]).exists()
    assert windows_rows == [
        {
            "ray_id": "fid_front_0_1",
            "image_type": "annotated",
            "image_path": r"C:\Users\kajiura\Desktop\arcGIS_data\detection_annotated\fid_front_0_1_annotated.jpg",
        }
    ]
