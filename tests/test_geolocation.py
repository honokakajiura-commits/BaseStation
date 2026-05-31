import json
import tempfile
from pathlib import Path

from tools.agent.geolocation import (
    bbox_center_xy,
    detection_center_to_local_angles,
    local_yaw_to_geo_azimuth,
    project_point,
    wrap360,
)
from tools.agent.gis_export import write_geojson_featurecollection


def test_wrap360_normalizes_angles():
    assert wrap360(0) == 0.0
    assert wrap360(360) == 0.0
    assert wrap360(-10) == 350.0
    assert wrap360(725) == 5.0


def test_project_point_moves_north_and_east():
    lon_n, lat_n = project_point(139.0, 35.0, 0.0, 100.0)
    lon_e, lat_e = project_point(139.0, 35.0, 90.0, 100.0)

    assert lat_n > 35.0
    assert abs(lon_n - 139.0) < 0.0001
    assert lon_e > 139.0
    assert abs(lat_e - 35.0) < 0.0001


def test_bbox_center_xy_returns_center():
    assert bbox_center_xy({"xyxy": [10, 20, 30, 60]}) == (20.0, 40.0)


def test_detection_center_angles_keep_center_view_direction():
    yaw, pitch = detection_center_to_local_angles(
        det={"xyxy": [630, 630, 650, 650]},
        yaw=25.0,
        pitch=10.0,
        fov=105.0,
        image_w=1280,
        image_h=1280,
    )
    assert abs(yaw - 25.0) < 1e-6
    assert abs(pitch - 10.0) < 1e-6


def test_local_yaw_to_geo_azimuth_uses_pano_zero_when_available():
    assert local_yaw_to_geo_azimuth(20.0, pano_zero_azimuth=350.0) == 10.0
    assert local_yaw_to_geo_azimuth(-20.0, pano_zero_azimuth=None) == 340.0


def test_write_geojson_featurecollection_outputs_valid_geojson():
    feature = {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [139.0, 35.0]},
        "properties": {"fid": "x"},
    }
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "points.geojson"
        write_geojson_featurecollection([feature], out_path)
        obj = json.loads(out_path.read_text(encoding="utf-8"))

    assert obj["type"] == "FeatureCollection"
    assert len(obj["features"]) == 1
    assert obj["features"][0]["geometry"]["type"] == "Point"
