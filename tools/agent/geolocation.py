#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Geolocation helpers for converting detections into map-checkable rays."""

from __future__ import annotations

import math
import re
from typing import Any, Dict, Optional, Tuple

from .spherical_camera import bbox_center_to_world_ray, ray_to_yaw_pitch


EARTH_RADIUS_M = 6371008.8
DEFAULT_CONF_HIGH_THRESHOLD = 0.60
DEFAULT_CONF_MEDIUM_THRESHOLD = 0.30
DEFAULT_OBSERVATION_OFFSET_M = 5.0
_REFINED_TEXT_TOKENS = ("refine", "refined", "retry", "recenter", "zoom")
_INITIAL_TEXT_TOKENS = ("initial", "init")
_NORMAL_INITIAL_VIEWS = {"front", "left", "right"}
_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9._-]+")


def wrap360(deg: float) -> float:
    """Normalize degrees into ``0 <= deg < 360``."""
    return float(float(deg) % 360.0)


def project_point(lon: float, lat: float, azimuth_deg: float, distance_m: float) -> Tuple[float, float]:
    """Project a WGS84 lon/lat by azimuth and distance using a spherical Earth.

    ``azimuth_deg`` follows GIS convention: 0 is north, 90 is east.
    The short distances used for detection rays do not need an added geodesy
    dependency; this approximation is within practical ArcGIS inspection needs.
    """
    lon1 = math.radians(float(lon))
    lat1 = math.radians(float(lat))
    az = math.radians(wrap360(azimuth_deg))
    angular_dist = float(distance_m) / EARTH_RADIUS_M

    sin_lat1 = math.sin(lat1)
    cos_lat1 = math.cos(lat1)
    sin_d = math.sin(angular_dist)
    cos_d = math.cos(angular_dist)

    lat2 = math.asin(sin_lat1 * cos_d + cos_lat1 * sin_d * math.cos(az))
    lon2 = lon1 + math.atan2(
        math.sin(az) * sin_d * cos_lat1,
        cos_d - sin_lat1 * math.sin(lat2),
    )

    lon_deg = (math.degrees(lon2) + 540.0) % 360.0 - 180.0
    lat_deg = math.degrees(lat2)
    return float(lon_deg), float(lat_deg)


def bbox_center_xy(det: Any) -> Tuple[float, float]:
    """Return bbox center from a YOLO-style detection dict or raw xyxy list."""
    raw = det.get("xyxy") if isinstance(det, dict) else det
    if raw is None or len(raw) != 4:
        raise ValueError(f"det must be xyxy or dict with xyxy: {det}")
    x1, y1, x2, y2 = [float(x) for x in raw]
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def detection_center_to_local_angles(
    det: Any,
    yaw: float,
    pitch: float,
    fov: float,
    image_w: int,
    image_h: int,
) -> Tuple[float, float]:
    """Convert a detection bbox center into panorama-local yaw/pitch angles."""
    ray = bbox_center_to_world_ray(
        bbox=det,
        yaw=float(yaw),
        pitch=float(pitch),
        roll=0.0,
        fov_x=float(fov),
        out_w=int(image_w),
        out_h=int(image_h),
    )
    target_yaw, target_pitch = ray_to_yaw_pitch(ray)
    return float(target_yaw), float(target_pitch)


def local_yaw_to_geo_azimuth(local_yaw: float, pano_zero_azimuth: Optional[float] = None) -> float:
    """Convert panorama-local yaw into geographic azimuth."""
    if pano_zero_azimuth is None:
        return wrap360(local_yaw)
    return wrap360(float(pano_zero_azimuth) + float(local_yaw))


def ray_to_bearing_elevation(ray: Any) -> Dict[str, float]:
    yaw_deg, pitch_deg = ray_to_yaw_pitch(ray)
    return {"bearing_deg": float(yaw_deg), "elevation_deg": float(pitch_deg)}


def bbox_to_bearing_elevation(
    bbox: Any,
    yaw: float,
    pitch: float,
    roll: float,
    fov_x: float,
    out_w: int,
    out_h: int,
) -> Dict[str, float]:
    ray = bbox_center_to_world_ray(
        bbox=bbox,
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        fov_x=fov_x,
        out_w=out_w,
        out_h=out_h,
    )
    return ray_to_bearing_elevation(ray)


def _safe_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def classify_confidence(
    conf: Any,
    high_threshold: float = DEFAULT_CONF_HIGH_THRESHOLD,
    medium_threshold: float = DEFAULT_CONF_MEDIUM_THRESHOLD,
) -> str:
    """Classify detection confidence for ArcGIS symbol rules."""
    value = _safe_float(conf)
    if value is None:
        return "unknown"
    if value >= float(high_threshold):
        return "high"
    if value >= float(medium_threshold):
        return "medium"
    return "low"


def _safe_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "refined"}:
        return True
    if text in {"0", "false", "no", "n", "initial"}:
        return False
    return None


def _text_has_any(value: Any, tokens: Tuple[str, ...]) -> bool:
    text = str(value or "").strip().lower()
    return bool(text) and any(token in text for token in tokens)


def _refine_status_from_step(step: Any) -> Optional[str]:
    if step is None or step == "":
        return None
    numeric = _safe_float(step)
    if numeric is not None:
        return "refined" if numeric > 0 else "initial"
    if _text_has_any(step, _REFINED_TEXT_TOKENS):
        return "refined"
    if _text_has_any(step, _INITIAL_TEXT_TOKENS):
        return "initial"
    return None


def _refine_status_from_crop_path(crop_path: Any) -> Optional[str]:
    text = str(crop_path or "").strip().lower()
    if not text:
        return None
    if _text_has_any(text, _REFINED_TEXT_TOKENS):
        return "refined"

    match = re.search(r"(?:^|__)r(\d+)(?:__|\.|$)", text)
    if match:
        return "refined" if int(match.group(1)) > 0 else "initial"
    if "__actinit" in text:
        return "initial"
    return None


def infer_refine_status(row: Dict[str, Any]) -> str:
    """Infer whether a detection came from the initial or refined crop."""
    explicit = str(row.get("refine_status") or "").strip().lower()
    if explicit in {"initial", "refined", "unknown"}:
        return explicit

    if "is_refined" in row:
        refined = _safe_bool(row.get("is_refined"))
        if refined is not None:
            return "refined" if refined else "initial"

    for key in ("refine_action", "retry_action", "recenter_action"):
        value = row.get(key)
        text = str(value or "").strip().lower()
        if text and text not in {"none", "false", "0", "initial"}:
            return "refined"

    step = row.get("step")
    if step in (None, ""):
        step = row.get("s")
    status = _refine_status_from_step(step)
    if status:
        return status

    status = _refine_status_from_crop_path(row.get("crop_path"))
    if status:
        return status

    view = str(row.get("view") or "").strip().lower()
    if view in _NORMAL_INITIAL_VIEWS:
        return "initial"
    return "unknown"


def is_refined_value(status: str) -> int:
    """Return an ArcGIS-friendly 0/1 flag from a refine status."""
    return 1 if str(status or "").strip().lower() == "refined" else 0


def safe_id(value: Any, max_length: int = 120) -> str:
    """Return a stable filename-safe identifier."""
    text = str(value or "").strip()
    text = _SAFE_ID_RE.sub("_", text)
    text = re.sub(r"_+", "_", text).strip("._-")
    if not text:
        return "ray"
    text = text[: int(max_length)].rstrip("._-")
    return text or "ray"


def make_ray_id(fid: Any, view: Any, step: Any, index: int) -> str:
    return safe_id(f"{fid}_{view}_{step}_{index}")


def _row_lon_lat(row: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    lon = _safe_float(row.get("lon"))
    lat = _safe_float(row.get("lat"))
    if lon is not None and lat is not None:
        return lon, lat

    props = row.get("properties") or {}
    lon = _safe_float(props.get("lon"))
    lat = _safe_float(props.get("lat"))
    if lon is not None and lat is not None:
        return lon, lat

    geom = row.get("geometry") or {}
    coords = geom.get("coordinates") if geom.get("type") == "Point" else None
    if isinstance(coords, (list, tuple)) and len(coords) >= 2:
        return _safe_float(coords[0]), _safe_float(coords[1])
    return None, None


def _row_view_azimuth(row: Dict[str, Any]) -> Tuple[Optional[float], str]:
    props = row.get("properties") or {}
    for key in ("view_azimuth", "view:azimuth", "azimuth"):
        val = _safe_float(row.get(key))
        if val is not None:
            return val, key
        val = _safe_float(props.get(key))
        if val is not None:
            return val, f"properties.{key}"
    return None, "local_yaw_fallback"


def make_detection_ray_record(
    detection: Dict[str, Any],
    index_row: Dict[str, Any],
    yaw_row: Optional[Dict[str, Any]] = None,
    image_w: int = 1280,
    image_h: int = 1280,
    ray_length_m: float = 100.0,
) -> Optional[Dict[str, Any]]:
    """Convert one detection row into an intermediate detection-ray record."""
    det = detection.get("best")
    if not det:
        return None

    fid = str(detection.get("fid") or index_row.get("fid") or index_row.get("id") or "")
    if not fid:
        return None

    camera_lon, camera_lat = _row_lon_lat(index_row)
    if camera_lon is None or camera_lat is None:
        return None

    yaw = _safe_float(detection.get("yaw"))
    if yaw is None:
        yaw_center = _safe_float((yaw_row or {}).get("yaw_center")) or 0.0
        view_offsets = {"front": 0.0, "left": -90.0, "right": 90.0}
        yaw = yaw_center + view_offsets.get(str(detection.get("view") or ""), 0.0)
    pitch = _safe_float(detection.get("pitch_deg"))
    if pitch is None:
        pitch = _safe_float(detection.get("pitch")) or 0.0
    fov = _safe_float(detection.get("fov")) or 105.0

    local_yaw, local_pitch = detection_center_to_local_angles(
        det=det,
        yaw=yaw,
        pitch=pitch,
        fov=fov,
        image_w=image_w,
        image_h=image_h,
    )
    pano_zero_azimuth, azimuth_source = _row_view_azimuth(index_row)
    geo_azimuth = local_yaw_to_geo_azimuth(local_yaw, pano_zero_azimuth)
    if pano_zero_azimuth is None:
        azimuth_source = "local_yaw_fallback"

    end_lon, end_lat = project_point(camera_lon, camera_lat, geo_azimuth, ray_length_m)
    cx, cy = bbox_center_xy(det)
    conf = _safe_float(det.get("conf") if isinstance(det, dict) else None)
    step = detection.get("s")
    if step in (None, ""):
        step = detection.get("step", "")
    refine_status = infer_refine_status(detection)

    return {
        "fid": fid,
        "view": detection.get("view", ""),
        "step": step,
        "conf": conf,
        "conf_class": classify_confidence(conf),
        "refine_status": refine_status,
        "is_refined": is_refined_value(refine_status),
        "local_yaw": float(local_yaw),
        "local_pitch": float(local_pitch),
        "geo_azimuth": float(geo_azimuth),
        "elevation": float(local_pitch),
        "azimuth_source": azimuth_source,
        "pano_zero_azimuth": pano_zero_azimuth,
        "ray_length_m": float(ray_length_m),
        "crop_path": detection.get("crop_path", ""),
        "annotated_path": detection.get("annotated_path", ""),
        "sequence_id": detection.get("sequence_id") or index_row.get("sequence_id", ""),
        "rank_in_collection": detection.get("rank_in_collection", index_row.get("rank_in_collection", None)),
        "yaw_center": _safe_float((yaw_row or {}).get("yaw_center")),
        "crop_yaw": float(yaw),
        "crop_pitch": float(pitch),
        "crop_fov": float(fov),
        "bbox_center": [float(cx), float(cy)],
        "start_lon": float(camera_lon),
        "start_lat": float(camera_lat),
        "camera_lon": float(camera_lon),
        "camera_lat": float(camera_lat),
        "end_lon": float(end_lon),
        "end_lat": float(end_lat),
    }


def make_observation_point_record(
    ray_record: Dict[str, Any],
    offset_m: float = DEFAULT_OBSERVATION_OFFSET_M,
) -> Dict[str, Any]:
    """Create the ArcGIS click target point for a detection ray."""
    lon, lat = project_point(
        lon=float(ray_record["camera_lon"]),
        lat=float(ray_record["camera_lat"]),
        azimuth_deg=float(ray_record["geo_azimuth"]),
        distance_m=float(offset_m),
    )
    keys = [
        "ray_id",
        "fid",
        "view",
        "step",
        "camera_lon",
        "camera_lat",
        "end_lon",
        "end_lat",
        "conf",
        "conf_class",
        "refine_status",
        "is_refined",
        "local_yaw",
        "local_pitch",
        "geo_azimuth",
        "elevation",
        "azimuth_source",
        "crop_path",
        "annotated_path",
        "sequence_id",
        "rank_in_collection",
    ]
    out = {key: ray_record.get(key, "") for key in keys}
    out["lon"] = float(lon)
    out["lat"] = float(lat)
    return out
