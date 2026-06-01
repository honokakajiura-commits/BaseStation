#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""GeoJSON export helpers for ArcGIS inspection."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from pathlib import PureWindowsPath
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .geolocation import make_detection_ray_record, make_observation_point_record, make_ray_id, safe_id
from .io_utils import read_jsonl, save_csv


CAMERA_POINT_CSV_FIELDS = [
    "fid",
    "lon",
    "lat",
    "sequence_id",
    "rank_in_collection",
    "yaw_center",
    "detection_count",
    "refined_detection_count",
]

DETECTION_RAY_CSV_FIELDS = [
    "ray_id",
    "fid",
    "view",
    "step",
    "start_lon",
    "start_lat",
    "end_lon",
    "end_lat",
    "camera_lon",
    "camera_lat",
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

OBSERVATION_POINT_CSV_FIELDS = [
    "ray_id",
    "fid",
    "view",
    "step",
    "lon",
    "lat",
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

ATTACHMENT_CSV_FIELDS = ["ray_id", "image_type", "image_path"]


def features_to_geojson(features: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    return {"type": "FeatureCollection", "features": list(features)}


def write_geojson_featurecollection(features: Iterable[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(features_to_geojson(features), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_geojson(path: Path, features: Iterable[Dict[str, Any]]) -> None:
    """Compatibility wrapper for older callers."""
    write_geojson_featurecollection(features, path)


def _clean_props(record: Dict[str, Any]) -> Dict[str, Any]:
    geometry_keys = {"lon", "lat", "camera_lon", "camera_lat", "start_lon", "start_lat", "end_lon", "end_lat"}
    return {k: v for k, v in record.items() if k not in geometry_keys}


def make_camera_point_records(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_fid: Dict[str, Dict[str, Any]] = {}
    for rec in records:
        fid = str(rec.get("fid", ""))
        if not fid:
            continue
        if fid not in by_fid:
            by_fid[fid] = {
                "fid": fid,
                "lon": float(rec["camera_lon"]),
                "lat": float(rec["camera_lat"]),
                "sequence_id": rec.get("sequence_id", ""),
                "rank_in_collection": rec.get("rank_in_collection", None),
                "yaw_center": rec.get("yaw_center", None),
                "detection_count": 0,
                "refined_detection_count": 0,
            }
        by_fid[fid]["detection_count"] += 1
        if int(rec.get("is_refined") or 0) == 1:
            by_fid[fid]["refined_detection_count"] += 1
    return list(by_fid.values())


def make_camera_points_geojson(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    features: List[Dict[str, Any]] = []
    for rec in make_camera_point_records(records):
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(rec["lon"]), float(rec["lat"])],
                },
                "properties": _clean_props(rec),
            }
        )
    return features


def make_detection_rays_geojson(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    features: List[Dict[str, Any]] = []
    for rec in records:
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [float(rec["camera_lon"]), float(rec["camera_lat"])],
                        [float(rec["end_lon"]), float(rec["end_lat"])],
                    ],
                },
                "properties": _clean_props(rec),
            }
        )
    return features


def make_detection_observation_points_geojson(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    features: List[Dict[str, Any]] = []
    for rec in records:
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(rec["lon"]), float(rec["lat"])],
                },
                "properties": _clean_props(rec),
            }
        )
    return features


def _index_by_fid(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        fid = str(row.get("fid") or row.get("id") or "")
        if fid:
            out[fid] = row
    return out


def _count_rows_with_best(rows: Iterable[Dict[str, Any]]) -> int:
    return sum(1 for row in rows if row.get("best"))


def _resolve_existing_path(path_str: Any, run_dir: Path) -> Optional[Path]:
    text = str(path_str or "").strip()
    if not text:
        return None

    path = Path(text).expanduser()
    candidates = [path]
    if not path.is_absolute():
        candidates.extend([run_dir / path, Path.cwd() / path])

    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate
    return None


def resolve_annotated_source(record: Dict[str, Any], run_dir: Path) -> Optional[Path]:
    direct = _resolve_existing_path(record.get("annotated_path"), run_dir)
    if direct is not None:
        return direct

    crop_path = str(record.get("crop_path") or "").strip()
    if not crop_path:
        return None
    crop_name = Path(crop_path).name
    if not crop_name:
        return None

    annotated_dir = run_dir / "annotated"
    exact = annotated_dir / crop_name
    if exact.exists():
        return exact
    if not annotated_dir.exists():
        return None

    crop_stem = Path(crop_name).stem
    crop_suffix = Path(crop_name).suffix.lower()
    for candidate in sorted(annotated_dir.iterdir()):
        if not candidate.is_file():
            continue
        if crop_suffix and candidate.suffix.lower() != crop_suffix:
            continue
        if candidate.stem == crop_stem or candidate.stem.startswith(f"{crop_stem}__v"):
            return candidate
    return None


def make_windows_image_path(image_path: Path, windows_base_dir: Optional[str] = None) -> str:
    if windows_base_dir:
        return str(PureWindowsPath(str(windows_base_dir)) / image_path.name)
    return str(image_path)


def prepare_annotated_attachments(
    records: Iterable[Dict[str, Any]],
    run_dir: Path,
    arcgis_annotated_dir: Optional[Path] = None,
    arcgis_windows_annotated_dir: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    windows_rows: List[Dict[str, Any]] = []

    if arcgis_annotated_dir is not None:
        arcgis_annotated_dir.mkdir(parents=True, exist_ok=True)

    for rec in records:
        source = resolve_annotated_source(rec, run_dir)
        if source is None:
            continue

        if arcgis_annotated_dir is not None:
            filename = f"{safe_id(rec.get('ray_id'))}_annotated.jpg"
            image_path = arcgis_annotated_dir / filename
            if source.resolve() != image_path.resolve():
                shutil.copy2(source, image_path)
        else:
            image_path = source

        rec["annotated_path"] = str(image_path)
        row = {"ray_id": rec.get("ray_id", ""), "image_type": "annotated", "image_path": str(image_path)}
        rows.append(row)
        windows_rows.append(
            {
                "ray_id": rec.get("ray_id", ""),
                "image_type": "annotated",
                "image_path": make_windows_image_path(image_path, arcgis_windows_annotated_dir),
            }
        )

    return rows, windows_rows


def export_detection_rays(
    detections_jsonl: Path,
    ordered_index: Path,
    yaw_map: Path,
    out_dir: Path,
    image_w: int,
    image_h: int,
    ray_length_m: float = 100.0,
    arcgis_annotated_dir: Optional[Path] = None,
    arcgis_windows_annotated_dir: Optional[str] = None,
) -> Dict[str, Any]:
    detections = read_jsonl(detections_jsonl)
    index_rows = _index_by_fid(read_jsonl(ordered_index))
    yaw_rows = _index_by_fid(read_jsonl(yaw_map)) if yaw_map.exists() else {}
    run_dir = out_dir.parent

    records: List[Dict[str, Any]] = []
    skipped_missing_index = 0
    skipped_unprojectable = 0

    for source_index, det_row in enumerate(detections, start=1):
        if not det_row.get("best"):
            continue
        fid = str(det_row.get("fid") or "")
        index_row = index_rows.get(fid)
        if index_row is None:
            skipped_missing_index += 1
            continue
        try:
            rec = make_detection_ray_record(
                detection=det_row,
                index_row=index_row,
                yaw_row=yaw_rows.get(fid),
                image_w=image_w,
                image_h=image_h,
                ray_length_m=ray_length_m,
            )
        except (KeyError, TypeError, ValueError):
            skipped_unprojectable += 1
            continue
        if rec is None:
            skipped_unprojectable += 1
            continue
        rec["ray_id"] = make_ray_id(rec.get("fid", fid), rec.get("view", ""), rec.get("step", ""), source_index)
        records.append(rec)

    attachment_rows, attachment_windows_rows = prepare_annotated_attachments(
        records=records,
        run_dir=run_dir,
        arcgis_annotated_dir=arcgis_annotated_dir,
        arcgis_windows_annotated_dir=arcgis_windows_annotated_dir,
    )
    observation_records = [make_observation_point_record(rec) for rec in records]
    camera_records = make_camera_point_records(records)
    camera_features = make_camera_points_geojson(records)
    ray_features = make_detection_rays_geojson(records)
    observation_features = make_detection_observation_points_geojson(observation_records)

    out_dir.mkdir(parents=True, exist_ok=True)
    camera_path = out_dir / "camera_points.geojson"
    rays_path = out_dir / "detection_rays.geojson"
    observation_path = out_dir / "detection_observation_points.geojson"
    camera_csv_path = out_dir / "camera_points.csv"
    rays_csv_path = out_dir / "detection_rays.csv"
    observation_csv_path = out_dir / "detection_observation_points.csv"
    attachments_csv_path = out_dir / "detection_annotated_attachments.csv"
    attachments_windows_csv_path = out_dir / "detection_annotated_attachments_windows.csv"

    write_geojson_featurecollection(camera_features, camera_path)
    write_geojson_featurecollection(ray_features, rays_path)
    write_geojson_featurecollection(observation_features, observation_path)
    save_csv(camera_csv_path, camera_records, CAMERA_POINT_CSV_FIELDS)
    save_csv(rays_csv_path, records, DETECTION_RAY_CSV_FIELDS)
    save_csv(observation_csv_path, observation_records, OBSERVATION_POINT_CSV_FIELDS)
    save_csv(attachments_csv_path, attachment_rows, ATTACHMENT_CSV_FIELDS)
    save_csv(attachments_windows_csv_path, attachment_windows_rows, ATTACHMENT_CSV_FIELDS)

    return {
        "camera_points_geojson": str(camera_path),
        "detection_rays_geojson": str(rays_path),
        "detection_observation_points_geojson": str(observation_path),
        "camera_points_csv": str(camera_csv_path),
        "detection_rays_csv": str(rays_csv_path),
        "detection_observation_points_csv": str(observation_csv_path),
        "detection_annotated_attachments_csv": str(attachments_csv_path),
        "detection_annotated_attachments_windows_csv": str(attachments_windows_csv_path),
        "camera_count": len(camera_features),
        "ray_count": len(ray_features),
        "observation_point_count": len(observation_records),
        "annotated_attachment_count": len(attachment_rows),
        "source_detection_count": _count_rows_with_best(detections),
        "skipped_missing_index": skipped_missing_index,
        "skipped_unprojectable": skipped_unprojectable,
        "ray_length_m": float(ray_length_m),
    }
