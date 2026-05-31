#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""GeoJSON export helpers for ArcGIS inspection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .geolocation import make_detection_ray_record
from .io_utils import read_jsonl


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
    geometry_keys = {"camera_lon", "camera_lat", "end_lon", "end_lat"}
    return {k: v for k, v in record.items() if k not in geometry_keys}


def make_camera_points_geojson(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_fid: Dict[str, Dict[str, Any]] = {}
    for rec in records:
        fid = str(rec.get("fid", ""))
        if not fid:
            continue
        if fid not in by_fid:
            by_fid[fid] = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(rec["camera_lon"]), float(rec["camera_lat"])],
                },
                "properties": {
                    "fid": fid,
                    "sequence_id": rec.get("sequence_id", ""),
                    "rank_in_collection": rec.get("rank_in_collection", None),
                    "yaw_center": rec.get("yaw_center", None),
                    "detection_count": 0,
                },
            }
        by_fid[fid]["properties"]["detection_count"] += 1
    return list(by_fid.values())


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


def _index_by_fid(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        fid = str(row.get("fid") or row.get("id") or "")
        if fid:
            out[fid] = row
    return out


def _count_rows_with_best(rows: Iterable[Dict[str, Any]]) -> int:
    return sum(1 for row in rows if row.get("best"))


def export_detection_rays(
    detections_jsonl: Path,
    ordered_index: Path,
    yaw_map: Path,
    out_dir: Path,
    image_w: int,
    image_h: int,
    ray_length_m: float = 100.0,
) -> Dict[str, Any]:
    detections = read_jsonl(detections_jsonl)
    index_rows = _index_by_fid(read_jsonl(ordered_index))
    yaw_rows = _index_by_fid(read_jsonl(yaw_map)) if yaw_map.exists() else {}

    records: List[Dict[str, Any]] = []
    skipped_missing_index = 0
    skipped_unprojectable = 0

    for det_row in detections:
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
        records.append(rec)

    camera_features = make_camera_points_geojson(records)
    ray_features = make_detection_rays_geojson(records)

    out_dir.mkdir(parents=True, exist_ok=True)
    camera_path = out_dir / "camera_points.geojson"
    rays_path = out_dir / "detection_rays.geojson"
    write_geojson_featurecollection(camera_features, camera_path)
    write_geojson_featurecollection(ray_features, rays_path)

    return {
        "camera_points_geojson": str(camera_path),
        "detection_rays_geojson": str(rays_path),
        "camera_count": len(camera_features),
        "ray_count": len(ray_features),
        "source_detection_count": _count_rows_with_best(detections),
        "skipped_missing_index": skipped_missing_index,
        "skipped_unprojectable": skipped_unprojectable,
        "ray_length_m": float(ray_length_m),
    }
