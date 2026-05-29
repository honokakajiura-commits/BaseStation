#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Panoramax fetch and download stages for the agent pipeline."""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any, Dict, List

import requests

from .io_utils import (
    append_stage_log,
    ensure_dir,
    load_images_map,
    load_ordered_index,
    load_points_features,
    safe_str,
    write_featurecollection,
    write_jsonl_records,
)


def fetch_points_stage(
    aoi_geojson: Path,
    points_jsonl: Path,
    points_geojson: Path,
    api_base: str,
    cell_deg: float,
    limit: int,
    timeout: int,
    sleep_sec: float,
    log_path: Path,
    overwrite: bool,
) -> Dict[str, Any]:
    from tools.archive.panoramax_fetch_points_in_aoi import (
        extract_features,
        load_aoi_union,
        make_grid_cells,
        normalize_feature_props,
        point_from_feature,
        post_search,
        top_next_link,
    )

    if points_jsonl.exists() and not overwrite:
        features = load_points_features(points_jsonl)
        if features:
            append_stage_log(
                log_path,
                step="fetch_points",
                status="skipped_existing",
                input_file=str(aoi_geojson),
                output_file=str(points_jsonl),
                params={"api_base": api_base, "cell_deg": cell_deg, "limit": limit},
                count=len(features),
            )
            if not points_geojson.exists():
                write_featurecollection(points_geojson, features)
            return {"count": len(features), "points_jsonl": str(points_jsonl), "points_geojson": str(points_geojson)}

    aoi_union = load_aoi_union(aoi_geojson)
    cells = make_grid_cells(aoi_union, cell_deg)
    session = requests.Session()
    session.headers.update({"User-Agent": "BaseStationComplete/fetch_points"})

    seen_ids = set()
    kept_features: List[dict] = []
    errors = 0
    pages_followed = 0

    for bbox in cells:
        payload = {"bbox": [bbox[0], bbox[1], bbox[2], bbox[3]], "limit": limit}
        try:
            resp = post_search(session, api_base, payload, timeout=timeout)
        except Exception as e:
            errors += 1
            append_stage_log(
                log_path,
                step="fetch_points_cell",
                status="fail",
                input_file=str(aoi_geojson),
                params={"bbox": payload["bbox"], "limit": limit},
                error=safe_str(e),
            )
            continue

        responses = [resp]
        nxt = top_next_link(resp)
        while nxt:
            try:
                r2 = session.get(nxt, timeout=timeout)
                r2.raise_for_status()
                resp2 = r2.json()
                responses.append(resp2)
                pages_followed += 1
                nxt = top_next_link(resp2)
                if sleep_sec > 0:
                    time.sleep(sleep_sec)
            except Exception as e:
                errors += 1
                append_stage_log(
                    log_path,
                    step="fetch_points_paging",
                    status="fail",
                    input_file=nxt,
                    params={"bbox": payload["bbox"]},
                    error=safe_str(e),
                )
                break

        for resp_obj in responses:
            for feature in extract_features(resp_obj):
                fid = feature.get("id")
                if not isinstance(fid, str) or not fid or fid in seen_ids:
                    continue
                pt = point_from_feature(feature)
                if pt is None or not aoi_union.covers(pt):
                    continue
                feature_copy = copy.deepcopy(feature)
                feature_copy["properties"] = normalize_feature_props(feature_copy, api_base)
                kept_features.append(feature_copy)
                seen_ids.add(fid)

        if sleep_sec > 0:
            time.sleep(sleep_sec)

    write_jsonl_records(points_jsonl, kept_features)
    write_featurecollection(points_geojson, kept_features)
    append_stage_log(
        log_path,
        step="fetch_points",
        status="ok",
        input_file=str(aoi_geojson),
        output_file=str(points_jsonl),
        params={"api_base": api_base, "cell_deg": cell_deg, "limit": limit, "timeout": timeout},
        count=len(kept_features),
        errors=errors,
        pages_followed=pages_followed,
    )
    return {"count": len(kept_features), "points_jsonl": str(points_jsonl), "points_geojson": str(points_geojson)}


def fetch_images_stage(
    points_jsonl: Path,
    images_jsonl: Path,
    api_base: str,
    image_base: str,
    timeout: int,
    sleep_sec: float,
    log_path: Path,
    overwrite: bool,
) -> Dict[str, Any]:
    from tools.archive.fetch_panos_ordered import get_feature_id, get_lonlat_from_feature
    from tools.archive.make_yolo_crops_from_panoramax import (
        extract_sequence_and_rank,
        fetch_picture_meta,
        find_item_url_from_feature,
        normalize_url,
        resolve_best_panoramax_image,
    )

    points = load_points_features(points_jsonl)
    existing = {} if overwrite else load_images_map(images_jsonl)
    rows = [] if overwrite else list(existing.values())
    session = requests.Session()
    session.headers.update({"User-Agent": "BaseStationComplete/fetch_images"})

    ok = 0
    fail = 0
    for feature in points:
        fid = get_feature_id(feature)
        if not fid or fid == "unknown":
            continue
        if fid in existing:
            ok += 1
            continue

        lon, lat = get_lonlat_from_feature(feature)
        item_url = normalize_url(find_item_url_from_feature(feature) or safe_str((feature.get("properties") or {}).get("item_url")))
        legacy_url = f"{image_base.rstrip('/')}/{fid}.jpg"

        rec: Dict[str, Any] = {
            "fid": fid,
            "lon": lon,
            "lat": lat,
            "item_url": item_url,
            "legacy_url": legacy_url,
            "status": "ok",
            "error": "",
            "sequence_id": "",
            "rank_in_collection": None,
            "img_url": "",
            "img_source": "",
            "selected_asset": None,
            "view_azimuth": (feature.get("properties") or {}).get("view:azimuth")
            or (feature.get("properties") or {}).get("view_azimuth")
            or (feature.get("properties") or {}).get("azimuth")
            or "",
        }
        try:
            meta = fetch_picture_meta(session, api_base=api_base, fid=fid, timeout=min(timeout, 45))
            seq, rank = extract_sequence_and_rank(meta)
            rec["sequence_id"] = seq
            rec["rank_in_collection"] = rank
            rec["picture_meta"] = {
                "collection": safe_str(meta.get("collection")),
                "datetime": safe_str((meta.get("properties") or {}).get("datetime") or (meta.get("properties") or {}).get("datetimetz")),
            }
            try:
                img_url, img_source, selected_asset = resolve_best_panoramax_image(session, feature, timeout=timeout)
                rec["img_url"] = img_url
                rec["img_source"] = img_source
                rec["selected_asset"] = selected_asset
            except Exception:
                rec["img_url"] = legacy_url
                rec["img_source"] = "legacy_image_base"
                rec["selected_asset"] = None
        except Exception as e:
            rec["status"] = "fail"
            rec["error"] = safe_str(e)
            fail += 1
            append_stage_log(
                log_path,
                step="fetch_images_item",
                status="fail",
                input_file=str(points_jsonl),
                output_file=str(images_jsonl),
                params={"fid": fid, "api_base": api_base},
                error=safe_str(e),
            )
            rows.append(rec)
            continue

        rows.append(rec)
        ok += 1
        if sleep_sec > 0:
            time.sleep(sleep_sec)

    write_jsonl_records(images_jsonl, rows)
    append_stage_log(
        log_path,
        step="fetch_images",
        status="ok" if fail == 0 else "partial",
        input_file=str(points_jsonl),
        output_file=str(images_jsonl),
        params={"api_base": api_base, "timeout": timeout},
        ok=ok,
        fail=fail,
    )
    return {"count": len(rows), "ok": ok, "fail": fail, "images_jsonl": str(images_jsonl)}


def download_panos_stage(
    ordered_index: Path,
    panos_dir: Path,
    api_base: str,
    image_base: str,
    log_path: Path,
) -> Dict[str, Any]:
    from tools.archive.agent_detect_only_agent2 import download_pano

    ensure_dir(panos_dir)
    rows = load_ordered_index(ordered_index)
    session = requests.Session()
    session.headers.update({"User-Agent": "BaseStationComplete/download_panos"})
    ok = 0
    fail = 0
    for row in rows:
        fid = row["fid"]
        dl_ok, dl_path, dl_meta = download_pano(
            fid=fid,
            panos_dir=panos_dir,
            api_base=api_base,
            image_base=image_base,
            session=session,
        )
        append_stage_log(
            log_path,
            step="download_pano",
            status="ok" if dl_ok else "fail",
            input_file=str(ordered_index),
            output_file=str(dl_path) if dl_path else "",
            params={"fid": fid, "api_base": api_base},
            error=safe_str(dl_meta.get("error")),
            fid=fid,
            meta=dl_meta,
        )
        if dl_ok:
            ok += 1
        else:
            fail += 1

    append_stage_log(
        log_path,
        step="download_panos",
        status="ok" if fail == 0 else "partial",
        input_file=str(ordered_index),
        output_file=str(panos_dir),
        params={"api_base": api_base, "image_base": image_base},
        ok=ok,
        fail=fail,
    )
    return {"ok": ok, "fail": fail, "panos_dir": str(panos_dir)}
