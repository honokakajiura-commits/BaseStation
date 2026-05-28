#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import copy
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import requests

from agent_detect_only_agent2 import (
    AgentConfig,
    YoloRunner,
    _build_crop_name,
    append_jsonl,
    best_det,
    clamp_pitch_deg,
    det_center_frac,
    draw_annot,
    draw_refine_compare,
    draw_status,
    download_pano,
    ensure_dir,
    estimate_yaw_center_auto,
    fit_next_fov_to_bbox,
    need_center_before_zoom,
    need_center_by_edge,
    px_to_angle_deg,
    py_to_angle_deg,
    read_jsonl,
    render_detection_crop,
    safe_str,
    save_json,
    unique_path,
    wrap_yaw_deg,
    yaw_delta_to_keep_bbox_in_next_fov,
)
from fetch_panos_ordered import (
    get_datetime_from_feature,
    get_feature_id,
    get_lonlat_from_feature,
    order_features_datetime,
    order_features_nearest,
    order_sequences_by_nearest,
    write_aoi_index_jsonl,
)
from make_yolo_crops_from_panoramax import (
    extract_sequence_and_rank,
    fetch_picture_meta,
    find_item_url_from_feature,
    find_pano_path,
    normalize_url,
    resolve_best_panoramax_image,
)
from panoramax_fetch_points_in_aoi import (
    extract_features,
    load_aoi_union,
    make_grid_cells,
    normalize_feature_props,
    point_from_feature,
    post_search,
    top_next_link,
)


def load_jsonl_records(path: Path) -> List[dict]:
    if not path.exists():
        return []
    return read_jsonl(path)


def write_jsonl_records(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_featurecollection(path: Path, features: List[dict]) -> None:
    path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def stage_enabled(run_flag: bool, skip_flag: bool, default: bool = True) -> bool:
    if run_flag and skip_flag:
        raise ValueError("conflicting stage flags")
    if run_flag:
        return True
    if skip_flag:
        return False
    return default


def append_stage_log(
    log_path: Path,
    step: str,
    status: str,
    input_file: str = "",
    output_file: str = "",
    params: Optional[dict] = None,
    error: str = "",
    **extra: Any,
) -> None:
    rec = {
        "step": step,
        "input_file": input_file,
        "output_file": output_file,
        "status": status,
        "params": params or {},
        "error": error,
    }
    rec.update(extra)
    append_jsonl(log_path, rec)


def load_points_features(points_jsonl: Path) -> List[dict]:
    return load_jsonl_records(points_jsonl)


def load_images_map(images_jsonl: Path) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for row in load_jsonl_records(images_jsonl):
        fid = safe_str(row.get("fid"))
        if fid:
            out[fid] = row
    return out


def load_ordered_index(path: Path) -> List[dict]:
    return load_jsonl_records(path)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_csv(path: Path, rows: List[dict], preferred: List[str]) -> None:
    ensure_parent(path)
    keys = set()
    for row in rows:
        keys.update(row.keys())
    fieldnames = [k for k in preferred if k in keys] + [k for k in sorted(keys) if k not in preferred]
    if not fieldnames:
        fieldnames = preferred
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


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


def order_points_with_sequence(
    features: List[dict],
    images_map: Dict[str, dict],
    session: requests.Session,
    api_base: str,
    timeout: int,
    sleep_sec: float,
) -> List[dict]:
    enriched: List[dict] = []
    for feature in features:
        feature_copy = copy.deepcopy(feature)
        props = feature_copy.get("properties") or {}
        fid = get_feature_id(feature_copy)
        img_row = images_map.get(fid, {})
        seq = safe_str(img_row.get("sequence_id"))
        rank = img_row.get("rank_in_collection")

        if not seq and fid and fid != "unknown":
            try:
                item = fetch_picture_meta(session, api_base=api_base, fid=fid, timeout=min(timeout, 45))
                seq, rank = extract_sequence_and_rank(item)
            except Exception:
                seq, rank = "", None
            if sleep_sec > 0:
                time.sleep(sleep_sec)

        props["sequence_id"] = seq
        props["rank_in_collection"] = rank
        feature_copy["properties"] = props
        enriched.append(feature_copy)

    groups: Dict[str, List[dict]] = {}
    for feature in enriched:
        seq = safe_str((feature.get("properties") or {}).get("sequence_id"))
        groups.setdefault(seq, []).append(feature)

    for _, group in groups.items():
        group.sort(
            key=lambda f: (
                float((f.get("properties") or {}).get("rank_in_collection"))
                if (f.get("properties") or {}).get("rank_in_collection") is not None
                else float("inf"),
                get_datetime_from_feature(f),
                get_feature_id(f),
            )
        )

    non_empty = [seq for seq in groups.keys() if seq]
    empty = [""] if "" in groups else []
    seq_order = order_sequences_by_nearest({seq: groups[seq] for seq in non_empty}) + empty

    ordered: List[dict] = []
    for seq in seq_order:
        ordered.extend(groups[seq])
    return ordered


def order_panos_stage(
    points_jsonl: Path,
    images_jsonl: Path,
    ordered_index: Path,
    order_mode: str,
    api_base: str,
    timeout: int,
    sleep_sec: float,
    log_path: Path,
    overwrite: bool,
) -> Dict[str, Any]:
    if ordered_index.exists() and not overwrite:
        rows = load_ordered_index(ordered_index)
        if rows:
            append_stage_log(
                log_path,
                step="order_panos",
                status="skipped_existing",
                input_file=str(points_jsonl),
                output_file=str(ordered_index),
                params={"order_mode": order_mode},
                count=len(rows),
            )
            return {"count": len(rows), "ordered_index": str(ordered_index)}

    features = load_points_features(points_jsonl)
    images_map = load_images_map(images_jsonl)
    session = requests.Session()
    session.headers.update({"User-Agent": "BaseStationComplete/order_panos"})

    if order_mode == "sequence":
        ordered = order_points_with_sequence(features, images_map, session, api_base, timeout, sleep_sec)
    elif order_mode == "datetime":
        ordered = order_features_datetime(features)
    else:
        ordered = order_features_nearest(features)

    write_aoi_index_jsonl(ordered_index, ordered)
    append_stage_log(
        log_path,
        step="order_panos",
        status="ok",
        input_file=str(points_jsonl),
        output_file=str(ordered_index),
        params={"order_mode": order_mode, "api_base": api_base},
        count=len(ordered),
    )
    return {"count": len(ordered), "ordered_index": str(ordered_index)}


def download_panos_stage(
    ordered_index: Path,
    panos_dir: Path,
    api_base: str,
    image_base: str,
    log_path: Path,
) -> Dict[str, Any]:
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


def ensure_yaw_map(
    ordered_rows: List[dict],
    panos_dir: Path,
    yaw_map_path: Path,
    pitch_deg: float,
    yaw_preview_fov: float,
    yaw_preview_w: int,
    yaw_preview_h: int,
    overwrite: bool,
    log_path: Path,
) -> Dict[str, float]:
    existing_map = {}
    if yaw_map_path.exists() and not overwrite:
        existing_map = {row["fid"]: float(row.get("yaw_center", 0.0)) for row in read_jsonl(yaw_map_path) if "fid" in row}
    if overwrite and yaw_map_path.exists():
        yaw_map_path.unlink()

    with yaw_map_path.open("a", encoding="utf-8") as f:
        for row in ordered_rows:
            fid = row["fid"]
            if fid in existing_map and not overwrite:
                continue
            pano_path = find_pano_path(panos_dir, fid)
            if pano_path is None:
                rec = {"fid": fid, "yaw_center": 0.0, "yaw_reason": "missing_pano"}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue
            pano = cv2.imread(str(pano_path))
            if pano is None:
                rec = {"fid": fid, "yaw_center": 0.0, "yaw_reason": "imread_failed"}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue
            yaw_center, reason, meta = estimate_yaw_center_auto(
                pano,
                pitch_deg=pitch_deg,
                view_azimuth=row.get("view_azimuth"),
                fov_preview=yaw_preview_fov,
                out_w=yaw_preview_w,
                out_h=yaw_preview_h,
            )
            rec = {"fid": fid, "yaw_center": float(yaw_center), "yaw_reason": reason, **meta}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            append_stage_log(
                log_path,
                step="yaw_map",
                status="ok",
                input_file=str(pano_path),
                output_file=str(yaw_map_path),
                params={"fid": fid},
                fid=fid,
                yaw_center=float(yaw_center),
                yaw_reason=reason,
            )
    return {row["fid"]: float(row.get("yaw_center", 0.0)) for row in read_jsonl(yaw_map_path) if "fid" in row}


def make_crops_stage(
    ordered_index: Path,
    panos_dir: Path,
    crops_dir: Path,
    yaw_map_path: Path,
    pitch_cli: float,
    hfov: float,
    crop_width: int,
    crop_height: int,
    crop_strategy: str,
    crop_supersample: float,
    crop_interpolation: str,
    overwrite: bool,
    log_path: Path,
) -> Dict[str, Any]:
    ensure_dir(crops_dir)
    rows = load_ordered_index(ordered_index)
    pitch_deg = -float(pitch_cli)
    yaw_map = ensure_yaw_map(
        ordered_rows=rows,
        panos_dir=panos_dir,
        yaw_map_path=yaw_map_path,
        pitch_deg=pitch_deg,
        yaw_preview_fov=110.0,
        yaw_preview_w=1024,
        yaw_preview_h=768,
        overwrite=overwrite,
        log_path=log_path,
    )
    made = 0
    skipped = 0
    crop_manifest_path = crops_dir.parent / "crops_manifest.jsonl"

    for i, row in enumerate(rows, start=1):
        fid = row["fid"]
        pano_path = find_pano_path(panos_dir, fid)
        if pano_path is None:
            continue
        pano = cv2.imread(str(pano_path))
        if pano is None:
            continue
        yaw_center = float(yaw_map.get(fid, 0.0))
        for view_name, yaw_off in [("front", 0.0), ("left", -90.0), ("right", 90.0)]:
            yaw = wrap_yaw_deg(yaw_center + yaw_off)
            crop_name = _build_crop_name(
                idx=i,
                fid=fid,
                view=view_name,
                step=0,
                yaw=yaw,
                fov=hfov,
                last_yaw_delta=0.0,
                last_zoom=False,
            )
            crop_path = crops_dir / crop_name
            if crop_path.exists() and not overwrite:
                skipped += 1
                continue
            crop, crop_meta = render_detection_crop(
                pano_bgr=pano,
                yaw_deg=yaw,
                pitch_deg=pitch_deg,
                fov_deg=hfov,
                out_w=crop_width,
                out_h=crop_height,
                crop_strategy=crop_strategy,
                supersample=crop_supersample,
                interpolation=crop_interpolation,
            )
            cv2.imwrite(str(crop_path), crop)
            append_jsonl(
                crop_manifest_path,
                {
                    "fid": fid,
                    "i": i,
                    "view": view_name,
                    "yaw": float(yaw),
                    "pitch_deg": float(pitch_deg),
                    "fov": float(hfov),
                    "crop_path": str(crop_path),
                    "crop_meta": crop_meta,
                },
            )
            append_stage_log(
                log_path,
                step="make_crop",
                status="ok",
                input_file=str(pano_path),
                output_file=str(crop_path),
                params={"fid": fid, "view": view_name, "hfov": hfov, "pitch_cli": pitch_cli},
                crop_meta=crop_meta,
            )
            made += 1

    append_stage_log(
        log_path,
        step="make_crops",
        status="ok",
        input_file=str(ordered_index),
        output_file=str(crops_dir),
        params={
            "crop_strategy": crop_strategy,
            "crop_supersample": crop_supersample,
            "crop_interpolation": crop_interpolation,
            "pitch_cli": pitch_cli,
            "hfov": hfov,
            "crop_width": crop_width,
            "crop_height": crop_height,
        },
        made=made,
        skipped=skipped,
    )
    return {"made": made, "skipped": skipped, "crops_dir": str(crops_dir)}


def detect_existing_crops_stage(
    crops_dir: Path,
    annotated_dir: Path,
    detections_jsonl: Path,
    weights: str,
    conf: float,
    imgsz: int,
    device: str,
    overwrite: bool,
    log_path: Path,
) -> Dict[str, Any]:
    ensure_dir(annotated_dir)
    yolo = YoloRunner(weights, conf=conf, imgsz=imgsz, device=device)
    crop_paths = sorted(
        [p for p in crops_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
    ) if crops_dir.exists() else []

    ok = 0
    fail = 0
    for crop_path in crop_paths:
        img = cv2.imread(str(crop_path))
        if img is None:
            fail += 1
            append_stage_log(
                log_path,
                step="detect_crop",
                status="fail",
                input_file=str(crop_path),
                error="imread_failed",
            )
            continue
        dets = yolo.infer(img)
        best = best_det(dets)
        ann = draw_annot(img, dets, topk=3) if dets else draw_status(img, ["NO DETECTION"])
        ann_path = unique_path(annotated_dir / crop_path.name, overwrite=overwrite)
        cv2.imwrite(str(ann_path), ann)
        append_jsonl(
            detections_jsonl,
            {
                "mode": "existing_crops",
                "crop_path": str(crop_path),
                "annotated_path": str(ann_path),
                "n": len(dets),
                "best": best,
            },
        )
        ok += 1

    append_stage_log(
        log_path,
        step="detect_existing_crops",
        status="ok" if fail == 0 else "partial",
        input_file=str(crops_dir),
        output_file=str(detections_jsonl),
        params={"weights": weights, "conf": conf, "imgsz": imgsz, "device": device},
        ok=ok,
        fail=fail,
    )
    return {"processed": ok, "fail": fail}


def detect_from_panos_stage(
    ordered_index: Path,
    panos_dir: Path,
    crops_dir: Path,
    annotated_dir: Path,
    compare_dir: Path,
    yaw_map_path: Path,
    detections_jsonl: Path,
    summary_path: Path,
    api_base: str,
    image_base: str,
    weights: str,
    conf: float,
    imgsz: int,
    device: str,
    pitch_cli: float,
    hfov: float,
    crop_width: int,
    crop_height: int,
    crop_strategy: str,
    crop_supersample: float,
    crop_interpolation: str,
    overwrite: bool,
    log_path: Path,
) -> Dict[str, Any]:
    ensure_dir(crops_dir)
    ensure_dir(annotated_dir)
    ensure_dir(compare_dir)

    rows = load_ordered_index(ordered_index)
    pitch_deg = -float(pitch_cli)
    yaw_map = ensure_yaw_map(
        ordered_rows=rows,
        panos_dir=panos_dir,
        yaw_map_path=yaw_map_path,
        pitch_deg=pitch_deg,
        yaw_preview_fov=110.0,
        yaw_preview_w=1024,
        yaw_preview_h=768,
        overwrite=overwrite,
        log_path=log_path,
    )

    cfg = AgentConfig(
        det_w=crop_width,
        det_h=crop_height,
        fov_front=hfov,
        fov_side=hfov,
        crop_strategy=crop_strategy,
        crop_supersample=crop_supersample,
        crop_interpolation=crop_interpolation,
    )
    yolo = YoloRunner(weights, conf=conf, imgsz=imgsz, device=device)
    total_panos = 0
    total_crops = 0
    confirmed = 0
    candidates = 0

    for i, row in enumerate(rows, start=1):
        fid = row["fid"]
        pano_path = find_pano_path(panos_dir, fid)
        if pano_path is None:
            append_stage_log(log_path, "detect_pano", "fail", input_file=str(panos_dir), error="missing_pano", fid=fid)
            continue
        pano = cv2.imread(str(pano_path))
        if pano is None:
            append_stage_log(log_path, "detect_pano", "fail", input_file=str(pano_path), error="imread_failed", fid=fid)
            continue

        total_panos += 1
        yaw_center = float(yaw_map.get(fid, 0.0))
        pano_confirmed = 0
        pano_candidate = 0

        for view_name, yaw_off in [("front", 0.0), ("left", -90.0), ("right", 90.0)]:
            cur_yaw = wrap_yaw_deg(yaw_center + yaw_off)
            cur_fov = float(hfov)
            cur_pitch = float(pitch_deg)
            last_yaw_delta = 0.0
            last_zoom = False
            prev_crop = None
            prev_dets: List[dict] = []
            prev_state = None

            for step in range(cfg.max_refine + 1):
                crop_name = _build_crop_name(
                    idx=i,
                    fid=fid,
                    view=view_name,
                    step=step,
                    yaw=cur_yaw,
                    fov=cur_fov,
                    last_yaw_delta=last_yaw_delta,
                    last_zoom=last_zoom,
                )
                crop_path = crops_dir / crop_name
                crop_meta: Dict[str, Any]
                if crop_path.exists() and not overwrite:
                    crop = cv2.imread(str(crop_path))
                    crop_meta = {"source": "existing_crop", "strategy": crop_strategy}
                    if crop is None:
                        crop, crop_meta = render_detection_crop(
                            pano_bgr=pano,
                            yaw_deg=cur_yaw,
                            pitch_deg=cur_pitch,
                            fov_deg=cur_fov,
                            out_w=cfg.det_w,
                            out_h=cfg.det_h,
                            crop_strategy=cfg.crop_strategy,
                            supersample=cfg.crop_supersample,
                            interpolation=cfg.crop_interpolation,
                        )
                        cv2.imwrite(str(crop_path), crop)
                else:
                    crop, crop_meta = render_detection_crop(
                        pano_bgr=pano,
                        yaw_deg=cur_yaw,
                        pitch_deg=cur_pitch,
                        fov_deg=cur_fov,
                        out_w=cfg.det_w,
                        out_h=cfg.det_h,
                        crop_strategy=cfg.crop_strategy,
                        supersample=cfg.crop_supersample,
                        interpolation=cfg.crop_interpolation,
                    )
                    cv2.imwrite(str(crop_path), crop)
                total_crops += 1

                dets = yolo.infer(crop)
                bd = best_det(dets)
                append_jsonl(
                    detections_jsonl,
                    {
                        "mode": "pano_refine",
                        "i": i,
                        "fid": fid,
                        "view": view_name,
                        "s": step,
                        "yaw_center": yaw_center,
                        "yaw": float(cur_yaw),
                        "yaw_off": float(yaw_off),
                        "pitch_cli": float(pitch_cli),
                        "pitch_deg": float(cur_pitch),
                        "fov": float(cur_fov),
                        "crop_meta": crop_meta,
                        "crop_path": str(crop_path),
                        "n": len(dets),
                        "best": bd,
                        "sequence_id": row.get("sequence_id", ""),
                        "rank_in_collection": row.get("rank_in_collection", None),
                    },
                )

                if not bd:
                    if step == 0:
                        break
                    ann0 = draw_status(
                        crop,
                        [
                            "NO DETECTION after refine",
                            f"view={view_name} step={step}",
                            f"yaw={cur_yaw:.1f} fov={cur_fov:.1f} pitch={cur_pitch:.1f}",
                        ],
                    )
                    ann_path0 = unique_path(annotated_dir / crop_path.name, overwrite=overwrite)
                    cv2.imwrite(str(ann_path0), ann0)
                    append_stage_log(
                        log_path,
                        step="refine_lost",
                        status="ok",
                        input_file=str(crop_path),
                        output_file=str(ann_path0),
                        params={"fid": fid, "view": view_name, "s": step},
                    )
                    break

                best_conf = float(bd["conf"])
                ann = draw_annot(crop, dets, topk=3)
                ann_path = unique_path(annotated_dir / crop_path.name, overwrite=overwrite)
                cv2.imwrite(str(ann_path), ann)

                cx_frac, cy_frac, area_frac = det_center_frac(bd, cfg.det_w, cfg.det_h)
                bbox_cx = (bd["xyxy"][0] + bd["xyxy"][2]) / 2.0
                bbox_cy = (bd["xyxy"][1] + bd["xyxy"][3]) / 2.0

                if prev_crop is not None and prev_state is not None:
                    compare_img = draw_refine_compare(
                        before_img=prev_crop,
                        before_dets=prev_dets,
                        after_img=crop,
                        after_dets=dets,
                        before_lines=[
                            f"before s={prev_state['step']} yaw={prev_state['yaw']:.1f} pitch={prev_state['pitch']:.1f}",
                            f"fov={prev_state['fov']:.1f} center=({prev_state['center_frac'][0]:.2f},{prev_state['center_frac'][1]:.2f})",
                        ],
                        after_lines=[
                            f"after s={step} yaw={cur_yaw:.1f} pitch={cur_pitch:.1f}",
                            f"fov={cur_fov:.1f} conf={best_conf:.2f}",
                        ],
                    )
                    compare_path = unique_path(compare_dir / f"{crop_path.stem}__compare.jpg", overwrite=overwrite)
                    cv2.imwrite(str(compare_path), compare_img)

                if best_conf >= cfg.high_conf:
                    confirmed += 1
                    pano_confirmed += 1
                    break
                if best_conf < cfg.low_conf:
                    break

                candidates += 1
                pano_candidate += 1
                yaw_delta_center = px_to_angle_deg(bbox_cx, cfg.det_w, cur_fov)
                pitch_delta_center = py_to_angle_deg(bbox_cy, cfg.det_h, cur_fov) if cfg.recenter_pitch else 0.0

                prev_fov = float(cur_fov)
                center_by_edge = need_center_by_edge(cx_frac, cfg.edge_center_margin)
                next_yaw = wrap_yaw_deg(cur_yaw + yaw_delta_center)
                next_pitch = clamp_pitch_deg(cur_pitch + pitch_delta_center)
                next_fov = float(cur_fov)
                zoom = False
                zoom_ratio = 1.0
                refine_action = "recenter_only"

                if center_by_edge:
                    refine_action = "recenter_only_edge"
                elif area_frac < cfg.large_area_frac:
                    zoom_ratio = cfg.refine_zoom_ratio_small if area_frac < cfg.small_area_frac else cfg.refine_zoom_ratio_medium
                    next_fov = max(cfg.zoom_min_fov, cur_fov * zoom_ratio)
                    zoom = next_fov != cur_fov
                    if zoom:
                        next_fov, zoom, _ = fit_next_fov_to_bbox(
                            cur_fov=cur_fov,
                            next_fov=next_fov,
                            det=bd,
                            w=cfg.det_w,
                            margin_deg=cfg.bbox_margin_deg,
                        )
                    if zoom:
                        refine_action = "recenter_and_zoom"

                append_stage_log(
                    log_path,
                    step="refine_plan",
                    status="ok",
                    input_file=str(crop_path),
                    params={"fid": fid, "view": view_name, "s": step},
                    refine_action=refine_action,
                    center_by_edge=bool(center_by_edge),
                    yaw_delta_center=float(yaw_delta_center),
                    pitch_delta_center=float(pitch_delta_center),
                    prev_fov=float(prev_fov),
                    next_fov=float(next_fov),
                    area_frac=float(area_frac),
                )

                if abs(yaw_delta_center) < 0.5 and abs(pitch_delta_center) < 0.5 and not zoom:
                    break

                prev_crop = crop.copy()
                prev_dets = list(dets)
                prev_state = {
                    "step": step,
                    "yaw": float(cur_yaw),
                    "pitch": float(cur_pitch),
                    "fov": float(cur_fov),
                    "center_frac": [float(cx_frac), float(cy_frac)],
                    "center_xy": [float(bbox_cx), float(bbox_cy)],
                }

                cur_yaw = next_yaw
                cur_pitch = next_pitch
                cur_fov = float(next_fov)
                last_yaw_delta = float(yaw_delta_center)
                last_zoom = bool(zoom)

        append_stage_log(
            log_path,
            step="detect_pano_done",
            status="ok",
            input_file=str(pano_path),
            params={"fid": fid},
            confirmed=pano_confirmed,
            candidate=pano_candidate,
        )

    summary = {
        "ordered_index": str(ordered_index),
        "panos_dir": str(panos_dir),
        "processed_panos": total_panos,
        "total_crops": total_crops,
        "confirmed": confirmed,
        "candidates": candidates,
        "params": {
            "pitch_cli": float(pitch_cli),
            "pitch_deg": float(pitch_deg),
            "crop_width": crop_width,
            "crop_height": crop_height,
            "hfov": float(hfov),
            "crop_strategy": crop_strategy,
            "crop_supersample": crop_supersample,
            "crop_interpolation": crop_interpolation,
            "weights": weights,
            "conf": float(conf),
            "imgsz": int(imgsz),
            "device": device,
            "api_base": api_base,
            "image_base": image_base,
        },
        "paths": {
            "crops": str(crops_dir),
            "annotated": str(annotated_dir),
            "refine_compare": str(compare_dir),
            "detections": str(detections_jsonl),
            "yaw_map": str(yaw_map_path),
            "log": str(log_path),
        },
    }
    save_json(summary_path, summary)
    append_stage_log(
        log_path,
        step="detect_from_panos",
        status="ok",
        input_file=str(ordered_index),
        output_file=str(detections_jsonl),
        params={"weights": weights, "conf": conf, "imgsz": imgsz, "device": device},
        processed_panos=total_panos,
        confirmed=confirmed,
        candidates=candidates,
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()

    ap.add_argument("--aoi_geojson", default="")
    ap.add_argument("--points_jsonl", default="")
    ap.add_argument("--images_jsonl", default="")
    ap.add_argument("--panos_dir", default="")
    ap.add_argument("--ordered_index", default="")
    ap.add_argument("--crops_dir", default="")
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--weights", default="")

    ap.add_argument("--fetch_points", action="store_true")
    ap.add_argument("--skip_fetch_points", action="store_true")
    ap.add_argument("--fetch_images", action="store_true")
    ap.add_argument("--skip_fetch_images", action="store_true")
    ap.add_argument("--download_panos", action="store_true")
    ap.add_argument("--skip_download_panos", action="store_true")
    ap.add_argument("--order_panos", action="store_true")
    ap.add_argument("--skip_order_panos", action="store_true")
    ap.add_argument("--make_crops", action="store_true")
    ap.add_argument("--skip_make_crops", action="store_true")
    ap.add_argument("--detect", action="store_true")
    ap.add_argument("--skip_detect", action="store_true")

    ap.add_argument("--crop_strategy", choices=["legacy", "ui_like"], default="ui_like")
    ap.add_argument("--crop_supersample", type=float, default=1.25)
    ap.add_argument("--crop_interpolation", choices=["linear", "cubic", "lanczos", "nearest"], default="cubic")
    ap.add_argument("--pitch_cli", type=float, default=40.0)
    ap.add_argument("--hfov", type=float, default=105.0)
    ap.add_argument("--crop_width", type=int, default=1280)
    ap.add_argument("--crop_height", type=int, default=1280)

    ap.add_argument("--conf", type=float, default=0.20)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default="")

    ap.add_argument("--api_base", default="https://api.panoramax.xyz")
    ap.add_argument("--image_base", default="https://panoramax.openstreetmap.fr/images")
    ap.add_argument("--order_mode", choices=["sequence", "datetime", "nearest"], default="sequence")
    ap.add_argument("--cell_deg", type=float, default=0.005)
    ap.add_argument("--search_limit", type=int, default=1000)
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--sleep", type=float, default=0.02)
    ap.add_argument("--overwrite", action="store_true")
    return ap


def main() -> None:
    args = build_parser().parse_args()

    run_dir = Path(args.run_dir)
    ensure_dir(run_dir)

    points_jsonl = Path(args.points_jsonl) if args.points_jsonl else (run_dir / "points.jsonl")
    images_jsonl = Path(args.images_jsonl) if args.images_jsonl else (run_dir / "images.jsonl")
    panos_dir = Path(args.panos_dir) if args.panos_dir else (run_dir / "panos")
    ordered_index = Path(args.ordered_index) if args.ordered_index else (run_dir / "aoi_index.jsonl")
    crops_dir = Path(args.crops_dir) if args.crops_dir else (run_dir / "crops")

    points_geojson = run_dir / "panoramax_points_in_aoi.geojson"
    annotated_dir = run_dir / "annotated"
    compare_dir = run_dir / "refine_compare"
    yaw_map_path = run_dir / "yaw_map.jsonl"
    detections_jsonl = run_dir / "detections.jsonl"
    detections_csv = run_dir / "detections.csv"
    summary_path = run_dir / "summary.json"
    log_path = run_dir / "agent_log.jsonl"

    if args.overwrite:
        for path in [images_jsonl, ordered_index, detections_jsonl, detections_csv, summary_path]:
            if path.exists():
                path.unlink()
        for path in [log_path, yaw_map_path]:
            if path.exists():
                path.unlink()

    do_fetch_points = stage_enabled(args.fetch_points, args.skip_fetch_points, default=True)
    do_fetch_images = stage_enabled(args.fetch_images, args.skip_fetch_images, default=True)
    do_download_panos = stage_enabled(args.download_panos, args.skip_download_panos, default=True)
    do_order_panos = stage_enabled(args.order_panos, args.skip_order_panos, default=True)
    do_make_crops = stage_enabled(args.make_crops, args.skip_make_crops, default=True)
    do_detect = stage_enabled(args.detect, args.skip_detect, default=True)

    if do_fetch_points and not args.aoi_geojson:
        raise ValueError("--aoi_geojson is required when fetch_points is enabled")
    if do_detect and not args.weights:
        raise ValueError("--weights is required when detect is enabled")

    summary: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "stages": {},
        "paths": {
            "points_jsonl": str(points_jsonl),
            "images_jsonl": str(images_jsonl),
            "panos_dir": str(panos_dir),
            "ordered_index": str(ordered_index),
            "crops_dir": str(crops_dir),
            "annotated_dir": str(annotated_dir),
            "compare_dir": str(compare_dir),
            "yaw_map": str(yaw_map_path),
            "detections_jsonl": str(detections_jsonl),
            "agent_log": str(log_path),
        },
    }

    if do_fetch_points:
        summary["stages"]["fetch_points"] = fetch_points_stage(
            aoi_geojson=Path(args.aoi_geojson),
            points_jsonl=points_jsonl,
            points_geojson=points_geojson,
            api_base=args.api_base,
            cell_deg=args.cell_deg,
            limit=args.search_limit,
            timeout=args.timeout,
            sleep_sec=args.sleep,
            log_path=log_path,
            overwrite=args.overwrite,
        )

    if do_fetch_images:
        summary["stages"]["fetch_images"] = fetch_images_stage(
            points_jsonl=points_jsonl,
            images_jsonl=images_jsonl,
            api_base=args.api_base,
            image_base=args.image_base,
            timeout=args.timeout,
            sleep_sec=args.sleep,
            log_path=log_path,
            overwrite=args.overwrite,
        )

    if do_order_panos:
        summary["stages"]["order_panos"] = order_panos_stage(
            points_jsonl=points_jsonl,
            images_jsonl=images_jsonl,
            ordered_index=ordered_index,
            order_mode=args.order_mode,
            api_base=args.api_base,
            timeout=args.timeout,
            sleep_sec=args.sleep,
            log_path=log_path,
            overwrite=args.overwrite,
        )

    if do_download_panos:
        summary["stages"]["download_panos"] = download_panos_stage(
            ordered_index=ordered_index,
            panos_dir=panos_dir,
            api_base=args.api_base,
            image_base=args.image_base,
            log_path=log_path,
        )

    if do_make_crops:
        summary["stages"]["make_crops"] = make_crops_stage(
            ordered_index=ordered_index,
            panos_dir=panos_dir,
            crops_dir=crops_dir,
            yaw_map_path=yaw_map_path,
            pitch_cli=args.pitch_cli,
            hfov=args.hfov,
            crop_width=args.crop_width,
            crop_height=args.crop_height,
            crop_strategy=args.crop_strategy,
            crop_supersample=args.crop_supersample,
            crop_interpolation=args.crop_interpolation,
            overwrite=args.overwrite,
            log_path=log_path,
        )

    if do_detect:
        use_existing_crops_only = (
            args.skip_make_crops
            and crops_dir.exists()
            and (not ordered_index.exists() or not panos_dir.exists())
        )
        if use_existing_crops_only:
            summary["stages"]["detect"] = detect_existing_crops_stage(
                crops_dir=crops_dir,
                annotated_dir=annotated_dir,
                detections_jsonl=detections_jsonl,
                weights=args.weights,
                conf=args.conf,
                imgsz=args.imgsz,
                device=args.device,
                overwrite=args.overwrite,
                log_path=log_path,
            )
        else:
            summary["stages"]["detect"] = detect_from_panos_stage(
                ordered_index=ordered_index,
                panos_dir=panos_dir,
                crops_dir=crops_dir,
                annotated_dir=annotated_dir,
                compare_dir=compare_dir,
                yaw_map_path=yaw_map_path,
                detections_jsonl=detections_jsonl,
                summary_path=summary_path,
                api_base=args.api_base,
                image_base=args.image_base,
                weights=args.weights,
                conf=args.conf,
                imgsz=args.imgsz,
                device=args.device,
                pitch_cli=args.pitch_cli,
                hfov=args.hfov,
                crop_width=args.crop_width,
                crop_height=args.crop_height,
                crop_strategy=args.crop_strategy,
                crop_supersample=args.crop_supersample,
                crop_interpolation=args.crop_interpolation,
                overwrite=args.overwrite,
                log_path=log_path,
            )

    detection_rows = read_jsonl(detections_jsonl) if detections_jsonl.exists() else []
    if detection_rows:
        save_csv(
            detections_csv,
            detection_rows,
            preferred=["mode", "i", "fid", "view", "s", "crop_path", "annotated_path", "n", "best"],
        )

    save_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
