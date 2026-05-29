#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Panorama ordering stages for the agent pipeline."""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any, Dict, List

import requests

from .io_utils import append_stage_log, load_images_map, load_ordered_index, load_points_features, safe_str


def order_points_with_sequence(
    features: List[dict],
    images_map: Dict[str, dict],
    session: requests.Session,
    api_base: str,
    timeout: int,
    sleep_sec: float,
) -> List[dict]:
    from tools.archive.fetch_panos_ordered import (
        get_datetime_from_feature,
        get_feature_id,
        order_sequences_by_nearest,
    )
    from tools.archive.make_yolo_crops_from_panoramax import extract_sequence_and_rank, fetch_picture_meta

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
    from tools.archive.fetch_panos_ordered import (
        order_features_datetime,
        order_features_nearest,
        write_aoi_index_jsonl,
    )

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
