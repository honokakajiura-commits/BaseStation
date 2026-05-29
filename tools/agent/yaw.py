#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Yaw-center estimation stage helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import cv2

from .io_utils import append_stage_log, ensure_dir, find_pano_path, read_jsonl


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
    from tools.archive.agent_detect_only_agent2 import estimate_yaw_center_auto

    existing_map = {}
    if yaw_map_path.exists() and not overwrite:
        existing_map = {row["fid"]: float(row.get("yaw_center", 0.0)) for row in read_jsonl(yaw_map_path) if "fid" in row}
    if overwrite and yaw_map_path.exists():
        yaw_map_path.unlink()

    ensure_dir(yaw_map_path.parent)
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
