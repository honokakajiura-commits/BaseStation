#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Execution stages for crop generation and detection."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2

from .config import AgentConfig
from .crop import build_crop_name as _build_crop_name, render_detection_crop
from .detector import YoloRunner, best_det, det_center_frac
from .io_utils import (
    append_jsonl,
    append_stage_log,
    ensure_dir,
    find_pano_path,
    load_ordered_index,
    save_json,
    unique_path,
)
from .leveling import estimate_pano_level_correction, make_level_rotation
from .refine_policy import need_center_by_edge, plan_refine_view
from .spherical_camera import clamp_pitch_deg, wrap_yaw_deg
from .visualize import draw_annot, draw_refine_compare, draw_status
from .yaw import ensure_yaw_map


def plan_detection_refine(*args, **kwargs):
    """Compatibility shim for older callers."""
    return plan_refine_view(*args, **kwargs)


def _disabled_level_meta(reason: str, **extra: Any) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "enabled": False,
        "roll_deg": 0.0,
        "confidence": 0.0,
        "sample_count": 0,
        "used_sample_count": 0,
        "samples": [],
        "method": "hough_lines",
        "reason": reason,
        "applied": False,
    }
    meta.update(extra)
    return meta


def _estimate_leveling_for_pano(
    pano_bgr: Any,
    level_horizon: bool,
    level_min_confidence: float,
    level_preview_fov: float,
    level_preview_w: int,
    level_preview_h: int,
) -> Tuple[Dict[str, Any], Optional[Any]]:
    if not bool(level_horizon):
        return _disabled_level_meta("disabled"), None

    try:
        level_meta = estimate_pano_level_correction(
            pano_bgr,
            preview_fov=float(level_preview_fov),
            preview_w=int(level_preview_w),
            preview_h=int(level_preview_h),
        )
    except Exception as exc:
        return _disabled_level_meta("estimate_failed", error=str(exc)), None

    level_meta = dict(level_meta)
    level_meta["min_confidence"] = float(level_min_confidence)
    level_confidence = float(level_meta.get("confidence", 0.0) or 0.0)
    applied = bool(level_meta.get("enabled", False)) and level_confidence >= float(level_min_confidence)
    level_meta["applied"] = bool(applied)
    if not applied:
        return level_meta, None
    return level_meta, make_level_rotation(float(level_meta.get("roll_deg", 0.0) or 0.0))


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
    level_horizon: bool = True,
    level_min_confidence: float = 0.25,
    level_preview_fov: float = 90.0,
    level_preview_w: int = 768,
    level_preview_h: int = 768,
) -> Dict[str, Any]:
    ensure_dir(crops_dir)
    rows = load_ordered_index(ordered_index)
    # Internal pitch is positive upward. Keep the CLI conversion in one place.
    pitch_deg = float(pitch_cli)
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
        level_meta, R_level = _estimate_leveling_for_pano(
            pano_bgr=pano,
            level_horizon=level_horizon,
            level_min_confidence=level_min_confidence,
            level_preview_fov=level_preview_fov,
            level_preview_w=level_preview_w,
            level_preview_h=level_preview_h,
        )
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
                R_level=R_level,
                roll_deg=0.0,
                level_meta=level_meta,
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
                params={
                    "fid": fid,
                    "view": view_name,
                    "hfov": hfov,
                    "pitch_cli": pitch_cli,
                    "level_horizon": bool(level_horizon),
                    "level_min_confidence": float(level_min_confidence),
                },
                level_meta=level_meta,
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
            "level_horizon": bool(level_horizon),
            "level_min_confidence": float(level_min_confidence),
            "level_preview_fov": float(level_preview_fov),
            "level_preview_w": int(level_preview_w),
            "level_preview_h": int(level_preview_h),
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
    level_horizon: bool = True,
    level_min_confidence: float = 0.25,
    level_preview_fov: float = 90.0,
    level_preview_w: int = 768,
    level_preview_h: int = 768,
) -> Dict[str, Any]:
    ensure_dir(crops_dir)
    ensure_dir(annotated_dir)
    ensure_dir(compare_dir)

    rows = load_ordered_index(ordered_index)
    # Internal pitch is positive upward. Keep the CLI conversion in one place.
    pitch_deg = float(pitch_cli)
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
        level_horizon=level_horizon,
        level_min_confidence=level_min_confidence,
        level_preview_fov=level_preview_fov,
        level_preview_w=level_preview_w,
        level_preview_h=level_preview_h,
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
        level_meta, R_level = _estimate_leveling_for_pano(
            pano_bgr=pano,
            level_horizon=cfg.level_horizon,
            level_min_confidence=cfg.level_min_confidence,
            level_preview_fov=cfg.level_preview_fov,
            level_preview_w=cfg.level_preview_w,
            level_preview_h=cfg.level_preview_h,
        )

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
                    crop_meta = {
                        "source": "existing_crop",
                        "strategy": crop_strategy,
                        "roll_deg": 0.0,
                        "leveling_enabled": False,
                        "level_roll_deg": float(level_meta.get("roll_deg", 0.0) or 0.0),
                        "level_confidence": float(level_meta.get("confidence", 0.0) or 0.0),
                        "level_sample_count": int(level_meta.get("sample_count", 0) or 0),
                        "level_used_sample_count": int(level_meta.get("used_sample_count", 0) or 0),
                        "R_level_applied": False,
                    }
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
                            R_level=R_level,
                            roll_deg=0.0,
                            level_meta=level_meta,
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
                        R_level=R_level,
                        roll_deg=0.0,
                        level_meta=level_meta,
                    )
                    cv2.imwrite(str(crop_path), crop)
                total_crops += 1

                dets = yolo.infer(crop)
                bd = best_det(dets)
                detection_row = {
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
                }

                if not bd:
                    if step == 0:
                        append_jsonl(detections_jsonl, detection_row)
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
                    detection_row["annotated_path"] = str(ann_path0)
                    append_jsonl(detections_jsonl, detection_row)
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
                detection_row["annotated_path"] = str(ann_path)
                append_jsonl(detections_jsonl, detection_row)

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

                if step >= cfg.max_refine:
                    break

                center_by_edge = need_center_by_edge(cx_frac, cfg.edge_center_margin)

                next_yaw, next_pitch, next_fov, refine_action, debug_info = plan_refine_view(
                    det=bd,
                    yaw=cur_yaw,
                    pitch=cur_pitch,
                    roll=0.0,
                    current_fov=cur_fov,
                    image_w=cfg.det_w,
                    image_h=cfg.det_h,
                    min_fov=cfg.zoom_min_fov,
                    margin_deg=cfg.bbox_margin_deg,
                    R_level=R_level,
                    recenter_pitch=cfg.recenter_pitch,
                    max_zoom_ratio=cfg.refine_zoom_ratio_small,
                )

                yaw_delta = wrap_yaw_deg(next_yaw - cur_yaw)
                pitch_delta = float(next_pitch) - float(cur_pitch)
                fov_delta = float(next_fov) - float(cur_fov)
                zoom = bool(float(next_fov) < float(cur_fov) - 0.5)

                if abs(yaw_delta) < 0.5 and abs(pitch_delta) < 0.5 and abs(fov_delta) < 0.5:
                    break

                append_stage_log(
                    log_path,
                    step="refine_plan",
                    status="ok",
                    input_file=str(crop_path),
                    params={
                        "fid": fid,
                        "view": view_name,
                        "s": step,
                        "level_horizon": bool(cfg.level_horizon),
                        "level_min_confidence": float(cfg.level_min_confidence),
                    },
                    level_meta=level_meta,
                    refine_action=refine_action,
                    previous_yaw=float(cur_yaw),
                    previous_pitch=float(cur_pitch),
                    previous_fov=float(cur_fov),
                    center_by_edge=bool(center_by_edge),
                    bbox_center=[float(bbox_cx), float(bbox_cy)],
                    bbox_area_ratio=float(debug_info["bbox_area_ratio"]),
                    target_yaw=float(debug_info["target_yaw"]),
                    target_pitch=float(debug_info["target_pitch"]),
                    next_yaw=float(next_yaw),
                    next_pitch=float(next_pitch),
                    next_fov=float(next_fov),
                    max_corner_angle=float(debug_info["max_corner_angle"]),
                    safe_fov=float(debug_info["safe_fov"]),
                    zoom_fov=float(debug_info["zoom_fov"]),
                    target_fov=float(debug_info["target_fov"]),
                    final_fov=float(debug_info["final_fov"]),
                    yaw_delta=float(yaw_delta),
                    pitch_delta=float(pitch_delta),
                    zoom=bool(zoom),
                    zoom_ratio_init=float(debug_info["max_zoom_ratio"]),
                    area_frac=float(area_frac),
                    debug_info=debug_info,
                )

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

                cur_yaw = wrap_yaw_deg(next_yaw)
                cur_pitch = clamp_pitch_deg(next_pitch)
                cur_fov = float(next_fov)
                last_yaw_delta = float(yaw_delta)
                last_zoom = bool(zoom)

        append_stage_log(
            log_path,
            step="detect_pano_done",
            status="ok",
            input_file=str(pano_path),
            params={"fid": fid, "level_horizon": bool(cfg.level_horizon)},
            level_meta=level_meta,
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
            "level_horizon": bool(cfg.level_horizon),
            "level_min_confidence": float(cfg.level_min_confidence),
            "level_preview_fov": float(cfg.level_preview_fov),
            "level_preview_w": int(cfg.level_preview_w),
            "level_preview_h": int(cfg.level_preview_h),
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
        params={
            "weights": weights,
            "conf": conf,
            "imgsz": imgsz,
            "device": device,
            "level_horizon": bool(cfg.level_horizon),
            "level_min_confidence": float(cfg.level_min_confidence),
        },
        processed_panos=total_panos,
        confirmed=confirmed,
        candidates=candidates,
    )
    return summary
