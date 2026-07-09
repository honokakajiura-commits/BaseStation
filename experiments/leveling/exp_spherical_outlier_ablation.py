#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# EXPERIMENTAL: outlier handling ablation for spherical upright leveling
# Safe to delete: remove experiments/leveling/ directory.
# It does not affect the main agent pipeline.

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.leveling.exp_spherical_upright_level import (  # noqa: E402
    GreatCircleLine,
    _collect_great_circles,
    _draw_label,
    _json_safe,
    _resize_to_height,
    _rotation_from_to,
    _wrap_yaw_deg,
)
from tools.agent.crop import render_detection_crop  # noqa: E402
from tools.agent.spherical_camera import normalize_ray  # noqa: E402


WORLD_UP = np.array([0.0, 1.0, 0.0], dtype=np.float64)
METHODS = ("no_outlier_handling", "ransac_inliers", "robust")


def _write_image(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), img):
        raise ValueError(f"failed to write image: {path}")


def _clamp_unit(x: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=np.float64), -1.0, 1.0)


def _weighted_median(values: Sequence[float], weights: Sequence[float]) -> float:
    vals = np.asarray(values, dtype=np.float64)
    wts = np.asarray(weights, dtype=np.float64)
    mask = np.isfinite(vals) & np.isfinite(wts) & (wts > 0.0)
    vals = vals[mask]
    wts = wts[mask]
    if vals.size == 0:
        return 0.0
    order = np.argsort(vals)
    vals = vals[order]
    wts = wts[order]
    total = float(np.sum(wts))
    if total <= 0.0:
        return float(np.median(vals))
    idx = int(np.searchsorted(np.cumsum(wts), total * 0.5, side="left"))
    return float(vals[min(idx, vals.size - 1)])


def _angular_residuals_rad(normals: np.ndarray, v: np.ndarray) -> np.ndarray:
    dots = normals @ np.asarray(v, dtype=np.float64)
    return np.abs(np.arcsin(_clamp_unit(dots)))


def _clone_records(records: Sequence[GreatCircleLine]) -> List[GreatCircleLine]:
    cloned: List[GreatCircleLine] = []
    for record in records:
        cloned.append(
            GreatCircleLine(
                preview_index=int(record.preview_index),
                yaw_sample=float(record.yaw_sample),
                x1=float(record.x1),
                y1=float(record.y1),
                x2=float(record.x2),
                y2=float(record.y2),
                length=float(record.length),
                angle_deg=float(record.angle_deg),
                normal=np.asarray(record.normal, dtype=np.float64).copy(),
                weight=float(record.weight),
                residual_deg=record.residual_deg,
                inlier=bool(record.inlier),
            )
        )
    return cloned


def _line_weight(record: GreatCircleLine, method: str) -> float:
    length = max(1.0, float(record.length))
    abs_angle = abs(float(record.angle_deg))
    if method == "no_outlier_handling":
        return length
    if method == "ransac_inliers":
        return length
    orient_weight = 1.0
    if abs_angle <= 3.0:
        orient_weight = 0.10
    elif abs_angle <= 12.0:
        orient_weight = 0.10 + 0.90 * ((abs_angle - 3.0) / 9.0)
    return math.sqrt(length) * orient_weight


def _weighted_svd_up(normals: np.ndarray, weights: np.ndarray) -> Optional[np.ndarray]:
    if normals.shape[0] < 2:
        return None
    w = np.asarray(weights, dtype=np.float64)
    w = np.where(np.isfinite(w) & (w > 0.0), w, 0.0)
    if float(np.sum(w)) <= 0.0:
        return None
    A = normals * np.sqrt(w)[:, None]
    try:
        _, _, vh = np.linalg.svd(A, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    v = np.asarray(vh[-1, :], dtype=np.float64)
    v = np.asarray(normalize_ray(v), dtype=np.float64)
    if float(v[1]) < 0.0:
        v = -v
    return v


def _candidate_from_normals(n1: np.ndarray, n2: np.ndarray) -> Optional[np.ndarray]:
    v = np.cross(np.asarray(n1, dtype=np.float64), np.asarray(n2, dtype=np.float64))
    norm = float(np.linalg.norm(v))
    if norm <= 1e-9:
        return None
    v = np.asarray(normalize_ray(v), dtype=np.float64)
    if float(v[1]) < 0.0:
        v = -v
    return v


def _estimate_method(records: Sequence[GreatCircleLine], args: argparse.Namespace, method: str) -> Dict[str, Any]:
    working = _clone_records(records)
    if len(working) < 2:
        return {
            "method": method,
            "ok": False,
            "applied": False,
            "reject_reason": "not_enough_great_circles",
            "v_up": None,
            "R_level": None,
            "angle_to_world_up_deg": None,
            "total_line_count": int(len(working)),
            "inlier_count": 0,
            "inlier_ratio": 0.0,
            "mean_residual_deg": None,
            "median_residual_deg": None,
            "residual_thresh_deg": float(args.residual_thresh_deg),
            "max_apply_deg": float(args.max_apply_deg),
        }

    normals = np.asarray([record.normal for record in working], dtype=np.float64)
    weights = np.asarray([_line_weight(record, method) for record in working], dtype=np.float64)
    threshold_rad = math.radians(float(args.residual_thresh_deg))
    rng = random.Random(int(args.seed) + (0 if method == "no_outlier_handling" else 97 if method == "ransac_inliers" else 193))

    v: Optional[np.ndarray] = None
    keep = np.ones(len(working), dtype=bool)

    if method == "no_outlier_handling":
        v = _weighted_svd_up(normals, weights)
        if v is None:
            return {
                "method": method,
                "ok": False,
                "applied": False,
                "reject_reason": "svd_failed",
                "v_up": None,
                "R_level": None,
                "angle_to_world_up_deg": None,
                "total_line_count": int(len(working)),
                "inlier_count": 0,
                "inlier_ratio": 0.0,
                "mean_residual_deg": None,
                "median_residual_deg": None,
                "residual_thresh_deg": float(args.residual_thresh_deg),
                "max_apply_deg": float(args.max_apply_deg),
            }
        residuals = _angular_residuals_rad(normals, v)
        keep = residuals <= threshold_rad
    else:
        nonzero_weights = np.where(weights > 0.0, weights, 0.0)
        if float(np.sum(nonzero_weights)) <= 0.0:
            return {
                "method": method,
                "ok": False,
                "applied": False,
                "reject_reason": "no_positive_weights",
                "v_up": None,
                "R_level": None,
                "angle_to_world_up_deg": None,
                "total_line_count": int(len(working)),
                "inlier_count": 0,
                "inlier_ratio": 0.0,
                "mean_residual_deg": None,
                "median_residual_deg": None,
                "residual_thresh_deg": float(args.residual_thresh_deg),
                "max_apply_deg": float(args.max_apply_deg),
            }

        best: Optional[Dict[str, Any]] = None
        sample_weights = nonzero_weights.tolist()
        for _ in range(max(1, int(args.ransac_iters))):
            i, j = rng.choices(range(len(working)), weights=sample_weights, k=2)
            if i == j:
                continue
            candidate = _candidate_from_normals(normals[i], normals[j])
            if candidate is None:
                continue
            residuals = _angular_residuals_rad(normals, candidate)
            candidate_keep = residuals <= threshold_rad
            inlier_count = int(np.count_nonzero(candidate_keep))
            if inlier_count < 2:
                continue
            inlier_weight = float(np.sum(nonzero_weights[candidate_keep]))
            median_residual = _weighted_median(
                residuals[candidate_keep].tolist(),
                nonzero_weights[candidate_keep].tolist(),
            )
            up_prior = max(0.0, float(candidate[1])) ** 4
            score = (up_prior, inlier_count, inlier_weight, -float(median_residual))
            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "v": candidate,
                    "keep": candidate_keep,
                }

        if best is None:
            return {
                "method": method,
                "ok": False,
                "applied": False,
                "reject_reason": "no_ransac_consensus",
                "v_up": None,
                "R_level": None,
                "angle_to_world_up_deg": None,
                "total_line_count": int(len(working)),
                "inlier_count": 0,
                "inlier_ratio": 0.0,
                "mean_residual_deg": None,
                "median_residual_deg": None,
                "residual_thresh_deg": float(args.residual_thresh_deg),
                "max_apply_deg": float(args.max_apply_deg),
            }

        keep = np.asarray(best["keep"], dtype=bool)
        v = _weighted_svd_up(normals[keep], weights[keep])
        if v is None:
            v = np.asarray(best["v"], dtype=np.float64)
        residuals = _angular_residuals_rad(normals, v)
        keep = residuals <= threshold_rad
        second_v = _weighted_svd_up(normals[keep], weights[keep])
        if second_v is not None:
            v = second_v
            residuals = _angular_residuals_rad(normals, v)
            keep = residuals <= threshold_rad

    assert v is not None
    residuals = _angular_residuals_rad(normals, v)
    keep = residuals <= threshold_rad
    for record, residual, is_inlier in zip(working, residuals.tolist(), keep.tolist()):
        record.residual_deg = math.degrees(float(residual))
        record.inlier = bool(is_inlier)

    inlier_count = int(np.count_nonzero(keep))
    inlier_ratio = float(inlier_count) / float(len(working)) if working else 0.0
    if inlier_count > 0:
        mean_residual_deg = math.degrees(float(np.average(residuals[keep], weights=weights[keep])))
        median_residual_deg = math.degrees(_weighted_median(residuals[keep].tolist(), weights[keep].tolist()))
    else:
        mean_residual_deg = None
        median_residual_deg = None

    angle_to_world_up_deg = math.degrees(math.acos(float(np.clip(np.dot(WORLD_UP, v), -1.0, 1.0))))
    R_level = _rotation_from_to(WORLD_UP, v)
    reject_reason = None
    applied = True
    if inlier_count < int(args.min_inliers):
        applied = False
        reject_reason = "not_enough_inliers"
    elif method == "robust" and angle_to_world_up_deg > float(args.max_apply_deg):
        applied = False
        reject_reason = "angle_exceeds_max_apply_deg"

    return {
        "method": method,
        "ok": bool(applied),
        "applied": bool(applied),
        "reject_reason": reject_reason,
        "v_up": _json_safe(v),
        "R_level": _json_safe(R_level),
        "angle_to_world_up_deg": float(angle_to_world_up_deg),
        "total_line_count": int(len(working)),
        "inlier_count": int(inlier_count),
        "inlier_ratio": float(inlier_ratio),
        "mean_residual_deg": None if mean_residual_deg is None else float(mean_residual_deg),
        "median_residual_deg": None if median_residual_deg is None else float(median_residual_deg),
        "residual_thresh_deg": float(args.residual_thresh_deg),
        "max_apply_deg": float(args.max_apply_deg),
        "records": working,
        "threshold_rad": float(threshold_rad),
    }


def _draw_preview_lines(preview_bgr: np.ndarray, records: Sequence[GreatCircleLine], mode: str) -> np.ndarray:
    out = preview_bgr.copy()
    for record in records:
        color = (150, 150, 150)
        thickness = 1
        if mode == "all":
            if record.inlier:
                color = (0, 140, 255)
                thickness = 2
            elif record.residual_deg is not None:
                color = (40, 40, 180)
        elif mode == "inlier" and record.inlier:
            color = (0, 140, 255)
            thickness = 2
        elif mode == "outlier" and not record.inlier:
            color = (40, 40, 180)
            thickness = 2
        else:
            continue
        cv2.line(
            out,
            (int(round(record.x1)), int(round(record.y1))),
            (int(round(record.x2)), int(round(record.y2))),
            color,
            thickness,
            cv2.LINE_AA,
        )
    return out


def _save_line_debug(out_dir: Path, previews: Sequence[Dict[str, Any]], robust_records: Sequence[GreatCircleLine]) -> None:
    debug_dir = out_dir / "line_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    records_by_idx: Dict[int, List[GreatCircleLine]] = defaultdict(list)
    for record in robust_records:
        records_by_idx[int(record.preview_index)].append(record)

    for preview_meta in previews:
        idx = int(preview_meta["preview_index"])
        preview_path = Path(preview_meta["preview_path"])
        preview = cv2.imread(str(preview_path))
        if preview is None:
            continue
        local_records = records_by_idx.get(idx, [])
        tag = preview_path.stem
        all_img = _draw_preview_lines(preview, local_records, "all")
        inlier_img = _draw_preview_lines(preview, local_records, "inlier")
        outlier_img = _draw_preview_lines(preview, local_records, "outlier")
        _write_image(debug_dir / f"{tag}_all_lines.jpg", all_img)
        _write_image(debug_dir / f"{tag}_inlier_lines.jpg", inlier_img)
        _write_image(debug_dir / f"{tag}_outlier_lines.jpg", outlier_img)


def _make_comparison(
    no_level: np.ndarray,
    no_outlier: np.ndarray,
    ransac: np.ndarray,
    robust: np.ndarray,
) -> np.ndarray:
    target_h = min(620, max(320, no_level.shape[0] // 2))
    panels = [
        _draw_label(_resize_to_height(no_level, target_h), "no_level"),
        _draw_label(_resize_to_height(no_outlier, target_h), "no_outlier_handling"),
        _draw_label(_resize_to_height(ransac, target_h), "ransac_inliers"),
        _draw_label(_resize_to_height(robust, target_h), "robust"),
    ]
    return np.hstack(panels)


def _render_view(
    pano: np.ndarray,
    pano_path: Path,
    out_dir: Path,
    view: str,
    yaw: float,
    pitch: float,
    fov: float,
    args: argparse.Namespace,
    methods: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    common = {
        "pano_bgr": pano,
        "yaw_deg": float(yaw),
        "pitch_deg": float(pitch),
        "fov_deg": float(fov),
        "out_w": int(args.crop_width),
        "out_h": int(args.crop_height),
        "crop_strategy": "ui_like",
        "supersample": 1.0,
        "interpolation": "cubic",
        "roll_deg": 0.0,
        "level_meta": None,
    }
    no_level, meta_no = render_detection_crop(R_level=None, **common)
    method_crops: Dict[str, np.ndarray] = {}
    method_meta: Dict[str, Any] = {}
    for method_name in METHODS:
        method = methods[method_name]
        R_level = None if not method["applied"] else np.asarray(method["R_level"], dtype=np.float64)
        crop, meta = render_detection_crop(R_level=R_level, **common)
        method_crops[method_name] = crop
        method_meta[method_name] = meta
        _write_image(out_dir / f"crop_{method_name}.jpg", crop)
    comparison = _make_comparison(
        no_level,
        method_crops["no_outlier_handling"],
        method_crops["ransac_inliers"],
        method_crops["robust"],
    )
    _write_image(out_dir / "crop_no_level.jpg", no_level)
    _write_image(out_dir / "comparison.jpg", comparison)
    return {
        "view": view,
        "yaw": float(yaw),
        "pitch": float(pitch),
        "fov": float(fov),
        "out_dir": str(out_dir),
        "crop_no_level": str(out_dir / "crop_no_level.jpg"),
        "crop_no_outlier_handling": str(out_dir / "crop_no_outlier_handling.jpg"),
        "crop_ransac_inliers": str(out_dir / "crop_ransac_inliers.jpg"),
        "crop_robust": str(out_dir / "crop_robust.jpg"),
        "comparison": str(out_dir / "comparison.jpg"),
        "crop_no_level_meta": _json_safe(meta_no),
        "method_meta": _json_safe(method_meta),
    }


def _summarize_method(method: str, method_result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "method": method,
        "applied": bool(method_result.get("applied", False)),
        "reject_reason": method_result.get("reject_reason"),
        "v_up": method_result.get("v_up"),
        "angle_to_world_up_deg": method_result.get("angle_to_world_up_deg"),
        "total_line_count": int(method_result.get("total_line_count", 0) or 0),
        "inlier_count": int(method_result.get("inlier_count", 0) or 0),
        "inlier_ratio": float(method_result.get("inlier_ratio", 0.0) or 0.0),
        "mean_residual_deg": method_result.get("mean_residual_deg"),
        "median_residual_deg": method_result.get("median_residual_deg"),
        "residual_thresh_deg": float(method_result.get("residual_thresh_deg", 0.0) or 0.0),
        "max_apply_deg": float(method_result.get("max_apply_deg", 0.0) or 0.0),
        "R_level": method_result.get("R_level"),
    }


def _run(args: argparse.Namespace) -> Dict[str, Any]:
    pano_path = Path(args.pano)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pano = cv2.imread(str(pano_path))
    if pano is None:
        raise ValueError(f"failed to read panorama image: {pano_path}")

    records, previews = _collect_great_circles(
        pano,
        out_dir,
        argparse.Namespace(
            yaw_center=float(args.yaw_center),
            pitch_detect=0.0,
            preview_fov=float(args.preview_fov),
            preview_width=int(args.preview_width),
            preview_height=int(args.preview_height),
        ),
    )

    method_results: Dict[str, Dict[str, Any]] = {}
    for method_name in METHODS:
        method_results[method_name] = _estimate_method(records, args, method_name)

    _save_line_debug(out_dir, previews, method_results["robust"].get("records", []))

    view_outputs = []
    for view, yaw, fov in [
        ("front", float(args.yaw_center), float(args.front_fov)),
        ("left", _wrap_yaw_deg(float(args.yaw_center) - 90.0), float(args.side_fov)),
        ("right", _wrap_yaw_deg(float(args.yaw_center) + 90.0), float(args.side_fov)),
    ]:
        view_out_dir = out_dir / view
        view_outputs.append(
            _render_view(
                pano=pano,
                pano_path=pano_path,
                out_dir=view_out_dir,
                view=view,
                yaw=yaw,
                pitch=float(args.crop_pitch),
                fov=fov,
                args=args,
                methods=method_results,
            )
        )

    summary = {
        "experiment": "EXPERIMENTAL outlier handling ablation for spherical upright leveling",
        "pano": str(pano_path),
        "yaw_center": float(args.yaw_center),
        "crop_pitch": float(args.crop_pitch),
        "front_fov": float(args.front_fov),
        "side_fov": float(args.side_fov),
        "preview_fov": float(args.preview_fov),
        "preview_width": int(args.preview_width),
        "preview_height": int(args.preview_height),
        "crop_width": int(args.crop_width),
        "crop_height": int(args.crop_height),
        "residual_thresh_deg": float(args.residual_thresh_deg),
        "max_apply_deg": float(args.max_apply_deg),
        "min_inliers": int(args.min_inliers),
        "ransac_iters": int(args.ransac_iters),
        "seed": int(args.seed),
        "total_line_count": int(len(records)),
        "methods": {method: _summarize_method(method, method_results[method]) for method in METHODS},
        "view_outputs": _json_safe(view_outputs),
        "line_debug_dir": str(out_dir / "line_debug"),
        "outputs": {
            "upright_meta_ablation": str(out_dir / "upright_meta_ablation.json"),
            "summary": str(out_dir / "summary.json"),
            "line_debug_dir": str(out_dir / "line_debug"),
            "comparison_front": str(out_dir / "front" / "comparison.jpg"),
            "comparison_left": str(out_dir / "left" / "comparison.jpg"),
            "comparison_right": str(out_dir / "right" / "comparison.jpg"),
        },
    }
    (out_dir / "upright_meta_ablation.json").write_text(
        json.dumps(_json_safe(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    method_summary: Dict[str, Any] = {}
    for method in METHODS:
        result = method_results[method]
        method_summary[method] = {
            "success_count": 1 if result.get("applied", False) else 0,
            "failure_count": 0 if result.get("applied", False) else 1,
            "reject_reason": result.get("reject_reason"),
            "inlier_count": int(result.get("inlier_count", 0) or 0),
            "total_line_count": int(result.get("total_line_count", 0) or 0),
            "mean_residual_deg": result.get("mean_residual_deg"),
            "median_residual_deg": result.get("median_residual_deg"),
            "angle_to_world_up_deg": result.get("angle_to_world_up_deg"),
        }
    (out_dir / "summary.json").write_text(
        json.dumps(_json_safe({"methods": method_summary}), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EXPERIMENTAL spherical upright outlier ablation.")
    parser.add_argument("--pano", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--yaw_center", type=float, default=0.0)
    parser.add_argument("--crop_pitch", type=float, default=40.0)
    parser.add_argument("--front_fov", type=float, default=105.0)
    parser.add_argument("--side_fov", type=float, default=90.0)
    parser.add_argument("--preview_fov", type=float, default=90.0)
    parser.add_argument("--preview_width", type=int, default=1024)
    parser.add_argument("--preview_height", type=int, default=768)
    parser.add_argument("--crop_width", type=int, default=1280)
    parser.add_argument("--crop_height", type=int, default=1280)
    parser.add_argument("--residual_thresh_deg", type=float, default=3.0)
    parser.add_argument("--max_apply_deg", type=float, default=5.0)
    parser.add_argument("--min_inliers", type=int, default=8)
    parser.add_argument("--ransac_iters", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    meta = _run(args)
    print(
        json.dumps(
            {
                "pano": meta["pano"],
                "out_dir": str(args.out_dir),
                "methods": meta["methods"],
                "upright_meta_ablation": meta["outputs"]["upright_meta_ablation"],
                "summary": meta["outputs"]["summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
