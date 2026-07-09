#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Create a before/after crop comparison for panorama horizon leveling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np

from .crop import render_detection_crop
from .leveling import estimate_pano_level_correction, make_level_rotation


def _draw_label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    pad = 10
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(out, (0, 0), (tw + pad * 2, th + baseline + pad * 2), (0, 0, 0), -1)
    cv2.putText(out, text, (pad, pad + th), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pano", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--yaw", type=float, default=0.0)
    ap.add_argument("--pitch", type=float, default=0.0)
    ap.add_argument("--fov", type=float, default=105.0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=1280)
    ap.add_argument("--crop_strategy", choices=["legacy", "ui_like"], default="ui_like")
    ap.add_argument("--crop_supersample", type=float, default=1.25)
    ap.add_argument("--crop_interpolation", choices=["linear", "cubic", "lanczos", "nearest"], default="cubic")
    ap.add_argument("--level_min_confidence", type=float, default=0.25)
    ap.add_argument("--level_preview_fov", type=float, default=90.0)
    ap.add_argument("--level_preview_w", type=int, default=768)
    ap.add_argument("--level_preview_h", type=int, default=768)
    return ap


def main() -> int:
    args = build_parser().parse_args()
    pano_path = Path(args.pano)
    out_path = Path(args.out)
    pano = cv2.imread(str(pano_path))
    if pano is None:
        raise ValueError(f"failed to read panorama: {pano_path}")

    level_meta = estimate_pano_level_correction(
        pano,
        preview_fov=args.level_preview_fov,
        preview_w=args.level_preview_w,
        preview_h=args.level_preview_h,
    )
    level_confidence = float(level_meta.get("confidence", 0.0) or 0.0)
    applied = bool(level_meta.get("enabled", False)) and level_confidence >= float(args.level_min_confidence)
    R_level = make_level_rotation(float(level_meta.get("roll_deg", 0.0) or 0.0)) if applied else None
    level_meta = dict(level_meta)
    level_meta["min_confidence"] = float(args.level_min_confidence)
    level_meta["applied"] = bool(applied)

    plain, plain_meta = render_detection_crop(
        pano_bgr=pano,
        yaw_deg=args.yaw,
        pitch_deg=args.pitch,
        fov_deg=args.fov,
        out_w=args.width,
        out_h=args.height,
        crop_strategy=args.crop_strategy,
        supersample=args.crop_supersample,
        interpolation=args.crop_interpolation,
        R_level=None,
        roll_deg=0.0,
        level_meta=level_meta,
    )
    leveled, leveled_meta = render_detection_crop(
        pano_bgr=pano,
        yaw_deg=args.yaw,
        pitch_deg=args.pitch,
        fov_deg=args.fov,
        out_w=args.width,
        out_h=args.height,
        crop_strategy=args.crop_strategy,
        supersample=args.crop_supersample,
        interpolation=args.crop_interpolation,
        R_level=R_level,
        roll_deg=0.0,
        level_meta=level_meta,
    )

    combined = np.hstack(
        [
            _draw_label(plain, "without leveling"),
            _draw_label(leveled, "with leveling" if R_level is not None else "with leveling (not applied)"),
        ]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), combined):
        raise ValueError(f"failed to write comparison image: {out_path}")

    result: Dict[str, Any] = {
        "pano": str(pano_path),
        "out": str(out_path),
        "yaw": float(args.yaw),
        "pitch": float(args.pitch),
        "fov": float(args.fov),
        "level_meta": level_meta,
        "plain_crop_meta": plain_meta,
        "leveled_crop_meta": leveled_meta,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
