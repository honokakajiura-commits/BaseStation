#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import importlib.util
from pathlib import Path
import cv2

FID = "9149cc87-15d3-43a6-aff9-ac595ff6a5be"

RUN_DIR = Path("runs/panoramax_trainset_std")
OUT_DIR = Path("runs/single_pano_roll_compare") / FID

PANO_DIR = RUN_DIR / "panos"

spec = importlib.util.spec_from_file_location(
    "roll_aligned",
    "tools/make_yolo_crops_from_panoramax_roll_aligned.py",
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

def find_pano_path(fid: str) -> Path:
    for ext in [".jpg", ".jpeg", ".png", ".webp"]:
        p = PANO_DIR / f"{fid}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"pano not found for fid={fid} in {PANO_DIR}")

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pano_path = find_pano_path(FID)
    pano = cv2.imread(str(pano_path))
    if pano is None:
        raise RuntimeError(f"failed to read pano: {pano_path}")

    pitch_cli = 30.0
    pitch_deg = -pitch_cli
    det_w = 1600
    det_h = 800
    fov_front = 115.0
    fov_side = 115.0
    fov_back = 115.0

    yaw_center, yaw_reason, yaw_meta = mod.estimate_yaw_center_auto(
        pano,
        pitch_deg=pitch_deg,
        view_azimuth=None,
        fov_preview=110.0,
        out_w=1024,
        out_h=768,
    )

    pano_roll_deg, pano_roll_reason, pano_roll_meta = mod.estimate_pano_roll_deg(
        pano,
        yaw_center_deg=yaw_center,
        pitch_deg=pitch_deg,
        out_w=det_w,
        out_h=det_h,
        fov_front=fov_front,
        fov_side=fov_side,
        fov_back=fov_back,
    )

    print("fid:", FID)
    print("pano_path:", pano_path)
    print("yaw_center:", yaw_center, "reason:", yaw_reason)
    print("pano_roll_deg:", pano_roll_deg, "reason:", pano_roll_reason)

    views = [
        ("front", 0.0, fov_front),
        ("left", -90.0, fov_side),
        ("right", 90.0, fov_side),
        ("back", 180.0, fov_back),
    ]

    variants = [
        ("nocorr", 0.0),
        ("normal", pano_roll_deg),
        ("inverse", -pano_roll_deg),
    ]

    for view_name, yaw_off, fov in views:
        yaw = mod.wrap_yaw_deg(yaw_center + yaw_off)

        for variant_name, roll_deg in variants:
            crop = mod.equirectangular_to_perspective(
                pano,
                yaw_deg=yaw,
                pitch_deg=pitch_deg,
                roll_deg=roll_deg,
                fov_deg=fov,
                out_w=det_w,
                out_h=det_h,
            )
            out_path = OUT_DIR / f"{FID}__{view_name}__{variant_name}__roll_{roll_deg:+.3f}.jpg"
            cv2.imwrite(str(out_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

            print("saved:", out_path)

if __name__ == "__main__":
    main()
