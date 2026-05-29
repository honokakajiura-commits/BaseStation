#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.agent.io_utils import ensure_dir, read_jsonl, save_csv, save_json, stage_enabled


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
    ap.add_argument("--pitch_cli", type=float, default=40.0, help="Pitch in degrees; positive is up")
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

    if do_fetch_points or do_fetch_images or do_download_panos:
        from tools.agent.panoramax_client import download_panos_stage, fetch_images_stage, fetch_points_stage
    if do_order_panos:
        from tools.agent.ordering import order_panos_stage
    if do_make_crops or do_detect:
        from tools.agent.pipeline import detect_existing_crops_stage, detect_from_panos_stage, make_crops_stage

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
