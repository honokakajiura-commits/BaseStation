#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# EXPERIMENTAL: batch runner for spherical upright adjustment validation.
# It only writes under outputs/experiments/leveling/spherical_upright_selected20
# unless the CLI overrides --out_dir.
# It does not affect the main agent pipeline.

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "experiments" / "leveling" / "exp_spherical_upright_level.py"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
MANDATORY_FIDS = [
    "0e4603f1-e7f1-460f-b42d-69c02d0cd3a6",
    "0ed22781-15fd-45d9-a707-cc39d398d2b5",
    "3f008858-45b2-4630-b357-41986cdf71e0",
    "6bed5786-9939-49b6-b1bc-9a6b1b7db521",
    "7f22a376-59e9-4043-a30d-e1e85a2b335e",
    "9b2a761d-a08f-4714-a62b-ef8d6e5849ef",
    "14a1af16-8218-486c-b679-d17c7b644b24",
    "31f75915-78e9-46b3-9217-8602995c6541",
]


def _read_yaw_map(path: Path) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        fid = str(row.get("fid") or "")
        if not fid:
            continue
        try:
            out[fid] = float(row.get("yaw_center", 0.0) or 0.0)
        except (TypeError, ValueError):
            out[fid] = 0.0
    return out


def _discover_panos(pano_dir: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for path in sorted(pano_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            out.setdefault(path.stem, path)
    return out


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _select_fids(
    pano_by_fid: Dict[str, Path],
    yaw_by_fid: Dict[str, float],
    count: int,
    seed: int,
) -> Dict[str, Any]:
    missing = [fid for fid in MANDATORY_FIDS if fid not in pano_by_fid]
    selected: List[str] = []
    seen = set()
    for fid in MANDATORY_FIDS:
        if fid in pano_by_fid and fid not in seen:
            selected.append(fid)
            seen.add(fid)

    rng = random.Random(int(seed))
    pool = sorted(fid for fid in yaw_by_fid if fid in pano_by_fid and fid not in seen)
    rng.shuffle(pool)
    for fid in pool:
        if len(selected) >= int(count):
            break
        selected.append(fid)
        seen.add(fid)

    if len(selected) < int(count):
        fallback = sorted(fid for fid in pano_by_fid if fid not in seen)
        rng.shuffle(fallback)
        for fid in fallback:
            if len(selected) >= int(count):
                break
            selected.append(fid)
            seen.add(fid)

    return {"selected": selected, "missing": missing}


def _run_one(row: Dict[str, Any], out_dir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    sample = int(row["sample"])
    fid = str(row["fid"])
    sample_out = out_dir / f"{sample:02d}_{fid}"
    cmd = [
        sys.executable,
        str(SCRIPT_PATH),
        "--pano",
        str(row["pano_path"]),
        "--out_dir",
        str(sample_out),
        "--yaw_center",
        str(float(row["yaw_center"])),
        "--pitch_detect",
        str(float(args.pitch_detect)),
        "--preview_fov",
        str(float(args.preview_fov)),
        "--preview_width",
        str(int(args.preview_width)),
        "--preview_height",
        str(int(args.preview_height)),
        "--crop_width",
        str(int(args.crop_width)),
        "--crop_height",
        str(int(args.crop_height)),
        "--crop_pitch",
        str(float(args.crop_pitch)),
        "--front_fov",
        str(float(args.front_fov)),
        "--side_fov",
        str(float(args.side_fov)),
        "--max_apply_deg",
        str(float(args.max_apply_deg)),
        "--ransac_iters",
        str(int(args.ransac_iters)),
        "--min_inliers",
        str(int(args.min_inliers)),
        "--seed",
        str(int(args.seed)),
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    result = {
        "sample": sample,
        "fid": fid,
        "out_dir": str(sample_out),
        "returncode": int(proc.returncode),
        "stdout": proc.stdout,
    }
    meta_path = sample_out / "upright_meta.json"
    if proc.returncode == 0 and meta_path.exists():
        result["meta_path"] = str(meta_path)
        result["meta"] = json.loads(meta_path.read_text(encoding="utf-8"))
    return result


def _copy_comparisons(out_dir: Path, run_rows: List[Dict[str, Any]]) -> None:
    comparison_dir = out_dir / "_comparison_all"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    for old in comparison_dir.iterdir():
        if old.is_file():
            old.unlink()

    for row in run_rows:
        sample = int(row["sample"])
        fid = str(row["fid"])
        sample_out = Path(row["out_dir"])
        for view in ("front", "left", "right"):
            src = sample_out / f"comparison_{view}.jpg"
            if src.exists():
                shutil.copy2(src, comparison_dir / f"{sample:02d}_{fid}_{view}_comparison.jpg")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EXPERIMENTAL selected20 spherical upright batch runner.")
    parser.add_argument("--pano_dir", type=Path, default=REPO_ROOT / "runs" / "full_test_TMU_east_best2" / "panos")
    parser.add_argument("--yaw_map", type=Path, default=REPO_ROOT / "runs" / "full_test_TMU_east_best2" / "yaw_map.jsonl")
    parser.add_argument("--out_dir", type=Path, default=REPO_ROOT / "outputs" / "experiments" / "leveling" / "spherical_upright_selected20")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pitch_detect", type=float, default=0.0)
    parser.add_argument("--preview_fov", type=float, default=90.0)
    parser.add_argument("--preview_width", type=int, default=1024)
    parser.add_argument("--preview_height", type=int, default=768)
    parser.add_argument("--crop_width", type=int, default=1280)
    parser.add_argument("--crop_height", type=int, default=1280)
    parser.add_argument("--crop_pitch", type=float, default=40.0)
    parser.add_argument("--front_fov", type=float, default=105.0)
    parser.add_argument("--side_fov", type=float, default=90.0)
    parser.add_argument("--max_apply_deg", type=float, default=5.0)
    parser.add_argument("--ransac_iters", type=int, default=1000)
    parser.add_argument("--min_inliers", type=int, default=8)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pano_by_fid = _discover_panos(Path(args.pano_dir))
    yaw_by_fid = _read_yaw_map(Path(args.yaw_map))
    selected_info = _select_fids(pano_by_fid, yaw_by_fid, int(args.count), int(args.seed))
    selected = selected_info["selected"]
    missing = selected_info["missing"]

    (out_dir / "selected_fids.txt").write_text("\n".join(selected) + "\n", encoding="utf-8")
    missing_path = out_dir / "missing_fids.txt"
    if missing:
        missing_path.write_text("\n".join(missing) + "\n", encoding="utf-8")
    elif missing_path.exists():
        missing_path.unlink()

    selected_rows: List[Dict[str, Any]] = []
    with (out_dir / "selected_rows.jsonl").open("w", encoding="utf-8") as f:
        for sample, fid in enumerate(selected, start=1):
            row = {
                "sample": int(sample),
                "fid": fid,
                "pano_path": str(pano_by_fid[fid]),
                "yaw_center": float(yaw_by_fid.get(fid, 0.0)),
                "mandatory": fid in MANDATORY_FIDS,
                "has_yaw_map": fid in yaw_by_fid,
            }
            selected_rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    run_rows: List[Dict[str, Any]] = []
    for row in selected_rows:
        print(f"[{int(row['sample']):02d}/{len(selected_rows):02d}] {row['fid']}", flush=True)
        result = _run_one(row, out_dir, args)
        run_rows.append(result)
        if result["returncode"] != 0:
            print(result["stdout"], flush=True)

    _copy_comparisons(out_dir, run_rows)

    successes = [row for row in run_rows if row.get("meta")]
    metas = [row["meta"] for row in successes]
    comparison_count = len(list((out_dir / "_comparison_all").glob("*.jpg")))
    index_rows = []
    for run_row in run_rows:
        meta: Optional[Dict[str, Any]] = run_row.get("meta")
        selected_row = selected_rows[int(run_row["sample"]) - 1]
        index_rows.append(
            {
                "sample": int(run_row["sample"]),
                "fid": str(run_row["fid"]),
                "pano_path": selected_row["pano_path"],
                "yaw_center": float(selected_row["yaw_center"]),
                "out_dir": run_row["out_dir"],
                "returncode": int(run_row["returncode"]),
                "upright_meta": run_row.get("meta_path"),
                "applied": None if meta is None else bool(meta.get("applied", False)),
                "reject_reason": None if meta is None else meta.get("reject_reason"),
                "angle_to_world_up_deg": None if meta is None else meta.get("angle_to_world_up_deg"),
                "inlier_count": None if meta is None else meta.get("inlier_count"),
                "total_line_count": None if meta is None else meta.get("total_line_count"),
                "mean_residual_deg": None if meta is None else meta.get("mean_residual_deg"),
                "median_residual_deg": None if meta is None else meta.get("median_residual_deg"),
                "comparison_front": str(Path(run_row["out_dir"]) / "comparison_front.jpg"),
                "comparison_left": str(Path(run_row["out_dir"]) / "comparison_left.jpg"),
                "comparison_right": str(Path(run_row["out_dir"]) / "comparison_right.jpg"),
            }
        )

    reject_reasons = Counter(str(meta.get("reject_reason") or "accepted") for meta in metas)
    index = {
        "experiment": "EXPERIMENTAL spherical upright selected20",
        "selected_count": len(selected),
        "mandatory_count": sum(1 for row in selected_rows if row["mandatory"]),
        "missing_mandatory_fids": missing,
        "success_count": len(successes),
        "failure_count": len(run_rows) - len(successes),
        "applied_count": sum(1 for meta in metas if bool(meta.get("applied", False))),
        "reject_reasons": dict(reject_reasons),
        "max_angle_to_world_up_deg": max((float(meta.get("angle_to_world_up_deg") or 0.0) for meta in metas), default=None),
        "max_inlier_count": max((int(meta.get("inlier_count", 0) or 0) for meta in metas), default=0),
        "comparison_all_count": comparison_count,
        "comparison_all_dir": str(out_dir / "_comparison_all"),
        "rows": index_rows,
    }
    _write_json(out_dir / "spherical_upright_index.json", index)
    print(
        json.dumps(
            {k: index[k] for k in index if k != "rows"},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if index["failure_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
