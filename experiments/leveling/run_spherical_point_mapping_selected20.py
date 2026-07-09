# EXPERIMENTAL: batch runner for spherical level point mapping validation.
# It only writes under outputs/experiments/leveling/spherical_point_mapping_selected20
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
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "experiments" / "leveling" / "exp_spherical_point_mapping.py"
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


def _select_fids(pano_by_fid: Dict[str, Path], yaw_by_fid: Dict[str, float], count: int, seed: int) -> Dict[str, Any]:
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


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _run_one(sample: int, fid: str, pano_path: Path, yaw: float, view: str, out_dir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    view_dir = out_dir / f"{sample:02d}_{fid}" / view
    yaw_for_view = float(yaw)
    fov_for_view = float(args.front_fov) if view == "front" else float(args.side_fov)
    if view == "left":
        yaw_for_view = float(yaw) - 90.0
    elif view == "right":
        yaw_for_view = float(yaw) + 90.0
    cmd = [
        sys.executable,
        str(SCRIPT_PATH),
        "--pano",
        str(pano_path),
        "--out_dir",
        str(view_dir),
        "--yaw",
        str(yaw_for_view),
        "--pitch",
        str(float(args.pitch)),
        "--fov",
        str(float(fov_for_view)),
        "--width",
        str(int(args.width)),
        "--height",
        str(int(args.height)),
        "--yaw_center",
        str(float(yaw)),
        "--max_points",
        str(int(args.max_points)),
        "--seed",
        str(int(args.seed)),
        "--max_apply_deg",
        str(float(args.max_apply_deg)),
    ]
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    result = {
        "sample": int(sample),
        "fid": fid,
        "view": view,
        "yaw_center": float(yaw),
        "yaw": float(yaw_for_view),
        "fov": float(fov_for_view),
        "pano_path": str(pano_path),
        "out_dir": str(view_dir),
        "returncode": int(proc.returncode),
        "stdout": proc.stdout,
    }
    meta_path = view_dir / "point_mapping_meta.json"
    if proc.returncode == 0 and meta_path.exists():
        result["meta_path"] = str(meta_path)
        result["meta"] = json.loads(meta_path.read_text(encoding="utf-8"))
    return result


def _copy_comparisons(out_dir: Path, rows: List[Dict[str, Any]]) -> None:
    comparison_dir = out_dir / "_comparison_all"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    for old in comparison_dir.iterdir():
        if old.is_file():
            old.unlink()

    for row in rows:
        src = Path(row["out_dir"]) / "comparison.jpg"
        if src.exists():
            shutil.copy2(src, comparison_dir / f'{int(row["sample"]):02d}_{row["fid"]}_{row["view"]}_comparison.jpg')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EXPERIMENTAL selected20 spherical point mapping batch runner.")
    parser.add_argument("--pano_dir", type=Path, default=REPO_ROOT / "runs" / "full_test_TMU_east_best2" / "panos")
    parser.add_argument("--yaw_map", type=Path, default=REPO_ROOT / "runs" / "full_test_TMU_east_best2" / "yaw_map.jsonl")
    parser.add_argument("--out_dir", type=Path, default=REPO_ROOT / "outputs" / "experiments" / "leveling" / "spherical_point_mapping_selected20")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pitch", type=float, default=40.0)
    parser.add_argument("--front_fov", type=float, default=105.0)
    parser.add_argument("--side_fov", type=float, default=90.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=1280)
    parser.add_argument("--max_points", type=int, default=20)
    parser.add_argument("--max_apply_deg", type=float, default=5.0)
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
        for view in ("front", "left", "right"):
            print(f"[{int(row['sample']):02d}/{len(selected_rows):02d}] {row['fid']} {view}", flush=True)
            result = _run_one(
                int(row["sample"]),
                str(row["fid"]),
                Path(row["pano_path"]),
                float(row["yaw_center"]),
                view,
                out_dir,
                args,
            )
            run_rows.append(result)
            if result["returncode"] != 0:
                print(result["stdout"], flush=True)

    _copy_comparisons(out_dir, run_rows)

    success_rows = [row for row in run_rows if row.get("meta")]
    reject_counter = Counter(str(row.get("meta", {}).get("status") or "failed") for row in run_rows)
    index_rows = []
    for row in run_rows:
        meta = row.get("meta")
        index_rows.append(
            {
                "sample": int(row["sample"]),
                "fid": str(row["fid"]),
                "view": str(row["view"]),
                "yaw_center": float(row["yaw_center"]),
                "yaw": float(row["yaw"]),
                "pitch": float(args.pitch),
                "fov": float(row["fov"]),
                "out_dir": row["out_dir"],
                "R_level_applied": None if meta is None else bool(meta.get("R_level_applied", False)),
                "v_up": None if meta is None else meta.get("v_up"),
                "angle_to_world_up_deg": None if meta is None else meta.get("angle_to_world_up_deg"),
                "inlier_count": None if meta is None else meta.get("inlier_count"),
                "total_line_count": None if meta is None else meta.get("total_line_count"),
                "mean_residual_deg": None if meta is None else meta.get("mean_residual_deg"),
                "median_residual_deg": None if meta is None else meta.get("median_residual_deg"),
                "point_count": None if meta is None else len(meta.get("points", []) or []),
                "error_mean_px": None if meta is None else meta.get("error_mean_px"),
                "error_median_px": None if meta is None else meta.get("error_median_px"),
                "error_max_px": None if meta is None else meta.get("error_max_px"),
                "status": None if meta is None else meta.get("status"),
                "point_mapping_meta": row.get("meta_path"),
                "comparison": str(Path(row["out_dir"]) / "comparison.jpg"),
            }
        )

    index = {
        "experiment": "EXPERIMENTAL spherical point mapping selected20",
        "selected_count": len(selected),
        "mandatory_count": sum(1 for row in selected_rows if row["mandatory"]),
        "missing_mandatory_fids": missing,
        "total_views": len(run_rows),
        "success_views": len(success_rows),
        "failure_views": len(run_rows) - len(success_rows),
        "status_counts": dict(reject_counter),
        "comparison_all_count": len(list((out_dir / "_comparison_all").glob("*.jpg"))),
        "comparison_all_dir": str(out_dir / "_comparison_all"),
        "rows": index_rows,
    }
    _write_json(out_dir / "point_mapping_index.json", index)
    print(
        json.dumps(
            {k: index[k] for k in index if k != "rows"},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if index["failure_views"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
