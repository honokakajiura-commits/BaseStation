#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Shared JSONL, JSON, CSV, path, and log helpers for agent scripts."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_str(x: Any) -> str:
    return "" if x is None else str(x)


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[dict]:
    out: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def read_jsonl_if_exists(path: Path) -> List[dict]:
    return read_jsonl(path) if path.exists() else []


def load_jsonl_records(path: Path) -> List[dict]:
    return read_jsonl_if_exists(path)


def write_jsonl_records(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_featurecollection(path: Path, features: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def stage_enabled(run_flag: bool, skip_flag: bool, default: bool = True) -> bool:
    if run_flag and skip_flag:
        raise ValueError("conflicting stage flags")
    if run_flag:
        return True
    if skip_flag:
        return False
    return default


def save_csv(path: Path, rows: List[dict], preferred: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def unique_path(dst: Path, overwrite: bool) -> Path:
    if overwrite or (not dst.exists()):
        return dst

    stem = dst.stem
    suffix = dst.suffix
    parent = dst.parent
    for k in range(1, 10000):
        cand = parent / f"{stem}__v{k:03d}{suffix}"
        if not cand.exists():
            return cand

    ts = int(time.time() * 1000)
    return parent / f"{stem}__v{ts}{suffix}"


def find_pano_path(panos_dir: Path, fid: str) -> Optional[Path]:
    for ext in [".jpg", ".jpeg", ".png", ".webp"]:
        p = panos_dir / f"{fid}{ext}"
        if p.exists():
            return p
    return None


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
