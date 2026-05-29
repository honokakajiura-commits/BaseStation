#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
比較・検証途中で増えた run ディレクトリを一箇所へ退避する。

デフォルトでは `runs/` 配下の比較・検証・スモーク系ディレクトリを
`runs/_archive/<archive_name>/` へ移動する。

使い方:
python tools/archive_experiment_runs.py --dry_run
python tools/archive_experiment_runs.py
"""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterable, List


DEFAULT_PATTERNS = [
    "compare_*",
    "verify_*",
    "global_upright_*",
    "single_pano_roll_compare",
    "pano_last",
    "panoramax_trainset_roll_aligned_last25_smoke",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="比較・検証系 run ディレクトリを退避する")
    parser.add_argument(
        "--runs_root",
        default="runs",
        help="run ディレクトリのルート",
    )
    parser.add_argument(
        "--archive_root",
        default=None,
        help="退避先ルート。未指定なら <runs_root>/_archive",
    )
    parser.add_argument(
        "--archive_name",
        default=None,
        help="今回の退避名。未指定なら archive_YYYYmmdd_HHMMSS",
    )
    parser.add_argument(
        "--pattern",
        action="append",
        default=[],
        help="退避対象の glob パターン。複数指定可。未指定なら既定パターンを使う",
    )
    parser.add_argument(
        "--include_tools_runs",
        action="store_true",
        help="tools/runs 直下の同名ディレクトリも退避する",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="実際には移動せず、対象だけ表示する",
    )
    return parser.parse_args()


def unique_paths(paths: Iterable[Path]) -> List[Path]:
    seen = set()
    out: List[Path] = []
    for path in sorted(paths):
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def collect_candidates(runs_root: Path, patterns: List[str]) -> List[Path]:
    candidates: List[Path] = []
    for pattern in patterns:
        for path in runs_root.glob(pattern):
            if path.is_dir() and not path.name.startswith("_archive"):
                candidates.append(path)
    return unique_paths(candidates)


def build_manifest(archive_dir: Path, moved: List[dict]) -> dict:
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "archive_dir": str(archive_dir),
        "moved": moved,
    }


def main() -> int:
    args = parse_args()
    runs_root = Path(args.runs_root).resolve()
    archive_root = Path(args.archive_root).resolve() if args.archive_root else (runs_root / "_archive")
    archive_name = args.archive_name or f"archive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    archive_dir = archive_root / archive_name
    patterns = args.pattern or list(DEFAULT_PATTERNS)

    if not runs_root.exists():
        raise SystemExit(f"runs_root が存在しません: {runs_root}")

    candidates = collect_candidates(runs_root, patterns)
    if not candidates:
        print("退避対象はありませんでした。")
        return 0

    moved: List[dict] = []
    for src in candidates:
        rel = src.relative_to(runs_root)
        dst = archive_dir / rel
        moved.append({"src": str(src), "dst": str(dst)})
        if args.include_tools_runs:
            tools_src = runs_root.parent / "tools" / "runs" / rel
            if tools_src.is_dir():
                tools_dst = archive_dir / "tools_runs" / rel
                moved.append({"src": str(tools_src), "dst": str(tools_dst)})

    print("退避対象:")
    for item in moved:
        print(f"- {item['src']} -> {item['dst']}")

    if args.dry_run:
        print("\n--dry_run のため移動は実行していません。")
        return 0

    archive_dir.mkdir(parents=True, exist_ok=True)
    for item in moved:
        src = Path(item["src"])
        dst = Path(item["dst"])
        if not src.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))

    manifest_path = archive_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(build_manifest(archive_dir, moved), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n退避完了: {archive_dir}")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
