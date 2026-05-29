#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AOI内のPanoramax撮影地点(Feature Point)を /api/search で取得して GeoJSON 保存する

例:
python3 tools/panoramax_fetch_points_in_aoi.py \
  --aoi data/TMU_east.geojson \
  --out_dir runs/TMu_east \
  --cell_deg 0.005 \
  --limit 1000 \
  --sleep 0.05
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

import requests
from shapely.geometry import shape, box, Point
from shapely.ops import unary_union


# -----------------------------
# AOI
# -----------------------------
def load_aoi_union(aoi_path: Path):
    data = json.loads(aoi_path.read_text(encoding="utf-8"))

    if isinstance(data, dict) and data.get("type") == "FeatureCollection":
        geoms = []
        for f in data.get("features", []):
            if not isinstance(f, dict):
                continue
            g = f.get("geometry")
            if isinstance(g, dict):
                geoms.append(shape(g))
        if not geoms:
            raise ValueError("AOI FeatureCollectionに geometry が見つかりません")
        return unary_union(geoms)

    if isinstance(data, dict) and data.get("type") == "Feature":
        g = data.get("geometry")
        if not isinstance(g, dict):
            raise ValueError("AOI Featureに geometry がありません")
        return shape(g)

    if isinstance(data, dict) and "type" in data and "coordinates" in data:
        return shape(data)

    raise ValueError("AOI GeoJSONの形式が想定外です（FeatureCollection/Feature/Geometry のいずれか）")


def frange(start: float, stop: float, step: float):
    x = start
    while x < stop - 1e-12:
        yield x
        x += step


def make_grid_cells(geom, cell_deg: float) -> List[Tuple[float, float, float, float]]:
    minx, miny, maxx, maxy = geom.bounds
    cells = []
    for x in frange(minx, maxx, cell_deg):
        for y in frange(miny, maxy, cell_deg):
            b = (x, y, min(x + cell_deg, maxx), min(y + cell_deg, maxy))
            if geom.intersects(box(*b)):
                cells.append(b)
    return cells


# -----------------------------
# Panoramax API
# -----------------------------
def safe_str(x) -> str:
    return "" if x is None else str(x)


def post_search(session: requests.Session, base_url: str, payload: Dict[str, Any], timeout: int) -> Dict[str, Any]:
    url = base_url.rstrip("/") + "/api/search"
    r = session.post(url, json=payload, timeout=timeout)
    if r.status_code in (404, 405):
        # fallback: some deployments accept GET
        r = session.get(url, params=payload, timeout=timeout)
    if r.status_code >= 400:
        raise RuntimeError(f"search failed {r.status_code}: {r.text[:300]}")
    return r.json()


def extract_features(resp: Dict[str, Any]) -> List[Dict[str, Any]]:
    feats = resp.get("features")
    if isinstance(feats, list):
        return [f for f in feats if isinstance(f, dict) and f.get("type") == "Feature"]
    return []


def top_next_link(resp: Dict[str, Any]) -> Optional[str]:
    links = resp.get("links")
    if not isinstance(links, list):
        return None
    for l in links:
        if isinstance(l, dict) and l.get("rel") == "next" and isinstance(l.get("href"), str):
            return l["href"]
    return None


def get_link_href(feature: Dict[str, Any], rel: str) -> Optional[str]:
    links = feature.get("links")
    if not isinstance(links, list):
        return None
    for l in links:
        if not isinstance(l, dict):
            continue
        if l.get("rel") == rel:
            href = l.get("href")
            if isinstance(href, str) and href:
                return href
    return None


def normalize_feature_props(feature: Dict[str, Any], base_url: str) -> Dict[str, Any]:
    """
    後段で使いやすいようにURLをpropertiesへ寄せる
    - item_url: rel=self を優先（STAC Item）
    - collection_url: rel=collection があれば
    - api_base: base_url
    """
    props = feature.get("properties")
    if not isinstance(props, dict):
        props = {}
    props = dict(props)

    item_url = get_link_href(feature, "self") or get_link_href(feature, "item")
    if item_url:
        props["item_url"] = item_url

    collection_url = get_link_href(feature, "collection")
    if collection_url:
        props["collection_url"] = collection_url

    props["api_base"] = base_url.rstrip("/")

    # pic_type がpropertiesに入ってることがあるので、lowerで正規化して追加（あれば）
    pt = props.get("pic_type") or props.get("picType") or props.get("type")
    if isinstance(pt, str) and pt:
        props["pic_type_norm"] = pt.lower()

    return props


def point_from_feature(feature: Dict[str, Any]) -> Optional[Point]:
    g = feature.get("geometry")
    if not (isinstance(g, dict) and g.get("type") == "Point"):
        return None
    coords = g.get("coordinates")
    if not (isinstance(coords, list) and len(coords) >= 2):
        return None
    lon, lat = coords[0], coords[1]
    try:
        return Point(float(lon), float(lat))
    except Exception:
        return None


# -----------------------------
# main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aoi", required=True, help="AOI geojson path (polygon)")
    ap.add_argument("--out_dir", required=True, help="output directory")
    ap.add_argument("--base", default="https://api.panoramax.xyz", help="Panoramax API base url")
    ap.add_argument("--cell_deg", type=float, default=0.005, help="grid size in degrees (0.005 ~ 500m)")
    ap.add_argument("--limit", type=int, default=1000, help="limit per bbox request")
    ap.add_argument("--timeout", type=int, default=60, help="request timeout seconds")
    ap.add_argument("--sleep", type=float, default=0.0, help="sleep seconds between requests")
    ap.add_argument("--debug_first", action="store_true", help="print first feature keys for debugging")
    args = ap.parse_args()

    aoi_path = Path(args.aoi)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    aoi_union = load_aoi_union(aoi_path)
    cells = make_grid_cells(aoi_union, args.cell_deg)

    session = requests.Session()
    session.headers.update({"User-Agent": "panoramax-aoi-fetch-points/3.0"})

    seen_ids: Set[str] = set()
    kept_features: List[Dict[str, Any]] = []

    stats = {
        "aoi": str(aoi_path),
        "base": args.base,
        "cell_deg": args.cell_deg,
        "limit": args.limit,
        "cells": len(cells),
        "unique_points_in_aoi": 0,
        "cells_hit_limit": 0,
        "pages_followed": 0,
        "errors": 0,
    }

    printed_debug = False

    for ci, bbox in enumerate(cells, start=1):
        payload = {"bbox": [bbox[0], bbox[1], bbox[2], bbox[3]], "limit": args.limit}

        try:
            resp = post_search(session, args.base, payload, timeout=args.timeout)
        except Exception as e:
            stats["errors"] += 1
            print(f"[{ci}/{len(cells)}] ERROR search bbox={bbox} -> {e}")
            continue

        responses = [resp]
        nxt = top_next_link(resp)
        while nxt:
            try:
                r2 = session.get(nxt, timeout=args.timeout)
                if r2.status_code >= 400:
                    raise RuntimeError(f"next failed {r2.status_code}: {r2.text[:200]}")
                resp2 = r2.json()
                responses.append(resp2)
                stats["pages_followed"] += 1
                nxt = top_next_link(resp2)
                if args.sleep > 0:
                    time.sleep(args.sleep)
            except Exception as e:
                stats["errors"] += 1
                print(f"[{ci}/{len(cells)}] ERROR paging next -> {e}")
                break

        raw_count_cell = 0
        for rj in responses:
            feats = extract_features(rj)
            raw_count_cell += len(feats)

            for f in feats:
                fid = f.get("id")
                if not isinstance(fid, str) or not fid:
                    continue
                if fid in seen_ids:
                    continue

                pt = point_from_feature(f)
                if pt is None:
                    continue

                # containsだと境界上を落とすので covers
                if not aoi_union.covers(pt):
                    continue

                # propertiesに正規化情報を付加（後段が楽）
                f2 = json.loads(json.dumps(f))  # deep copy
                f2["properties"] = normalize_feature_props(f2, args.base)

                seen_ids.add(fid)
                kept_features.append(f2)

                if args.debug_first and not printed_debug:
                    printed_debug = True
                    print("DEBUG first feature keys:", list(f2.keys()))
                    print("DEBUG first feature properties keys:", list((f2.get("properties") or {}).keys()))
                    print("DEBUG first feature links:", f2.get("links"))

        if raw_count_cell >= args.limit:
            stats["cells_hit_limit"] += 1

        if ci % 25 == 0 or ci == 1 or ci == len(cells):
            print(f"[{ci}/{len(cells)}] unique_points={len(kept_features)}")

        if args.sleep > 0:
            time.sleep(args.sleep)

    stats["unique_points_in_aoi"] = len(kept_features)

    # outputs
    out_points = out_dir / "panoramax_points_in_aoi.geojson"
    out_points.write_text(
        json.dumps({"type": "FeatureCollection", "features": kept_features}, ensure_ascii=False),
        encoding="utf-8",
    )

    out_stats = out_dir / "stats.json"
    out_stats.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print("---- DONE ----")
    print("Wrote:")
    print(" ", out_points)
    print(" ", out_stats)
    print("Stats:")
    for k in ["unique_points_in_aoi", "cells_hit_limit", "pages_followed", "errors"]:
        print(f"  {k}: {stats[k]}")
    if stats["cells_hit_limit"] > 0:
        print(f"WARNING: {stats['cells_hit_limit']} cells returned >= limit ({args.limit}).")
        print("  -> cell_deg を小さくする / limit を上げる と欠けが減ります")


if __name__ == "__main__":
    main()
