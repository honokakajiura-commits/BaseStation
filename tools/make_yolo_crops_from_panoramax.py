#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Panoramax から画像を取得しつつ、YOLO学習用に
front / left / right / back の4方向を切り抜くスクリプト

できること:
1. points GeoJSON を読む
2. 必要なら Panoramax から全天球画像を保存
3. aoi_index.jsonl を生成
4. yaw を自動推定
5. front / left / right / back を切り抜いて保存
  --order_mode sequence 

入力:
- --points: Panoramaxの撮影地点 FeatureCollection (GeoJSON)
- --run_dir: 出力先

python tools/make_yolo_crops_from_panoramax.py \
  --points runs/TMU_east/panoramax_points_in_aoi.geojson \
  --run_dir runs/toyota_yolo_dataset \\
  --meta_cache_jsonl runs/toyota_yolo_dataset/picture_meta_cache.jsonl \
  --pitch_cli 40 \
  --det_w 1280 \
  --det_h 1280 \
  --fov_front 105 \
  --fov_side 90 \
  --fov_back 105 \
  --skip_existing
"""

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set
from urllib.parse import urlparse

import cv2
import numpy as np
import requests


# -------------------------
# utils
# -------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def safe_str(x) -> str:
    return "" if x is None else str(x)

def append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def read_jsonl(path: Path) -> List[dict]:
    out: List[dict] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out

def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def load_featurecollection(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("type") != "FeatureCollection":
        raise ValueError("points は GeoJSON FeatureCollection である必要があります")
    if "features" not in data or not isinstance(data["features"], list):
        raise ValueError("points の features が見つかりません")
    return data

def get_feature_id(f: Dict[str, Any]) -> str:
    fid = f.get("id")
    if isinstance(fid, str) and fid:
        return fid
    props = f.get("properties") or {}
    for k in ["pic_id", "picId", "picture_id", "uuid", "id", "fid"]:
        v = props.get(k)
        if isinstance(v, str) and v:
            return v
    return "unknown"

def get_lonlat_from_feature(f: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    geom = f.get("geometry") or {}
    if isinstance(geom, dict) and geom.get("type") == "Point":
        coords = geom.get("coordinates")
        if isinstance(coords, (list, tuple)) and len(coords) >= 2:
            lon, lat = coords[0], coords[1]
            try:
                return float(lon), float(lat)
            except Exception:
                return None, None
    return None, None

def get_datetime_from_feature(f: Dict[str, Any]) -> str:
    props = f.get("properties") or {}
    for k in ["datetimetz", "datetime", "captured_at", "created", "timestamp"]:
        v = props.get(k)
        if isinstance(v, str) and v:
            return v
    return ""

def normalize_url(u: str) -> str:
    if not u:
        return ""
    u = u.strip()
    for sep in ["#", "?"]:
        if sep in u:
            u = u.split(sep, 1)[0]
    return u

def find_item_url_from_feature(f: Dict[str, Any]) -> Optional[str]:
    for key in ["links", "assets"]:
        v = f.get(key)
        if isinstance(v, list):
            for it in v:
                if isinstance(it, dict):
                    href = it.get("href")
                    if isinstance(href, str) and "/items/" in href:
                        return href

    links = f.get("links")
    if isinstance(links, list):
        for lk in links:
            if not isinstance(lk, dict):
                continue
            if lk.get("rel") == "self":
                href = lk.get("href")
                if isinstance(href, str) and href:
                    return href
        for lk in links:
            if not isinstance(lk, dict):
                continue
            href = lk.get("href")
            if isinstance(href, str) and "/items/" in href:
                return href

    props = f.get("properties") or {}
    for k in ["href", "item", "item_url", "url"]:
        href = props.get(k)
        if isinstance(href, str) and "/items/" in href:
            return href

    return None

def is_image_url(u: str) -> bool:
    low = safe_str(u).lower()
    return low.endswith(".jpg") or low.endswith(".jpeg") or low.endswith(".png") or low.endswith(".webp")

def choose_best_image_href_from_assets_dict(assets: Dict[str, Any]) -> Optional[str]:
    candidates: List[Tuple[int, str]] = []

    def score_asset(a: Dict[str, Any]) -> int:
        score = 0
        roles = a.get("roles") or []
        if isinstance(roles, list):
            if "visual" in roles:
                score += 20
            if "data" in roles:
                score += 10
            if "thumbnail" in roles:
                score += 1
        typ = safe_str(a.get("type")).lower()
        if "image/jpeg" in typ:
            score += 5
        if "image/webp" in typ:
            score += 4
        href = safe_str(a.get("href"))
        if "{z}" in href or "{x}" in href or "{y}" in href:
            score -= 100
        if is_image_url(href):
            score += 3
        return score

    for _, a in assets.items():
        if not isinstance(a, dict):
            continue
        href = a.get("href")
        if not isinstance(href, str) or not href:
            continue
        candidates.append((score_asset(a), href))

    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[0][1]

def find_direct_image_url_from_feature(f: Dict[str, Any]) -> Optional[str]:
    assets = f.get("assets")
    if isinstance(assets, dict):
        href = choose_best_image_href_from_assets_dict(assets)
        if href and is_image_url(href):
            return href

    if isinstance(assets, list):
        for it in assets:
            if not isinstance(it, dict):
                continue
            href = it.get("href")
            if isinstance(href, str) and href and is_image_url(href):
                return href

    links = f.get("links")
    if isinstance(links, list):
        for lk in links:
            if not isinstance(lk, dict):
                continue
            href = lk.get("href")
            if isinstance(href, str) and href and is_image_url(href):
                return href

    props = f.get("properties") or {}
    for k in ["image", "image_url", "img_url", "href", "url", "download", "original"]:
        v = props.get(k)
        if isinstance(v, str) and v and is_image_url(v):
            return v

    return None

def choose_best_asset_href(item: Dict[str, Any]) -> Optional[str]:
    assets = item.get("assets")
    if not isinstance(assets, dict):
        return None
    return choose_best_image_href_from_assets_dict(assets)

def request_get_with_retry(session: requests.Session, url: str, timeout: int, max_tries: int = 4) -> requests.Response:
    last_err: Optional[Exception] = None
    for i in range(max_tries):
        try:
            r = session.get(url, timeout=timeout)
            if 500 <= r.status_code <= 599:
                raise requests.HTTPError(f"HTTP {r.status_code} for {url}")
            r.raise_for_status()
            return r
        except Exception as e:
            last_err = e
            time.sleep(0.6 * (2 ** i))
    raise RuntimeError(f"GET failed after retries: {url} ({last_err})")

def infer_image_ext(img_url: str, content_type: str = "") -> str:
    path = urlparse(img_url).path.lower()
    suf = Path(path).suffix.lower()
    if suf in [".jpg", ".jpeg", ".png", ".webp"]:
        return suf

    content_type = safe_str(content_type).lower().split(";", 1)[0].strip()
    if content_type == "image/jpeg":
        return ".jpg"
    if content_type == "image/png":
        return ".png"
    if content_type == "image/webp":
        return ".webp"
    return ".jpg"

def download_image_bytes(session: requests.Session, img_url: str, timeout: int = 60) -> Tuple[bytes, str]:
    r = request_get_with_retry(session, img_url, timeout=timeout, max_tries=4)
    img_bytes = r.content
    ext = infer_image_ext(img_url, r.headers.get("Content-Type", ""))
    return img_bytes, ext

def resolve_image_url_via_item(
    session: requests.Session,
    item_url: str,
    timeout: int = 60,
    visited: Optional[Set[str]] = None,
    depth: int = 0,
    max_depth: int = 8
) -> Optional[str]:
    if visited is None:
        visited = set()
    if depth > max_depth or item_url in visited:
        return None
    visited.add(item_url)

    u = safe_str(item_url)
    if not u:
        return None
    if is_image_url(u):
        return u

    r = request_get_with_retry(session, u, timeout=timeout, max_tries=4)
    ct = safe_str(r.headers.get("content-type")).lower()
    if "image/" in ct:
        return u

    item = r.json()
    if isinstance(item, dict):
        href = choose_best_asset_href(item)
        if href:
            return href

        links = item.get("links")
        if isinstance(links, list):
            for lk in links:
                if not isinstance(lk, dict):
                    continue
                href2 = lk.get("href")
                if isinstance(href2, str) and href2 and ("/items/" in href2 or href2.endswith(".json")):
                    out = resolve_image_url_via_item(
                        session, href2, timeout=timeout, visited=visited, depth=depth + 1, max_depth=max_depth
                    )
                    if out:
                        return out
    return None

def find_pano_path(panos_dir: Path, fid: str) -> Optional[Path]:
    for ext in [".jpg", ".jpeg", ".png", ".webp"]:
        p = panos_dir / f"{fid}{ext}"
        if p.exists():
            return p
    return None

def find_all_pano_paths(panos_dir: Path, fid: str) -> List[Path]:
    out: List[Path] = []
    for ext in [".jpg", ".jpeg", ".png", ".webp"]:
        p = panos_dir / f"{fid}{ext}"
        if p.exists():
            out.append(p)
    return out


# -------------------------
# ordering
# -------------------------

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))

def fetch_picture_meta(session: requests.Session, api_base: str, fid: str, timeout: int = 30) -> Dict[str, Any]:
    url = f"{api_base.rstrip('/')}/api/pictures/{fid}"
    r = request_get_with_retry(session, url, timeout=timeout, max_tries=4)
    return r.json()

def extract_sequence_and_rank(item: Dict[str, Any]) -> Tuple[str, Optional[float]]:
    seq = str(item.get("collection") or "")
    props = item.get("properties") or {}
    rank = props.get("geovisio:rank_in_collection")
    try:
        rank_f = float(rank) if rank is not None else None
    except Exception:
        rank_f = None
    return seq, rank_f

def load_meta_cache_jsonl(path: Path) -> Dict[str, Dict[str, Any]]:
    cache: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return cache
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            fid = o.get("fid")
            if isinstance(fid, str) and fid:
                cache[fid] = o
    return cache

def append_meta_cache(path: Path, rec: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def order_sequences_by_nearest(groups: Dict[str, List[Dict[str, Any]]]) -> List[str]:
    seq_ids = list(groups.keys())
    if not seq_ids:
        return []

    def endpoints(seq: str) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
        feats = groups[seq]
        if not feats:
            return None, None
        lon0, lat0 = get_lonlat_from_feature(feats[0])
        lon1, lat1 = get_lonlat_from_feature(feats[-1])
        s = (lat0, lon0) if (lat0 is not None and lon0 is not None) else None
        e = (lat1, lon1) if (lat1 is not None and lon1 is not None) else None
        return s, e

    def sw_key(seq: str) -> Tuple[float, float]:
        s, e = endpoints(seq)
        p = s or e
        if p is None:
            return (1e9, 1e9)
        return (p[0], p[1])

    remaining = seq_ids[:]
    remaining.sort(key=sw_key)
    order = [remaining.pop(0)]

    while remaining:
        cur = order[-1]
        cur_s, cur_e = endpoints(cur)
        cur_p = cur_e or cur_s

        best_i = 0
        best_d = float("inf")
        for i, cand in enumerate(remaining):
            s, e = endpoints(cand)
            ds = de = float("inf")
            if cur_p and s:
                ds = haversine_m(cur_p[0], cur_p[1], s[0], s[1])
            if cur_p and e:
                de = haversine_m(cur_p[0], cur_p[1], e[0], e[1])
            d = min(ds, de)
            if d < best_d:
                best_d = d
                best_i = i

        order.append(remaining.pop(best_i))
    return order

def order_features_datetime(features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        features,
        key=lambda f: (
            get_datetime_from_feature(f) == "",
            get_datetime_from_feature(f),
            get_feature_id(f),
        ),
    )

def order_features_nearest(features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    feats = features[:]

    def latlon_key(f):
        lon, lat = get_lonlat_from_feature(f)
        return (lat if lat is not None else 999.0, lon if lon is not None else 999.0)

    feats.sort(key=latlon_key)
    if not feats:
        return feats

    ordered = [feats.pop(0)]
    while feats:
        lon0, lat0 = get_lonlat_from_feature(ordered[-1])
        if lon0 is None or lat0 is None:
            ordered.append(feats.pop(0))
            continue
        best_i, best_d = 0, float("inf")
        for i, f in enumerate(feats):
            lon1, lat1 = get_lonlat_from_feature(f)
            if lon1 is None or lat1 is None:
                continue
            d = haversine_m(lat0, lon0, lat1, lon1)
            if d < best_d:
                best_d, best_i = d, i
        ordered.append(feats.pop(best_i))
    return ordered

def order_features_sequence(
    session: requests.Session,
    features: List[Dict[str, Any]],
    api_base: str,
    timeout: int,
    sleep: float,
    cache_path: Optional[Path],
) -> List[Dict[str, Any]]:
    cache: Dict[str, Dict[str, Any]] = {}
    if cache_path:
        cache = load_meta_cache_jsonl(cache_path)

    enriched: List[Dict[str, Any]] = []
    for ft in features:
        fid = get_feature_id(ft)
        if fid == "unknown":
            ff = json.loads(json.dumps(ft))
            props = ff.get("properties") or {}
            props.setdefault("sequence_id", "")
            props.setdefault("rank_in_collection", None)
            ff["properties"] = props
            enriched.append(ff)
            continue

        if fid in cache:
            seq = safe_str(cache[fid].get("sequence_id"))
            rank = cache[fid].get("rank_in_collection", None)
            try:
                rank = float(rank) if rank is not None else None
            except Exception:
                rank = None
        else:
            try:
                item = fetch_picture_meta(session, api_base=api_base, fid=fid, timeout=timeout)
                seq, rank = extract_sequence_and_rank(item)
                rec = {"fid": fid, "sequence_id": seq, "rank_in_collection": rank}
                cache[fid] = rec
                if cache_path:
                    append_meta_cache(cache_path, rec)
            except Exception:
                seq, rank = "", None
            if sleep > 0:
                time.sleep(sleep)

        ff = json.loads(json.dumps(ft))
        props = ff.get("properties") or {}
        props["sequence_id"] = seq
        props["rank_in_collection"] = rank
        ff["properties"] = props
        enriched.append(ff)

    groups: Dict[str, List[Dict[str, Any]]] = {}
    for ft in enriched:
        seq = safe_str((ft.get("properties") or {}).get("sequence_id"))
        groups.setdefault(seq, []).append(ft)

    for seq, g in groups.items():
        def key_inside(f: Dict[str, Any]):
            p = f.get("properties") or {}
            r = p.get("rank_in_collection")
            try:
                rf = float(r) if r is not None else float("inf")
            except Exception:
                rf = float("inf")
            dt = get_datetime_from_feature(f)
            fid = get_feature_id(f)
            return (rf, dt, fid)
        g.sort(key=key_inside)

    seq_ids = list(groups.keys())
    non_empty = [s for s in seq_ids if s != ""]
    empty = [""] if "" in groups else []
    seq_order = order_sequences_by_nearest({k: groups[k] for k in non_empty}) + empty

    ordered: List[Dict[str, Any]] = []
    for s in seq_order:
        ordered.extend(groups.get(s, []))
    return ordered

def write_aoi_index_jsonl(out_path: Path, ordered_features: List[Dict[str, Any]]):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for ft in ordered_features:
            fid = get_feature_id(ft)
            lon, lat = get_lonlat_from_feature(ft)
            dt = get_datetime_from_feature(ft)
            props = ft.get("properties") or {}
            view_az = props.get("view:azimuth") or props.get("view_azimuth") or props.get("azimuth") or ""
            seq = props.get("sequence_id", "")
            rank = props.get("rank_in_collection", None)
            rec = {
                "fid": fid,
                "lon": lon,
                "lat": lat,
                "datetime": dt,
                "view_azimuth": view_az,
                "sequence_id": seq,
                "rank_in_collection": rank,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ----------------------------
# crop
# ----------------------------

def unique_path(dst: Path, overwrite: bool) -> Path:
    if overwrite or (not dst.exists()):
        return dst

    stem = dst.stem
    suf = dst.suffix
    parent = dst.parent
    for k in range(1, 10000):
        cand = parent / f"{stem}__v{k:03d}{suf}"
        if not cand.exists():
            return cand
    ts = int(time.time() * 1000)
    return parent / f"{stem}__v{ts}{suf}"

def _fmt_deg_tag(x: float, ndigits: int = 0) -> str:
    sign = "p" if x >= 0 else "m"
    ax = abs(float(x))
    if ndigits <= 0:
        return f"{sign}{int(round(ax))}"
    scale = 10 ** ndigits
    v = int(round(ax * scale))
    whole = v // scale
    frac = v % scale
    return f"{sign}{whole}p{frac}"

def build_crop_name(idx: int, fid: str, view: str, yaw: float, fov: float, ext: str = "jpg") -> str:
    return (
        f"{idx:05d}__{fid}__{view}"
        f"__yaw{_fmt_deg_tag(yaw)}"
        f"__fov{_fmt_deg_tag(fov)}.{ext}"
    )

def wrap_yaw_deg(y: float) -> float:
    return (y + 180.0) % 360.0 - 180.0

def equirectangular_to_perspective(
    img_bgr: np.ndarray,
    yaw_deg: float,
    pitch_deg: float,
    fov_deg: float,
    out_w: int,
    out_h: int,
) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    fov = math.radians(fov_deg)
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)

    fx = (out_w / 2) / math.tan(fov / 2)
    fy = fx
    cx = out_w / 2
    cy = out_h / 2

    xs = np.arange(out_w)
    ys = np.arange(out_h)
    xv, yv = np.meshgrid(xs, ys)

    x_cam = (xv - cx) / fx
    y_cam = -(yv - cy) / fy
    z_cam = np.ones_like(x_cam)

    norm = np.sqrt(x_cam**2 + y_cam**2 + z_cam**2)
    x_cam /= norm
    y_cam /= norm
    z_cam /= norm

    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    cos_p, sin_p = math.cos(pitch), math.sin(pitch)

    x1 = x_cam
    y1 = cos_p * y_cam - sin_p * z_cam
    z1 = sin_p * y_cam + cos_p * z_cam

    x2 = cos_y * x1 + sin_y * z1
    y2 = y1
    z2 = -sin_y * x1 + cos_y * z1

    lon = np.arctan2(x2, z2)
    lat = np.arcsin(np.clip(y2, -1.0, 1.0))

    u = (lon / (2 * math.pi) + 0.5) * w
    v = (0.5 - lat / math.pi) * h

    return cv2.remap(
        img_bgr,
        u.astype(np.float32),
        v.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP,
    )


# ----------------------------
# yaw推定
# ----------------------------

def intersect_lines(l1, l2) -> Optional[Tuple[float, float]]:
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-6:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
    return (px, py)

def estimate_vanishing_point_x(
    img_bgr: np.ndarray,
    roi_top_ratio: float = 0.10,
    roi_bottom_ratio: float = 0.95,
) -> Tuple[Optional[float], int]:
    h, w = img_bgr.shape[:2]
    y0 = int(h * roi_top_ratio)
    y1 = int(h * roi_bottom_ratio)
    roi = img_bgr[y0:y1, :]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 60, 160)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=70,
        minLineLength=60,
        maxLineGap=10
    )
    if lines is None:
        return None, 0

    segs: List[Tuple[int, int, int, int]] = []
    for (x1, y1_, x2, y2_) in lines[:, 0]:
        dx = x2 - x1
        dy = y2_ - y1_
        length = math.hypot(dx, dy)
        if length < 60:
            continue
        ang = abs(math.degrees(math.atan2(dy, dx)))
        if ang < 15 or ang > 88:
            continue
        segs.append((x1, y1_, x2, y2_))

    if len(segs) < 6:
        return None, len(segs)

    inter_x = []
    cap = min(len(segs), 90)
    for i in range(cap):
        for j in range(i + 1, cap):
            p = intersect_lines(segs[i], segs[j])
            if p is None:
                continue
            px, py = p
            if px < -w or px > 2 * w or py < -h or py > 2 * h:
                continue
            inter_x.append(px)

    if len(inter_x) < 12:
        return None, len(segs)

    return float(np.median(np.array(inter_x))), len(segs)

def vp_x_to_yaw_offset_deg(vp_x: float, out_w: int, fov_deg: float) -> float:
    cx = out_w / 2.0
    fov = math.radians(fov_deg)
    fx = (out_w / 2) / math.tan(fov / 2)
    ang = math.atan((vp_x - cx) / fx)
    return math.degrees(ang)

def yaw_from_view_azimuth(az: Any, default_if_missing: float = 0.0) -> float:
    try:
        az = float(az)
    except Exception:
        az = float(default_if_missing)
    return ((az + 180.0) % 360.0) - 180.0

def estimate_yaw_center_auto(
    pano_bgr: np.ndarray,
    pitch_deg: float,
    view_azimuth: Any,
    fov_preview: float = 110.0,
    out_w: int = 1024,
    out_h: int = 768,
) -> Tuple[float, str, dict]:
    yaw0 = 0.0
    persp = equirectangular_to_perspective(
        pano_bgr,
        yaw_deg=yaw0,
        pitch_deg=pitch_deg,
        fov_deg=fov_preview,
        out_w=out_w,
        out_h=out_h
    )
    vp_x, n_lines = estimate_vanishing_point_x(persp)
    if vp_x is not None:
        yaw_off = vp_x_to_yaw_offset_deg(vp_x, out_w=out_w, fov_deg=fov_preview)
        yaw_center = wrap_yaw_deg(yaw0 + yaw_off)
        meta = {
            "vp_x": float(vp_x),
            "n_lines": int(n_lines),
            "fov_preview": float(fov_preview),
            "out_w": out_w,
            "out_h": out_h
        }
        return float(yaw_center), "vanishing_point", meta

    if view_azimuth is not None and safe_str(view_azimuth) != "":
        yaw_center = yaw_from_view_azimuth(view_azimuth, default_if_missing=0.0)
        meta = {"n_lines": int(n_lines)}
        return float(yaw_center), "view_azimuth", meta

    return 0.0, "fallback_zero", {"n_lines": int(n_lines)}


# ----------------------------
# main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", required=True, help="Panoramax points GeoJSON (FeatureCollection)")
    ap.add_argument("--run_dir", required=True)

    ap.add_argument("--order_mode", choices=["datetime", "nearest", "sequence"], default="sequence")
    ap.add_argument("--api_base", default="https://api.panoramax.xyz")
    ap.add_argument("--meta_cache_jsonl", default="")

    ap.add_argument("--pitch_cli", type=float, required=True, help="CLI pitch (up is negative)")
    ap.add_argument("--det_w", type=int, default=1280)
    ap.add_argument("--det_h", type=int, default=1280)
    ap.add_argument("--fov_front", type=float, default=105.0)
    ap.add_argument("--fov_side", type=float, default=90.0)
    ap.add_argument("--fov_back", type=float, default=105.0)

    ap.add_argument("--yaw_preview_fov", type=float, default=110.0)
    ap.add_argument("--yaw_preview_w", type=int, default=1024)
    ap.add_argument("--yaw_preview_h", type=int, default=768)

    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=0.02)
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--crop_format", choices=["jpg", "png"], default="jpg")
    ap.add_argument("--crop_jpeg_quality", type=int, default=95)

    args = ap.parse_args()

    points_path = Path(args.points)
    run_dir = Path(args.run_dir)
    ensure_dir(run_dir)

    panos_dir = run_dir / "panos"
    crops_dir = run_dir / "crops"
    aoi_index_path = run_dir / "aoi_index.jsonl"
    yaw_map_path = run_dir / "yaw_map.jsonl"
    crop_log_path = run_dir / "crop_log.jsonl"
    summary_path = run_dir / "summary.json"

    for d in [panos_dir, crops_dir]:
        ensure_dir(d)

    if args.overwrite:
        for p in [aoi_index_path, yaw_map_path, crop_log_path]:
            if p.exists():
                p.unlink()

    fc = load_featurecollection(points_path)
    feats: List[Dict[str, Any]] = fc["features"]

    session = requests.Session()
    session.headers.update({"User-Agent": "panoramax-fetch-and-crop/1.0"})

    cache_path = Path(args.meta_cache_jsonl) if args.meta_cache_jsonl else None
    if args.order_mode == "sequence":
        feats = order_features_sequence(
            session=session,
            features=feats,
            api_base=args.api_base,
            timeout=min(args.timeout, 40),
            sleep=max(0.0, args.sleep),
            cache_path=cache_path,
        )
    elif args.order_mode == "datetime":
        feats = order_features_datetime(feats)
    else:
        feats = order_features_nearest(feats)

    if args.limit and args.limit > 0:
        feats = feats[:args.limit]

    write_aoi_index_jsonl(aoi_index_path, feats)

    ok_dl = fail_dl = skip_dl = 0
    for f in feats:
        fid = get_feature_id(f)
        if fid == "unknown":
            fail_dl += 1
            continue

        existing_pano_path = find_pano_path(panos_dir, fid)
        if args.skip_existing and existing_pano_path and existing_pano_path.stat().st_size > 20_000:
            skip_dl += 1
            continue

        direct_img_url = find_direct_image_url_from_feature(f)
        direct_img_url = normalize_url(direct_img_url) if direct_img_url else ""

        item_url = find_item_url_from_feature(f)
        item_url = normalize_url(item_url) if item_url else ""

        try:
            if direct_img_url:
                img_url = direct_img_url
            else:
                if not item_url:
                    raise RuntimeError("no direct image url and no item url")
                resolved = resolve_image_url_via_item(session, item_url, timeout=args.timeout)
                if not resolved:
                    raise RuntimeError("cannot resolve image url via item")
                img_url = normalize_url(resolved)

            img_bytes, pano_ext = download_image_bytes(session, img_url, timeout=args.timeout)
            pano_path = panos_dir / f"{fid}{pano_ext}"
            for old_path in find_all_pano_paths(panos_dir, fid):
                if old_path != pano_path:
                    old_path.unlink()
            pano_path.write_bytes(img_bytes)
            ok_dl += 1

            if args.sleep > 0:
                time.sleep(args.sleep)

        except Exception as e:
            fail_dl += 1
            print(f"[download fail] fid={fid}: {e}")

    pitch_deg = -float(args.pitch_cli)
    index_recs = read_jsonl(aoi_index_path)

    yaw_map = {}
    for r in index_recs:
        fid = r["fid"]

        pano_path = find_pano_path(panos_dir, fid)
        if pano_path is None:
            rec = {"fid": fid, "yaw_center": 0.0, "yaw_reason": "missing_pano"}
            append_jsonl(yaw_map_path, rec)
            yaw_map[fid] = 0.0
            continue

        pano = cv2.imread(str(pano_path))
        if pano is None:
            rec = {"fid": fid, "yaw_center": 0.0, "yaw_reason": "imread_failed"}
            append_jsonl(yaw_map_path, rec)
            yaw_map[fid] = 0.0
            continue

        yaw_center, reason, meta = estimate_yaw_center_auto(
            pano,
            pitch_deg=pitch_deg,
            view_azimuth=r.get("view_azimuth"),
            fov_preview=args.yaw_preview_fov,
            out_w=args.yaw_preview_w,
            out_h=args.yaw_preview_h,
        )
        rec = {"fid": fid, "yaw_center": float(yaw_center), "yaw_reason": reason, **meta}
        append_jsonl(yaw_map_path, rec)
        yaw_map[fid] = float(yaw_center)

    total_panos = 0
    total_crops = 0
    fail_missing = 0
    fail_imread = 0

    for i, r in enumerate(index_recs, start=1):
        fid = r["fid"]
        pano_path = find_pano_path(panos_dir, fid)
        if pano_path is None:
            fail_missing += 1
            append_jsonl(crop_log_path, {
                "step": "pano",
                "i": i,
                "fid": fid,
                "status": "fail",
                "reason": "missing_pano"
            })
            continue

        pano = cv2.imread(str(pano_path))
        if pano is None:
            fail_imread += 1
            append_jsonl(crop_log_path, {
                "step": "pano",
                "i": i,
                "fid": fid,
                "status": "fail",
                "reason": "imread_failed"
            })
            continue

        total_panos += 1
        yaw_center = float(yaw_map.get(fid, 0.0))

        views = [
            ("front", 0.0, args.fov_front),
            ("left", -90.0, args.fov_side),
            ("right", 90.0, args.fov_side),
            ("back", 180.0, args.fov_back),
        ]

        for view_name, yaw_off, fov in views:
            yaw = wrap_yaw_deg(yaw_center + yaw_off)

            crop = equirectangular_to_perspective(
                pano,
                yaw_deg=yaw,
                pitch_deg=pitch_deg,
                fov_deg=fov,
                out_w=args.det_w,
                out_h=args.det_h,
            )

            crop_name = build_crop_name(
                idx=i,
                fid=fid,
                view=view_name,
                yaw=yaw,
                fov=fov,
                ext=args.crop_format,
            )
            crop_path = unique_path(crops_dir / crop_name, overwrite=args.overwrite)
            if args.crop_format == "png":
                ok = cv2.imwrite(str(crop_path), crop)
            else:
                ok = cv2.imwrite(
                    str(crop_path),
                    crop,
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(args.crop_jpeg_quality)],
                )

            append_jsonl(crop_log_path, {
                "step": "crop",
                "i": i,
                "fid": fid,
                "view": view_name,
                "yaw_center": yaw_center,
                "yaw": float(yaw),
                "yaw_off": float(yaw_off),
                "pitch_cli": float(args.pitch_cli),
                "pitch_deg": float(pitch_deg),
                "fov": float(fov),
                "crop_path": str(crop_path),
                "status": "ok" if ok else "fail",
                "sequence_id": r.get("sequence_id", ""),
                "rank_in_collection": r.get("rank_in_collection", None),
            })

            if ok:
                total_crops += 1

    summary = {
        "points": str(points_path),
        "aoi_index": str(aoi_index_path),
        "panos_dir": str(panos_dir),
        "processed_panos": total_panos,
        "total_crops": total_crops,
        "download_ok": ok_dl,
        "download_skip": skip_dl,
        "download_fail": fail_dl,
        "fail_missing_pano": fail_missing,
        "fail_imread": fail_imread,
        "params": {
            "order_mode": args.order_mode,
            "pitch_cli": float(args.pitch_cli),
            "pitch_deg": float(pitch_deg),
            "det_w": int(args.det_w),
            "det_h": int(args.det_h),
            "crop_format": args.crop_format,
            "crop_jpeg_quality": int(args.crop_jpeg_quality),
            "fov_front": float(args.fov_front),
            "fov_side": float(args.fov_side),
            "fov_back": float(args.fov_back),
            "yaw_preview_fov": float(args.yaw_preview_fov),
            "yaw_preview_w": int(args.yaw_preview_w),
            "yaw_preview_h": int(args.yaw_preview_h),
            "api_base": args.api_base,
        },
        "paths": {
            "run_dir": str(run_dir),
            "panos": str(panos_dir),
            "crops": str(crops_dir),
            "aoi_index": str(aoi_index_path),
            "yaw_map": str(yaw_map_path),
            "crop_log": str(crop_log_path),
        }
    }
    save_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
