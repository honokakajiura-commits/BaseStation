#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import io
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

import requests
from PIL import Image
from PIL import ImageOps


# crop(投影)で numpy を使う
import numpy as np


# -------------------------
# utils
# -------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def safe_str(x) -> str:
    return "" if x is None else str(x)


def load_featurecollection(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("type") != "FeatureCollection":
        raise ValueError("points は GeoJSON FeatureCollection である必要があります")
    if "features" not in data or not isinstance(data["features"], list):
        raise ValueError("points の features が見つかりません")
    return data


def get_feature_id(f: Dict[str, Any]) -> str:
    # /api/search の feature は id が pic_id と同じことが多い
    fid = f.get("id")
    if isinstance(fid, str) and fid:
        return fid
    props = f.get("properties") or {}
    for k in ["pic_id", "picId", "picture_id", "uuid", "id"]:
        v = props.get(k)
        if isinstance(v, str) and v:
            return v
    # fallback
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


def normalize_url(u: str) -> str:
    """
    URLキーのブレを減らす（クエリ/フラグメント除去）
    """
    if not u:
        return ""
    u = u.strip()
    for sep in ["#", "?"]:
        if sep in u:
            u = u.split(sep, 1)[0]
    return u


def find_item_url_from_feature(f: Dict[str, Any]) -> Optional[str]:
    """
    feature から item(JSON) のURLを拾う
    - links の rel=self が item であることが多い
    - href に /items/ が含まれるものも優先
    """
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


def get_pic_type_from_feature(f: Dict[str, Any]) -> str:
    props = f.get("properties") or {}
    for k in ["pic_type", "picType", "type", "picture_type"]:
        v = props.get(k)
        if isinstance(v, str) and v:
            return v.lower()
    return ""


def is_image_url(u: str) -> bool:
    low = safe_str(u).lower()
    return low.endswith(".jpg") or low.endswith(".jpeg") or low.endswith(".png") or low.endswith(".webp")


def choose_best_image_href_from_assets_dict(assets: Dict[str, Any]) -> Optional[str]:
    """
    assets(dict) から画像hrefを選ぶ（featureでもitemでも使える）
    """
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
        # 明らかに画像拡張子なら加点
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
    best = candidates[0][1]
    return best if best else None


def find_direct_image_url_from_feature(f: Dict[str, Any]) -> Optional[str]:
    """
    item API を叩かず、feature の中だけから直接画像URLを探す（最優先）
    """
    # 1) feature.assets が dict(STAC風)なら
    assets = f.get("assets")
    if isinstance(assets, dict):
        href = choose_best_image_href_from_assets_dict(assets)
        if href and is_image_url(href):
            return href
        # tileテンプレ等除外後に残ったhrefでも、画像拡張子じゃないことがあるのでその場合は保留

    # 2) assets が list の場合
    if isinstance(assets, list):
        for it in assets:
            if not isinstance(it, dict):
                continue
            href = it.get("href")
            if isinstance(href, str) and href and is_image_url(href):
                return href

    # 3) links の中に直接画像URLが入っている場合
    links = f.get("links")
    if isinstance(links, list):
        for lk in links:
            if not isinstance(lk, dict):
                continue
            href = lk.get("href")
            if isinstance(href, str) and href and is_image_url(href):
                return href

    # 4) properties 直書き
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


def request_get_with_retry(session: requests.Session, url: str, timeout: int, max_tries: int = 3) -> requests.Response:
    last_err: Optional[Exception] = None
    for i in range(max_tries):
        try:
            r = session.get(url, timeout=timeout)
            # 5xx はリトライしたい（ただし 4xx は基本即死）
            if 500 <= r.status_code <= 599:
                raise requests.HTTPError(f"HTTP {r.status_code} for {url}")
            r.raise_for_status()
            return r
        except Exception as e:
            last_err = e
            # backoff: 0.5, 1.0, 2.0...
            time.sleep(0.5 * (2 ** i))
    raise RuntimeError(f"GET failed after retries: {url} ({last_err})")


def resolve_image_url_via_item(session: requests.Session, item_url: str, timeout: int = 60,
                              visited: Optional[Set[str]] = None, depth: int = 0, max_depth: int = 8) -> Optional[str]:
    """
    item(JSON) を GET して assets から画像hrefを返す（fallback）
    links を少しだけ辿る保険付き
    """
    if visited is None:
        visited = set()
    if depth > max_depth:
        return None
    if item_url in visited:
        return None
    visited.add(item_url)

    u = safe_str(item_url)
    if not u:
        return None

    if is_image_url(u):
        return u

    r = request_get_with_retry(session, u, timeout=timeout, max_tries=3)

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
                if isinstance(href2, str) and href2:
                    if "/items/" in href2 or href2.endswith(".json"):
                        out = resolve_image_url_via_item(
                            session, href2, timeout=timeout, visited=visited, depth=depth + 1, max_depth=max_depth
                        )
                        if out:
                            return out

    return None


def download_image(session: requests.Session, img_url: str, timeout: int = 60) -> Image.Image:
    r = request_get_with_retry(session, img_url, timeout=timeout, max_tries=3)
    img = Image.open(io.BytesIO(r.content))
    img = ImageOps.exif_transpose(img).convert("RGB")
    return img


def infer_kind_from_image(img: Image.Image) -> str:
    w, h = img.size
    if h <= 0:
        return "unknown"
    ratio = w / h
    if 1.85 <= ratio <= 2.20:
        return "equirectangular"
    if ratio >= 2.20:
        return "panoramic"
    return "flat"


def normalize_kind(from_feature: str, inferred: str) -> str:
    s = safe_str(from_feature).lower()
    if "equirect" in s or "360" in s:
        return "equirectangular"
    if "panor" in s:
        return "panoramic"
    if "flat" in s or "photo" in s or "image" in s:
        return "flat"
    return inferred if inferred else "unknown"


def make_point_key_for_join(f: Dict[str, Any], item_url: str, fid: str, lon: Optional[float], lat: Optional[float]) -> str:
    """
    ArcGIS統合用の join キー（安定性重視）
    1) item_url（恒久寄り） 2) fid(pic_id相当) 3) lonlat丸め
    """
    iu = normalize_url(item_url)
    if iu:
        return f"item:{iu}"
    if fid and fid != "unknown":
        return f"pic:{fid}"
    if lon is not None and lat is not None:
        return f"ll:{lon:.7f},{lat:.7f}"
    return "unknown"


def write_manifest_csv(path: Path, rows: List[Dict[str, Any]]):
    ensure_dir(path.parent)
    preferred = [
        "run_id", "idx", "fid", "point_key",
        "lon", "lat",
        "status", "error",
        "kind", "url_source",
        "item_url", "img_url",
        "yaw", "pitch", "fov",
        "crop_w", "crop_h",
        "src_w", "src_h", "src_ratio",
        "pano_relpath", "crop_relpath",
    ]
    keys: Set[str] = set()
    for r in rows:
        keys.update(r.keys())
    fieldnames = [k for k in preferred if k in keys] + [k for k in sorted(keys) if k not in preferred]
    if not fieldnames:
        fieldnames = preferred

    with path.open("w", encoding="utf-8", newline="") as fw:
        w = csv.DictWriter(fw, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: safe_str(r.get(k)) for k in fieldnames})


def build_points_all_geojson(points_fc: Dict[str, Any], manifest_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    AOI内全点に、manifest（crop単位ログ）を突き合わせて downloaded/n_crops 等を付与
    ※ rerunで二重計上しないよう crop_relpath でユニーク集計
    """
    agg: Dict[str, Dict[str, Any]] = {}
    for r in manifest_rows:
        pk = safe_str(r.get("point_key"))
        if not pk:
            continue
        a = agg.setdefault(pk, {
            "downloaded": 0,
            "crop_paths": set(),
            "kinds": set(),
            "yaw_set": set(),
            "img_url": "",
            "item_url": "",
            "url_source": "",
            "last_error": "",
        })

        status = safe_str(r.get("status"))
        if status == "ok" or status == "cached":
            a["downloaded"] = 1
            cr = safe_str(r.get("crop_relpath"))
            if cr:
                a["crop_paths"].add(cr)
            k = safe_str(r.get("kind"))
            if k:
                a["kinds"].add(k)
            yaw = safe_str(r.get("yaw"))
            if yaw != "":
                a["yaw_set"].add(yaw)
            if safe_str(r.get("img_url")):
                a["img_url"] = safe_str(r.get("img_url"))
            if safe_str(r.get("item_url")):
                a["item_url"] = safe_str(r.get("item_url"))
            if safe_str(r.get("url_source")):
                a["url_source"] = safe_str(r.get("url_source"))
        else:
            err = safe_str(r.get("error"))
            if err:
                a["last_error"] = err

    out_fc = {"type": "FeatureCollection", "features": []}
    for f in points_fc.get("features", []):
        ff = json.loads(json.dumps(f))
        props = ff.get("properties") or {}

        fid = get_feature_id(ff)
        lon, lat = get_lonlat_from_feature(ff)

        # points側に item_url があるとは限らないので、propertiesから拾える範囲は拾う
        item_url = safe_str(props.get("item_url") or props.get("item") or props.get("href") or "")
        pk = make_point_key_for_join(ff, item_url=item_url, fid=fid, lon=lon, lat=lat)
        props.setdefault("point_key", pk)

        props.setdefault("downloaded", 0)
        props.setdefault("n_crops", 0)
        props.setdefault("kind", "")
        props.setdefault("img_url", "")
        props.setdefault("item_url", item_url)
        props.setdefault("url_source", "")
        props.setdefault("yaw_list", "")
        props.setdefault("last_error", "")

        a = agg.get(pk)
        if a:
            props["downloaded"] = int(a["downloaded"])
            props["n_crops"] = int(len(a["crop_paths"]))
            kinds = list(a["kinds"])
            if kinds:
                props["kind"] = kinds[0]
            if a.get("img_url"):
                props["img_url"] = a["img_url"]
            if a.get("item_url"):
                props["item_url"] = a["item_url"]
            if a.get("url_source"):
                props["url_source"] = a["url_source"]
            if a.get("yaw_set"):
                try:
                    props["yaw_list"] = ",".join(sorted(a["yaw_set"], key=lambda x: float(x)))
                except Exception:
                    props["yaw_list"] = ",".join(sorted(a["yaw_set"]))
            if a.get("last_error"):
                props["last_error"] = a["last_error"]

        ff["properties"] = props
        out_fc["features"].append(ff)

    return out_fc


# -------------------------
# projection crop (equirectangular/panoramic)
# -------------------------

def equirectangular_to_perspective(
    img: Image.Image,
    yaw_deg: float,
    pitch_deg: float,
    fov_deg: float,
    out_w: int,
    out_h: int,
    flip_v: bool = False,
) -> Image.Image:
    w, h = img.size
    img_np = np.array(img)

    fov = math.radians(fov_deg)
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)

    xs = (np.linspace(0, out_w - 1, out_w) / (out_w - 1)) * 2 - 1
    ys = (np.linspace(0, out_h - 1, out_h) / (out_h - 1)) * 2 - 1
    xv, yv = np.meshgrid(xs, ys)

    if flip_v:
        yv = -yv

    x = xv * math.tan(fov / 2)
    y = -yv * math.tan(fov / 2)
    z = np.ones_like(x)

    norm = np.sqrt(x * x + y * y + z * z)
    x /= norm
    y /= norm
    z /= norm

    cp, sp = math.cos(pitch), math.sin(pitch)
    y2 = y * cp - z * sp
    z2 = y * sp + z * cp
    x2 = x

    cy, sy = math.cos(yaw), math.sin(yaw)
    x3 = x2 * cy + z2 * sy
    z3 = -x2 * sy + z2 * cy
    y3 = y2

    lon = np.arctan2(x3, z3)
    lat = np.arcsin(y3)

    uf = (lon / (2 * math.pi) + 0.5) * (w - 1)
    vf = (0.5 - lat / math.pi) * (h - 1)

    u0 = np.floor(uf).astype(np.int32)
    v0 = np.floor(vf).astype(np.int32)
    u1 = (u0 + 1) % w
    v1 = np.clip(v0 + 1, 0, h - 1)

    du = (uf - u0)[..., None]
    dv = (vf - v0)[..., None]

    p00 = img_np[v0, u0]
    p10 = img_np[v0, u1]
    p01 = img_np[v1, u0]
    p11 = img_np[v1, u1]

    out = (1 - du) * (1 - dv) * p00 + du * (1 - dv) * p10 + (1 - du) * dv * p01 + du * dv * p11
    out = np.clip(out, 0, 255).astype(np.uint8)

    return Image.fromarray(out)


def make_equirect_like_for_panoramic(img: Image.Image) -> Image.Image:
    w, h = img.size
    target_h = max(h, int(w / 2))
    if target_h == h:
        return img
    canvas = Image.new("RGB", (w, target_h), (0, 0, 0))
    top = (target_h - h) // 2
    canvas.paste(img, (0, top))
    return canvas


# -------------------------
# main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", required=True, help="input points geojson (FeatureCollection)")
    ap.add_argument("--out_dir", required=True, help="output directory")
    ap.add_argument("--aoi", default="", help="(optional) AOI geojson path to copy into run dir")

    ap.add_argument("--every", type=int, default=1, help="process every N points (1=all)")
    ap.add_argument("--yaw", default="0", help='comma-separated yaws (default "0")')
    # あなたの現状安定設定へ寄せる（上向きはマイナス）
    ap.add_argument("--pitch", type=float, default=-40.0, help="pitch in degrees (negative = up)")
    ap.add_argument("--fov", type=float, default=90.0, help="fov in degrees (wider front crop)")
    ap.add_argument("--crop_w", type=int, default=1024)
    ap.add_argument("--crop_h", type=int, default=768)

    ap.add_argument("--limit", type=int, default=0, help="max selected points to process (0=no limit)")
    ap.add_argument("--sleep", type=float, default=0.0, help="sleep seconds between requests")
    ap.add_argument("--timeout", type=int, default=60, help="request timeout seconds")

    ap.add_argument("--no_crops", action="store_true", help="disable crops output")
    ap.add_argument("--flip_v", action="store_true", help="flip vertical mapping (fix upside-down if needed)")

    # 運用系
    ap.add_argument("--skip_existing", action="store_true", help="skip if pano/crop already exists (resume-friendly)")
    ap.add_argument("--max_retries", type=int, default=3, help="max retries for HTTP GET")

    args = ap.parse_args()

    points_path = Path(args.points)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    run_id = out_dir.name

    # 出力ディレクトリ構成
    panos_dir = out_dir / "panos"
    crops_dir = out_dir / "crops"
    ensure_dir(panos_dir)
    ensure_dir(crops_dir)

    # 再現性: AOI を保存（任意）
    if args.aoi:
        aoi_path = Path(args.aoi)
        if aoi_path.exists():
            (out_dir / "aoi.geojson").write_text(aoi_path.read_text(encoding="utf-8"), encoding="utf-8")

    # yaw parse（ただし最初は正面のみ推奨）
    yaws: List[float] = []
    for s in safe_str(args.yaw).split(","):
        s = s.strip()
        if not s:
            continue
        yaws.append(float(s))
    if not yaws:
        yaws = [0.0]

    fc = load_featurecollection(points_path)
    feats: List[Dict[str, Any]] = fc["features"]
    input_points = len(feats)

    every = max(1, int(args.every))
    selected: List[Dict[str, Any]] = feats[::every]
    if args.limit and args.limit > 0:
        selected = selected[: args.limit]

    print(f"Input points: {input_points}")
    print(f"Selected (every={every}): {len(selected)}")
    print("----")

    session = requests.Session()
    session.headers.update({"User-Agent": "panoramax-fetch-images/2.0"})

    stats = {
        "run_id": run_id,
        "input_points": input_points,
        "selected_points": len(selected),
        "every": every,
        "panos_ok": 0,
        "panos_fail": 0,
        "crops_ok": 0,
        "crops_fail": 0,
        "crops_skip": 0,
        "cache_hits_pano": 0,
        "cache_hits_crop": 0,
        "kinds": {"equirectangular": 0, "panoramic": 0, "flat": 0, "unknown": 0},
        "params": {
            "yaws": yaws,
            "pitch": args.pitch,
            "fov": args.fov,
            "crop_w": args.crop_w,
            "crop_h": args.crop_h,
        }
    }

    downloaded_features: List[Dict[str, Any]] = []
    manifest_rows: List[Dict[str, Any]] = []

    try:
        for idx, f in enumerate(selected, start=1):
            fid = get_feature_id(f)
            lon, lat = get_lonlat_from_feature(f)

            # まず feature から直画像URLを探す（最優先）
            direct_img_url = find_direct_image_url_from_feature(f)
            direct_img_url = normalize_url(direct_img_url) if direct_img_url else None

            # item_url は fallback 用
            item_url = find_item_url_from_feature(f)
            item_url = normalize_url(item_url) if item_url else ""

            # joinキー（item_urlがあれば強い）
            point_key = make_point_key_for_join(f, item_url=item_url, fid=fid, lon=lon, lat=lat)

            pano_name = f"{fid}.jpg"
            pano_path = panos_dir / pano_name
            pano_rel = str(pano_path.relative_to(out_dir))

            try:
                img_url = ""
                url_source = ""

                # 1) 既にローカルがあれば、それを使う（再開対応）
                if args.skip_existing and pano_path.exists():
                    img = Image.open(pano_path)
                    img = ImageOps.exif_transpose(img).convert("RGB")
                    stats["cache_hits_pano"] += 1
                    url_source = "cached"
                    img_url = ""
                else:
                    # 2) 直URLがあれば最優先
                    if direct_img_url:
                        img_url = direct_img_url
                        url_source = "feature_direct"
                    else:
                        # 3) なければ item fallback
                        if not item_url:
                            raise RuntimeError("no direct image url in feature and no item url")
                        resolved = resolve_image_url_via_item(session, item_url, timeout=args.timeout)
                        if not resolved:
                            raise RuntimeError("cannot resolve image url via item (no assets.href)")
                        img_url = normalize_url(resolved)
                        url_source = "item_fallback"

                    # download
                    img = download_image(session, img_url, timeout=args.timeout)

                    # save original
                    img.save(pano_path, quality=92)
                    stats["panos_ok"] += 1

                inferred = infer_kind_from_image(img)
                kind = normalize_kind(get_pic_type_from_feature(f), inferred)
                stats["kinds"][kind if kind in stats["kinds"] else "unknown"] += 1

                src_w, src_h = img.size
                src_ratio = (src_w / src_h) if src_h else ""
                crop_written = 0

                if args.no_crops:
                    stats["crops_skip"] += 1
                else:
                    if kind in ("equirectangular", "panoramic"):
                        img_for_proj = img if kind == "equirectangular" else make_equirect_like_for_panoramic(img)

                        for yaw in yaws:
                            out_name = (
                                f"{fid}__yaw{int(yaw)}_pitch{int(args.pitch)}_fov{int(args.fov)}_"
                                f"{args.crop_w}x{args.crop_h}.jpg"
                            )
                            crop_path = crops_dir / out_name
                            crop_rel = str(crop_path.relative_to(out_dir))

                            # cropが既にあるならスキップ（再開）
                            if args.skip_existing and crop_path.exists():
                                stats["cache_hits_crop"] += 1
                                stats["crops_ok"] += 1
                                crop_written += 1
                                manifest_rows.append({
                                    "run_id": run_id,
                                    "idx": idx,
                                    "fid": fid,
                                    "point_key": point_key,
                                    "lon": lon,
                                    "lat": lat,
                                    "status": "cached",
                                    "error": "",
                                    "kind": kind,
                                    "url_source": url_source,
                                    "item_url": item_url,
                                    "img_url": img_url,
                                    "yaw": yaw,
                                    "pitch": args.pitch,
                                    "fov": args.fov,
                                    "crop_w": args.crop_w,
                                    "crop_h": args.crop_h,
                                    "src_w": src_w,
                                    "src_h": src_h,
                                    "src_ratio": src_ratio,
                                    "pano_relpath": pano_rel,
                                    "crop_relpath": crop_rel,
                                })
                                continue

                            try:
                                crop = equirectangular_to_perspective(
                                    img_for_proj,
                                    yaw_deg=yaw,
                                    pitch_deg=args.pitch,
                                    fov_deg=args.fov,
                                    out_w=args.crop_w,
                                    out_h=args.crop_h,
                                    flip_v=args.flip_v,
                                )
                                crop.save(crop_path, quality=92)
                                stats["crops_ok"] += 1
                                crop_written += 1

                                manifest_rows.append({
                                    "run_id": run_id,
                                    "idx": idx,
                                    "fid": fid,
                                    "point_key": point_key,
                                    "lon": lon,
                                    "lat": lat,
                                    "status": "ok",
                                    "error": "",
                                    "kind": kind,
                                    "url_source": url_source,
                                    "item_url": item_url,
                                    "img_url": img_url,
                                    "yaw": yaw,
                                    "pitch": args.pitch,
                                    "fov": args.fov,
                                    "crop_w": args.crop_w,
                                    "crop_h": args.crop_h,
                                    "src_w": src_w,
                                    "src_h": src_h,
                                    "src_ratio": src_ratio,
                                    "pano_relpath": pano_rel,
                                    "crop_relpath": crop_rel,
                                })

                            except Exception as e:
                                stats["crops_fail"] += 1
                                manifest_rows.append({
                                    "run_id": run_id,
                                    "idx": idx,
                                    "fid": fid,
                                    "point_key": point_key,
                                    "lon": lon,
                                    "lat": lat,
                                    "status": "failed",
                                    "error": str(e),
                                    "kind": kind,
                                    "url_source": url_source,
                                    "item_url": item_url,
                                    "img_url": img_url,
                                    "yaw": yaw,
                                    "pitch": args.pitch,
                                    "fov": args.fov,
                                    "crop_w": args.crop_w,
                                    "crop_h": args.crop_h,
                                    "src_w": src_w,
                                    "src_h": src_h,
                                    "src_ratio": src_ratio,
                                    "pano_relpath": pano_rel,
                                    "crop_relpath": crop_rel,
                                })

                    else:
                        # flat/unknown → crops にコピー（YOLOが crops を見るだけでOK）
                        out_name = f"{fid}__flat.jpg"
                        crop_path = crops_dir / out_name
                        crop_rel = str(crop_path.relative_to(out_dir))

                        if args.skip_existing and crop_path.exists():
                            stats["cache_hits_crop"] += 1
                            stats["crops_ok"] += 1
                            crop_written += 1
                            manifest_rows.append({
                                "run_id": run_id,
                                "idx": idx,
                                "fid": fid,
                                "point_key": point_key,
                                "lon": lon,
                                "lat": lat,
                                "status": "cached",
                                "error": "",
                                "kind": kind,
                                "url_source": url_source,
                                "item_url": item_url,
                                "img_url": img_url,
                                "yaw": "",
                                "pitch": "",
                                "fov": "",
                                "crop_w": "",
                                "crop_h": "",
                                "src_w": src_w,
                                "src_h": src_h,
                                "src_ratio": src_ratio,
                                "pano_relpath": pano_rel,
                                "crop_relpath": crop_rel,
                            })
                        else:
                            try:
                                crop_path.write_bytes(pano_path.read_bytes())
                                stats["crops_ok"] += 1
                                crop_written += 1
                                manifest_rows.append({
                                    "run_id": run_id,
                                    "idx": idx,
                                    "fid": fid,
                                    "point_key": point_key,
                                    "lon": lon,
                                    "lat": lat,
                                    "status": "ok",
                                    "error": "",
                                    "kind": kind,
                                    "url_source": url_source,
                                    "item_url": item_url,
                                    "img_url": img_url,
                                    "yaw": "",
                                    "pitch": "",
                                    "fov": "",
                                    "crop_w": "",
                                    "crop_h": "",
                                    "src_w": src_w,
                                    "src_h": src_h,
                                    "src_ratio": src_ratio,
                                    "pano_relpath": pano_rel,
                                    "crop_relpath": crop_rel,
                                })
                            except Exception as e:
                                stats["crops_fail"] += 1
                                manifest_rows.append({
                                    "run_id": run_id,
                                    "idx": idx,
                                    "fid": fid,
                                    "point_key": point_key,
                                    "lon": lon,
                                    "lat": lat,
                                    "status": "failed",
                                    "error": str(e),
                                    "kind": kind,
                                    "url_source": url_source,
                                    "item_url": item_url,
                                    "img_url": img_url,
                                    "yaw": "",
                                    "pitch": "",
                                    "fov": "",
                                    "crop_w": "",
                                    "crop_h": "",
                                    "src_w": src_w,
                                    "src_h": src_h,
                                    "src_ratio": src_ratio,
                                    "pano_relpath": pano_rel,
                                    "crop_relpath": crop_rel,
                                })

                # record downloaded feature（あなたの従来出力も維持）
                ff = json.loads(json.dumps(f))
                props = ff.get("properties") or {}
                props["kind"] = kind
                props["img_url"] = img_url
                props["item_url"] = item_url
                props["url_source"] = url_source
                props["point_key"] = point_key
                props["pano_file"] = pano_rel
                props["n_crops"] = crop_written
                if crop_written:
                    props["crops_dir"] = str(crops_dir.relative_to(out_dir))
                ff["properties"] = props
                downloaded_features.append(ff)

                if idx % 20 == 0 or idx == 1 or idx == len(selected):
                    print(
                        f"[{idx}/{len(selected)}] ok fid={fid} kind={kind} source={url_source} "
                        f"(panos_ok={stats['panos_ok']} crops_ok={stats['crops_ok']})"
                    )

                if args.sleep > 0:
                    time.sleep(args.sleep)

            except Exception as e:
                stats["panos_fail"] += 1
                err = str(e)
                manifest_rows.append({
                    "run_id": run_id,
                    "idx": idx,
                    "fid": fid,
                    "point_key": point_key,
                    "lon": lon,
                    "lat": lat,
                    "status": "failed",
                    "error": err,
                    "kind": "",
                    "url_source": "",
                    "item_url": item_url,
                    "img_url": direct_img_url or "",
                    "yaw": "",
                    "pitch": "",
                    "fov": "",
                    "crop_w": "",
                    "crop_h": "",
                    "pano_relpath": "",
                    "crop_relpath": "",
                })
                print(f"[{idx}/{len(selected)}] ERROR fid={fid}: {e}")

    finally:
        # outputs（中断されても必ず出す）
        downloaded_fc = {"type": "FeatureCollection", "features": downloaded_features}
        (out_dir / "downloaded_points.geojson").write_text(
            json.dumps(downloaded_fc, ensure_ascii=False),
            encoding="utf-8",
        )
        (out_dir / "run_summary.json").write_text(
            json.dumps(stats, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # 追加出力: crops_manifest.csv
        write_manifest_csv(out_dir / "crops_manifest.csv", manifest_rows)

        # 追加出力: points_all_in_aoi.geojson（AOI内全点に downloaded 属性などを付与）
        try:
            points_fc = load_featurecollection(points_path)
            points_all = build_points_all_geojson(points_fc, manifest_rows)
            (out_dir / "points_all_in_aoi.geojson").write_text(
                json.dumps(points_all, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            print("WARN: failed to write points_all_in_aoi.geojson:", e)

        print("----")
        print(f"DONE. outputs in {out_dir}")
        print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
