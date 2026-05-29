#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GeoJSON読み込み
→ featureごとにIDや位置を取得
→ sequence情報で順序付け
→ 各featureから画像URLを解決
→ 画像ダウンロード
→ panos/ に保存
→ aoi_index.jsonl と fetch_summary.json を出力


python tools/fetch_panos_ordered.py \
  --points runs/aoi_TMU_east_points2/panoramax_points_in_aoi.geojson \
  --out_dir runs/fetch_full_seq \
  --order_mode sequence \
  --meta_cache_jsonl runs/aoi_TMU_east_points2/picture_meta_cache.jsonl \
  --skip_existing \
  --name_with_index
"""
import argparse
import io
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

import requests
from PIL import Image, ImageOps


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
    """returns (lon, lat)"""
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
    return candidates[0][1] if candidates[0][1] else None


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
                if isinstance(href2, str) and href2:
                    if "/items/" in href2 or href2.endswith(".json"):
                        out = resolve_image_url_via_item(
                            session, href2, timeout=timeout, visited=visited, depth=depth + 1, max_depth=max_depth
                        )
                        if out:
                            return out
    return None


def download_image(session: requests.Session, img_url: str, timeout: int = 60) -> Image.Image:
    r = request_get_with_retry(session, img_url, timeout=timeout, max_tries=4)
    img = Image.open(io.BytesIO(r.content))
    img = ImageOps.exif_transpose(img).convert("RGB")
    return img


# -------------------------
# ordering: datetime / nearest / sequence
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
        return (p[0], p[1])  # lat, lon

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
    n_ok = n_fail = 0

    for i, ft in enumerate(features, start=1):
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
                n_ok += 1
            except Exception:
                seq, rank = "", None
                n_fail += 1

            if sleep > 0:
                time.sleep(sleep)

        ff = json.loads(json.dumps(ft))
        props = ff.get("properties") or {}
        props["sequence_id"] = seq
        props["rank_in_collection"] = rank
        ff["properties"] = props
        enriched.append(ff)

        if i % 300 == 0:
            print(f"[sequence-meta] {i}/{len(features)} ok={n_ok} fail={n_fail}")

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

    print(f"[sequence] sequences={len(non_empty)} (+empty={1 if empty else 0}) meta_ok={n_ok} meta_fail={n_fail}")
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


# -------------------------
# main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", required=True, help="input points geojson (FeatureCollection)")
    ap.add_argument("--out_dir", required=True, help="output directory (will create panos/ etc)")

    ap.add_argument("--order_mode", choices=["datetime", "nearest", "sequence"], default="sequence")
    ap.add_argument("--api_base", default="https://api.panoramax.xyz", help="used only when order_mode=sequence")
    ap.add_argument("--meta_cache_jsonl", default="", help="cache for sequence meta (resume-friendly)")

    ap.add_argument("--limit", type=int, default=0, help="max points (0=no limit)")
    ap.add_argument("--sleep", type=float, default=0.02, help="sleep between API calls / downloads")
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--skip_existing", action="store_true", help="skip if panos/{fid}.jpg exists")
    ap.add_argument("--name_with_index", action="store_true",
                    help="also save panos as panos/{index}__{fid}.jpg for human-visible order")
    args = ap.parse_args()

    points_path = Path(args.points)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)
    panos_dir = out_dir / "panos"
    ensure_dir(panos_dir)

    run_id = out_dir.name

    fc = load_featurecollection(points_path)
    feats: List[Dict[str, Any]] = fc["features"]

    session = requests.Session()
    session.headers.update({"User-Agent": "panoramax-fetch-panos-ordered/2.1"})

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
        feats = feats[: args.limit]

    aoi_index_path = out_dir / "aoi_index.jsonl"
    write_aoi_index_jsonl(aoi_index_path, feats)

    ok = fail = skip = 0

    for idx, f in enumerate(feats, start=1):
        fid = get_feature_id(f)
        if fid == "unknown":
            fail += 1
            if idx == 1 or idx % 200 == 0 or idx == len(feats):
                print(f"[{idx}/{len(feats)}] FAIL unknown fid (ok={ok} skip={skip} fail={fail})")
            continue

        # 基本の保存名（agent互換）
        pano_path = panos_dir / f"{fid}.jpg"

        # 人間が順番で見たい時用の保存名（任意）
        pano_path_idx = panos_dir / f"{idx:06d}__{fid}.jpg" if args.name_with_index else None

        if args.skip_existing and pano_path.exists() and pano_path.stat().st_size > 20_000:
            skip += 1
            # name_with_index の方だけ欠けてる場合は作っておく
            if args.name_with_index and pano_path_idx and (not pano_path_idx.exists()):
                try:
                    pano_path_idx.write_bytes(pano_path.read_bytes())
                except Exception:
                    pass
            if idx == 1 or idx % 200 == 0 or idx == len(feats):
                print(f"[{idx}/{len(feats)}] skip={skip} ok={ok} fail={fail}")
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

            img = download_image(session, img_url, timeout=args.timeout)
            img.save(pano_path, quality=92)

            if args.name_with_index and pano_path_idx:
                try:
                    img.save(pano_path_idx, quality=92)
                except Exception:
                    # 失敗しても本体があればOK
                    pass

            ok += 1

            if idx == 1 or idx % 200 == 0 or idx == len(feats):
                props = f.get("properties") or {}
                seq = safe_str(props.get("sequence_id"))
                rnk = props.get("rank_in_collection", None)
                print(f"[{idx}/{len(feats)}] ok fid={fid} seq={seq} rank={rnk} (ok={ok} skip={skip} fail={fail})")

            if args.sleep > 0:
                time.sleep(args.sleep)

        except Exception as e:
            fail += 1
            print(f"[{idx}/{len(feats)}] FAIL fid={fid}: {e}")

    summary = {
        "run_id": run_id,
        "points_in": len(fc["features"]),
        "points_used": len(feats),
        "order_mode": args.order_mode,
        "api_base": args.api_base if args.order_mode == "sequence" else "",
        "meta_cache_jsonl": str(cache_path) if cache_path else "",
        "ok": ok,
        "skip": skip,
        "fail": fail,
        "aoi_index": str(aoi_index_path),
        "panos_dir": str(panos_dir),
        "name_with_index": bool(args.name_with_index),
    }
    (out_dir / "fetch_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("DONE")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
