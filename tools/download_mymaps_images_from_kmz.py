import argparse
import csv
import hashlib
import re
import time
import zipfile
from pathlib import Path
from urllib.parse import unquote

import requests


def safe_name(text, max_len=40):
    text = text.strip()
    text = re.sub(r'[\\/:*?"<>|]', "_", text)
    text = re.sub(r"\s+", "_", text)
    return (text or "no_name")[:max_len]


def extract_kml_from_kmz(kmz_path, out_dir):
    work_dir = out_dir / "_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(kmz_path, "r") as z:
        kml_files = [n for n in z.namelist() if n.lower().endswith(".kml")]
        if not kml_files:
            raise RuntimeError("KMZの中にKMLが見つかりません")

        kml_name = kml_files[0]
        kml_path = work_dir / "doc.kml"

        with z.open(kml_name) as src:
            kml_path.write_bytes(src.read())

    return kml_path


def extract_placemarks(kml_text):
    placemark_blocks = re.findall(
        r"<Placemark\b.*?</Placemark>",
        kml_text,
        flags=re.S | re.I,
    )

    records = []

    for point_index, block in enumerate(placemark_blocks, start=1):
        name_match = re.search(r"<name>(.*?)</name>", block, flags=re.S | re.I)
        name = name_match.group(1).strip() if name_match else f"point_{point_index:04d}"
        name = re.sub(r"<.*?>", "", name)

        coord_match = re.search(r"<coordinates>(.*?)</coordinates>", block, flags=re.S | re.I)
        lon = ""
        lat = ""
        if coord_match:
            coord = coord_match.group(1).strip()
            parts = coord.split(",")
            if len(parts) >= 2:
                lon = parts[0].strip()
                lat = parts[1].strip()

        text = unquote(block).replace("&amp;", "&")

        urls = re.findall(r'https?://[^\s<>"\']+', text)

        image_urls = []
        for u in urls:
            u = u.strip()
            u = u.replace("]]", "")
            u = u.rstrip("]")
            u = u.rstrip(")")
            ul = u.lower()

            if "mymaps.usercontent.google.com/hostedimage" in ul:
                image_urls.append(u)
            elif "googleusercontent.com" in ul and "hostedimage" in ul:
                image_urls.append(u)
            elif "ggpht.com" in ul:
                image_urls.append(u)
            elif any(ext in ul for ext in [".jpg", ".jpeg", ".png", ".webp"]):
                image_urls.append(u)

        # 重複削除
        image_urls = list(dict.fromkeys(image_urls))

        for image_index, url in enumerate(image_urls, start=1):
            records.append({
                "point_index": point_index,
                "point_name": name,
                "image_index": image_index,
                "lat": lat,
                "lon": lon,
                "url": url,
            })

    return records


def download_one(url, out_path):
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    r = requests.get(url, headers=headers, timeout=40)
    r.raise_for_status()

    content_type = r.headers.get("Content-Type", "").lower()

    if "png" in content_type:
        ext = ".png"
    elif "webp" in content_type:
        ext = ".webp"
    else:
        ext = ".jpg"

    final_path = out_path.with_suffix(ext)
    final_path.write_bytes(r.content)

    return final_path, content_type, len(r.content)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kmz", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.3)
    args = parser.parse_args()

    kmz_path = Path(args.kmz).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    images_dir = out_dir / "images"

    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"KMZ: {kmz_path}")
    print(f"OUT: {out_dir}")

    kml_path = extract_kml_from_kmz(kmz_path, out_dir)
    kml_text = kml_path.read_text(encoding="utf-8", errors="ignore")

    records = extract_placemarks(kml_text)

    print(f"抽出した画像URL数: {len(records)}")

    metadata_rows = []
    failed_rows = []

    for i, rec in enumerate(records, start=1):
        point_name_safe = safe_name(rec["point_name"])
        url_hash = hashlib.md5(rec["url"].encode("utf-8")).hexdigest()[:8]
        base = f"{rec['point_index']:04d}_{point_name_safe}_{rec['image_index']:02d}_{url_hash}"
        out_path = images_dir / base

        if args.dry_run:
            metadata_rows.append({
                **rec,
                "filename": "",
                "content_type": "",
                "size_bytes": "",
                "status": "dry_run",
            })
            continue

        print(f"[{i}/{len(records)}] {rec['point_name']}")

        try:
            saved_path, content_type, size_bytes = download_one(rec["url"], out_path)
            metadata_rows.append({
                **rec,
                "filename": str(saved_path.relative_to(out_dir)),
                "content_type": content_type,
                "size_bytes": size_bytes,
                "status": "ok",
            })
            time.sleep(args.sleep)
        except Exception as e:
            print(f"  failed: {e}")
            failed_rows.append({
                **rec,
                "error": str(e),
            })

    metadata_path = out_dir / "metadata.csv"
    failed_path = out_dir / "failed.csv"

    with metadata_path.open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "point_index",
            "point_name",
            "image_index",
            "lat",
            "lon",
            "url",
            "filename",
            "content_type",
            "size_bytes",
            "status",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_rows)

    with failed_path.open("w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "point_index",
            "point_name",
            "image_index",
            "lat",
            "lon",
            "url",
            "error",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(failed_rows)

    print("完了")
    print(f"保存成功: {len(metadata_rows)}")
    print(f"失敗: {len(failed_rows)}")
    print(f"metadata: {metadata_path}")
    print(f"failed: {failed_path}")


if __name__ == "__main__":
    main()
