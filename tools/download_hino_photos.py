import csv
import re
import time
from pathlib import Path
from urllib.request import Request, urlopen

INPUT_CSV = "data/hino_basestation_photo_urls_for_arcgis.csv"
OUT_DIR = Path("data/hino_basestation_photos")
ATTACH_CSV = OUT_DIR / "attachments.csv"


OUT_DIR.mkdir(parents=True, exist_ok=True)

def safe_name(s):
    s = str(s)
    s = re.sub(r'[\\/:*?"<>|]', "_", s)
    s = re.sub(r"\s+", "_", s)
    return s[:80]

rows_out = []

with open(INPUT_CSV, newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)

    for row in reader:
        name = row.get("name") or row.get("Name")
        if not name:
            continue

        for i in range(1, 5):
            url = row.get(f"photo_url_{i}")
            if not url or url == "<Null>":
                continue

            filename = f"{safe_name(name)}_{i}.jpg"
            out_path = OUT_DIR / filename

            try:
                req = Request(
                    url,
                    headers={
                        "User-Agent": "Mozilla/5.0"
                    },
                )
                with urlopen(req, timeout=30) as r:
                    data = r.read()

                out_path.write_bytes(data)

                rows_out.append({
                    "Name": name,
                    "image_path": str(out_path).replace("/", "\\").replace("\\mnt\\c", "C:")
                })

                print("saved:", out_path)

                time.sleep(0.2)

            except Exception as e:
                print("failed:", name, i, e)

with open(ATTACH_CSV, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=["Name", "image_path"])
    writer.writeheader()
    writer.writerows(rows_out)

print("attachments csv:", ATTACH_CSV)
print("attachments:", len(rows_out))