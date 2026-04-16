#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python tools/agent_detect_only_agent.py \
  --aoi_index runs/fetch_full_seq/aoi_index.jsonl \
  --input_panos_dir runs/fetch_full_seq/panos \
  --run_dir runs/agent_full_seq \
  --skip_download \
  --pitch_cli 40 \
  --weights runs/過去結果/detect/v3_aug_new_m_960/weights/best.pt \
  --conf 0.20 \
  --imgsz 1280
"""

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
import requests


# ----------------------------
# IO utils
# ----------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def append_jsonl(path: Path, obj: dict) -> None:
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

def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def safe_str(x) -> str:
    return "" if x is None else str(x)

def find_pano_path(panos_dir: Path, fid: str) -> Optional[Path]:
    for ext in [".jpg", ".jpeg", ".png", ".webp"]:
        p = panos_dir / f"{fid}{ext}"
        if p.exists():
            return p
    return None

def unique_path(dst: Path, overwrite: bool) -> Path:
    """
    overwrite=False の場合:
      既に dst が存在するなら dst の stem に __v001, __v002... を付けて空きを探す
    overwrite=True の場合:
      dst をそのまま返す
    """
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


# ----------------------------
# filename helpers (traceable exploration)
# ----------------------------

def _fmt_deg_tag(x: float, ndigits: int = 0) -> str:
    """
    Filename-safe signed degree tag.
    Examples:
      12.3 -> p12 (ndigits=0) / p12p3 (ndigits=1)
      -8.0 -> m8
    """
    sign = "p" if x >= 0 else "m"
    ax = abs(float(x))
    if ndigits <= 0:
        return f"{sign}{int(round(ax))}"
    scale = 10 ** ndigits
    v = int(round(ax * scale))
    whole = v // scale
    frac = v % scale
    return f"{sign}{whole}p{frac}"

def _build_action_tag(step: int, last_yaw_delta: float, last_zoom: bool) -> str:
    if step == 0:
        return "init"
    parts = []
    if abs(last_yaw_delta) > 1e-6:
        parts.append(f"yaw_{_fmt_deg_tag(last_yaw_delta)}")
    if last_zoom:
        parts.append("zoom")
    if not parts:
        parts.append("keep")
    return "_".join(parts)

def _build_crop_name(
    idx: int,
    fid: str,
    view: str,
    step: int,
    yaw: float,
    fov: float,
    last_yaw_delta: float,
    last_zoom: bool,
) -> str:
    act = _build_action_tag(step, last_yaw_delta, last_zoom)
    return (
        f"{idx:05d}__{fid}__{view}"
        f"__r{step}"
        f"__yaw{_fmt_deg_tag(yaw)}"
        f"__fov{_fmt_deg_tag(fov)}"
        f"__act{act}.jpg"
    )


# ----------------------------
# download (optional)
# ----------------------------

def download_pano(fid: str, dst: Path, image_base: str, session: requests.Session, retries: int = 5) -> bool:
    if dst.exists() and dst.stat().st_size > 20_000:
        return True
    url = f"{image_base.rstrip('/')}/{fid}.jpg"
    for k in range(retries):
        try:
            r = session.get(url, timeout=45)
            if r.status_code == 200 and len(r.content) > 20_000:
                dst.write_bytes(r.content)
                return True
        except Exception:
            pass
        time.sleep(0.6 * (k + 1))
    return False


# ----------------------------
# geometry / crop
# ----------------------------

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
    """Rectilinear projection. yaw:+right, pitch:+up."""
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
    y_cam = -(yv - cy) / fy  # 上下反転補正
    z_cam = np.ones_like(x_cam)

    norm = np.sqrt(x_cam**2 + y_cam**2 + z_cam**2)
    x_cam /= norm
    y_cam /= norm
    z_cam /= norm

    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    cos_p, sin_p = math.cos(pitch), math.sin(pitch)

    # pitch (X)
    x1 = x_cam
    y1 = cos_p * y_cam - sin_p * z_cam
    z1 = sin_p * y_cam + cos_p * z_cam

    # yaw (Y)
    x2 = cos_y * x1 + sin_y * z1
    y2 = y1
    z2 = -sin_y * x1 + cos_y * z1

    lon = np.arctan2(x2, z2)
    lat = np.arcsin(np.clip(y2, -1.0, 1.0))

    u = (lon / (2 * math.pi) + 0.5) * w
    v = (0.5 - lat / math.pi) * h

    u = u.astype(np.float32)
    v = v.astype(np.float32)

    return cv2.remap(img_bgr, u, v, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


# ----------------------------
# yaw_center estimation (vanishing point + fallback azimuth)
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
        edges, rho=1, theta=np.pi / 180,
        threshold=70, minLineLength=60, maxLineGap=10
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
            if px < -w or px > 2 * w:
                continue
            if py < -h or py > 2 * h:
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
        pano_bgr, yaw_deg=yaw0, pitch_deg=pitch_deg,
        fov_deg=fov_preview, out_w=out_w, out_h=out_h
    )
    vp_x, n_lines = estimate_vanishing_point_x(persp)
    if vp_x is not None:
        yaw_off = vp_x_to_yaw_offset_deg(vp_x, out_w=out_w, fov_deg=fov_preview)
        yaw_center = wrap_yaw_deg(yaw0 + yaw_off)
        meta = {"vp_x": float(vp_x), "n_lines": int(n_lines), "fov_preview": float(fov_preview), "out_w": out_w, "out_h": out_h}
        return float(yaw_center), "vanishing_point", meta

    if view_azimuth is not None and safe_str(view_azimuth) != "":
        yaw_center = yaw_from_view_azimuth(view_azimuth, default_if_missing=0.0)
        meta = {"n_lines": int(n_lines)}
        return float(yaw_center), "view_azimuth", meta

    return 0.0, "fallback_zero", {"n_lines": int(n_lines)}


# ----------------------------
# YOLO
# ----------------------------

class YoloRunner:
    def __init__(self, weights: str, conf: float, imgsz: int, device: str = ""):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.conf = conf
        self.imgsz = imgsz
        self.device = device

    def infer(self, img_bgr: np.ndarray) -> List[dict]:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = self.model.predict(
            source=img_rgb,
            conf=self.conf,
            imgsz=self.imgsz,
            verbose=False,
            device=self.device if self.device else None,
        )
        out: List[dict] = []
        r = results[0]
        if r.boxes is None:
            return out
        for b in r.boxes:
            xyxy = b.xyxy[0].cpu().numpy().tolist()
            conf = float(b.conf[0].cpu().numpy().item())
            cls = int(b.cls[0].cpu().numpy().item())
            out.append({"cls": cls, "conf": conf, "xyxy": [float(x) for x in xyxy]})
        out.sort(key=lambda d: d["conf"], reverse=True)
        return out


# ----------------------------
# Agent config + helpers
# ----------------------------

@dataclass
class AgentConfig:
    det_w: int = 1280
    det_h: int = 1280
    fov_front: float = 105.0
    fov_side: float = 90.0

    zoom_min_fov: float = 50.0
    high_conf: float = 0.60
    low_conf: float = 0.20

    # bboxサイズ判定
    small_area_frac: float = 0.02
    large_area_frac: float = 0.08

    # 既存方針：bbox中心が端すぎる時だけ「中心寄せ」
    edge_center_margin: float = 0.20

    # ズームで消えそうなら中心寄せ（安全率）
    zoom_safe_factor: float = 0.90

    # ズーム後に bbox が切れないための角度余白（度）
    bbox_margin_deg: float = 3.0

    max_refine: int = 2
    yaw_side_deg: float = 90.0


def best_det(dets: List[dict]) -> Optional[dict]:
    return dets[0] if dets else None

def det_center_frac(det: dict, w: int, h: int) -> Tuple[float, float, float]:
    x1, y1, x2, y2 = det["xyxy"]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
    return cx / w, cy / h, area / (w * h)

def yaw_adjust_from_px(cx_frac: float, hfov_deg: float) -> float:
    dx = (cx_frac - 0.5)
    return dx * hfov_deg

def draw_annot(img_bgr: np.ndarray, dets: List[dict], topk: int = 3) -> np.ndarray:
    out = img_bgr.copy()
    for d in dets[:topk]:
        x1, y1, x2, y2 = map(int, d["xyxy"])
        conf = float(d["conf"])
        cls = int(d["cls"])
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            out,
            f"cls={cls} conf={conf:.2f}",
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
    return out

def draw_status(img_bgr: np.ndarray, lines: List[str]) -> np.ndarray:
    out = img_bgr.copy()
    h, w = out.shape[:2]

    pad = 10
    line_h = 28
    box_h = pad * 2 + line_h * len(lines)
    box_h = min(box_h, h)

    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, box_h), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.45, out, 0.55, 0)

    y = pad + 20
    for s in lines:
        cv2.putText(out, s, (pad, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        y += line_h

    return out

def need_center_by_edge(cx_frac: float, margin: float) -> bool:
    return (cx_frac < margin) or (cx_frac > 1.0 - margin)

def need_center_before_zoom(cx_frac: float, cur_fov: float, next_fov: float, safe_factor: float) -> bool:
    """
    ズーム後に bbox がフレーム外へ出そうなら中心寄せする、の粗い判定。
    （下の角度ベースの補正が主役だが、保険として残す）
    """
    if next_fov >= cur_fov:
        return False
    r = max(1e-6, float(next_fov) / float(cur_fov))
    safe_half = 0.5 * r * float(safe_factor)
    dx = abs(float(cx_frac) - 0.5)
    return dx > safe_half

def px_to_angle_deg(x_px: float, w: int, fov_deg: float) -> float:
    """画像中心からの角度(度)。右が正。"""
    cx = w / 2.0
    fov = math.radians(fov_deg)
    fx = (w / 2.0) / math.tan(fov / 2.0)
    ang = math.atan((x_px - cx) / fx)
    return math.degrees(ang)

def bbox_lr_angles_deg(det: dict, w: int, fov_deg: float) -> Tuple[float, float]:
    x1, y1, x2, y2 = det["xyxy"]
    a1 = px_to_angle_deg(float(x1), w, fov_deg)
    a2 = px_to_angle_deg(float(x2), w, fov_deg)
    return (min(a1, a2), max(a1, a2))

def fit_next_fov_to_bbox(cur_fov: float, next_fov: float, det: dict, w: int, margin_deg: float) -> Tuple[float, bool, float]:
    """
    bbox の左右端が next_fov に収まる可能性があるように next_fov を調整する。
    - bboxの角度幅が next_fov を超えるなら、next_fov を広げる（ただし cur_fov を上限）
    - 結果として next_fov が cur_fov と同じになれば「ズームなし」に近い状態になる
    戻り値: (adjusted_next_fov, zoom_flag, bbox_width_deg)
    """
    left, right = bbox_lr_angles_deg(det, w, cur_fov)
    width = max(0.0, right - left)
    need = width + 2.0 * float(margin_deg)

    adj = float(next_fov)
    if need > adj:
        adj = min(float(cur_fov), float(need))
    zoom = (abs(adj - float(cur_fov)) > 1e-6) and (adj < float(cur_fov) - 1e-6)
    return adj, bool(zoom), float(width)

def yaw_delta_to_keep_bbox_in_next_fov(
    det: dict,
    w: int,
    cur_fov: float,
    next_fov: float,
    margin_deg: float = 3.0,
) -> float:
    """
    次FOV(next_fov)に bbox の左右端が margin_deg 付きで収まるようにする追加yaw(度)。
    角度ベースで「端が切れる」問題を減らすための補正。
    """
    left, right = bbox_lr_angles_deg(det, w, cur_fov)

    half_next = float(next_fov) / 2.0
    allow_left = -half_next + float(margin_deg)
    allow_right = half_next - float(margin_deg)

    need_shift_left = right - allow_right   # >0 なら右がはみ出す
    need_shift_right = allow_left - left    # >0 なら左がはみ出す

    shift = 0.0
    if need_shift_left > 0:
        shift = -need_shift_left
    elif need_shift_right > 0:
        shift = need_shift_right

    # shift は「bbox角度を動かす量」→ 視線yawは逆向きに効くので符号反転
    return float(-shift)


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--aoi_index", required=True, help="fetch_panos_ordered.py が作った aoi_index.jsonl")
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--input_panos_dir", default="", help="(optional) panos dir; default: <run_dir>/panos")

    ap.add_argument("--image_base", default="https://panoramax.openstreetmap.fr/images")
    ap.add_argument("--skip_download", action="store_true")

    ap.add_argument("--pitch_cli", type=float, required=True, help="CLI pitch (up is negative)")

    ap.add_argument("--weights", required=True)
    ap.add_argument("--conf", type=float, default=0.20)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default="")

    ap.add_argument("--det_w", type=int, default=1280)
    ap.add_argument("--det_h", type=int, default=1280)
    ap.add_argument("--fov_front", type=float, default=105.0)
    ap.add_argument("--fov_side", type=float, default=90.0)

    ap.add_argument("--high_conf", type=float, default=0.60)
    ap.add_argument("--low_conf", type=float, default=0.20)
    ap.add_argument("--max_refine", type=int, default=2)

    ap.add_argument("--small_area_frac", type=float, default=0.02)
    ap.add_argument("--large_area_frac", type=float, default=0.08)
    ap.add_argument("--zoom_min_fov", type=float, default=50.0)

    ap.add_argument("--edge_center_margin", type=float, default=0.20)
    ap.add_argument("--zoom_safe_factor", type=float, default=0.90)
    ap.add_argument("--bbox_margin_deg", type=float, default=3.0)

    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")

    ap.add_argument("--yaw_preview_fov", type=float, default=110.0)
    ap.add_argument("--yaw_preview_w", type=int, default=1024)
    ap.add_argument("--yaw_preview_h", type=int, default=768)

    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    ensure_dir(run_dir)

    aoi_index_path = Path(args.aoi_index)
    panos_dir = Path(args.input_panos_dir) if args.input_panos_dir else (run_dir / "panos")

    crops_dir = run_dir / "crops"
    ann_dir = run_dir / "annotated"
    log_path = run_dir / "agent_log.jsonl"
    yaw_map_path = run_dir / "yaw_map.jsonl"
    summary_path = run_dir / "summary.json"

    for d in [panos_dir, crops_dir, ann_dir]:
        ensure_dir(d)

    if args.overwrite:
        if log_path.exists():
            log_path.unlink()
        if yaw_map_path.exists():
            yaw_map_path.unlink()

    # pitch convention: crop expects +up, CLI is -up
    pitch_deg = -float(args.pitch_cli)

    index_recs = read_jsonl(aoi_index_path)
    if args.limit and args.limit > 0:
        index_recs = index_recs[: args.limit]

    sess = requests.Session()
    sess.headers.update({"User-Agent": "BaseStation2/agent_detect_only_agent"})

    # download (optional)
    if not args.skip_download:
        ok = fail = 0
        for r in index_recs:
            fid = r["fid"]
            dst = panos_dir / f"{fid}.jpg"
            if download_pano(fid, dst, args.image_base, sess):
                ok += 1
            else:
                fail += 1
        print(f"[download] ok={ok} fail={fail} dir={panos_dir}")

    # yaw_center map (auto)
    yaw_done = set()
    if yaw_map_path.exists() and (not args.overwrite):
        for o in read_jsonl(yaw_map_path):
            if "fid" in o:
                yaw_done.add(o["fid"])

    with yaw_map_path.open("a", encoding="utf-8") as f:
        for r in index_recs:
            fid = r["fid"]
            if fid in yaw_done and (not args.overwrite):
                continue

            pano_path = find_pano_path(panos_dir, fid)
            if pano_path is None:
                f.write(json.dumps({"fid": fid, "yaw_center": 0.0, "yaw_reason": "missing_pano"}, ensure_ascii=False) + "\n")
                continue

            pano = cv2.imread(str(pano_path))
            if pano is None:
                f.write(json.dumps({"fid": fid, "yaw_center": 0.0, "yaw_reason": "imread_failed"}, ensure_ascii=False) + "\n")
                continue

            yaw_center, reason, meta = estimate_yaw_center_auto(
                pano,
                pitch_deg=pitch_deg,
                view_azimuth=r.get("view_azimuth"),
                fov_preview=args.yaw_preview_fov,
                out_w=args.yaw_preview_w,
                out_h=args.yaw_preview_h,
            )
            rec_out = {"fid": fid, "yaw_center": float(yaw_center), "yaw_reason": reason, **meta}
            f.write(json.dumps(rec_out, ensure_ascii=False) + "\n")

    yaw_map = {o["fid"]: float(o.get("yaw_center", 0.0)) for o in read_jsonl(yaw_map_path) if "fid" in o}

    cfg = AgentConfig(
        det_w=args.det_w,
        det_h=args.det_h,
        fov_front=args.fov_front,
        fov_side=args.fov_side,
        zoom_min_fov=args.zoom_min_fov,
        high_conf=args.high_conf,
        low_conf=args.low_conf,
        small_area_frac=args.small_area_frac,
        large_area_frac=args.large_area_frac,
        edge_center_margin=args.edge_center_margin,
        zoom_safe_factor=args.zoom_safe_factor,
        bbox_margin_deg=args.bbox_margin_deg,
        max_refine=args.max_refine,
    )

    yolo = YoloRunner(args.weights, conf=args.conf, imgsz=args.imgsz, device=args.device)

    total_panos = 0
    total_crops = 0
    confirmed = 0
    candidates = 0

    # 1-based 統一：ログもファイル名も同じ i
    for i, r in enumerate(index_recs, start=1):
        fid = r["fid"]
        pano_path = find_pano_path(panos_dir, fid)
        if pano_path is None:
            append_jsonl(log_path, {"step": "pano", "i": i, "fid": fid, "status": "fail", "reason": "missing_pano"})
            continue

        pano = cv2.imread(str(pano_path))
        if pano is None:
            append_jsonl(log_path, {"step": "pano", "i": i, "fid": fid, "status": "fail", "reason": "imread_failed"})
            continue

        total_panos += 1
        yaw_center = float(yaw_map.get(fid, 0.0))

        views = [
            ("front", 0.0, cfg.fov_front),
            ("left", -cfg.yaw_side_deg, cfg.fov_side),
            ("right", cfg.yaw_side_deg, cfg.fov_side),
        ]

        pano_confirmed = 0
        pano_candidate = 0

        for view_name, yaw_off, fov0 in views:
            cur_yaw = wrap_yaw_deg(yaw_center + yaw_off)
            cur_fov = float(fov0)

            last_yaw_delta = 0.0
            last_zoom = False

            for step in range(cfg.max_refine + 1):
                crop = equirectangular_to_perspective(
                    pano,
                    yaw_deg=cur_yaw,
                    pitch_deg=pitch_deg,
                    fov_deg=cur_fov,
                    out_w=cfg.det_w,
                    out_h=cfg.det_h,
                )

                crop_name = _build_crop_name(
                    idx=i,
                    fid=fid,
                    view=view_name,
                    step=step,
                    yaw=cur_yaw,
                    fov=cur_fov,
                    last_yaw_delta=last_yaw_delta,
                    last_zoom=last_zoom,
                )

                crop_path = unique_path(crops_dir / crop_name, overwrite=args.overwrite)
                cv2.imwrite(str(crop_path), crop)
                total_crops += 1

                dets = yolo.infer(crop)
                bd = best_det(dets)

                append_jsonl(log_path, {
                    "step": "infer",
                    "i": i,
                    "fid": fid,
                    "view": view_name,
                    "s": step,
                    "yaw_center": yaw_center,
                    "yaw": float(cur_yaw),
                    "yaw_off": float(yaw_off),
                    "pitch_cli": float(args.pitch_cli),
                    "pitch_deg": float(pitch_deg),
                    "fov": float(cur_fov),
                    "crop_path": str(crop_path),
                    "n": len(dets),
                    "best": bd,
                    "sequence_id": r.get("sequence_id", ""),
                    "rank_in_collection": r.get("rank_in_collection", None),
                })

                # 検出0の場合
                if not bd:
                    # step0(初回)で検出0なら annotated は作らない（従来通り）
                    if step == 0:
                        break

                    # refine後(step>0)で検出0になった場合は annotated を保存して痕跡を残す
                    msg_lines = [
                        "NO DETECTION after refine",
                        f"view={view_name} step={step}",
                        f"yaw={cur_yaw:.1f} fov={cur_fov:.1f} pitch={pitch_deg:.1f}",
                    ]
                    ann0 = draw_status(crop, msg_lines)
                    ann_path0 = unique_path(ann_dir / crop_path.name, overwrite=args.overwrite)
                    cv2.imwrite(str(ann_path0), ann0)

                    append_jsonl(log_path, {
                        "step": "refine_lost",
                        "i": i,
                        "fid": fid,
                        "view": view_name,
                        "s": step,
                        "status": "no_detection_after_refine",
                        "ann_path": str(ann_path0),
                        "yaw": float(cur_yaw),
                        "fov": float(cur_fov),
                    })
                    break

                best_conf = float(bd["conf"])

                # ★検出が1件でもあれば annotated 保存（confが低くても残す）
                ann = draw_annot(crop, dets, topk=3)
                ann_path = unique_path(ann_dir / crop_path.name, overwrite=args.overwrite)
                cv2.imwrite(str(ann_path), ann)

                if best_conf >= cfg.high_conf:
                    confirmed += 1
                    pano_confirmed += 1
                    break

                if best_conf < cfg.low_conf:
                    # 低すぎはここで打ち切り（再探索しない）
                    break

                # --- 中程度(confがlow〜high) ---
                candidates += 1
                pano_candidate += 1

                cx_frac, cy_frac, area_frac = det_center_frac(bd, cfg.det_w, cfg.det_h)

                # 1) ズーム計画（bboxが大きい時はズームしない）
                next_fov = cur_fov
                zoom = False
                zoom_ratio = 1.0
                if area_frac < cfg.large_area_frac:
                    zoom_ratio = 0.55 if area_frac < cfg.small_area_frac else 0.75
                    next_fov = max(cfg.zoom_min_fov, cur_fov * zoom_ratio)
                    zoom = (next_fov != cur_fov)

                # 2) bbox幅が次FOVに収まらないなら、next_fovを「収まる程度まで」緩める
                #    （それでも cur_fov と同じになったら実質ズームしない）
                bbox_width_deg = 0.0
                if zoom:
                    next_fov, zoom, bbox_width_deg = fit_next_fov_to_bbox(
                        cur_fov=cur_fov,
                        next_fov=next_fov,
                        det=bd,
                        w=cfg.det_w,
                        margin_deg=cfg.bbox_margin_deg,
                    )

                # 3) 中心寄せ判定（既存：端すぎるなら中心寄せ）
                center_by_edge = need_center_by_edge(cx_frac, cfg.edge_center_margin)

                # 4) 追加：ズームしたら消えそうなら中心寄せ（粗い保険）
                center_for_zoom = False
                if zoom:
                    center_for_zoom = need_center_before_zoom(
                        cx_frac=cx_frac,
                        cur_fov=cur_fov,
                        next_fov=next_fov,
                        safe_factor=cfg.zoom_safe_factor,
                    )

                # 5) 追加：角度ベースで「bbox左右端が次FOV内に収まる」ための yaw 補正
                yaw_delta_keep = 0.0
                if zoom:
                    yaw_delta_keep = yaw_delta_to_keep_bbox_in_next_fov(
                        det=bd,
                        w=cfg.det_w,
                        cur_fov=cur_fov,
                        next_fov=next_fov,
                        margin_deg=cfg.bbox_margin_deg,
                    )

                need_center = bool(center_by_edge or center_for_zoom or (abs(yaw_delta_keep) > 1e-6))

                yaw_delta = 0.0
                if need_center:
                    # 端寄りなら中心に寄せる（従来）
                    if center_by_edge or center_for_zoom:
                        yaw_delta += yaw_adjust_from_px(cx_frac, cur_fov)
                    # ズームで端が切れそうなら、角度ベースで追加補正
                    yaw_delta += yaw_delta_keep

                # 6) 変化がないなら打ち切り
                if abs(yaw_delta) < 0.5 and not zoom:
                    break

                cur_yaw = wrap_yaw_deg(cur_yaw + yaw_delta)
                cur_fov = float(next_fov)

                last_yaw_delta = float(yaw_delta)
                last_zoom = bool(zoom)

                append_jsonl(log_path, {
                    "step": "refine_plan",
                    "i": i,
                    "fid": fid,
                    "view": view_name,
                    "from_s": step,
                    "to_s": step + 1,
                    "cx_frac": float(cx_frac),
                    "cy_frac": float(cy_frac),
                    "area_frac": float(area_frac),
                    "center_by_edge": bool(center_by_edge),
                    "center_for_zoom": bool(center_for_zoom),
                    "yaw_delta_keep": float(yaw_delta_keep),
                    "bbox_width_deg": float(bbox_width_deg),
                    "bbox_margin_deg": float(cfg.bbox_margin_deg),
                    "need_center": bool(need_center),
                    "yaw_delta": float(yaw_delta),
                    "next_yaw": float(cur_yaw),
                    "cur_fov": float(cur_fov),
                    "next_fov": float(next_fov),
                    "zoom": bool(zoom),
                    "zoom_ratio_init": float(zoom_ratio),
                })

        append_jsonl(log_path, {
            "step": "pano_done",
            "i": i,
            "fid": fid,
            "yaw_center": yaw_center,
            "confirmed": pano_confirmed,
            "candidate": pano_candidate,
            "sequence_id": r.get("sequence_id", ""),
            "rank_in_collection": r.get("rank_in_collection", None),
        })

    summary = {
        "aoi_index": str(aoi_index_path),
        "panos_dir": str(panos_dir),
        "processed_panos": total_panos,
        "total_crops": total_crops,
        "confirmed": confirmed,
        "candidates": candidates,
        "params": {
            "pitch_cli": float(args.pitch_cli),
            "pitch_deg": float(pitch_deg),
            "det_w": cfg.det_w,
            "det_h": cfg.det_h,
            "fov_front": cfg.fov_front,
            "fov_side": cfg.fov_side,
            "high_conf": cfg.high_conf,
            "low_conf": cfg.low_conf,
            "max_refine": cfg.max_refine,
            "small_area_frac": cfg.small_area_frac,
            "large_area_frac": cfg.large_area_frac,
            "edge_center_margin": cfg.edge_center_margin,
            "zoom_safe_factor": cfg.zoom_safe_factor,
            "bbox_margin_deg": cfg.bbox_margin_deg,
            "zoom_min_fov": cfg.zoom_min_fov,
            "weights": args.weights,
            "conf": float(args.conf),
            "imgsz": int(args.imgsz),
        },
        "paths": {
            "run_dir": str(run_dir),
            "yaw_map": str(yaw_map_path),
            "crops": str(crops_dir),
            "annotated": str(ann_dir),
            "log": str(log_path),
        }
    }
    save_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
