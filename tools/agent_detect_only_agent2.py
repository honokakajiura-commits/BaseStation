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

from make_yolo_crops_from_panoramax import (
    download_image_bytes as panoramax_download_image_bytes,
    fetch_picture_meta as panoramax_fetch_picture_meta,
    render_panoramax_crop,
    resolve_best_panoramax_image as resolve_best_panoramax_image_record,
)
from spherical_camera import (
    compute_next_view_from_bbox,
    equirect_to_perspective as spherical_equirect_to_perspective,
)


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

def download_pano(
    fid: str,
    panos_dir: Path,
    api_base: str,
    image_base: str,
    session: requests.Session,
    retries: int = 5,
) -> Tuple[bool, Optional[Path], dict]:
    existing = find_pano_path(panos_dir, fid)
    if existing is not None and existing.stat().st_size > 20_000:
        return True, existing, {"source": "existing_local"}

    last_err = ""
    for k in range(retries):
        try:
            item = panoramax_fetch_picture_meta(session, api_base=api_base, fid=fid, timeout=45)
            img_url, selected_source, selected_asset = resolve_best_panoramax_image_record(
                session,
                item,
                timeout=45,
            )
            img_bytes, ext = panoramax_download_image_bytes(session, img_url, timeout=45)
            if len(img_bytes) <= 20_000:
                raise RuntimeError(f"downloaded image too small: {len(img_bytes)} bytes")
            dst = panos_dir / f"{fid}{ext}"
            dst.write_bytes(img_bytes)
            return True, dst, {
                "source": selected_source,
                "img_url": img_url,
                "selected_asset": selected_asset,
            }
        except Exception as e:
            last_err = safe_str(e)

        legacy_url = f"{image_base.rstrip('/')}/{fid}.jpg"
        try:
            r = session.get(legacy_url, timeout=45)
            if r.status_code == 200 and len(r.content) > 20_000:
                dst = panos_dir / f"{fid}.jpg"
                dst.write_bytes(r.content)
                return True, dst, {
                    "source": "legacy_image_base",
                    "img_url": legacy_url,
                }
        except Exception as e:
            last_err = safe_str(e)

        time.sleep(0.6 * (k + 1))

    return False, None, {"error": last_err}


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
    return spherical_equirect_to_perspective(
        img_bgr,
        yaw=float(yaw_deg),
        pitch=float(pitch_deg),
        roll=0.0,
        fov_x=float(fov_deg),
        out_w=int(out_w),
        out_h=int(out_h),
        R_level=None,
        interpolation=cv2.INTER_LINEAR,
    )


def render_detection_crop(
    pano_bgr: np.ndarray,
    yaw_deg: float,
    pitch_deg: float,
    fov_deg: float,
    out_w: int,
    out_h: int,
    crop_strategy: str,
    supersample: float,
    interpolation: str,
) -> Tuple[np.ndarray, dict]:
    crop, meta = render_panoramax_crop(
        pano_bgr=pano_bgr,
        yaw_deg=yaw_deg,
        pitch_deg=pitch_deg,
        fov_deg=fov_deg,
        out_w=out_w,
        out_h=out_h,
        crop_strategy=crop_strategy,
        supersample=supersample,
        interpolation=interpolation,
    )
    if "remap_interpolation" in meta and "interpolation" not in meta:
        meta["interpolation"] = meta["remap_interpolation"]
    return crop, meta


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
    crop_strategy: str = "ui_like"
    crop_supersample: float = 1.25
    crop_interpolation: str = "cubic"

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
    recenter_pitch: bool = True
    refine_zoom_ratio_small: float = 0.55
    refine_zoom_ratio_medium: float = 0.75

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


def py_to_angle_deg(y_px: float, h: int, fov_deg: float) -> float:
    cy = h / 2.0
    fov = math.radians(fov_deg)
    fy = (h / 2.0) / math.tan(fov / 2.0)
    ang = math.atan((cy - y_px) / fy)
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


def clamp_pitch_deg(pitch_deg: float, margin_deg: float = 1.0) -> float:
    return max(-89.0 + margin_deg, min(89.0 - margin_deg, float(pitch_deg)))


def draw_refine_compare(
    before_img: np.ndarray,
    before_dets: List[dict],
    after_img: np.ndarray,
    after_dets: List[dict],
    before_lines: List[str],
    after_lines: List[str],
) -> np.ndarray:
    left = draw_status(draw_annot(before_img, before_dets, topk=3), before_lines)
    right = draw_status(draw_annot(after_img, after_dets, topk=3), after_lines)
    return np.concatenate([left, right], axis=1)


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--aoi_index", required=True, help="fetch_panos_ordered.py が作った aoi_index.jsonl")
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--input_panos_dir", default="", help="(optional) panos dir; default: <run_dir>/panos")

    ap.add_argument("--api_base", default="https://api.panoramax.xyz")
    ap.add_argument("--image_base", default="https://panoramax.openstreetmap.fr/images")
    ap.add_argument("--skip_download", action="store_true")

    ap.add_argument("--pitch_cli", type=float, required=True, help="CLI pitch in degrees (positive is up)")

    ap.add_argument("--weights", required=True)
    ap.add_argument("--conf", type=float, default=0.20)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default="")

    ap.add_argument("--det_w", type=int, default=1280)
    ap.add_argument("--det_h", type=int, default=1280)
    ap.add_argument("--fov_front", type=float, default=105.0)
    ap.add_argument("--fov_side", type=float, default=90.0)
    ap.add_argument("--crop_strategy", choices=["legacy", "ui_like"], default="ui_like")
    ap.add_argument("--crop_supersample", type=float, default=1.25)
    ap.add_argument("--crop_interpolation", choices=["linear", "cubic", "lanczos", "nearest"], default="cubic")

    ap.add_argument("--high_conf", type=float, default=0.60)
    ap.add_argument("--low_conf", type=float, default=0.20)
    ap.add_argument("--max_refine", type=int, default=2)

    ap.add_argument("--small_area_frac", type=float, default=0.02)
    ap.add_argument("--large_area_frac", type=float, default=0.08)
    ap.add_argument("--zoom_min_fov", type=float, default=50.0)

    ap.add_argument("--edge_center_margin", type=float, default=0.20)
    ap.add_argument("--zoom_safe_factor", type=float, default=0.90)
    ap.add_argument("--bbox_margin_deg", type=float, default=3.0)
    ap.add_argument("--refine_zoom_ratio_small", type=float, default=0.55)
    ap.add_argument("--refine_zoom_ratio_medium", type=float, default=0.75)
    ap.add_argument("--disable_recenter_pitch", action="store_true")

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
    compare_dir = run_dir / "refine_compare"
    log_path = run_dir / "agent_log.jsonl"
    yaw_map_path = run_dir / "yaw_map.jsonl"
    summary_path = run_dir / "summary.json"

    for d in [panos_dir, crops_dir, ann_dir, compare_dir]:
        ensure_dir(d)

    if args.overwrite:
        if log_path.exists():
            log_path.unlink()
        if yaw_map_path.exists():
            yaw_map_path.unlink()

    # Internal pitch is positive upward. Keep the CLI conversion in one place.
    pitch_deg = float(args.pitch_cli)

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
            dl_ok, _, dl_meta = download_pano(
                fid=fid,
                panos_dir=panos_dir,
                api_base=args.api_base,
                image_base=args.image_base,
                session=sess,
            )
            append_jsonl(log_path, {
                "step": "download",
                "fid": fid,
                "status": "ok" if dl_ok else "fail",
                **dl_meta,
            })
            if dl_ok:
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
        crop_strategy=args.crop_strategy,
        crop_supersample=args.crop_supersample,
        crop_interpolation=args.crop_interpolation,
        zoom_min_fov=args.zoom_min_fov,
        high_conf=args.high_conf,
        low_conf=args.low_conf,
        small_area_frac=args.small_area_frac,
        large_area_frac=args.large_area_frac,
        edge_center_margin=args.edge_center_margin,
        zoom_safe_factor=args.zoom_safe_factor,
        bbox_margin_deg=args.bbox_margin_deg,
        recenter_pitch=not args.disable_recenter_pitch,
        refine_zoom_ratio_small=args.refine_zoom_ratio_small,
        refine_zoom_ratio_medium=args.refine_zoom_ratio_medium,
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
            cur_pitch = float(pitch_deg)

            last_yaw_delta = 0.0
            last_zoom = False
            prev_crop: Optional[np.ndarray] = None
            prev_dets: List[dict] = []
            prev_state: Optional[dict] = None

            for step in range(cfg.max_refine + 1):
                crop, crop_meta = render_detection_crop(
                    pano,
                    yaw_deg=cur_yaw,
                    pitch_deg=cur_pitch,
                    fov_deg=cur_fov,
                    out_w=cfg.det_w,
                    out_h=cfg.det_h,
                    crop_strategy=cfg.crop_strategy,
                    supersample=cfg.crop_supersample,
                    interpolation=cfg.crop_interpolation,
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
                    "pitch_deg": float(cur_pitch),
                    "fov": float(cur_fov),
                    "crop_meta": crop_meta,
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
                        f"yaw={cur_yaw:.1f} fov={cur_fov:.1f} pitch={cur_pitch:.1f}",
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

                cx_frac, cy_frac, area_frac = det_center_frac(bd, cfg.det_w, cfg.det_h)
                bbox_cx = (bd["xyxy"][0] + bd["xyxy"][2]) / 2.0
                bbox_cy = (bd["xyxy"][1] + bd["xyxy"][3]) / 2.0

                if prev_crop is not None and prev_state is not None:
                    compare_img = draw_refine_compare(
                        before_img=prev_crop,
                        before_dets=prev_dets,
                        after_img=crop,
                        after_dets=dets,
                        before_lines=[
                            f"before s={prev_state['step']} yaw={prev_state['yaw']:.1f} pitch={prev_state['pitch']:.1f}",
                            f"fov={prev_state['fov']:.1f} center=({prev_state['center_frac'][0]:.2f},{prev_state['center_frac'][1]:.2f})",
                        ],
                        after_lines=[
                            f"after s={step} yaw={cur_yaw:.1f} pitch={cur_pitch:.1f}",
                            f"fov={cur_fov:.1f} conf={best_conf:.2f}",
                        ],
                    )
                    compare_name = crop_path.stem + "__compare.jpg"
                    compare_path = unique_path(compare_dir / compare_name, overwrite=args.overwrite)
                    cv2.imwrite(str(compare_path), compare_img)
                    append_jsonl(log_path, {
                        "step": "refine_compare",
                        "i": i,
                        "fid": fid,
                        "view": view_name,
                        "from_s": prev_state["step"],
                        "to_s": step,
                        "compare_path": str(compare_path),
                        "before_center_frac": prev_state["center_frac"],
                        "before_center_xy": prev_state["center_xy"],
                        "after_center_frac": [float(cx_frac), float(cy_frac)],
                        "after_center_xy": [float(bbox_cx), float(bbox_cy)],
                        "before_yaw": float(prev_state["yaw"]),
                        "after_yaw": float(cur_yaw),
                        "before_pitch": float(prev_state["pitch"]),
                        "after_pitch": float(cur_pitch),
                        "before_fov": float(prev_state["fov"]),
                        "after_fov": float(cur_fov),
                    })

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

                if step >= cfg.max_refine:
                    break

                zoom_ratio = 1.0
                if area_frac < cfg.large_area_frac:
                    zoom_ratio = cfg.refine_zoom_ratio_small if area_frac < cfg.small_area_frac else cfg.refine_zoom_ratio_medium

                next_yaw, next_pitch, next_fov, debug_info = compute_next_view_from_bbox(
                    bbox=bd,
                    yaw=cur_yaw,
                    pitch=cur_pitch,
                    roll=0.0,
                    fov_x=cur_fov,
                    out_w=cfg.det_w,
                    out_h=cfg.det_h,
                    zoom_ratio=zoom_ratio,
                    min_fov=cfg.zoom_min_fov,
                    margin_deg=cfg.bbox_margin_deg,
                    R_level=None,
                )
                if not cfg.recenter_pitch:
                    next_pitch = float(cur_pitch)
                    debug_info["next_pitch"] = float(next_pitch)
                    debug_info["final_pitch"] = float(next_pitch)
                    debug_info["pitch_delta"] = 0.0
                    debug_info["recenter_pitch"] = False
                else:
                    debug_info["recenter_pitch"] = True

                yaw_delta = wrap_yaw_deg(next_yaw - cur_yaw)
                pitch_delta = float(next_pitch) - float(cur_pitch)
                fov_delta = float(next_fov) - float(cur_fov)
                zoom = bool(float(next_fov) < float(cur_fov) - 0.5)
                center_by_edge = need_center_by_edge(cx_frac, cfg.edge_center_margin)
                refine_action = safe_str(debug_info.get("refine_action"))

                if abs(yaw_delta) < 0.5 and abs(pitch_delta) < 0.5 and abs(fov_delta) < 0.5:
                    break

                prev_crop = crop.copy()
                prev_dets = list(dets)
                prev_state = {
                    "step": step,
                    "yaw": float(cur_yaw),
                    "pitch": float(cur_pitch),
                    "fov": float(cur_fov),
                    "center_frac": [float(cx_frac), float(cy_frac)],
                    "center_xy": [float(bbox_cx), float(bbox_cy)],
                }

                append_jsonl(log_path, {
                    "step": "refine_plan",
                    "i": i,
                    "fid": fid,
                    "view": view_name,
                    "from_s": step,
                    "to_s": step + 1,
                    "previous_yaw": float(cur_yaw),
                    "previous_pitch": float(cur_pitch),
                    "previous_fov": float(cur_fov),
                    "cx_frac": float(cx_frac),
                    "cy_frac": float(cy_frac),
                    "area_frac": float(area_frac),
                    "center_by_edge": bool(center_by_edge),
                    "bbox_center": [float(bbox_cx), float(bbox_cy)],
                    "bbox_margin_deg": float(cfg.bbox_margin_deg),
                    "target_yaw": float(debug_info["target_yaw"]),
                    "target_pitch": float(debug_info["target_pitch"]),
                    "yaw_delta": float(yaw_delta),
                    "pitch_delta": float(pitch_delta),
                    "next_yaw": float(next_yaw),
                    "next_pitch": float(next_pitch),
                    "next_fov": float(next_fov),
                    "max_corner_angle": float(debug_info["max_corner_angle"]),
                    "safe_fov": float(debug_info["safe_fov"]),
                    "zoom_fov": float(debug_info["zoom_fov"]),
                    "final_fov": float(debug_info["final_fov"]),
                    "zoom": bool(zoom),
                    "zoom_ratio_init": float(zoom_ratio),
                    "refine_action": refine_action,
                    "debug_info": debug_info,
                })

                cur_yaw = wrap_yaw_deg(next_yaw)
                cur_pitch = clamp_pitch_deg(next_pitch)
                cur_fov = float(next_fov)

                last_yaw_delta = float(yaw_delta)
                last_zoom = bool(zoom)


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
            "crop_strategy": cfg.crop_strategy,
            "crop_supersample": cfg.crop_supersample,
            "crop_interpolation": cfg.crop_interpolation,
            "high_conf": cfg.high_conf,
            "low_conf": cfg.low_conf,
            "max_refine": cfg.max_refine,
            "small_area_frac": cfg.small_area_frac,
            "large_area_frac": cfg.large_area_frac,
            "edge_center_margin": cfg.edge_center_margin,
            "zoom_safe_factor": cfg.zoom_safe_factor,
            "bbox_margin_deg": cfg.bbox_margin_deg,
            "zoom_min_fov": cfg.zoom_min_fov,
            "recenter_pitch": cfg.recenter_pitch,
            "refine_zoom_ratio_small": cfg.refine_zoom_ratio_small,
            "refine_zoom_ratio_medium": cfg.refine_zoom_ratio_medium,
            "weights": args.weights,
            "conf": float(args.conf),
            "imgsz": int(args.imgsz),
            "api_base": args.api_base,
        },
        "paths": {
            "run_dir": str(run_dir),
            "yaw_map": str(yaw_map_path),
            "crops": str(crops_dir),
            "annotated": str(ann_dir),
            "refine_compare": str(compare_dir),
            "log": str(log_path),
        }
    }
    save_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
