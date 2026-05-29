#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""YOLO detection helpers used by the agent."""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np


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


def best_det(dets: List[dict]) -> Optional[dict]:
    return dets[0] if dets else None


def det_center_frac(det: dict, w: int, h: int) -> Tuple[float, float, float]:
    x1, y1, x2, y2 = det["xyxy"]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
    return cx / w, cy / h, area / (w * h)

