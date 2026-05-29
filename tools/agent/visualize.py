#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Visualization helpers for detections and refine comparisons."""

from __future__ import annotations

from typing import List

import cv2
import numpy as np


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
    box_h = min(pad * 2 + line_h * len(lines), h)

    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, box_h), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.45, out, 0.55, 0)

    y = pad + 20
    for s in lines:
        cv2.putText(out, s, (pad, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        y += line_h
    return out


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

