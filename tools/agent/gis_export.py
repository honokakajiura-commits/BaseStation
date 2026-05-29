#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Minimal GIS export helpers for future ArcGIS/GeoJSON integration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


# TODO: Add ArcGIS-specific fields and CRS handling when geolocation estimates
# become part of the detection pipeline output.


def features_to_geojson(features: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    return {"type": "FeatureCollection", "features": list(features)}


def write_geojson(path: Path, features: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(features_to_geojson(features), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
